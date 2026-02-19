"""
DG solvers for incompressible Navier-Stokes: Projection and Mixed methods.

Both use the same DG infrastructure (SIPG viscous, Rusanov convective)
but differ in how they handle the pressure-velocity coupling:

1. Penalty-Projection: Explicit predictor + AC pressure relaxation
   - No global linear solve needed
   - Fully explicit, DG-native

2. Mixed DG: Equal-order velocity-pressure with Brezzi-Pitkäranta stabilization
   - Uses AC pseudo-timestepping for pressure (avoids fragile Poisson solve)
   - Pressure stabilization prevents spurious modes

Both achieve divergence-free velocity at steady state.

References
----------
Guermond & Minev (2006), penalty-projection methods.
Brezzi & Pitkäranta (1984), pressure stabilization.
"""

import numpy as np
from numba import njit
import time


# =========================================================================
# Shared DG operators
# =========================================================================

@njit(cache=True)
def _compute_divergence(u, v, Dr, Ds, rx, ry, sx, sy, K, Np):
    """∇·(u,v) = ∂u/∂x + ∂v/∂y."""
    div = np.empty((K, Np), dtype=np.float64)
    for k in range(K):
        rx_k, ry_k = rx[k, 0], ry[k, 0]
        sx_k, sy_k = sx[k, 0], sy[k, 0]
        for i in range(Np):
            dudr = 0.0; duds = 0.0; dvdr = 0.0; dvds = 0.0
            for j in range(Np):
                dudr += Dr[i,j]*u[k,j]; duds += Ds[i,j]*u[k,j]
                dvdr += Dr[i,j]*v[k,j]; dvds += Ds[i,j]*v[k,j]
            div[k,i] = rx_k*dudr + sx_k*duds + ry_k*dvdr + sy_k*dvds
    return div


@njit(cache=True)
def _compute_gradient(phi, Dr, Ds, rx, ry, sx, sy, K, Np):
    """∇φ = (∂φ/∂x, ∂φ/∂y)."""
    dpdx = np.empty((K, Np), dtype=np.float64)
    dpdy = np.empty((K, Np), dtype=np.float64)
    for k in range(K):
        rx_k, ry_k = rx[k, 0], ry[k, 0]
        sx_k, sy_k = sx[k, 0], sy[k, 0]
        for i in range(Np):
            dpdr = 0.0; dpds = 0.0
            for j in range(Np):
                dpdr += Dr[i,j]*phi[k,j]; dpds += Ds[i,j]*phi[k,j]
            dpdx[k,i] = rx_k*dpdr + sx_k*dpds
            dpdy[k,i] = ry_k*dpdr + sy_k*dpds
    return dpdx, dpdy


@njit(cache=True)
def _projection_rhs(q_u, q_v, q_p, Dr, Ds, rx, ry, sx, sy,
                    LIFT, Fmask_flat, nx_f, ny_f, Fscale,
                    vmapP_k, vmapP_n, bc_per_node,
                    nu, lid_vel, sigma_ip, beta_penalty,
                    K, Np, Nfp):
    """
    Full RHS for penalty-projection INS.

    Momentum: du/dt = -∇·(u⊗u) - ∇p + ν∇²u
    Pressure: dp/dt = -β²∇·u  (AC relaxation)

    This is essentially the AC method but rewritten as a projection:
    the pressure equation relaxes toward divergence-free.

    Returns (rhs_u, rhs_v, rhs_p).
    """
    n_fp = 3 * Nfp
    rhs_u = np.empty((K, Np), dtype=np.float64)
    rhs_v = np.empty((K, Np), dtype=np.float64)
    rhs_p = np.empty((K, Np), dtype=np.float64)
    b2 = beta_penalty * beta_penalty

    for k in range(K):
        rx_k, ry_k = rx[k, 0], ry[k, 0]
        sx_k, sy_k = sx[k, 0], sy[k, 0]

        # ---- Volume fluxes ----
        # Momentum: F_u = (u²+p, uv), F_v = (uv, v²+p)
        # Continuity: F_p = (β²u, β²v)
        Fx = np.empty((Np, 3), dtype=np.float64)
        Fy = np.empty((Np, 3), dtype=np.float64)
        for i in range(Np):
            u = q_u[k, i]; v = q_v[k, i]; p = q_p[k, i]
            Fx[i, 0] = u*u + p;  Fy[i, 0] = u*v
            Fx[i, 1] = u*v;      Fy[i, 1] = v*v + p
            Fx[i, 2] = b2*u;     Fy[i, 2] = b2*v

        # ---- Face numerical fluxes (Rusanov) ----
        fn = np.empty((n_fp, 3), dtype=np.float64)
        u_face_int = np.empty((n_fp, 2), dtype=np.float64)
        u_face_ext = np.empty((n_fp, 2), dtype=np.float64)

        for fpt in range(n_fp):
            vol_idx = Fmask_flat[fpt]
            ui = q_u[k, vol_idx]; vi = q_v[k, vol_idx]; pi = q_p[k, vol_idx]
            nxi = nx_f[k, fpt]; nyi = ny_f[k, fpt]

            bc = bc_per_node[k, fpt]
            if bc == 0:
                pk = vmapP_k[k, fpt]; pn = vmapP_n[k, fpt]
                ue = q_u[pk, pn]; ve = q_v[pk, pn]; pe = q_p[pk, pn]
            elif bc == 1:
                ue = -ui; ve = -vi; pe = pi
            else:
                ue = 2.0*lid_vel - ui; ve = -vi; pe = pi

            u_face_int[fpt, 0] = ui; u_face_int[fpt, 1] = vi
            u_face_ext[fpt, 0] = ue; u_face_ext[fpt, 1] = ve

            Vn_i = ui*nxi + vi*nyi; Vn_e = ue*nxi + ve*nyi
            lam_i = abs(Vn_i) + np.sqrt(Vn_i*Vn_i + b2)
            lam_e = abs(Vn_e) + np.sqrt(Vn_e*Vn_e + b2)
            lam = max(lam_i, lam_e)

            for c in range(3):
                qi = (ui if c==0 else (vi if c==1 else pi))
                qe = (ue if c==0 else (ve if c==1 else pe))
                fxi = Fx[vol_idx, c]; fyi = Fy[vol_idx, c]
                fni_c = fxi*nxi + fyi*nyi
                # Exterior flux
                if c == 0: fxe = ue*ue+pe; fye = ue*ve
                elif c == 1: fxe = ue*ve; fye = ve*ve+pe
                else: fxe = b2*ue; fye = b2*ve
                fne_c = fxe*nxi + fye*nyi
                fn[fpt, c] = 0.5*(fni_c + fne_c) - 0.5*lam*(qe - qi)

        # ---- Volume divergence + surface for all 3 equations ----
        for c in range(3):
            for i in range(Np):
                dfdr = 0.0; dfds = 0.0; dgdr = 0.0; dgds = 0.0
                for j in range(Np):
                    dfdr += Dr[i,j]*Fx[j,c]; dfds += Ds[i,j]*Fx[j,c]
                    dgdr += Dr[i,j]*Fy[j,c]; dgds += Ds[i,j]*Fy[j,c]
                val = -(rx_k*dfdr + sx_k*dfds + ry_k*dgdr + sy_k*dgds)
                for fpt in range(n_fp):
                    vidx = Fmask_flat[fpt]
                    f_phys_n = Fx[vidx,c]*nx_f[k,fpt] + Fy[vidx,c]*ny_f[k,fpt]
                    val += LIFT[i,fpt] * Fscale[k,fpt] * (f_phys_n - fn[fpt,c])
                if c == 0: rhs_u[k,i] = val
                elif c == 1: rhs_v[k,i] = val
                else: rhs_p[k,i] = val

        # ---- SIPG viscous for u,v ----
        for v_idx in range(2):
            field = q_u if v_idx == 0 else q_v
            ux = np.empty(Np, dtype=np.float64)
            uy = np.empty(Np, dtype=np.float64)
            for i in range(Np):
                dudr = 0.0; duds = 0.0
                for j in range(Np):
                    dudr += Dr[i,j]*field[k,j]; duds += Ds[i,j]*field[k,j]
                ux[i] = rx_k*dudr + sx_k*duds
                uy[i] = ry_k*dudr + sy_k*duds

            vol_v = np.empty(Np, dtype=np.float64)
            for i in range(Np):
                d2x_dr = 0.0; d2x_ds = 0.0; d2y_dr = 0.0; d2y_ds = 0.0
                for j in range(Np):
                    d2x_dr -= Dr[i,j]*nu*ux[j]; d2x_ds -= Ds[i,j]*nu*ux[j]
                    d2y_dr -= Dr[i,j]*nu*uy[j]; d2y_ds -= Ds[i,j]*nu*uy[j]
                vol_v[i] = -(rx_k*d2x_dr + sx_k*d2x_ds + ry_k*d2y_dr + sy_k*d2y_ds)

            surf_v = np.empty(n_fp, dtype=np.float64)
            for fpt in range(n_fp):
                vol_idx = Fmask_flat[fpt]
                grad_n_int = ux[vol_idx]*nx_f[k,fpt] + uy[vol_idx]*ny_f[k,fpt]
                bc = bc_per_node[k, fpt]
                if bc == 0:
                    pk2 = vmapP_k[k,fpt]; pn2 = vmapP_n[k,fpt]
                    rx_e = rx[pk2,0]; ry_e = ry[pk2,0]; sx_e = sx[pk2,0]; sy_e = sy[pk2,0]
                    dudr_e = 0.0; duds_e = 0.0
                    f2 = q_u if v_idx==0 else q_v
                    for j in range(Np):
                        dudr_e += Dr[pn2,j]*f2[pk2,j]; duds_e += Ds[pn2,j]*f2[pk2,j]
                    ux_e = rx_e*dudr_e + sx_e*duds_e; uy_e = ry_e*dudr_e + sy_e*duds_e
                    grad_n_ext = ux_e*nx_f[k,fpt] + uy_e*ny_f[k,fpt]
                else:
                    grad_n_ext = -grad_n_int

                u_int_v = u_face_int[fpt, v_idx]
                u_ext_v = u_face_ext[fpt, v_idx]
                jump_u = u_int_v - u_ext_v
                avg_grad_n = 0.5*(grad_n_int + grad_n_ext)
                surf_v[fpt] = Fscale[k,fpt]*(nu*(avg_grad_n - grad_n_int) - sigma_ip*jump_u)

            for i in range(Np):
                val = vol_v[i]
                for fpt in range(n_fp):
                    val += LIFT[i,fpt]*surf_v[fpt]
                if v_idx == 0: rhs_u[k,i] += val
                else: rhs_v[k,i] += val

    return rhs_u, rhs_v, rhs_p


def _setup_mesh(nx, p):
    """Common mesh setup. Returns dict of all arrays."""
    from .reference_triangle import RefTriangle
    from .mesh.triangle_mesh import (
        TriangleMesh, build_face_connectivity,
        geometric_factors, face_normals,
    )

    K = 2 * nx * nx
    mesh = TriangleMesh.rectangle((0, 1), (0, 1), nx, nx)
    ref = RefTriangle(p)
    Np = ref.Np; Nfp = ref.Nfp; n_fp = 3 * Nfp

    EToE, EToF = build_face_connectivity(mesh)
    x, y, rx, ry, sx, sy, J = geometric_factors(mesh, ref, p)
    nx_f, ny_f, sJ, Fscale = face_normals(mesh, ref, x, y, J)
    Fmask_flat = np.concatenate(ref.Fmask).astype(np.int64)
    Fmask = ref.Fmask
    Dr = np.ascontiguousarray(ref.Dr)
    Ds = np.ascontiguousarray(ref.Ds)
    LIFT = np.ascontiguousarray(ref.LIFT)

    vmapP_k = np.zeros((K, n_fp), dtype=np.int64)
    vmapP_n = np.zeros((K, n_fp), dtype=np.int64)
    for k in range(K):
        for f in range(3):
            k2, f2 = EToE[k, f], EToF[k, f]
            for i in range(Nfp):
                idx = f * Nfp + i
                if k2 == k and f2 == f:
                    vmapP_k[k, idx] = k; vmapP_n[k, idx] = Fmask[f][i]
                else:
                    vmapP_k[k, idx] = k2; vmapP_n[k, idx] = Fmask[f2][Nfp-1-i]

    bc_per_node = np.zeros((K, n_fp), dtype=np.int32)
    for k in range(K):
        for f in range(3):
            if EToE[k, f] == k and EToF[k, f] == f:
                face_y = np.mean(y[k, Fmask[f]])
                bc_per_node[k, f*Nfp:(f+1)*Nfp] = 2 if face_y > 1.0 - 1e-8 else 1

    h_min = np.min(2.0 * np.abs(J[:, 0]) /
                   np.max(sJ.reshape(K, 3, Nfp), axis=2).max(axis=1))

    return dict(K=K, Np=Np, Nfp=Nfp, n_fp=n_fp, x=x, y=y, J=J,
                rx=rx, ry=ry, sx=sx, sy=sy,
                nx_f=nx_f, ny_f=ny_f, Fscale=Fscale,
                Dr=Dr, Ds=Ds, LIFT=LIFT, Fmask_flat=Fmask_flat,
                vmapP_k=vmapP_k, vmapP_n=vmapP_n,
                bc_per_node=bc_per_node, h_min=h_min, ref=ref, sJ=sJ)


def run_projection_cavity(nx=12, p=2, Re=100.0, beta=1.0, cfl=0.15,
                          t_final=30.0, lid_velocity=1.0, print_every=5000):
    """
    Lid-driven cavity via penalty-projection (AC-based) method.

    This IS essentially the AC method, but framed as a projection:
    - Momentum includes explicit pressure gradient
    - Pressure relaxes via dp/dt = -β²∇·u
    - At steady state: ∇·u → 0 (divergence-free)

    The key difference from the `cavity_solver.py` AC solver is that this
    separates the operators cleanly (convective, pressure, viscous).
    """
    m = _setup_mesh(nx, p)
    K, Np, Nfp = m['K'], m['Np'], m['Nfp']
    nu = 1.0 / Re
    sigma_ip = 4.0 * (p+1) * (p+2) * nu / 2.0

    print(f"Projection Cavity Re={Re}, β={beta}, nx={nx}, P{p}")
    print(f"  {K} triangles, {Np} nodes/elem, {K*Np*3} DOFs")

    # Warmup
    u_d = np.zeros((K, Np)); v_d = np.zeros((K, Np)); p_d = np.zeros((K, Np))
    _projection_rhs(u_d, v_d, p_d, m['Dr'], m['Ds'], m['rx'], m['ry'], m['sx'], m['sy'],
                    m['LIFT'], m['Fmask_flat'], m['nx_f'], m['ny_f'], m['Fscale'],
                    m['vmapP_k'], m['vmapP_n'], m['bc_per_node'],
                    nu, lid_velocity, sigma_ip, beta, K, Np, Nfp)

    u = np.zeros((K, Np)); v = np.zeros((K, Np)); pressure = np.zeros((K, Np))

    t0_wall = time.perf_counter()
    t_sim = 0.0; step = 0
    residuals = []

    print("  Solving...", flush=True)

    while t_sim < t_final - 1e-14 and step < 2_000_000:
        # CFL with AC wavespeed
        u_max_arr = np.abs(u) + np.abs(v)
        s2 = u_max_arr ** 2
        a_max = np.max(np.sqrt(s2) + np.sqrt(s2 + beta*beta)) + 1e-14
        dt = min(cfl * m['h_min'] / ((2*p+1) * a_max), t_final - t_sim)

        # SSP-RK3
        def rhs(qu, qv, qp):
            return _projection_rhs(qu, qv, qp, m['Dr'], m['Ds'],
                                    m['rx'], m['ry'], m['sx'], m['sy'],
                                    m['LIFT'], m['Fmask_flat'],
                                    m['nx_f'], m['ny_f'], m['Fscale'],
                                    m['vmapP_k'], m['vmapP_n'], m['bc_per_node'],
                                    nu, lid_velocity, sigma_ip, beta, K, Np, Nfp)

        ru1, rv1, rp1 = rhs(u, v, pressure)
        u1 = u + dt*ru1; v1 = v + dt*rv1; p1 = pressure + dt*rp1

        ru2, rv2, rp2 = rhs(u1, v1, p1)
        u2 = 0.75*u + 0.25*(u1 + dt*ru2)
        v2 = 0.75*v + 0.25*(v1 + dt*rv2)
        p2 = 0.75*pressure + 0.25*(p1 + dt*rp2)

        ru3, rv3, rp3 = rhs(u2, v2, p2)
        u = (1./3)*u + (2./3)*(u2 + dt*ru3)
        v = (1./3)*v + (2./3)*(v2 + dt*rv3)
        pressure = (1./3)*pressure + (2./3)*(p2 + dt*rp3)

        t_sim += dt; step += 1

        if step % print_every == 0:
            res = max(np.max(np.abs(ru3)), np.max(np.abs(rv3)))
            div_u = _compute_divergence(u, v, m['Dr'], m['Ds'],
                                         m['rx'], m['ry'], m['sx'], m['sy'], K, Np)
            max_div = np.max(np.abs(div_u))
            residuals.append((t_sim, res, max_div))
            elapsed = time.perf_counter() - t0_wall
            print(f"  step={step:6d}  t={t_sim:7.3f}  res={res:.4e}  "
                  f"|div|={max_div:.4e}  |u|={np.max(np.abs(u)):.4f}  |v|={np.max(np.abs(v)):.4f}  "
                  f"wall={elapsed:.1f}s  {step/elapsed:.0f} step/s", flush=True)

            if np.any(np.isnan(u)) or res > 1e6:
                raise RuntimeError("Blowup")

    elapsed = time.perf_counter() - t0_wall
    print(f"\n  Done: {step} steps, t={t_sim:.3f}, {elapsed:.1f}s ({step/elapsed:.0f} step/s)")

    q = np.zeros((K, Np, 3))
    q[:, :, 0] = u; q[:, :, 1] = v; q[:, :, 2] = pressure
    return q, m['x'], m['y'], m['ref'], m['J'], np.array(residuals) if residuals else np.zeros((0, 3))


def run_mixed_cavity(nx=12, p=2, Re=100.0, beta=1.0, cfl=0.15,
                     t_final=30.0, lid_velocity=1.0, print_every=5000,
                     gamma_p=0.01):
    """
    Lid-driven cavity via mixed DG formulation.

    Same as penalty-projection but with explicit pressure jump stabilization
    (Brezzi-Pitkäranta) to suppress spurious pressure modes in equal-order DG.

    The stabilization adds: γ_p * h² * penalty on pressure jumps at faces.
    This is applied as an additional damping term in the pressure equation.
    """
    m = _setup_mesh(nx, p)
    K, Np, Nfp = m['K'], m['Np'], m['Nfp']
    nu = 1.0 / Re
    sigma_ip = 4.0 * (p+1) * (p+2) * nu / 2.0

    print(f"Mixed DG Cavity Re={Re}, β={beta}, γ_p={gamma_p}, nx={nx}, P{p}")
    print(f"  {K} triangles, {Np} nodes/elem, {K*Np*3} DOFs")

    # Warmup
    u_d = np.zeros((K, Np))
    _projection_rhs(u_d, u_d, u_d, m['Dr'], m['Ds'], m['rx'], m['ry'], m['sx'], m['sy'],
                    m['LIFT'], m['Fmask_flat'], m['nx_f'], m['ny_f'], m['Fscale'],
                    m['vmapP_k'], m['vmapP_n'], m['bc_per_node'],
                    nu, lid_velocity, sigma_ip, beta, K, Np, Nfp)
    _pressure_stabilization(u_d, m['Fmask_flat'], m['nx_f'], m['ny_f'], m['Fscale'],
                            m['vmapP_k'], m['vmapP_n'], m['bc_per_node'],
                            m['LIFT'], gamma_p, m['h_min'], K, Np, Nfp)

    u = np.zeros((K, Np)); v = np.zeros((K, Np)); pressure = np.zeros((K, Np))

    t0_wall = time.perf_counter()
    t_sim = 0.0; step = 0
    residuals = []

    print("  Solving...", flush=True)

    while t_sim < t_final - 1e-14 and step < 2_000_000:
        u_max_arr = np.abs(u) + np.abs(v)
        s2 = u_max_arr ** 2
        a_max = np.max(np.sqrt(s2) + np.sqrt(s2 + beta*beta)) + 1e-14
        dt = min(cfl * m['h_min'] / ((2*p+1) * a_max), t_final - t_sim)

        def rhs(qu, qv, qp):
            ru, rv, rp = _projection_rhs(qu, qv, qp, m['Dr'], m['Ds'],
                                          m['rx'], m['ry'], m['sx'], m['sy'],
                                          m['LIFT'], m['Fmask_flat'],
                                          m['nx_f'], m['ny_f'], m['Fscale'],
                                          m['vmapP_k'], m['vmapP_n'], m['bc_per_node'],
                                          nu, lid_velocity, sigma_ip, beta, K, Np, Nfp)
            # Add pressure stabilization
            rp += _pressure_stabilization(qp, m['Fmask_flat'], m['nx_f'], m['ny_f'],
                                           m['Fscale'], m['vmapP_k'], m['vmapP_n'],
                                           m['bc_per_node'], m['LIFT'],
                                           gamma_p, m['h_min'], K, Np, Nfp)
            return ru, rv, rp

        ru1, rv1, rp1 = rhs(u, v, pressure)
        u1 = u + dt*ru1; v1 = v + dt*rv1; p1 = pressure + dt*rp1

        ru2, rv2, rp2 = rhs(u1, v1, p1)
        u2 = 0.75*u + 0.25*(u1 + dt*ru2)
        v2 = 0.75*v + 0.25*(v1 + dt*rv2)
        p2 = 0.75*pressure + 0.25*(p1 + dt*rp2)

        ru3, rv3, rp3 = rhs(u2, v2, p2)
        u = (1./3)*u + (2./3)*(u2 + dt*ru3)
        v = (1./3)*v + (2./3)*(v2 + dt*rv3)
        pressure = (1./3)*pressure + (2./3)*(p2 + dt*rp3)

        t_sim += dt; step += 1

        if step % print_every == 0:
            res = max(np.max(np.abs(ru3)), np.max(np.abs(rv3)))
            div_u = _compute_divergence(u, v, m['Dr'], m['Ds'],
                                         m['rx'], m['ry'], m['sx'], m['sy'], K, Np)
            max_div = np.max(np.abs(div_u))
            residuals.append((t_sim, res, max_div))
            elapsed = time.perf_counter() - t0_wall
            print(f"  step={step:6d}  t={t_sim:7.3f}  res={res:.4e}  "
                  f"|div|={max_div:.4e}  |u|={np.max(np.abs(u)):.4f}  |v|={np.max(np.abs(v)):.4f}  "
                  f"wall={elapsed:.1f}s  {step/elapsed:.0f} step/s", flush=True)

            if np.any(np.isnan(u)) or res > 1e6:
                raise RuntimeError("Blowup")

    elapsed = time.perf_counter() - t0_wall
    print(f"\n  Done: {step} steps, t={t_sim:.3f}, {elapsed:.1f}s ({step/elapsed:.0f} step/s)")

    q = np.zeros((K, Np, 3))
    q[:, :, 0] = u; q[:, :, 1] = v; q[:, :, 2] = pressure
    return q, m['x'], m['y'], m['ref'], m['J'], np.array(residuals) if residuals else np.zeros((0, 3))


@njit(cache=True)
def _pressure_stabilization(p_field, Fmask_flat, nx_f, ny_f, Fscale,
                            vmapP_k, vmapP_n, bc_per_node, LIFT,
                            gamma_p, h_min, K, Np, Nfp):
    """
    Brezzi-Pitkäranta pressure jump stabilization.

    Adds -γ_p * h² * LIFT(Fscale * [[p]]) to pressure equation.
    This penalizes pressure jumps at element interfaces, suppressing
    spurious pressure modes in equal-order velocity-pressure DG.
    """
    n_fp = 3 * Nfp
    rhs = np.empty((K, Np), dtype=np.float64)
    coeff = gamma_p * h_min * h_min

    for k in range(K):
        surf = np.empty(n_fp, dtype=np.float64)
        for fpt in range(n_fp):
            vol_idx = Fmask_flat[fpt]
            pi = p_field[k, vol_idx]
            bc = bc_per_node[k, fpt]
            if bc == 0:
                pk = vmapP_k[k, fpt]; pn = vmapP_n[k, fpt]
                pe = p_field[pk, pn]
            else:
                pe = pi  # Neumann: no jump at boundary
            surf[fpt] = -coeff * Fscale[k, fpt] * (pi - pe)

        for i in range(Np):
            val = 0.0
            for fpt in range(n_fp):
                val += LIFT[i, fpt] * surf[fpt]
            rhs[k, i] = val

    return rhs
