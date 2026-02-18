"""
Fused DG solver for lid-driven cavity with artificial compressibility.

Single Numba kernel computes full RHS (convective + viscous) in one pass,
eliminating Python call overhead from lambdas, face extraction, and BC functions.

For 288 elements P2: targets <100µs/step (vs 530µs with component approach).
"""

import numpy as np
from numba import njit, prange
import time


@njit(cache=True)
def _ac_max_wavespeed(q, beta, K, Np):
    """Global max wavespeed for CFL."""
    b2 = beta * beta
    wmax = 0.0
    for k in range(K):
        for i in range(Np):
            u = q[k, i, 0]
            v = q[k, i, 1]
            s2 = u * u + v * v
            w = np.sqrt(s2) + np.sqrt(s2 + b2)
            if w > wmax:
                wmax = w
    return wmax


@njit(cache=True, parallel=True)
def _cavity_rhs(q, Dr, Ds, rx, ry, sx, sy, LIFT, Fmask_flat,
                nx_f, ny_f, Fscale, vmapP_k, vmapP_n,
                bc_per_node, face_x, beta, nu, sigma_ip, lid_vel,
                K, Np, Nfp, n_fp):
    """
    Fused convective + viscous RHS for AC lid-driven cavity.

    Computes in one parallel loop per element:
    1. Physical fluxes F(q), G(q)
    2. Face extraction with BC ghost values
    3. Rusanov numerical flux
    4. Volume divergence + LIFT surface correction (convective)
    5. SIPG viscous operator for u,v components
    """
    rhs = np.empty((K, Np, 3), dtype=np.float64)
    b2 = beta * beta

    for k in prange(K):
        rx_k = rx[k, 0]
        ry_k = ry[k, 0]
        sx_k = sx[k, 0]
        sy_k = sy[k, 0]

        # ---- Physical fluxes at volume nodes ----
        Fx = np.empty((Np, 3), dtype=np.float64)
        Fy = np.empty((Np, 3), dtype=np.float64)
        for i in range(Np):
            u = q[k, i, 0]
            v = q[k, i, 1]
            p = q[k, i, 2]
            Fx[i, 0] = u * u + p
            Fx[i, 1] = u * v
            Fx[i, 2] = b2 * u
            Fy[i, 0] = u * v
            Fy[i, 1] = v * v + p
            Fy[i, 2] = b2 * v

        # ---- Face extraction + BCs + Rusanov flux ----
        f_num = np.empty((n_fp, 3), dtype=np.float64)
        # Also store face values for viscous operator
        u_face_int = np.empty((n_fp, 2), dtype=np.float64)
        u_face_ext = np.empty((n_fp, 2), dtype=np.float64)

        for fpt in range(n_fp):
            vol_idx = Fmask_flat[fpt]
            nx_ki = nx_f[k, fpt]
            ny_ki = ny_f[k, fpt]

            # Interior values
            ui = q[k, vol_idx, 0]
            vi = q[k, vol_idx, 1]
            pi = q[k, vol_idx, 2]

            # Exterior values (neighbor or ghost)
            bc = bc_per_node[k, fpt]
            if bc == 0:
                pk = vmapP_k[k, fpt]
                pn = vmapP_n[k, fpt]
                ue = q[pk, pn, 0]
                ve = q[pk, pn, 1]
                pe = q[pk, pn, 2]
            elif bc == 1:
                # Wall: no-slip
                ue = -ui
                ve = -vi
                pe = pi
            else:
                # Lid: regularized
                x = face_x[k, fpt]
                u_lid = lid_vel * 16.0 * x * x * (1.0 - x) * (1.0 - x)
                ue = 2.0 * u_lid - ui
                ve = -vi
                pe = pi

            # Store for viscous
            u_face_int[fpt, 0] = ui
            u_face_int[fpt, 1] = vi
            u_face_ext[fpt, 0] = ue
            u_face_ext[fpt, 1] = ve

            # Rusanov flux
            Vn_i = ui * nx_ki + vi * ny_ki
            Vn_e = ue * nx_ki + ve * ny_ki
            lam_i = abs(Vn_i) + np.sqrt(Vn_i * Vn_i + b2)
            lam_e = abs(Vn_e) + np.sqrt(Vn_e * Vn_e + b2)
            lam = max(lam_i, lam_e)

            fni0 = (ui * ui + pi) * nx_ki + (ui * vi) * ny_ki
            fni1 = (ui * vi) * nx_ki + (vi * vi + pi) * ny_ki
            fni2 = b2 * (ui * nx_ki + vi * ny_ki)
            fne0 = (ue * ue + pe) * nx_ki + (ue * ve) * ny_ki
            fne1 = (ue * ve) * nx_ki + (ve * ve + pe) * ny_ki
            fne2 = b2 * (ue * nx_ki + ve * ny_ki)

            f_num[fpt, 0] = 0.5 * (fni0 + fne0) - 0.5 * lam * (ue - ui)
            f_num[fpt, 1] = 0.5 * (fni1 + fne1) - 0.5 * lam * (ve - vi)
            f_num[fpt, 2] = 0.5 * (fni2 + fne2) - 0.5 * lam * (pe - pi)

        # ---- Convective RHS: volume + surface ----
        for v in range(3):
            vol = np.empty(Np, dtype=np.float64)
            for i in range(Np):
                dfdr = 0.0
                dfds = 0.0
                dgdr = 0.0
                dgds = 0.0
                for j in range(Np):
                    dfdr += Dr[i, j] * Fx[j, v]
                    dfds += Ds[i, j] * Fx[j, v]
                    dgdr += Dr[i, j] * Fy[j, v]
                    dgds += Ds[i, j] * Fy[j, v]
                dfdx = rx_k * dfdr + sx_k * dfds
                dgdy = ry_k * dgdr + sy_k * dgds
                vol[i] = -(dfdx + dgdy)

            du_surf = np.empty(n_fp, dtype=np.float64)
            for fpt in range(n_fp):
                vol_idx = Fmask_flat[fpt]
                f_phys_n = (Fx[vol_idx, v] * nx_f[k, fpt]
                            + Fy[vol_idx, v] * ny_f[k, fpt])
                du_surf[fpt] = Fscale[k, fpt] * (f_phys_n - f_num[fpt, v])

            for i in range(Np):
                val = vol[i]
                for fpt in range(n_fp):
                    val += LIFT[i, fpt] * du_surf[fpt]
                rhs[k, i, v] = val

        # ---- SIPG viscous RHS for u,v only ----
        for v_idx in range(2):
            # Volume gradient
            ux = np.empty(Np, dtype=np.float64)
            uy = np.empty(Np, dtype=np.float64)
            for i in range(Np):
                dudr = 0.0
                duds = 0.0
                for j in range(Np):
                    dudr += Dr[i, j] * q[k, j, v_idx]
                    duds += Ds[i, j] * q[k, j, v_idx]
                ux[i] = rx_k * dudr + sx_k * duds
                uy[i] = ry_k * dudr + sy_k * duds

            # Volume Laplacian
            vol_v = np.empty(Np, dtype=np.float64)
            for i in range(Np):
                d_fx_dr = 0.0
                d_fx_ds = 0.0
                d_fy_dr = 0.0
                d_fy_ds = 0.0
                for j in range(Np):
                    d_fx_dr -= Dr[i, j] * nu * ux[j]
                    d_fx_ds -= Ds[i, j] * nu * ux[j]
                    d_fy_dr -= Dr[i, j] * nu * uy[j]
                    d_fy_ds -= Ds[i, j] * nu * uy[j]
                div_fx = rx_k * d_fx_dr + sx_k * d_fx_ds
                div_fy = ry_k * d_fy_dr + sy_k * d_fy_ds
                vol_v[i] = -(div_fx + div_fy)

            # Surface correction — use neighbor gradient from pre-stored arrays
            surf_v = np.empty(n_fp, dtype=np.float64)
            for fpt in range(n_fp):
                vol_idx = Fmask_flat[fpt]
                grad_n_int = ux[vol_idx] * nx_f[k, fpt] + uy[vol_idx] * ny_f[k, fpt]

                bc = bc_per_node[k, fpt]
                if bc == 0:
                    pk = vmapP_k[k, fpt]
                    pn = vmapP_n[k, fpt]
                    # Compute neighbor gradient at the matching node
                    rx_e = rx[pk, 0]
                    ry_e = ry[pk, 0]
                    sx_e = sx[pk, 0]
                    sy_e = sy[pk, 0]
                    dudr_e = 0.0
                    duds_e = 0.0
                    for j in range(Np):
                        dudr_e += Dr[pn, j] * q[pk, j, v_idx]
                        duds_e += Ds[pn, j] * q[pk, j, v_idx]
                    ux_ext = rx_e * dudr_e + sx_e * duds_e
                    uy_ext = ry_e * dudr_e + sy_e * duds_e
                    grad_n_ext = ux_ext * nx_f[k, fpt] + uy_ext * ny_f[k, fpt]
                else:
                    grad_n_ext = -grad_n_int

                u_int_v = u_face_int[fpt, v_idx]
                u_ext_v = u_face_ext[fpt, v_idx]
                jump_u = u_int_v - u_ext_v
                avg_grad_n = 0.5 * (grad_n_int + grad_n_ext)

                surf_v[fpt] = Fscale[k, fpt] * (
                    nu * (avg_grad_n - grad_n_int) - sigma_ip * jump_u
                )

            for i in range(Np):
                val = vol_v[i]
                for fpt in range(n_fp):
                    val += LIFT[i, fpt] * surf_v[fpt]
                rhs[k, i, v_idx] += val

    return rhs


def run_cavity(nx=12, p=2, Re=100.0, beta=1.0, cfl=0.008,
               t_final=20.0, lid_velocity=1.0, print_every=10000):
    """Run lid-driven cavity with fused kernel."""
    from .reference_triangle import RefTriangle
    from .mesh.triangle_mesh import (
        TriangleMesh, build_face_connectivity,
        geometric_factors, face_normals,
    )

    K = 2 * nx * nx
    Np_val = (p + 1) * (p + 2) // 2
    print(f"Cavity Re={Re}, β={beta}, nx={nx}, P{p}")
    print(f"  {K} triangles, {Np_val} nodes/elem, {K*Np_val*3} DOFs")

    mesh = TriangleMesh.rectangle((0, 1), (0, 1), nx, nx)
    ref = RefTriangle(p)
    Np = ref.Np
    Nfp = ref.Nfp
    n_fp = 3 * Nfp

    EToE, EToF = build_face_connectivity(mesh)
    x, y, rx, ry, sx, sy, J = geometric_factors(mesh, ref, p)
    nx_f, ny_f, sJ, Fscale = face_normals(mesh, ref, x, y, J)
    Fmask_flat = np.concatenate(ref.Fmask).astype(np.int64)
    Fmask = ref.Fmask
    Dr = np.ascontiguousarray(ref.Dr)
    Ds = np.ascontiguousarray(ref.Ds)
    LIFT = np.ascontiguousarray(ref.LIFT)

    # Build maps
    vmapP_k = np.zeros((K, n_fp), dtype=np.int64)
    vmapP_n = np.zeros((K, n_fp), dtype=np.int64)
    for k in range(K):
        for f in range(3):
            k2, f2 = EToE[k, f], EToF[k, f]
            for i in range(Nfp):
                idx = f * Nfp + i
                if k2 == k and f2 == f:
                    vmapP_k[k, idx] = k
                    vmapP_n[k, idx] = Fmask[f][i]
                else:
                    vmapP_k[k, idx] = k2
                    vmapP_n[k, idx] = Fmask[f2][Nfp - 1 - i]

    # BC tags
    bc_tags = np.zeros((K, 3), dtype=np.int32)
    for k in range(K):
        for f in range(3):
            if EToE[k, f] == k and EToF[k, f] == f:
                face_y = np.mean(y[k, Fmask[f]])
                bc_tags[k, f] = 2 if face_y > 1.0 - 1e-8 else 1

    bc_per_node = np.zeros((K, n_fp), dtype=np.int32)
    for f in range(3):
        bc_per_node[:, f * Nfp:(f + 1) * Nfp] = bc_tags[:, f:f + 1]

    face_x = np.zeros((K, n_fp))
    for i in range(n_fp):
        face_x[:, i] = x[:, Fmask_flat[i]]

    nu = 1.0 / Re
    C_ip = 4.0
    sigma_ip = C_ip * (p + 1) * (p + 2) * nu / 2.0

    # h_min for CFL
    h_min = np.min(2.0 * np.abs(J[:, 0]) /
                   np.max(sJ.reshape(K, 3, Nfp), axis=2).max(axis=1))

    # Initial condition
    q = np.zeros((K, Np, 3))

    # Warmup
    _cavity_rhs(q, Dr, Ds, rx, ry, sx, sy, LIFT, Fmask_flat,
                nx_f, ny_f, Fscale, vmapP_k, vmapP_n,
                bc_per_node, face_x, beta, nu, sigma_ip, lid_velocity,
                K, Np, Nfp, n_fp)

    print("  Solving...", flush=True)
    t0_wall = time.perf_counter()
    t_sim = 0.0
    step = 0
    residuals = []

    def rhs_func(q_in, t):
        return _cavity_rhs(q_in, Dr, Ds, rx, ry, sx, sy, LIFT, Fmask_flat,
                           nx_f, ny_f, Fscale, vmapP_k, vmapP_n,
                           bc_per_node, face_x, beta, nu, sigma_ip, lid_velocity,
                           K, Np, Nfp, n_fp)

    while t_sim < t_final - 1e-14 and step < 2_000_000:
        a_max = _ac_max_wavespeed(q, beta, K, Np) + 1e-14
        dt = min(cfl * h_min / ((2 * p + 1) * a_max), t_final - t_sim)

        # SSP-RK3 inlined to avoid function call overhead
        rhs1 = rhs_func(q, t_sim)
        u1 = q + dt * rhs1
        rhs2 = rhs_func(u1, t_sim + dt)
        u2 = 0.75 * q + 0.25 * (u1 + dt * rhs2)
        rhs3 = rhs_func(u2, t_sim + 0.5 * dt)
        q = (1.0 / 3.0) * q + (2.0 / 3.0) * (u2 + dt * rhs3)

        t_sim += dt
        step += 1

        if step % print_every == 0:
            res = np.max(np.abs(rhs3))
            residuals.append((t_sim, res))
            elapsed = time.perf_counter() - t0_wall
            u_max = np.max(np.abs(q[:, :, 0]))
            v_max = np.max(np.abs(q[:, :, 1]))
            rate = step / elapsed
            print(f"  step={step:7d}  t={t_sim:7.3f}  res={res:.4e}  "
                  f"|u|={u_max:.4f}  |v|={v_max:.4f}  "
                  f"wall={elapsed:.1f}s  {rate:.0f} step/s", flush=True)

            if np.any(np.isnan(q)) or res > 1e6:
                raise RuntimeError("Blowup")

    elapsed = time.perf_counter() - t0_wall
    print(f"\n  Done: {step} steps, t={t_sim:.3f}, {elapsed:.1f}s "
          f"({step/elapsed:.0f} step/s)")
    if residuals:
        print(f"  Final residual: {residuals[-1][1]:.4e}")

    return q, x, y, ref, J, residuals
