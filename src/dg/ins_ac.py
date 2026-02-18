"""
Incompressible Navier-Stokes via Artificial Compressibility (AC) method.

State vector: q = [u, v, p] (velocity components + pressure)

The AC-perturbed incompressible NS:
    ∂u/∂t + ∂(u²+p)/∂x + ∂(uv)/∂y = (1/Re)(∂²u/∂x² + ∂²u/∂y²)
    ∂v/∂t + ∂(uv)/∂x + ∂(v²+p)/∂y = (1/Re)(∂²v/∂x² + ∂²v/∂y²)
    (1/β²)∂p/∂t + ∂u/∂x + ∂v/∂y = 0

At steady state, ∂p/∂t → 0, recovering ∇·u = 0.

β is the artificial compressibility parameter (pseudo-sound speed).

Convective fluxes:
    F(q) = [u²+p, uv, β²u]     G(q) = [uv, v²+p, β²v]

Numerical flux: Rusanov with eigenvalue-based wavespeed.
Eigenvalues of dF/dq: u ± √(u² + β²), u
Eigenvalues of dG/dq: v ± √(v² + β²), v

References
----------
Bassi, Crivellini, Di Pietro & Rebay (2006), JCP 227(12).
Chorin (1967), JCP 2, pp. 12-26.
"""

import numpy as np
from numba import njit, prange


# =============================================================================
# Physical fluxes
# =============================================================================

@njit(cache=True, parallel=True)
def ac_flux_x(q, beta=1.0):
    """x-direction convective flux F(q). q: (K, Np, 3)."""
    K, Np, _ = q.shape
    Fx = np.empty((K, Np, 3), dtype=np.float64)
    b2 = beta * beta
    for k in prange(K):
        for i in range(Np):
            u = q[k, i, 0]
            v = q[k, i, 1]
            p = q[k, i, 2]
            Fx[k, i, 0] = u * u + p
            Fx[k, i, 1] = u * v
            Fx[k, i, 2] = b2 * u
    return Fx


@njit(cache=True, parallel=True)
def ac_flux_y(q, beta=1.0):
    """y-direction convective flux G(q). q: (K, Np, 3)."""
    K, Np, _ = q.shape
    Fy = np.empty((K, Np, 3), dtype=np.float64)
    b2 = beta * beta
    for k in prange(K):
        for i in range(Np):
            u = q[k, i, 0]
            v = q[k, i, 1]
            p = q[k, i, 2]
            Fy[k, i, 0] = u * v
            Fy[k, i, 1] = v * v + p
            Fy[k, i, 2] = b2 * v
    return Fy


# =============================================================================
# Numerical flux (Rusanov)
# =============================================================================

@njit(cache=True, parallel=True)
def ac_numerical_flux(q_int, q_ext, nx, ny, beta=1.0):
    """
    Rusanov numerical flux for the AC system.

    F̂·n = ½[(F_int + F_ext)·n] - ½λ_max(q_ext - q_int)

    λ_max = max(|V·n| + √((V·n)² + β²)) over int/ext states.
    """
    K, n_fp, nv = q_int.shape
    b2 = beta * beta
    f_num = np.empty((K, n_fp, nv), dtype=np.float64)

    for k in prange(K):
        for i in range(n_fp):
            ui = q_int[k, i, 0]
            vi = q_int[k, i, 1]
            pi = q_int[k, i, 2]
            ue = q_ext[k, i, 0]
            ve = q_ext[k, i, 1]
            pe = q_ext[k, i, 2]
            nx_ki = nx[k, i]
            ny_ki = ny[k, i]

            Vn_i = ui * nx_ki + vi * ny_ki
            Vn_e = ue * nx_ki + ve * ny_ki

            lam_i = abs(Vn_i) + np.sqrt(Vn_i * Vn_i + b2)
            lam_e = abs(Vn_e) + np.sqrt(Vn_e * Vn_e + b2)
            lam = max(lam_i, lam_e)

            # F·n interior
            fni0 = (ui * ui + pi) * nx_ki + (ui * vi) * ny_ki
            fni1 = (ui * vi) * nx_ki + (vi * vi + pi) * ny_ki
            fni2 = b2 * (ui * nx_ki + vi * ny_ki)
            # F·n exterior
            fne0 = (ue * ue + pe) * nx_ki + (ue * ve) * ny_ki
            fne1 = (ue * ve) * nx_ki + (ve * ve + pe) * ny_ki
            fne2 = b2 * (ue * nx_ki + ve * ny_ki)

            f_num[k, i, 0] = 0.5 * (fni0 + fne0) - 0.5 * lam * (ue - ui)
            f_num[k, i, 1] = 0.5 * (fni1 + fne1) - 0.5 * lam * (ve - vi)
            f_num[k, i, 2] = 0.5 * (fni2 + fne2) - 0.5 * lam * (pe - pi)

    return f_num


# =============================================================================
# Max wavespeed for CFL
# =============================================================================

@njit(cache=True)
def ac_max_wavespeed(q, beta=1.0):
    """Global maximum wavespeed for CFL computation."""
    K, Np, _ = q.shape
    b2 = beta * beta
    wmax = 0.0
    for k in range(K):
        for i in range(Np):
            u = q[k, i, 0]
            v = q[k, i, 1]
            s2 = u * u + v * v
            s = np.sqrt(s2)
            w = s + np.sqrt(s2 + b2)
            if w > wmax:
                wmax = w
    return wmax


# =============================================================================
# Boundary conditions
# =============================================================================

@njit(cache=True, parallel=True)
def cavity_bc(q_int, q_ext, bc_tags, face_nx, face_ny,
              lid_velocity=1.0, face_x=None):
    """
    Apply lid-driven cavity BCs (Numba-accelerated).

    bc_tags: (K, n_fp) int — 0=interior, 1=wall, 2=lid
    face_x: (K, n_fp) float — x-coords at face nodes (for regularized lid)
    """
    K, n_fp, nv = q_int.shape
    q_out = q_ext.copy()

    for k in prange(K):
        for i in range(n_fp):
            bc = bc_tags[k, i]
            if bc == 1:
                # Wall: no-slip
                q_out[k, i, 0] = -q_int[k, i, 0]
                q_out[k, i, 1] = -q_int[k, i, 1]
                q_out[k, i, 2] = q_int[k, i, 2]
            elif bc == 2:
                # Lid: regularized moving wall
                if face_x is not None:
                    x = face_x[k, i]
                    u_lid = lid_velocity * 16.0 * x * x * (1.0 - x) * (1.0 - x)
                else:
                    u_lid = lid_velocity
                q_out[k, i, 0] = 2.0 * u_lid - q_int[k, i, 0]
                q_out[k, i, 1] = -q_int[k, i, 1]
                q_out[k, i, 2] = q_int[k, i, 2]

    return q_out


# =============================================================================
# Simple viscous operator (BR1 / averaging)
# =============================================================================

def make_viscous_rhs(solver, Re: float):
    """
    Create a viscous RHS function using the BR1 (averaging) approach.

    For the velocity components only (not pressure equation).

    ∇²u ≈ ∇·(∇u) computed via:
    1. Compute gradient σ = ∇u using DG differentiation
    2. Add face correction via LIFT
    3. Take divergence of σ

    This is a simplified viscous operator. For production use BR2.
    """
    Dr = solver.Dr
    Ds = solver.Ds
    LIFT = solver.LIFT
    rx, ry, sx, sy = solver.rx, solver.ry, solver.sx, solver.sy
    Fmask_flat = solver.Fmask_flat
    nx_arr, ny_arr = solver.nx, solver.ny
    Fscale = solver.Fscale
    K, Np, Nfp = solver.K, solver.Np, solver.Nfp
    nu = 1.0 / Re

    def viscous_rhs_func(q, t):
        rhs = np.zeros_like(q)

        for v_idx in range(2):  # Only u and v, not pressure
            u = q[:, :, v_idx]

            # Compute gradient on reference element
            dudr = u @ Dr.T  # (K, Np)
            duds = u @ Ds.T

            # Map to physical gradient
            dudx = rx[:, 0:1] * dudr + sx[:, 0:1] * duds
            dudy = ry[:, 0:1] * dudr + sy[:, 0:1] * duds

            # Compute divergence of gradient (Laplacian)
            d2udr = dudx @ Dr.T
            d2uds = dudx @ Ds.T
            lap_x = rx[:, 0:1] * d2udr + sx[:, 0:1] * d2uds

            d2vdr = dudy @ Dr.T
            d2vds = dudy @ Ds.T
            lap_y = ry[:, 0:1] * d2vdr + sy[:, 0:1] * d2vds

            lap = lap_x + lap_y

            # Face correction (penalty for jumps)
            u_face_int = np.zeros((K, 3 * Nfp))
            u_face_ext = np.zeros((K, 3 * Nfp))
            for i in range(3 * Nfp):
                vol_idx = Fmask_flat[i]
                u_face_int[:, i] = u[:, vol_idx]
                pk = solver.vmapP_k[:, i]
                pn = solver.vmapP_n[:, i]
                u_face_ext[:, i] = u[pk, pn]

            # Apply BCs to exterior values
            if solver.bc_tags is not None and solver.bc_func is not None:
                bc_per_node = np.zeros((K, 3 * Nfp), dtype=np.int32)
                for f in range(3):
                    bc_per_node[:, f * Nfp:(f + 1) * Nfp] = solver.bc_tags[:, f:f + 1]

                # For viscous BCs, mirror velocity at walls
                wall = bc_per_node == 1
                lid = bc_per_node == 2

                if v_idx == 0:  # u component
                    u_face_ext[wall] = -u_face_int[wall]
                    u_face_ext[lid] = 2.0 - u_face_int[lid]  # lid velocity = 1
                else:  # v component
                    u_face_ext[wall] = -u_face_int[wall]
                    u_face_ext[lid] = -u_face_int[lid]

            # Jump penalty: τ/h * [[u]]
            jump = u_face_ext - u_face_int
            # Average gradient at faces
            dudx_face = np.zeros((K, 3 * Nfp))
            dudy_face = np.zeros((K, 3 * Nfp))
            for i in range(3 * Nfp):
                vol_idx = Fmask_flat[i]
                dudx_face[:, i] = dudx[:, vol_idx]
                dudy_face[:, i] = dudy[:, vol_idx]

            # Penalty parameter
            tau = (solver.p + 1)**2 / np.min(2 * np.abs(solver.J[:, 0]))

            # Surface correction
            surf_corr = Fscale * (0.5 * (dudx_face * nx_arr + dudy_face * ny_arr)
                                   + tau * jump)
            penalty = (LIFT @ surf_corr.T).T  # This is wrong shape-wise, need per-element

            # Per-element LIFT
            penalty2 = np.zeros((K, Np))
            for k in range(K):
                penalty2[k] = LIFT @ surf_corr[k]

            rhs[:, :, v_idx] = nu * (lap + penalty2)

        return rhs

    return viscous_rhs_func
