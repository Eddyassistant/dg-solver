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
from numba import njit


# =============================================================================
# Physical fluxes
# =============================================================================

def ac_flux_x(q: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """
    x-direction convective flux F(q).

    q: (K, Np, 3) with q[:,:,0]=u, q[:,:,1]=v, q[:,:,2]=p
    """
    u = q[:, :, 0]
    v = q[:, :, 1]
    p = q[:, :, 2]
    Fx = np.empty_like(q)
    Fx[:, :, 0] = u * u + p
    Fx[:, :, 1] = u * v
    Fx[:, :, 2] = beta**2 * u
    return Fx


def ac_flux_y(q: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """
    y-direction convective flux G(q).
    """
    u = q[:, :, 0]
    v = q[:, :, 1]
    p = q[:, :, 2]
    Fy = np.empty_like(q)
    Fy[:, :, 0] = u * v
    Fy[:, :, 1] = v * v + p
    Fy[:, :, 2] = beta**2 * v
    return Fy


# =============================================================================
# Numerical flux (Rusanov)
# =============================================================================

def ac_numerical_flux(q_int: np.ndarray, q_ext: np.ndarray,
                      nx: np.ndarray, ny: np.ndarray,
                      beta: float = 1.0) -> np.ndarray:
    """
    Rusanov numerical flux for the AC system.

    F̂·n = ½[(F_int + F_ext)·n] - ½λ_max(q_ext - q_int)

    λ_max = max(|V·n| + √((V·n)² + β²)) over int/ext states
    where V = (u, v) is the velocity.
    """
    K, n_fp, nv = q_int.shape

    u_int = q_int[:, :, 0]
    v_int = q_int[:, :, 1]
    p_int = q_int[:, :, 2]
    u_ext = q_ext[:, :, 0]
    v_ext = q_ext[:, :, 1]
    p_ext = q_ext[:, :, 2]

    # Normal velocity
    Vn_int = u_int * nx + v_int * ny
    Vn_ext = u_ext * nx + v_ext * ny

    # Max wavespeed
    lam_int = np.abs(Vn_int) + np.sqrt(Vn_int**2 + beta**2)
    lam_ext = np.abs(Vn_ext) + np.sqrt(Vn_ext**2 + beta**2)
    lam_max = np.maximum(lam_int, lam_ext)

    # Physical flux · n for interior
    fn_int = np.empty((K, n_fp, nv))
    fn_int[:, :, 0] = (u_int**2 + p_int) * nx + (u_int * v_int) * ny
    fn_int[:, :, 1] = (u_int * v_int) * nx + (v_int**2 + p_int) * ny
    fn_int[:, :, 2] = beta**2 * (u_int * nx + v_int * ny)

    fn_ext = np.empty((K, n_fp, nv))
    fn_ext[:, :, 0] = (u_ext**2 + p_ext) * nx + (u_ext * v_ext) * ny
    fn_ext[:, :, 1] = (u_ext * v_ext) * nx + (v_ext**2 + p_ext) * ny
    fn_ext[:, :, 2] = beta**2 * (u_ext * nx + v_ext * ny)

    # Rusanov
    f_num = np.empty((K, n_fp, nv))
    for v_idx in range(nv):
        f_num[:, :, v_idx] = (0.5 * (fn_int[:, :, v_idx] + fn_ext[:, :, v_idx])
                               - 0.5 * lam_max * (q_ext[:, :, v_idx] - q_int[:, :, v_idx]))

    return f_num


# =============================================================================
# Max wavespeed for CFL
# =============================================================================

def ac_max_wavespeed(q: np.ndarray, beta: float = 1.0) -> float:
    """Global maximum wavespeed for CFL computation."""
    u = q[:, :, 0]
    v = q[:, :, 1]
    speed = np.sqrt(u**2 + v**2)
    return float(np.max(speed + np.sqrt(speed**2 + beta**2)))


# =============================================================================
# Boundary conditions
# =============================================================================

def cavity_bc(q_int, q_ext, bc_tags, face_nx, face_ny,
              lid_velocity=1.0):
    """
    Apply lid-driven cavity BCs.

    bc_tags: (K, 3*Nfp) int array
        0 = interior (use q_ext as-is from neighbor)
        1 = wall (no-slip)
        2 = lid (moving wall)

    For wall: u_ext = -u_int, v_ext = -v_int, p_ext = p_int
    For lid: u_ext = 2*U_lid - u_int, v_ext = -v_int, p_ext = p_int
    """
    q_out = q_ext.copy()

    wall = bc_tags == 1
    lid = bc_tags == 2

    # No-slip wall
    if np.any(wall):
        q_out[wall, 0] = -q_int[wall, 0]
        q_out[wall, 1] = -q_int[wall, 1]
        q_out[wall, 2] = q_int[wall, 2]

    # Lid (moving wall)
    if np.any(lid):
        q_out[lid, 0] = 2.0 * lid_velocity - q_int[lid, 0]
        q_out[lid, 1] = -q_int[lid, 1]
        q_out[lid, 2] = q_int[lid, 2]

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
