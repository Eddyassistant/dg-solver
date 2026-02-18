"""
SIPG viscous operator for INS system DG solver.

Treats viscous diffusion as u_t = +ν∇²u, implemented in the strong-form DG
framework by writing the viscous flux as F_visc = -ν∇u (negative gradient flux),
so that -∇·F_visc = ν∇²u (correct positive diffusion).

The SIPG numerical flux for F_visc = -ν∇u:
    F̂·n = -ν{{∇u}}·n + σ_IP·[[u]]

Strong-form surface correction (physical - numerical):
    F_phys·n - F̂·n = -ν∇u⁻·n - (-ν{{∇u}}·n + σ_IP[[u]])
                   = ν({{∇u}} - ∇u⁻)·n - σ_IP[[u]]
                   = ν·½(∇u⁺-∇u⁻)·n - σ_IP·(u⁻-u⁺)

With negative penalty (-σ_IP·(u⁻-u⁺)), high-u regions are damped (stable).

References
----------
Arnold et al. (2002), Unified analysis of DG for elliptic problems, SIAM J. Numer. Anal.
Shahbazi (2005), Short note on penalty parameter, JCP.
"""

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def _sipg_viscous_component(u_field, Dr, Ds, rx, ry, sx, sy,
                            LIFT, Fmask_flat, nx_f, ny_f, Fscale,
                            vmapP_k, vmapP_n, bc_per_node, face_x,
                            v_idx, nu, lid_vel, sigma_ip,
                            K, Np, Nfp):
    """
    SIPG viscous RHS for one scalar velocity component.

    Volume: -∇·(-ν∇u) = ν∇²u (positive diffusion)
    Surface: Fscale * [ν·½(∇u⁺-∇u⁻)·n - σ_IP·(u⁻-u⁺)]

    The negative penalty (-σ_IP·jump_u) damps jumps: provides energy removal
    at locations where u⁻ > u⁺.
    """
    n_fp = 3 * Nfp
    rhs = np.empty((K, Np), dtype=np.float64)

    # Step 1: Element-wise volume gradient (no lifting)
    ux = np.empty((K, Np), dtype=np.float64)
    uy = np.empty((K, Np), dtype=np.float64)

    for k in prange(K):
        rx_k = rx[k, 0]
        ry_k = ry[k, 0]
        sx_k = sx[k, 0]
        sy_k = sy[k, 0]

        for i in range(Np):
            dudr = 0.0
            duds = 0.0
            for j in range(Np):
                dudr += Dr[i, j] * u_field[k, j]
                duds += Ds[i, j] * u_field[k, j]
            ux[k, i] = rx_k * dudr + sx_k * duds
            uy[k, i] = ry_k * dudr + sy_k * duds

    # Step 2: Volume Laplacian + surface correction
    for k in prange(K):
        rx_k = rx[k, 0]
        ry_k = ry[k, 0]
        sx_k = sx[k, 0]
        sy_k = sy[k, 0]

        # Volume: -∇·(-ν∇u) = ν∇²u
        # Compute -ν·∂(ux)/∂r etc. and then -∂()/∂x
        # = ν*(∂²u/∂x² + ∂²u/∂y²)
        vol = np.empty(Np, dtype=np.float64)
        for i in range(Np):
            d_fx_dr = 0.0  # ∂(-ν*ux)/∂r
            d_fx_ds = 0.0
            d_fy_dr = 0.0  # ∂(-ν*uy)/∂r
            d_fy_ds = 0.0
            for j in range(Np):
                d_fx_dr -= Dr[i, j] * nu * ux[k, j]   # -ν*∂ux/∂r
                d_fx_ds -= Ds[i, j] * nu * ux[k, j]
                d_fy_dr -= Dr[i, j] * nu * uy[k, j]   # -ν*∂uy/∂r
                d_fy_ds -= Ds[i, j] * nu * uy[k, j]
            # ∂(-ν*ux)/∂x and ∂(-ν*uy)/∂y
            div_fx = rx_k * d_fx_dr + sx_k * d_fx_ds
            div_fy = ry_k * d_fy_dr + sy_k * d_fy_ds
            # -∇·F_visc = -∇·(-ν∇u) = ν∇²u
            vol[i] = -(div_fx + div_fy)

        # Surface correction: Fscale * [ν·½(∇u⁺-∇u⁻)·n - σ_IP·(u⁻-u⁺)]
        surf = np.empty(n_fp, dtype=np.float64)
        for fpt in range(n_fp):
            vol_idx = Fmask_flat[fpt]
            u_int = u_field[k, vol_idx]
            grad_n_int = ux[k, vol_idx] * nx_f[k, fpt] + uy[k, vol_idx] * ny_f[k, fpt]

            bc = bc_per_node[k, fpt]
            if bc == 0:
                # Interior face: use actual neighbor
                pk = vmapP_k[k, fpt]
                pn = vmapP_n[k, fpt]
                u_ext = u_field[pk, pn]
                grad_n_ext = ux[pk, pn] * nx_f[k, fpt] + uy[pk, pn] * ny_f[k, fpt]
            elif bc == 1:
                # Wall (no-slip): ghost u_ext = -u_int → avg = 0
                u_ext = -u_int
                # Mirror gradient (anti-symmetric reflection)
                grad_n_ext = -grad_n_int
            else:
                # Lid: ghost u_ext = 2*U_lid - u_int → avg = U_lid
                x = face_x[k, fpt]
                u_lid = lid_vel * 16.0 * x * x * (1.0 - x) * (1.0 - x)
                if v_idx == 0:
                    u_ext = 2.0 * u_lid - u_int
                else:
                    u_ext = -u_int  # v=0 at lid
                grad_n_ext = -grad_n_int  # Mirror

            # [[u]] = u⁻ - u⁺ (jump from interior perspective)
            jump_u = u_int - u_ext

            # {{∇u}}·n = ½(∇u⁻+∇u⁺)·n
            avg_grad_n = 0.5 * (grad_n_int + grad_n_ext)

            # F_phys·n - F_num·n = -ν∇u⁻·n - (-ν{{∇u}}·n + σ_IP[[u]])
            #   = ν(avg_grad_n - grad_n_int) - σ_IP * jump_u
            #   = ν*½(grad_n_ext - grad_n_int) - σ_IP * (u_int - u_ext)
            surf[fpt] = Fscale[k, fpt] * (
                nu * (avg_grad_n - grad_n_int) - sigma_ip * jump_u
            )

        for i in range(Np):
            val = vol[i]
            for fpt in range(n_fp):
                val += LIFT[i, fpt] * surf[fpt]
            rhs[k, i] = val

    return rhs


def make_system_viscous_rhs(solver, Re, lid_velocity=1.0):
    """
    SIPG viscous RHS for DG2DSystem (velocity components only).

    Parameters
    ----------
    solver : DG2DSystem
    Re : float  (ν = 1/Re)
    lid_velocity : float

    Returns
    -------
    viscous_rhs : callable (q, t) -> (K, Np, 3)
    """
    Dr = np.ascontiguousarray(solver.Dr)
    Ds = np.ascontiguousarray(solver.Ds)
    LIFT = np.ascontiguousarray(solver.LIFT)
    rx, ry = solver.rx, solver.ry
    sx, sy = solver.sx, solver.sy
    Fmask_flat = solver.Fmask_flat
    nx_arr, ny_arr = solver.nx, solver.ny
    Fscale = solver.Fscale
    vmapP_k, vmapP_n = solver.vmapP_k, solver.vmapP_n
    K, Np, Nfp = solver.K, solver.Np, solver.Nfp
    p = solver.p
    nu = 1.0 / Re
    lid_vel = lid_velocity

    # Face x-coordinates for regularized lid BC
    face_x = np.zeros((K, 3 * Nfp))
    for i in range(3 * Nfp):
        face_x[:, i] = solver.x[:, Fmask_flat[i]]

    # SIPG penalty (Shahbazi 2005): σ_IP = C*(p+1)*(p+2)/2 * ν
    # Large enough for stability with explicit time stepping.
    # Fscale incorporates 1/h, so we keep σ_IP in physical units.
    C_ip = 4.0  # Safety factor
    sigma_ip = C_ip * (p + 1) * (p + 2) * nu / 2.0

    # Build bc_per_node
    bc_tags = solver.bc_tags
    if bc_tags is not None:
        bc_per_node = np.zeros((K, 3 * Nfp), dtype=np.int32)
        for f in range(3):
            bc_per_node[:, f * Nfp:(f + 1) * Nfp] = bc_tags[:, f:f + 1]
    else:
        bc_per_node = np.zeros((K, 3 * Nfp), dtype=np.int32)

    # Warmup
    u_dummy = np.zeros((K, Np))
    for vi in range(2):
        _sipg_viscous_component(
            u_dummy, Dr, Ds, rx, ry, sx, sy,
            LIFT, Fmask_flat, nx_arr, ny_arr, Fscale,
            vmapP_k, vmapP_n, bc_per_node, face_x,
            vi, nu, lid_vel, sigma_ip, K, Np, Nfp)

    def viscous_rhs(q, t):
        rhs = np.zeros((K, Np, 3), dtype=np.float64)
        for v_idx in range(2):
            rhs[:, :, v_idx] = _sipg_viscous_component(
                q[:, :, v_idx], Dr, Ds, rx, ry, sx, sy,
                LIFT, Fmask_flat, nx_arr, ny_arr, Fscale,
                vmapP_k, vmapP_n, bc_per_node, face_x,
                v_idx, nu, lid_vel, sigma_ip, K, Np, Nfp)
        return rhs

    return viscous_rhs