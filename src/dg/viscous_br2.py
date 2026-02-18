"""
BR2 (Bassi-Rebay 2) viscous flux operator for 2D DG.

The BR2 method computes viscous fluxes using an auxiliary variable σ = ∇u,
computed via a lifting operator that accounts for solution jumps across faces.

References
----------
Bassi & Rebay (2000), J. Comput. Phys. 131, pp. 267-279.
Hesthaven & Warburton (2008), Ch. 7 (viscous terms).
"""

import numpy as np
from numba import njit, prange
from typing import Optional

from .reference_triangle import RefTriangle
from .mesh.triangle_mesh import (
    TriangleMesh, build_face_connectivity, geometric_factors, face_normals
)
from .timestepping.runge_kutta import ssp_rk3


# =============================================================================
# Numba kernels for performance-critical operations
# =============================================================================

@njit(cache=True, parallel=True)
def _compute_lifted_gradient_kernel(u, Dr, Ds, rx, ry, sx, sy,
                                    LIFT, Fmask_flat, nx, ny, Fscale,
                                    vmapP_k, vmapP_n, K, Np, Nfp):
    """
    Compute BR2 lifting corrections for gradients.
    
    Returns σ = ∇u with BR2 lifting for jumps.
    
    The auxiliary variable σ is computed as:
        σ = ∇u - r_e([u])
    where r_e is the lifting operator applied to jumps at faces.
    
    For each element k:
        - First compute the volume gradient ∇u
        - Then compute face jumps and apply lifting
    """
    # Storage for gradient components (ux, uy)
    ux = np.empty((K, Np), dtype=np.float64)
    uy = np.empty((K, Np), dtype=np.float64)
    
    n_face_pts = 3 * Nfp
    
    for k in prange(K):
        # --- Volume gradient: ∇u on reference, then map to physical ---
        rx_k = rx[k, 0]
        ry_k = ry[k, 0]
        sx_k = sx[k, 0]
        sy_k = sy[k, 0]
        
        for i in range(Np):
            dudr = 0.0
            duds = 0.0
            for j in range(Np):
                dudr += Dr[i, j] * u[k, j]
                duds += Ds[i, j] * u[k, j]
            ux[k, i] = rx_k * dudr + sx_k * duds
            uy[k, i] = ry_k * dudr + sy_k * duds
        
        # --- Face corrections: compute jumps and lift them ---
        # Jump at face: [u] = u_int - u_ext (normal is outward from int)
        # BR2 lifting: r_e applied to average of traces
        # For DG viscous, we need: {{∇u}} - η r_e([u])
        
        # Compute jumps for each face point
        jumps = np.empty(n_face_pts, dtype=np.float64)
        for i in range(n_face_pts):
            vol_idx = Fmask_flat[i]
            pk = vmapP_k[k, i]
            pn = vmapP_n[k, i]
            u_int = u[k, vol_idx]
            u_ext = u[pk, pn]
            # Outward normal from element k
            jumps[i] = u_int - u_ext  # [u] = u^- - u^+
        
        # LIFT @ (Fscale * jumps / 2) - correction for averaging
        # The BR2 method uses: σ = ∇u - LIFT @ (Fscale * [u] / 2 * n)
        # We compute this as a correction to the gradient
        for i in range(Np):
            lift_x = 0.0
            lift_y = 0.0
            for j in range(n_face_pts):
                # LIFT contribution weighted by face normal
                # The jump [u] is scalar; we need to project onto normal
                # Correction to gradient: r_e([u]) contributes to both components
                lift_weight = LIFT[i, j] * Fscale[k, j] * 0.5 * jumps[j]
                lift_x += lift_weight * nx[k, j]
                lift_y += lift_weight * ny[k, j]
            ux[k, i] -= lift_x
            uy[k, i] -= lift_y
    
    return ux, uy


@njit(cache=True, parallel=True)
def _compute_viscous_rhs_kernel(ux, uy, nu, Dr, Ds, rx, ry, sx, sy,
                                LIFT, Fmask_flat, nx, ny, Fscale,
                                vmapP_k, vmapP_n, K, Np, Nfp):
    """
    Compute viscous RHS: ∇·(ν∇u) using BR2 formulation.
    
    Returns rhs_visc[k] = ∫_K ∇φ·(ν∇u) dA - face_corrections
    
    Using integration by parts (weak form):
        ∫_K φ ∇·(ν∇u) dV = -∫_K ∇φ·(ν∇u) dV + ∮_∂K φ (ν∇u·n) ds
    
    In DG form with numerical flux for ∇u:
        rhs = -∫_K ∇φ·(νσ) dV + ∮_∂K φ (νσ̂·n) ds
    
    where σ̂ = {{σ}} is the average of the BR2-corrected gradients.
    """
    rhs = np.empty((K, Np), dtype=np.float64)
    n_face_pts = 3 * Nfp
    
    for k in prange(K):
        rx_k = rx[k, 0]
        ry_k = ry[k, 0]
        sx_k = sx[k, 0]
        sy_k = sy[k, 0]
        
        # --- Volume term: -∫ ∇φ·(νσ) dV ---
        # = -|J| * (rx*Dr + sx*Ds) @ (ν*ux) - |J| * (ry*Dr + sy*Ds) @ (ν*uy)
        vol = np.empty(Np, dtype=np.float64)
        
        for i in range(Np):
            dfx_dr = 0.0
            dfx_ds = 0.0
            dfy_dr = 0.0
            dfy_ds = 0.0
            nu_ux = nu * ux[k, :]
            nu_uy = nu * uy[k, :]
            for j in range(Np):
                dfx_dr += Dr[i, j] * nu_ux[j]
                dfx_ds += Ds[i, j] * nu_ux[j]
                dfy_dr += Dr[i, j] * nu_uy[j]
                dfy_ds += Ds[i, j] * nu_uy[j]
            
            # ∇·(ν∇u) divergence
            d_nux_dx = rx_k * dfx_dr + sx_k * dfx_ds
            d_nuy_dy = ry_k * dfy_dr + sy_k * dfy_ds
            vol[i] = -(d_nux_dx + d_nuy_dy)  # Negative for weak form
        
        # --- Surface term: +∮ φ (νσ̂·n) ds ---
        # Numerical flux for viscous: {{νσ}}·n = ½(νσ⁻·n + νσ⁺·n)
        surf_corr = np.empty(n_face_pts, dtype=np.float64)
        
        for i in range(n_face_pts):
            vol_idx = Fmask_flat[i]
            pk = vmapP_k[k, i]
            pn = vmapP_n[k, i]
            
            # Interior trace
            sigma_n_int = ux[k, vol_idx] * nx[k, i] + uy[k, vol_idx] * ny[k, i]
            
            # Exterior trace
            sigma_n_ext = ux[pk, pn] * nx[k, i] + uy[pk, pn] * ny[k, i]
            
            # Average flux {{σ}}·n
            sigma_n_avg = 0.5 * (sigma_n_int + sigma_n_ext)
            
            # Viscous flux at face: ν * {{σ}}·n
            visc_flux = nu * sigma_n_avg
            
            # Physical flux minus numerical flux for DG residual
            # The residual contribution is: (phys_flux - num_flux)
            # For viscous with average flux, phys_flux = νσ⁻·n, num_flux = ν{{σ}}·n
            phys_flux = nu * sigma_n_int
            surf_corr[i] = Fscale[k, i] * (visc_flux - phys_flux)
            # Note: The sign convention gives us (num_flux - phys_flux) for the jump
            # Actually we want: +num_flux in the surface integral
            # Let me reconsider...
        
        # Recompute surface contribution correctly
        for i in range(n_face_pts):
            vol_idx = Fmask_flat[i]
            pk = vmapP_k[k, i]
            pn = vmapP_n[k, i]
            
            # Interior trace of σ·n
            sigma_n_int = ux[k, vol_idx] * nx[k, i] + uy[k, vol_idx] * ny[k, i]
            
            # Exterior trace (note: neighbor's outward normal is opposite)
            # For the neighbor, the normal is flipped, but we use same (nx, ny)
            # so σ⁺·n here means the value from neighbor dotted with our normal
            sigma_n_ext = ux[pk, pn] * nx[k, i] + uy[pk, pn] * ny[k, i]
            
            # Average: {{σ}}·n = ½(σ⁻·n + σ⁺·n)
            sigma_n_avg = 0.5 * (sigma_n_int + sigma_n_ext)
            
            # DG surface term: Fscale * ({{νσ}}·n - νσ⁻·n)
            # = Fscale * ν * ({{σ}}·n - σ⁻·n)
            # = Fscale * ν * ½(σ⁺·n - σ⁻·n)
            surf_corr[i] = Fscale[k, i] * nu * (sigma_n_avg - sigma_n_int)
        
        # LIFT @ surf_corr + volume term
        for i in range(Np):
            val = vol[i]
            for j in range(n_face_pts):
                val += LIFT[i, j] * surf_corr[j]
            rhs[k, i] = val
    
    return rhs


@njit(cache=True, parallel=True)
def _compute_laplacian_direct_kernel(u, Dr, Ds, rx, ry, sx, sy,
                                     LIFT, Fmask_flat, nx, ny, Fscale,
                                     vmapP_k, vmapP_n, nu, K, Np, Nfp):
    """
    Direct Laplacian computation for comparison/testing.
    
    This computes ν∇²u using BR2-corrected gradients for the viscous flux.
    """
    rhs = np.empty((K, Np), dtype=np.float64)
    n_face_pts = 3 * Nfp
    
    for k in prange(K):
        # --- Step 1: Compute BR2-corrected gradient σ = ∇u ---
        rx_k = rx[k, 0]
        ry_k = ry[k, 0]
        sx_k = sx[k, 0]
        sy_k = sy[k, 0]
        
        # Volume gradient
        ux = np.empty(Np, dtype=np.float64)
        uy = np.empty(Np, dtype=np.float64)
        
        for i in range(Np):
            dudr = 0.0
            duds = 0.0
            for j in range(Np):
                dudr += Dr[i, j] * u[k, j]
                duds += Ds[i, j] * u[k, j]
            ux[i] = rx_k * dudr + sx_k * duds
            uy[i] = ry_k * dudr + sy_k * duds
        
        # --- Step 2: Lifting corrections (BR2) ---
        for i in range(n_face_pts):
            vol_idx = Fmask_flat[i]
            pk = vmapP_k[k, i]
            pn = vmapP_n[k, i]
            jump = u[k, vol_idx] - u[pk, pn]
            
            # Apply lifting correction to gradient
            for j in range(Np):
                lift_weight = LIFT[j, i] * Fscale[k, i] * 0.5 * jump
                ux[j] -= lift_weight * nx[k, i]
                uy[j] -= lift_weight * ny[k, i]
        
        # --- Step 3: Compute divergence of (ν∇u) ---
        # Volume term: -∇·(νσ)
        for i in range(Np):
            d_nux_dr = 0.0
            d_nux_ds = 0.0
            d_nuy_dr = 0.0
            d_nuy_ds = 0.0
            
            for j in range(Np):
                d_nux_dr += Dr[i, j] * ux[j]
                d_nux_ds += Ds[i, j] * ux[j]
                d_nuy_dr += Dr[i, j] * uy[j]
                d_nuy_ds += Ds[i, j] * uy[j]
            
            d_nux_dx = rx_k * d_nux_dr + sx_k * d_nux_ds
            d_nuy_dy = ry_k * d_nuy_dr + sy_k * d_nuy_ds
            rhs[k, i] = -nu * (d_nux_dx + d_nuy_dy)
        
        # Surface term: correction for ∇u at faces
        # We need to apply the lifting again for the test function
        # This gives the symmetric interior penalty-like correction
        surf = np.empty(n_face_pts, dtype=np.float64)
        
        for i in range(n_face_pts):
            vol_idx = Fmask_flat[i]
            pk = vmapP_k[k, i]
            pn = vmapP_n[k, i]
            
            # Traces of corrected gradient
            sigma_n_int = ux[vol_idx] * nx[k, i] + uy[vol_idx] * ny[k, i]
            
            # Need neighbor's corrected gradient for average
            # For simplicity in this direct version, use uncorrected neighbor
            sigma_n_ext = (ux[pk, pn] if pk < K else ux[vol_idx]) * nx[k, i] + \
                          (uy[pk, pn] if pk < K else uy[vol_idx]) * ny[k, i]
            
            # Average minus interior
            avg_jump = 0.5 * (sigma_n_ext - sigma_n_int)
            surf[i] = Fscale[k, i] * nu * avg_jump
        
        # Add surface contribution
        for i in range(Np):
            for j in range(n_face_pts):
                rhs[k, i] += LIFT[i, j] * surf[j]
    
    return rhs


# =============================================================================
# BR2 Viscous Flux Class
# =============================================================================

class BR2ViscousFlux:
    """
    BR2 viscous flux operator for 2D DG on triangular meshes.
    
    This implements the diffusion equation:
        u_t = ν∇²u
    
    using the Bassi-Rebay 2 (BR2) method which introduces an auxiliary
    variable σ = ∇u computed with proper lifting of jumps.
    
    Parameters
    ----------
    mesh : TriangleMesh
        Triangular mesh.
    p : int
        Polynomial degree.
    nu : float
        Viscosity coefficient.
    periodic : bool
        Use periodic boundary conditions.
    x_range, y_range : tuple
        Domain ranges for periodic BCs.
    """
    
    def __init__(
        self,
        mesh: TriangleMesh,
        p: int,
        nu: float,
        periodic: bool = True,
        x_range: tuple = (0.0, 1.0),
        y_range: tuple = (0.0, 1.0),
    ):
        self.mesh = mesh
        self.p = p
        self.nu = float(nu)
        self.K = mesh.n_elem
        
        # Reference element
        self.ref = RefTriangle(p)
        self.Np = self.ref.Np
        self.Nfp = self.ref.Nfp
        
        # Connectivity
        self.EToE, self.EToF = build_face_connectivity(
            mesh, periodic_x=periodic, periodic_y=periodic,
            x_range=x_range, y_range=y_range,
        )
        
        # Geometric factors
        self.x, self.y, self.rx, self.ry, self.sx, self.sy, self.J = \
            geometric_factors(mesh, self.ref, p)
        
        self.nx, self.ny, self.sJ, self.Fscale = \
            face_normals(mesh, self.ref, self.x, self.y, self.J)
        
        # Flatten Fmask
        self.Fmask_flat = np.concatenate(self.ref.Fmask).astype(np.int64)
        
        # Build maps
        self._build_maps()
        
        # Contiguous matrices for Numba
        self.Dr = np.ascontiguousarray(self.ref.Dr)
        self.Ds = np.ascontiguousarray(self.ref.Ds)
        self.LIFT = np.ascontiguousarray(self.ref.LIFT)
        
        # JIT warmup
        self._warmup()
    
    def _build_maps(self):
        """Build face node maps using reversed ordering for neighbor faces."""
        K, Nfp = self.K, self.Nfp
        Fmask = self.ref.Fmask
        EToE, EToF = self.EToE, self.EToF
        n_fp = 3 * Nfp
        
        vmapP_k = np.zeros((K, n_fp), dtype=np.int64)
        vmapP_n = np.zeros((K, n_fp), dtype=np.int64)
        
        for k in range(K):
            for f in range(3):
                k2 = EToE[k, f]
                f2 = EToF[k, f]
                for i in range(Nfp):
                    idx = f * Nfp + i
                    if k2 == k and f2 == f:
                        # Boundary: self
                        vmapP_k[k, idx] = k
                        vmapP_n[k, idx] = Fmask[f][i]
                    else:
                        # Neighbor face nodes are in reverse order
                        vmapP_k[k, idx] = k2
                        vmapP_n[k, idx] = Fmask[f2][Nfp - 1 - i]
        
        self.vmapP_k = vmapP_k
        self.vmapP_n = vmapP_n
    
    def _warmup(self):
        """JIT compile kernels."""
        K, Np, Nfp = self.K, self.Np, self.Nfp
        u = np.zeros((K, Np))
        
        # Warmup gradient kernel
        _compute_lifted_gradient_kernel(
            u, self.Dr, self.Ds, self.rx, self.ry, self.sx, self.sy,
            self.LIFT, self.Fmask_flat, self.nx, self.ny, self.Fscale,
            self.vmapP_k, self.vmapP_n, K, Np, Nfp
        )
        
        # Warmup RHS kernel
        ux = np.zeros((K, Np))
        _compute_viscous_rhs_kernel(
            ux, ux, self.nu, self.Dr, self.Ds, self.rx, self.ry, self.sx, self.sy,
            self.LIFT, self.Fmask_flat, self.nx, self.ny, self.Fscale,
            self.vmapP_k, self.vmapP_n, K, Np, Nfp
        )
    
    def compute_lifted_gradient(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute BR2-corrected gradient σ = ∇u.
        
        The gradient includes lifting corrections from face jumps:
            σ = ∇u - r_e([u])
        
        Parameters
        ----------
        u : ndarray, shape (K, Np)
            Solution field.
        
        Returns
        -------
        ux, uy : tuple of ndarray, shape (K, Np)
            Gradient components with BR2 corrections.
        """
        return _compute_lifted_gradient_kernel(
            u, self.Dr, self.Ds, self.rx, self.ry, self.sx, self.sy,
            self.LIFT, self.Fmask_flat, self.nx, self.ny, self.Fscale,
            self.vmapP_k, self.vmapP_n, self.K, self.Np, self.Nfp
        )
    
    def compute_rhs(self, u: np.ndarray, t: float = 0.0) -> np.ndarray:
        """
        Compute viscous RHS: ν∇²u using BR2 method.
        
        Parameters
        ----------
        u : ndarray, shape (K, Np)
            Solution field.
        t : float
            Current time (unused, for API compatibility).
        
        Returns
        -------
        rhs : ndarray, shape (K, Np)
            Right-hand side: du/dt = ν∇²u
        """
        # Step 1: Compute BR2-corrected gradient
        ux, uy = self.compute_lifted_gradient(u)
        
        # Step 2: Compute divergence
        rhs = _compute_viscous_rhs_kernel(
            ux, uy, self.nu, self.Dr, self.Ds, self.rx, self.ry, self.sx, self.sy,
            self.LIFT, self.Fmask_flat, self.nx, self.ny, self.Fscale,
            self.vmapP_k, self.vmapP_n, self.K, self.Np, self.Nfp
        )
        
        return rhs
    
    def compute_laplacian(self, u: np.ndarray) -> np.ndarray:
        """
        Compute Laplacian ∇²u directly.
        
        This is a convenience method that returns ∇²u (not scaled by ν).
        
        Parameters
        ----------
        u : ndarray, shape (K, Np)
            Solution field.
        
        Returns
        -------
        laplacian : ndarray, shape (K, Np)
            Laplacian of u.
        """
        rhs = self.compute_rhs(u) / self.nu
        return rhs
    
    def solve_diffusion(
        self,
        u0_func,
        t_final: float,
        cfl: float = 0.1,
    ) -> tuple[np.ndarray, float]:
        """
        Solve diffusion equation u_t = ν∇²u from t=0 to t_final.
        
        Parameters
        ----------
        u0_func : callable
            Initial condition function u0(x, y) -> ndarray.
        t_final : float
            Final time.
        cfl : float
            CFL number (default 0.1, diffusion-limited).
        
        Returns
        -------
        u : ndarray, shape (K, Np)
            Solution at t_final.
        t : float
            Actual final time.
        """
        # Project initial condition
        u = u0_func(self.x, self.y)
        t = 0.0
        
        # Characteristic element size for diffusion
        h_min = np.min(2.0 * np.abs(self.J[:, 0]) /
                       np.max(self.sJ.reshape(self.K, 3, self.Nfp), axis=2).max(axis=1))
        
        # Diffusion CFL: dt ≤ dx² / (2ν) for explicit
        dt_diff = cfl * h_min**2 / (2.0 * self.nu)
        
        step = 0
        while t < t_final - 1e-14:
            dt = min(dt_diff, t_final - t)
            u = ssp_rk3(u, dt, self.compute_rhs, t)
            t += dt
            step += 1
        
        return u, t
    
    def l2_error(self, u: np.ndarray, exact_func, t: float = 0.0) -> float:
        """
        Compute L2 error against exact solution.
        
        Parameters
        ----------
        u : ndarray, shape (K, Np)
            Numerical solution.
        exact_func : callable
            Exact solution function exact(x, y, t) -> ndarray.
        t : float
            Current time.
        
        Returns
        -------
        error : float
            L2 norm of error.
        """
        u_exact = exact_func(self.x, self.y, t)
        err = u - u_exact
        err_modal = err @ self.ref.Vinv.T
        return np.sqrt(np.sum(err_modal**2 * np.abs(self.J)))
