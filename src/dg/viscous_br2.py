"""
BR2 (Bassi-Rebay 2) viscous flux operator for 2D DG.

The BR2 method computes viscous fluxes using an auxiliary variable σ = ∇u,
computed via a lifting operator that accounts for solution jumps across faces.

BR2 formulation:
    1. Compute auxiliary variable σ = ∇u - r_e([[u]])
       where r_e is the lifting operator applied to jumps [[u]] = u⁻ - u⁺
    
    2. Compute viscous flux using σ in the weak form:
       ∫_K u_t φ dV = -∫_K νσ · ∇φ dV + ∮_∂K ν{{σ}}·n φ ds

The lifting operator in matrix form (nodal DG):
    r_e([[u]]) = LIFT @ (Fscale * [[u]] * n / 2)

where the factor 1/2 comes from the averaging operator in the lifting definition.

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
def _compute_volume_gradient(u, Dr, Ds, rx, ry, sx, sy, K, Np):
    """
    Compute local element gradients (no lifting corrections).
    
    Returns volume gradients (∂u/∂x, ∂u/∂y) for each element.
    """
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
                dudr += Dr[i, j] * u[k, j]
                duds += Ds[i, j] * u[k, j]
            ux[k, i] = rx_k * dudr + sx_k * duds
            uy[k, i] = ry_k * dudr + sy_k * duds
    
    return ux, uy


@njit(cache=True, parallel=True)
def _compute_br2_gradient(u, Dr, Ds, rx, ry, sx, sy,
                          LIFT, Fmask_flat, nx, ny, Fscale,
                          vmapP_k, vmapP_n, K, Np, Nfp):
    """
    Compute BR2-corrected gradient σ = ∇u - r_e([[u]]).
    
    The BR2 lifting operator r_e applied to the jump [[u]] gives:
        r_e([[u]]) = LIFT @ (Fscale * [[u]] * n / 2)
    
    The factor 1/2 is from the averaging operator {{v}} in the lifting definition.
    """
    # First compute volume gradients
    ux, uy = _compute_volume_gradient(u, Dr, Ds, rx, ry, sx, sy, K, Np)
    
    n_face_pts = 3 * Nfp
    
    # Apply BR2 lifting corrections
    for k in prange(K):
        # Gather face jumps: [[u]] = u⁻ - u⁺ at each face point
        for fpt in range(n_face_pts):
            vol_idx = Fmask_flat[fpt]
            pk = vmapP_k[k, fpt]
            pn = vmapP_n[k, fpt]
            jump = u[k, vol_idx] - u[pk, pn]  # [[u]] = u⁻ - u⁺
            
            # LIFT contribution: Fscale * [[u]] * n / 2
            # This gets applied to all nodes in the element
            lift_weight = Fscale[k, fpt] * 0.5 * jump
            
            for i in range(Np):
                ux[k, i] -= LIFT[i, fpt] * lift_weight * nx[k, fpt]
                uy[k, i] -= LIFT[i, fpt] * lift_weight * ny[k, fpt]
    
    return ux, uy


@njit(cache=True, parallel=True)
def _compute_viscous_rhs_volume(ux, uy, nu, Dr, Ds, rx, ry, sx, sy, K, Np):
    """
    Compute volume term: ∫_K ∇·(νσ) φ dV (integrated by parts from weak form).
    
    For diffusion u_t = ∇·(ν∇u), using BR2 σ ≈ ∇u:
        u_t = ∇·(νσ)
    
    In strong form DG:
        RHS = (rx*Dr + sx*Ds) @ (ν*ux) + (ry*Dr + sy*Ds) @ (ν*uy)
            = ∇·(νσ)  (the divergence)
    
    Returns array of shape (K, Np) containing the volume RHS contribution.
    """
    rhs_vol = np.empty((K, Np), dtype=np.float64)
    
    for k in prange(K):
        rx_k = rx[k, 0]
        ry_k = ry[k, 0]
        sx_k = sx[k, 0]
        sy_k = sy[k, 0]
        
        for i in range(Np):
            # Compute ∂(ν*ux)/∂x and ∂(ν*uy)/∂y using chain rule
            d_nux_dr = 0.0
            d_nux_ds = 0.0
            d_nuy_dr = 0.0
            d_nuy_ds = 0.0
            
            for j in range(Np):
                val_x = nu * ux[k, j]
                val_y = nu * uy[k, j]
                d_nux_dr += Dr[i, j] * val_x
                d_nux_ds += Ds[i, j] * val_x
                d_nuy_dr += Dr[i, j] * val_y
                d_nuy_ds += Ds[i, j] * val_y
            
            # Transform to physical coordinates
            d_nux_dx = rx_k * d_nux_dr + sx_k * d_nux_ds
            d_nuy_dy = ry_k * d_nuy_dr + sy_k * d_nuy_ds
            
            # Volume term: ∂(ν*ux)/∂x + ∂(ν*uy)/∂y = ∇·(νσ)
            # This gives u_t = ∇·(ν∇u)
            rhs_vol[k, i] = d_nux_dx + d_nuy_dy
    
    return rhs_vol


@njit(cache=True, parallel=True)
def _compute_viscous_rhs_surface(ux, uy, nu, LIFT, Fmask_flat, nx, ny, Fscale,
                                 vmapP_k, vmapP_n, K, Np, Nfp):
    """
    Compute surface correction term for viscous RHS.
    
    Surface contribution: ∮_∂K φ (ν{{σ}}·n - νσ⁻·n) ds
                        = LIFT @ (Fscale * ν * ({{σ}}·n - σ⁻·n))
    
    where {{σ}} = (σ⁻ + σ⁺)/2 is the average of traces.
    """
    n_face_pts = 3 * Nfp
    rhs_surf = np.zeros((K, Np), dtype=np.float64)
    
    for k in prange(K):
        # Compute surface flux correction for each face point
        for fpt in range(n_face_pts):
            vol_idx = Fmask_flat[fpt]
            pk = vmapP_k[k, fpt]
            pn = vmapP_n[k, fpt]
            
            # Interior trace: σ⁻ · n
            sigma_n_minus = ux[k, vol_idx] * nx[k, fpt] + uy[k, vol_idx] * ny[k, fpt]
            
            # Exterior trace: σ⁺ · n (neighbor's value dotted with same normal)
            sigma_n_plus = ux[pk, pn] * nx[k, fpt] + uy[pk, pn] * ny[k, fpt]
            
            # Average: {{σ}} · n = (σ⁻·n + σ⁺·n) / 2
            sigma_n_avg = 0.5 * (sigma_n_minus + sigma_n_plus)
            
            # Surface flux correction: Fscale * ν * ({{σ}}·n - σ⁻·n)
            # = Fscale * ν * (σ⁺·n - σ⁻·n) / 2
            flux_corr = Fscale[k, fpt] * nu * (sigma_n_avg - sigma_n_minus)
            
            # Distribute to all volume nodes using LIFT
            for i in range(Np):
                rhs_surf[k, i] += LIFT[i, fpt] * flux_corr
    
    return rhs_surf


@njit(cache=True, parallel=True)
def _compute_viscous_rhs(u, ux, uy, nu, Dr, Ds, rx, ry, sx, sy,
                         LIFT, Fmask_flat, nx, ny, Fscale,
                         vmapP_k, vmapP_n, K, Np, Nfp):
    """
    Compute full viscous RHS combining volume and surface terms.
    
    For diffusion equation u_t = ∇·(ν∇u):
        Volume: ∇·(νσ) (divergence of BR2-corrected flux)
        Surface: LIFT @ (Fscale * ({{νσ}}·n - νσ⁻·n)) (interface corrections)
    
    Returns: rhs = ∇·(νσ) + surface_corrections
    """
    # Volume term
    rhs_vol = _compute_viscous_rhs_volume(ux, uy, nu, Dr, Ds, rx, ry, sx, sy, K, Np)
    
    # Surface term
    rhs_surf = _compute_viscous_rhs_surface(ux, uy, nu, LIFT, Fmask_flat, nx, ny, Fscale,
                                            vmapP_k, vmapP_n, K, Np, Nfp)
    
    # Combine
    rhs = np.empty((K, Np), dtype=np.float64)
    for k in prange(K):
        for i in range(Np):
            rhs[k, i] = rhs_vol[k, i] + rhs_surf[k, i]
    
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
        _compute_br2_gradient(
            u, self.Dr, self.Ds, self.rx, self.ry, self.sx, self.sy,
            self.LIFT, self.Fmask_flat, self.nx, self.ny, self.Fscale,
            self.vmapP_k, self.vmapP_n, K, Np, Nfp
        )
        
        # Warmup RHS kernel
        ux = np.zeros((K, Np))
        _compute_viscous_rhs(
            u, ux, ux, self.nu, self.Dr, self.Ds, self.rx, self.ry, self.sx, self.sy,
            self.LIFT, self.Fmask_flat, self.nx, self.ny, self.Fscale,
            self.vmapP_k, self.vmapP_n, K, Np, Nfp
        )
    
    def compute_gradient(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute element-wise gradient (no BR2 lifting).
        
        Parameters
        ----------
        u : ndarray, shape (K, Np)
            Solution field.
        
        Returns
        -------
        ux, uy : tuple of ndarray, shape (K, Np)
            Gradient components without lifting corrections.
        """
        return _compute_volume_gradient(
            u, self.Dr, self.Ds, self.rx, self.ry, self.sx, self.sy,
            self.K, self.Np
        )
    
    def compute_lifted_gradient(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute BR2-corrected gradient σ = ∇u.
        
        The gradient includes lifting corrections from face jumps:
            σ = ∇u - r_e([[u]])
        
        Parameters
        ----------
        u : ndarray, shape (K, Np)
            Solution field.
        
        Returns
        -------
        ux, uy : tuple of ndarray, shape (K, Np)
            Gradient components with BR2 corrections.
        """
        return _compute_br2_gradient(
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
        rhs = _compute_viscous_rhs(
            u, ux, uy, self.nu, self.Dr, self.Ds, self.rx, self.ry, self.sx, self.sy,
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
        rhs = self.compute_rhs(u)
        return rhs / self.nu
    
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
