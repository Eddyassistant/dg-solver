"""
Mixed DG formulation for incompressible Navier-Stokes.

Pressure-Poisson formulation:
    1. Predictor: u* = u^n + dt * [-conv(u^n) + ν∇²u^n]
    2. Pressure Poisson: ∇²p^{n+1} = (1/dt) ∇·u*
    3. Corrector: u^{n+1} = u* - dt * ∇p^{n+1}

This is a projection method implemented in the DG framework.

Key features:
- Semi-implicit: convection explicit, viscous semi-implicit
- Pressure Poisson solved with scipy.sparse.linalg (GMRES/CG)
- Sharp lid BC (constant u=1 at top wall)
- Equal-order P_p spaces with pressure stabilization

References
----------
Karniadakis & Sherwin (2005), Spectral/hp Element Methods, Ch. 8.
Cockburn, Kanschat & Schötzau (2005), Math. Comp. 74(249), pp. 1067-1095.
"""

import numpy as np
from numba import njit, prange
from scipy import sparse
from scipy.sparse import csr_matrix, block_diag
from scipy.sparse.linalg import spsolve, cg, gmres, LinearOperator
import time

from .dg_operators import (
    build_mass_matrix, apply_mass_inv,
    compute_divergence, compute_gradient,
    build_weak_laplacian, build_pressure_stabilization
)


@njit(cache=True, parallel=True)
def _convection_rhs_numba(u, v, Dr, Ds, rx, ry, sx, sy, LIFT,
                          nx_f, ny_f, Fscale, Fmask_flat,
                          vmapP_k, vmapP_n, bc_per_node,
                          K, Np, n_fp, lid_vel):
    """
    Compute convection term: -(u·∇)u using weak-form DG.
    
    Returns RHS contribution: -∇·(u u) for each component.
    Uses Rusanov numerical flux.
    """
    rhs_u = np.zeros((K, Np), dtype=np.float64)
    rhs_v = np.zeros((K, Np), dtype=np.float64)
    
    for k in prange(K):
        rx_k = rx[k, 0]
        ry_k = ry[k, 0]
        sx_k = sx[k, 0]
        sy_k = sy[k, 0]
        
        # Physical fluxes: F = u*u, G = u*v
        Fx_u = np.empty(Np, dtype=np.float64)  # u*u
        Fy_u = np.empty(Np, dtype=np.float64)  # u*v
        Fx_v = np.empty(Np, dtype=np.float64)  # u*v  
        Fy_v = np.empty(Np, dtype=np.float64)  # v*v
        
        for i in range(Np):
            ui = u[k, i]
            vi = v[k, i]
            Fx_u[i] = ui * ui
            Fy_u[i] = ui * vi
            Fx_v[i] = ui * vi
            Fy_v[i] = vi * vi
        
        # Volume divergence: ∇·F = ∂Fx/∂x + ∂Fy/∂y
        vol_u = np.empty(Np, dtype=np.float64)
        vol_v = np.empty(Np, dtype=np.float64)
        
        for i in range(Np):
            dFx_dr = 0.0
            dFx_ds = 0.0
            dFy_dr = 0.0
            dFy_ds = 0.0
            
            for j in range(Np):
                dFx_dr += Dr[i, j] * Fx_u[j]
                dFx_ds += Ds[i, j] * Fx_u[j]
                dFy_dr += Dr[i, j] * Fy_u[j]
                dFy_ds += Ds[i, j] * Fy_u[j]
            
            dFx_dx = rx_k * dFx_dr + sx_k * dFx_ds
            dFy_dy = ry_k * dFy_dr + sy_k * dFy_ds
            vol_u[i] = -(dFx_dx + dFy_dy)
            
            dFx_dr = 0.0
            dFx_ds = 0.0
            dFy_dr = 0.0
            dFy_ds = 0.0
            
            for j in range(Np):
                dFx_dr += Dr[i, j] * Fx_v[j]
                dFx_ds += Ds[i, j] * Fx_v[j]
                dFy_dr += Dr[i, j] * Fy_v[j]
                dFy_ds += Ds[i, j] * Fy_v[j]
            
            dFx_dx = rx_k * dFx_dr + sx_k * dFx_ds
            dFy_dy = ry_k * dFy_dr + sy_k * dFy_ds
            vol_v[i] = -(dFx_dx + dFy_dy)
        
        # Surface fluxes with Rusanov
        surf_u = np.empty(n_fp, dtype=np.float64)
        surf_v = np.empty(n_fp, dtype=np.float64)
        
        for fpt in range(n_fp):
            vol_idx = Fmask_flat[fpt]
            nx = nx_f[k, fpt]
            ny = ny_f[k, fpt]
            
            ui = u[k, vol_idx]
            vi = v[k, vol_idx]
            
            # Physical flux dot n
            fni_u = (ui * ui) * nx + (ui * vi) * ny
            fni_v = (ui * vi) * nx + (vi * vi) * ny
            
            # Wavespeed for Rusanov
            Vi = abs(ui * nx + vi * ny)
            
            # Exterior values
            bc = bc_per_node[k, fpt]
            if bc == 0:
                pk = vmapP_k[k, fpt]
                pn = vmapP_n[k, fpt]
                ue = u[pk, pn]
                ve = v[pk, pn]
            elif bc == 1:
                # Wall: no-slip
                ue = -ui
                ve = -vi
            else:
                # Lid: sharp BC
                ue = 2.0 * lid_vel - ui
                ve = -vi
            
            Ve = abs(ue * nx + ve * ny)
            lam = max(Vi, Ve)
            
            fne_u = (ue * ue) * nx + (ue * ve) * ny
            fne_v = (ue * ve) * nx + (ve * ve) * ny
            
            # Rusanov flux
            fhat_u = 0.5 * (fni_u + fne_u) - 0.5 * lam * (ue - ui)
            fhat_v = 0.5 * (fni_v + fne_v) - 0.5 * lam * (ve - vi)
            
            # Surface correction
            surf_u[fpt] = Fscale[k, fpt] * (fni_u - fhat_u)
            surf_v[fpt] = Fscale[k, fpt] * (fni_v - fhat_v)
        
        # Lift to volume
        for i in range(Np):
            val_u = vol_u[i]
            val_v = vol_v[i]
            for fpt in range(n_fp):
                val_u += LIFT[i, fpt] * surf_u[fpt]
                val_v += LIFT[i, fpt] * surf_v[fpt]
            rhs_u[k, i] = val_u
            rhs_v[k, i] = val_v
    
    return rhs_u, rhs_v


@njit(cache=True, parallel=True)
def _viscous_rhs_numba(u, v, Dr, Ds, rx, ry, sx, sy, LIFT,
                       nx_f, ny_f, Fscale, Fmask_flat,
                       vmapP_k, vmapP_n, bc_per_node,
                       K, Np, n_fp, nu, sigma_ip, lid_vel):
    """
    SIPG viscous RHS for velocity components.
    Returns ν∇²u (already includes viscosity).
    """
    rhs_u = np.zeros((K, Np), dtype=np.float64)
    rhs_v = np.zeros((K, Np), dtype=np.float64)
    
    for k in prange(K):
        rx_k = rx[k, 0]
        ry_k = ry[k, 0]
        sx_k = sx[k, 0]
        sy_k = sy[k, 0]
        
        for vcomp in range(2):
            u_field = u if vcomp == 0 else v
            
            # Volume gradient
            ux = np.empty(Np, dtype=np.float64)
            uy = np.empty(Np, dtype=np.float64)
            
            for i in range(Np):
                dudr = 0.0
                duds = 0.0
                for j in range(Np):
                    dudr += Dr[i, j] * u_field[k, j]
                    duds += Ds[i, j] * u_field[k, j]
                ux[i] = rx_k * dudr + sx_k * duds
                uy[i] = ry_k * dudr + sy_k * duds
            
            # Volume Laplacian
            vol = np.empty(Np, dtype=np.float64)
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
                vol[i] = -(div_fx + div_fy)
            
            # Surface correction
            surf = np.empty(n_fp, dtype=np.float64)
            for fpt in range(n_fp):
                vol_idx = Fmask_flat[fpt]
                nx = nx_f[k, fpt]
                ny = ny_f[k, fpt]
                
                u_int = u_field[k, vol_idx]
                grad_n_int = ux[vol_idx] * nx + uy[vol_idx] * ny
                
                bc = bc_per_node[k, fpt]
                if bc == 0:
                    pk = vmapP_k[k, fpt]
                    pn = vmapP_n[k, fpt]
                    u_ext = u_field[pk, pn]
                    # Compute neighbor gradient
                    rx_e = rx[pk, 0]
                    ry_e = ry[pk, 0]
                    sx_e = sx[pk, 0]
                    sy_e = sy[pk, 0]
                    dudr_e = 0.0
                    duds_e = 0.0
                    for j in range(Np):
                        dudr_e += Dr[pn, j] * u_field[pk, j]
                        duds_e += Ds[pn, j] * u_field[pk, j]
                    ux_ext = rx_e * dudr_e + sx_e * duds_e
                    uy_ext = ry_e * dudr_e + sy_e * duds_e
                    grad_n_ext = ux_ext * nx + uy_ext * ny
                elif bc == 1:
                    u_ext = -u_int
                    grad_n_ext = -grad_n_int
                else:
                    if vcomp == 0:
                        u_ext = 2.0 * lid_vel - u_int
                    else:
                        u_ext = -u_int
                    grad_n_ext = -grad_n_int
                
                jump_u = u_int - u_ext
                avg_grad_n = 0.5 * (grad_n_int + grad_n_ext)
                
                surf[fpt] = Fscale[k, fpt] * (
                    nu * (avg_grad_n - grad_n_int) - sigma_ip * jump_u
                )
            
            for i in range(Np):
                val = vol[i]
                for fpt in range(n_fp):
                    val += LIFT[i, fpt] * surf[fpt]
                if vcomp == 0:
                    rhs_u[k, i] = val
                else:
                    rhs_v[k, i] = val
    
    return rhs_u, rhs_v


class MixedDGCavitySolver:
    """
    Mixed DG solver for incompressible Navier-Stokes using pressure-Poisson projection.
    
    Time stepping:
        1. u* = u^n + dt * [-conv(u^n) + ν∇²u^n]
        2. Solve ∇²p^{n+1} = (1/dt) ∇·u*  (with Neumann BC from u*)
        3. u^{n+1} = u* - dt * ∇p^{n+1}
    
    Attributes
    ----------
    mesh : TriangleMesh
    p : int
        Polynomial degree
    Re : float
        Reynolds number
    nu : float
        Kinematic viscosity (1/Re)
    solver : DG2DSystem
        Base DG system
    """
    
    def __init__(self, mesh, p=2, Re=100.0, gamma_p=0.1):
        """
        Initialize mixed DG cavity solver.
        
        Parameters
        ----------
        mesh : TriangleMesh
        p : int
            Polynomial degree
        Re : float
            Reynolds number
        gamma_p : float
            Pressure stabilization parameter
        """
        from .solver2d_system import DG2DSystem
        
        self.mesh = mesh
        self.p = p
        self.Re = Re
        self.nu = 1.0 / Re
        self.gamma_p = gamma_p
        
        # Initialize base solver (for geometry, operators)
        self.solver = DG2DSystem(
            mesh, p, n_vars=3,
            flux_x=lambda q: np.zeros_like(q),
            flux_y=lambda q: np.zeros_like(q),
            numerical_flux=lambda qi, qe, nx, ny: np.zeros_like(qi),
            max_wavespeed=lambda q: 1.0,
            periodic=False,
        )
        
        # Build boundary condition tags
        self._build_bc_tags()
        
        # SIPG penalty parameter
        C_ip = 4.0
        self.sigma_ip = C_ip * (p + 1) * (p + 2) * self.nu / 2.0
        
        # Build Poisson operator
        self._build_poisson_operator()
        
        # Precompute geometry
        self._precompute_geometry()
        
    def _build_bc_tags(self):
        """Tag boundary faces: 0=interior, 1=wall, 2=lid."""
        K = self.solver.K
        Fmask = self.solver.ref.Fmask
        tol = 1e-8
        
        bc_tags = np.zeros((K, 3), dtype=np.int32)
        
        for k in range(K):
            for f in range(3):
                if self.solver.EToE[k, f] == k and self.solver.EToF[k, f] == f:
                    face_y = np.mean(self.solver.y[k, Fmask[f]])
                    if face_y > 1.0 - tol:
                        bc_tags[k, f] = 2  # lid
                    else:
                        bc_tags[k, f] = 1  # wall
        
        self.bc_tags = bc_tags
        
        # Expand to per-node
        Nfp = self.solver.ref.Nfp
        self.bc_per_node = np.zeros((K, 3 * Nfp), dtype=np.int32)
        for f in range(3):
            self.bc_per_node[:, f * Nfp:(f + 1) * Nfp] = bc_tags[:, f:f + 1]
    
    def _build_poisson_operator(self):
        """Build pressure Poisson operator using BR2-like approach."""
        # Use SIPG Laplacian from dg_operators
        from .dg_operators import build_sipg_laplacian
        
        # Build Laplacian with unit viscosity (we scale RHS instead)
        self.Lap = build_sipg_laplacian(self.solver, nu=1.0)
        
        # Add pressure stabilization for equal-order spaces
        self.J_stab = build_pressure_stabilization(self.solver, gamma_p=self.gamma_p)
        
        # Combined operator
        self.A_poisson = self.Lap + self.J_stab
        
        # Add small regularization to make operator invertible
        n_dof = self.solver.K * self.solver.Np
        eps_reg = 1e-8
        self.A_poisson = self.A_poisson + eps_reg * sparse.eye(n_dof, format='csr')
        
        # Build mass matrix (for projections)
        self.M = build_mass_matrix(self.solver)
        
        # Store shape
        self.n_dof = n_dof
        
    def _precompute_geometry(self):
        """Precompute commonly used geometric quantities."""
        self.Fmask_flat = np.concatenate(self.solver.ref.Fmask).astype(np.int64)
        
        # h_min for CFL
        self.h_min = np.min(
            2.0 * np.abs(self.solver.J[:, 0]) /
            np.max(self.solver.Fscale.reshape(self.solver.K, 3, self.solver.ref.Nfp), axis=2).max(axis=1)
        )
    
    def _compute_rhs(self, u, v, lid_vel=1.0):
        """
        Compute RHS: -conv(u) + ν∇²u.
        
        Parameters
        ----------
        u, v : ndarray, shape (K, Np)
            Velocity components
        
        Returns
        -------
        rhs_u, rhs_v : ndarray, shape (K, Np)
            RHS for velocity components
        """
        K, Np = self.solver.K, self.solver.Np
        n_fp = 3 * self.solver.ref.Nfp
        
        # Convection term
        conv_u, conv_v = _convection_rhs_numba(
            u, v, self.solver.Dr, self.solver.Ds,
            self.solver.rx, self.solver.ry, self.solver.sx, self.solver.sy,
            self.solver.LIFT, self.solver.nx, self.solver.ny, self.solver.Fscale,
            self.Fmask_flat, self.solver.vmapP_k, self.solver.vmapP_n,
            self.bc_per_node, K, Np, n_fp, lid_vel
        )
        
        # Viscous term
        visc_u, visc_v = _viscous_rhs_numba(
            u, v, self.solver.Dr, self.solver.Ds,
            self.solver.rx, self.solver.ry, self.solver.sx, self.solver.sy,
            self.solver.LIFT, self.solver.nx, self.solver.ny, self.solver.Fscale,
            self.Fmask_flat, self.solver.vmapP_k, self.solver.vmapP_n,
            self.bc_per_node, K, Np, n_fp, self.nu, self.sigma_ip, lid_vel
        )
        
        return conv_u + visc_u, conv_v + visc_v
    
    def _compute_divergence(self, u, v):
        """Compute ∇·u."""
        return compute_divergence(self.solver, u, v)
    
    def _compute_gradient(self, p):
        """Compute ∇p."""
        return compute_gradient(self.solver, p)
    
    def _solve_poisson(self, rhs_flat, tol=1e-6, maxiter=500):
        """
        Solve Poisson equation A @ p = rhs using CG.
        
        Uses a simpler approach with direct sparse solve.
        """
        from scipy.sparse.linalg import spsolve
        
        # For small problems, use direct solve
        if self.n_dof < 1000:
            p_flat = spsolve(self.A_poisson, rhs_flat)
        else:
            # For larger problems, use CG with diagonal preconditioner
            M_diag = self.A_poisson.diagonal()
            M_diag[M_diag == 0] = 1.0
            M_inv = 1.0 / M_diag
            
            def precondition(r):
                return M_inv * r
            
            M_op = LinearOperator((self.n_dof, self.n_dof), matvec=precondition)
            
            p_flat, info = cg(self.A_poisson, rhs_flat, rtol=tol, maxiter=maxiter, M=M_op)
            
            if info != 0:
                # Fall back to GMRES if CG fails
                p_flat, info = gmres(self.A_poisson, rhs_flat, rtol=tol, restart=min(100, self.n_dof))
        
        return p_flat
    
    def step(self, u, v, p, dt, lid_vel=1.0):
        """
        Take one time step using projection method.
        
        Parameters
        ----------
        u, v, p : ndarray, shape (K, Np)
            Current velocity and pressure
        dt : float
            Time step
        lid_vel : float
            Lid velocity
        
        Returns
        -------
        u_new, v_new, p_new : ndarray, shape (K, Np)
            Updated velocity and pressure
        """
        K, Np = self.solver.K, self.solver.Np
        
        # --- Step 1: Predictor ---
        rhs_u, rhs_v = self._compute_rhs(u, v, lid_vel)
        
        u_star = u + dt * rhs_u
        v_star = v + dt * rhs_v
        
        # --- Step 2: Pressure Poisson ---
        # RHS: (1/dt) ∇·u*
        div_u_star = self._compute_divergence(u_star, v_star)
        rhs_poisson = div_u_star.ravel() / dt
        
        # Solve for pressure correction
        p_corr_flat = self._solve_poisson(rhs_poisson)
        p_corr = p_corr_flat.reshape(K, Np)
        
        p_new = p + p_corr
        
        # --- Step 3: Corrector ---
        grad_px, grad_py = self._compute_gradient(p_corr)
        
        u_new = u_star - dt * grad_px
        v_new = v_star - dt * grad_py
        
        # Enforce BCs strongly on velocity
        self._apply_velocity_bc(u_new, v_new, lid_vel)
        
        return u_new, v_new, p_new
    
    def _apply_velocity_bc(self, u, v, lid_vel):
        """Apply Dirichlet BCs strongly to face nodes."""
        Fmask = self.solver.ref.Fmask
        
        # First apply lid BC
        for k in range(self.solver.K):
            for f in range(3):
                if self.bc_tags[k, f] == 2:  # Lid
                    fids = Fmask[f]
                    u[k, fids] = lid_vel
                    v[k, fids] = 0.0
        
        # Then apply wall BC (overwrites corners to enforce no-slip at corners)
        for k in range(self.solver.K):
            for f in range(3):
                if self.bc_tags[k, f] == 1:  # Wall
                    fids = Fmask[f]
                    u[k, fids] = 0.0
                    v[k, fids] = 0.0
    
    def solve(self, t_final=30.0, cfl=0.1, dt_max=None, lid_vel=1.0,
              print_every=500, callback=None):
        """
        Solve to steady state.
        
        Parameters
        ----------
        t_final : float
            Final time
        cfl : float
            CFL number for explicit convection
        dt_max : float, optional
            Maximum time step
        lid_vel : float
            Lid velocity
        print_every : int
            Print frequency
        callback : callable, optional
            Callback(q, t, step, dt)
        
        Returns
        -------
        u, v, p : ndarray, shape (K, Np)
            Final velocity and pressure
        t : float
            Final time
        n_steps : int
            Number of steps
        """
        K, Np = self.solver.K, self.solver.Np
        
        # Initial condition: quiescent
        u = np.zeros((K, Np))
        v = np.zeros((K, Np))
        p = np.zeros((K, Np))
        
        # Apply BCs
        self._apply_velocity_bc(u, v, lid_vel)
        
        t = 0.0
        n_steps = 0
        residuals = []
        
        print(f"Mixed DG Cavity: Re={self.Re}, P{self.p}, {K} elements")
        print(f"  CFL={cfl}, gamma_p={self.gamma_p}")
        print("  Solving...", flush=True)
        
        t0_wall = time.perf_counter()
        
        while t < t_final - 1e-14 and n_steps < 500_000:
            # Compute max velocity for CFL
            u_max = np.max(np.abs(u))
            v_max = np.max(np.abs(v))
            vel_max = max(u_max, v_max, 1e-14)
            
            # CFL condition for convection
            dt_conv = cfl * self.h_min / ((2 * self.p + 1) * vel_max)
            
            # Viscous stability limit: dt <= h^2 / (2*nu)
            dt_visc = self.h_min**2 / (2 * self.nu) if self.nu > 0 else 1e10
            
            # Take the more restrictive
            dt = min(dt_conv, dt_visc)
            
            # Cap maximum dt to prevent blowup during startup
            dt_max_safe = 0.01 if n_steps < 100 else 0.1
            dt = min(dt, dt_max_safe)
            
            if dt_max is not None:
                dt = min(dt, dt_max)
            dt = min(dt, t_final - t)
            
            # Time step
            u, v, p = self.step(u, v, p, dt, lid_vel)
            
            t += dt
            n_steps += 1
            
            # Compute residual
            if n_steps % print_every == 0:
                rhs_u, rhs_v = self._compute_rhs(u, v, lid_vel)
                res = max(np.max(np.abs(rhs_u)), np.max(np.abs(rhs_v)))
                residuals.append((t, res))
                
                elapsed = time.perf_counter() - t0_wall
                div_max = np.max(np.abs(self._compute_divergence(u, v)))
                
                print(f"  step={n_steps:6d}  t={t:7.3f}  dt={dt:.4e}  "
                      f"res={res:.4e}  max|div|={div_max:.4e}  "
                      f"wall={elapsed:.1f}s", flush=True)
                
                if callback:
                    q = np.stack([u, v, p], axis=-1)
                    callback(q, t, n_steps, dt)
                
                if np.any(np.isnan(u)) or res > 1e6:
                    raise RuntimeError("Solution blew up")
                
                if res < 1e-8:
                    print(f"  Converged at step {n_steps}")
                    break
        
        elapsed = time.perf_counter() - t0_wall
        print(f"\n  Done: {n_steps} steps, t={t:.3f}, wall={elapsed:.1f}s")
        if residuals:
            print(f"  Final residual: {residuals[-1][1]:.4e}")
        
        return u, v, p, t, n_steps


# =============================================================================
# Convenience function for running cavity
# =============================================================================

def run_mixed_cavity(nx=16, p=2, Re=100.0, gamma_p=0.1, cfl=0.1,
                     t_final=30.0, print_every=500):
    """
    Run lid-driven cavity with mixed DG formulation.
    
    Parameters
    ----------
    nx : int
        Number of elements in each direction
    p : int
        Polynomial degree
    Re : float
        Reynolds number
    gamma_p : float
        Pressure stabilization parameter
    cfl : float
        CFL number
    t_final : float
        Final time
    print_every : int
        Print frequency
    
    Returns
    -------
    solver : MixedDGCavitySolver
    u, v, p : ndarray
        Final solution
    t : float
        Final time
    n_steps : int
        Number of steps
    """
    from .mesh.triangle_mesh import TriangleMesh
    
    mesh = TriangleMesh.rectangle((0.0, 1.0), (0.0, 1.0), nx, nx)
    
    solver = MixedDGCavitySolver(mesh, p=p, Re=Re, gamma_p=gamma_p)
    
    u, v, p, t, n_steps = solver.solve(
        t_final=t_final, cfl=cfl, print_every=print_every
    )
    
    return solver, u, v, p, t, n_steps
