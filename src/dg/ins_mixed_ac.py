"""
Mixed DG solver using Artificial Compressibility (AC) for pressure.

This avoids the Poisson solve entirely by using the AC pseudo-timestepping.

Key difference from projection method:
- Projection: solve ∇²p = (1/dt)∇·u* each step
- AC: update p via ∂p/∂t + β²∇·u = 0 (pseudo-timestepping)

We use a dual-timestepping approach:
1. Physical timestep for momentum equations
2. Sub-iterations for pressure to enforce ∇·u ≈ 0
"""

import numpy as np
from numba import njit, prange
from scipy import sparse
from scipy.sparse.linalg import spsolve
import time

from .dg_operators import (
    build_mass_matrix, apply_mass_inv,
    compute_divergence, compute_gradient
)
from .ins_mixed import (
    _convection_rhs_numba, _viscous_rhs_numba,
    MixedDGCavitySolver as BaseMixedSolver
)


class ACMixedDGCavitySolver(BaseMixedSolver):
    """
    Mixed DG solver using Artificial Compressibility for pressure.
    
    Instead of solving pressure Poisson, we use:
        ∂p/∂τ + β²∇·u = 0
    
    where τ is pseudo-time. At each physical timestep, we sub-iterate
    pressure to drive divergence to zero.
    
    Attributes
    ----------
    beta : float
        Artificial compressibility parameter
    n_subiter : int
        Number of pressure sub-iterations per timestep
    """
    
    def __init__(self, mesh, p=2, Re=100.0, beta=10.0, n_subiter=5, gamma_p=0.1):
        """
        Initialize AC-based mixed DG cavity solver.
        
        Parameters
        ----------
        mesh : TriangleMesh
        p : int
            Polynomial degree
        Re : float
            Reynolds number
        beta : float
            Artificial compressibility parameter (higher = more incompressible)
        n_subiter : int
            Number of pressure sub-iterations per timestep
        gamma_p : float
            Unused (for compatibility)
        """
        from .solver2d_system import DG2DSystem
        
        self.mesh = mesh
        self.p = p
        self.Re = Re
        self.nu = 1.0 / Re
        self.beta = beta
        self.n_subiter = n_subiter
        self.gamma_p = 0.0  # For compatibility with base class
        
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
        
        # Precompute geometry
        self._precompute_geometry()
        
        # Build mass matrix
        self.M = build_mass_matrix(self.solver)
        self.M_inv = sparse.diags(1.0 / self.M.diagonal(), format='csr')
        
        self.n_dof = self.solver.K * self.solver.Np
        
    def step(self, u, v, p, dt, lid_vel=1.0):
        """
        Take one time step using AC method.
        
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
        
        # --- Step 1: Velocity predictor (no pressure) ---
        rhs_u, rhs_v = self._compute_rhs(u, v, lid_vel)
        
        u_star = u + dt * rhs_u
        v_star = v + dt * rhs_v
        
        # --- Step 2: Pressure sub-iterations ---
        u_new, v_new, p_new = self._pressure_subiter(
            u_star, v_star, p, dt, lid_vel
        )
        
        return u_new, v_new, p_new
    
    def _pressure_subiter(self, u_star, v_star, p, dt, lid_vel):
        """
        Sub-iterate pressure to enforce ∇·u ≈ 0.
        
        Using AC: ∂p/∂τ = -β²∇·u
        """
        K, Np = self.solver.K, self.solver.Np
        
        u = u_star.copy()
        v = v_star.copy()
        p_new = p.copy()
        
        # Pseudo-timestep for pressure (local scaling)
        dtau = dt / self.n_subiter
        
        for _ in range(self.n_subiter):
            # Compute divergence
            div = self._compute_divergence(u, v)
            
            # Update pressure: p_new = p - dtau * beta² * div
            # Project div onto pressure space (mass matrix inverse)
            div_flat = div.ravel()
            M_div = self.M @ div_flat
            dpdt = -self.beta**2 * (self.M_inv @ M_div)
            
            p_flat = p_new.ravel() + dtau * dpdt
            p_new = p_flat.reshape(K, Np)
            
            # Update velocity to match new pressure
            # u = u_star - dt * ∇p
            grad_px, grad_py = self._compute_gradient(p_new - p)
            u = u_star - dt * grad_px
            v = v_star - dt * grad_py
            
            # Enforce BCs
            self._apply_velocity_bc(u, v, lid_vel)
        
        return u, v, p_new
    
    def _build_poisson_operator(self):
        """Not used for AC method."""
        pass
    
    def _solve_poisson(self, rhs_flat, tol=1e-8, maxiter=2000):
        """Not used for AC method."""
        pass


# =============================================================================
# Convenience function
# =============================================================================

def run_ac_cavity(nx=16, p=2, Re=100.0, beta=10.0, n_subiter=5,
                  cfl=0.1, t_final=30.0, print_every=500):
    """
    Run lid-driven cavity with AC-based mixed DG formulation.
    
    Parameters
    ----------
    nx : int
        Number of elements in each direction
    p : int
        Polynomial degree
    Re : float
        Reynolds number
    beta : float
        Artificial compressibility parameter
    n_subiter : int
        Number of pressure sub-iterations per timestep
    cfl : float
        CFL number
    t_final : float
        Final time
    print_every : int
        Print frequency
    
    Returns
    -------
    solver : ACMixedDGCavitySolver
    u, v, p : ndarray
        Final solution
    t : float
        Final time
    n_steps : int
        Number of steps
    """
    from .mesh.triangle_mesh import TriangleMesh
    
    mesh = TriangleMesh.rectangle((0.0, 1.0), (0.0, 1.0), nx, nx)
    
    solver = ACMixedDGCavitySolver(
        mesh, p=p, Re=Re, beta=beta, n_subiter=n_subiter
    )
    
    u, v, p, t, n_steps = solver.solve(
        t_final=t_final, cfl=cfl, print_every=print_every
    )
    
    return solver, u, v, p, t, n_steps
