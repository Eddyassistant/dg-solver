"""
Projection method DG solver for incompressible Navier-Stokes.

Uses Chorin's projection method with DG discretization:
1. Predictor: u* = u^n + dt*(-N(u^n) + nu*L(u^n))
2. Pressure Poisson: ∇²φ = (1/dt)∇·u*  with ∂φ/∂n = 0 (Neumann)
3. Correction: u^{n+1} = u* - dt*∇φ

The pressure Poisson equation uses SIPG discretization, yielding
a symmetric positive semi-definite (SPD) system. We pin one DOF
to make it non-singular.

References
----------
Chorin (1968), Math. Comp. 22, pp. 745-762
Shahbazi, Fischer, Ethier (2007), JCP 227(1), pp. 112-140
"""

import numpy as np
from numba import njit, prange
from scipy import sparse
from scipy.sparse.linalg import cg
import time

from .reference_triangle import RefTriangle
from .mesh.triangle_mesh import TriangleMesh, build_face_connectivity, geometric_factors, face_normals
from .timestepping.runge_kutta import ssp_rk3


# =============================================================================
# Convective fluxes (velocity only, no pressure coupling)
# =============================================================================

@njit(cache=True, parallel=True)
def conv_flux_x(u, v):
    """x-direction convective flux F = [u², uv]."""
    K, Np = u.shape
    Fx = np.empty((K, Np, 2), dtype=np.float64)
    for k in prange(K):
        for i in range(Np):
            Fx[k, i, 0] = u[k, i] * u[k, i]
            Fx[k, i, 1] = u[k, i] * v[k, i]
    return Fx


@njit(cache=True, parallel=True)
def conv_flux_y(u, v):
    """y-direction convective flux G = [uv, v²]."""
    K, Np = u.shape
    Fy = np.empty((K, Np, 2), dtype=np.float64)
    for k in prange(K):
        for i in range(Np):
            Fy[k, i, 0] = u[k, i] * v[k, i]
            Fy[k, i, 1] = v[k, i] * v[k, i]
    return Fy


@njit(cache=True, parallel=True)
def conv_numerical_flux(u_int, v_int, u_ext, v_ext, nx, ny):
    """
    Rusanov numerical flux for convection.
    
    F̂·n = ½(F_int + F_ext)·n - ½λ_max(q_ext - q_int)
    
    λ_max = max(|V·n| + |V·n|) over int/ext states.
    """
    K, n_fp = u_int.shape
    f_num_u = np.empty((K, n_fp), dtype=np.float64)
    f_num_v = np.empty((K, n_fp), dtype=np.float64)
    
    for k in prange(K):
        for i in range(n_fp):
            ui, vi = u_int[k, i], v_int[k, i]
            ue, ve = u_ext[k, i], v_ext[k, i]
            nx_ki, ny_ki = nx[k, i], ny[k, i]
            
            # V·n at interior and exterior
            Vn_i = ui * nx_ki + vi * ny_ki
            Vn_e = ue * nx_ki + ve * ny_ki
            
            # Wavespeed (max of |V·n| on both sides)
            lam = max(abs(Vn_i), abs(Vn_e))
            
            # F·n for interior
            fni_u = ui * ui * nx_ki + ui * vi * ny_ki
            fni_v = ui * vi * nx_ki + vi * vi * ny_ki
            
            # F·n for exterior
            fne_u = ue * ue * nx_ki + ue * ve * ny_ki
            fne_v = ue * ve * nx_ki + ve * ve * ny_ki
            
            # Rusanov flux
            f_num_u[k, i] = 0.5 * (fni_u + fne_u) - 0.5 * lam * (ue - ui)
            f_num_v[k, i] = 0.5 * (fni_v + fne_v) - 0.5 * lam * (ve - vi)
    
    return f_num_u, f_num_v


@njit(cache=True)
def conv_max_wavespeed(u, v):
    """Global maximum wavespeed for CFL computation."""
    K, Np = u.shape
    wmax = 0.0
    for k in range(K):
        for i in range(Np):
            vel = np.sqrt(u[k, i]**2 + v[k, i]**2)
            if vel > wmax:
                wmax = vel
    return wmax


# =============================================================================
# Divergence and gradient (local operations)
# =============================================================================

@njit(cache=True, parallel=True)
def compute_divergence(u, v, Dr, Ds, rx, sx, ry, sy, K, Np):
    """
    Compute ∇·(u,v) = ∂u/∂x + ∂v/∂y at volume nodes.
    
    Returns div: (K, Np)
    """
    div = np.empty((K, Np), dtype=np.float64)
    
    for k in prange(K):
        rx_k, sx_k = rx[k, 0], sx[k, 0]
        ry_k, sy_k = ry[k, 0], sy[k, 0]
        
        for i in range(Np):
            # ∂u/∂r, ∂u/∂s
            dudr = 0.0
            duds = 0.0
            dvdr = 0.0
            dvds = 0.0
            
            for j in range(Np):
                dudr += Dr[i, j] * u[k, j]
                duds += Ds[i, j] * u[k, j]
                dvdr += Dr[i, j] * v[k, j]
                dvds += Ds[i, j] * v[k, j]
            
            # Transform to physical
            dudx = rx_k * dudr + sx_k * duds
            dvdy = ry_k * dvdr + sy_k * dvds
            
            div[k, i] = dudx + dvdy
    
    return div


@njit(cache=True, parallel=True)
def compute_gradient(phi, Dr, Ds, rx, sx, ry, sy, K, Np):
    """
    Compute ∇φ = (∂φ/∂x, ∂φ/∂y) at volume nodes.
    
    Returns grad_x, grad_y: each (K, Np)
    """
    grad_x = np.empty((K, Np), dtype=np.float64)
    grad_y = np.empty((K, Np), dtype=np.float64)
    
    for k in prange(K):
        rx_k, sx_k = rx[k, 0], sx[k, 0]
        ry_k, sy_k = ry[k, 0], sy[k, 0]
        
        for i in range(Np):
            # ∂φ/∂r, ∂φ/∂s
            dpdr = 0.0
            dpds = 0.0
            
            for j in range(Np):
                dpdr += Dr[i, j] * phi[k, j]
                dpds += Ds[i, j] * phi[k, j]
            
            # Transform to physical
            grad_x[k, i] = rx_k * dpdr + sx_k * dpds
            grad_y[k, i] = ry_k * dpdr + sy_k * dpds
    
    return grad_x, grad_y


# =============================================================================
# SIPG Laplacian matrix assembly
# =============================================================================

@njit(cache=True)
def _sipg_laplacian_kernel(
    Dr, Ds, LIFT, Fmask_flat,
    rx, ry, sx, sy, J, Fscale,
    nx, ny, vmapP_k, vmapP_n, is_boundary,
    sigma_ip, K, Np, Nfp, n_dof
):
    """
    Assemble global SIPG Laplacian matrix entries.
    
    The bilinear form for -∇²φ with SIPG:
    a(φ,ψ) = Σ_E ∫∇φ·∇ψ dx - Σ_f ∫{{∇φ}}·n[[ψ]] ds - Σ_f ∫{{∇ψ}}·n[[φ]] ds
             + Σ_f σ_IP/h_f ∫[[φ]][[ψ]] ds
    
    Returns row_ind, col_ind, data for COO format.
    """
    n_fp = 3 * Nfp
    
    # Estimate nonzeros: each row has Np (volume) + n_face_neighbors (surface)
    # Conservative estimate: 4*Np per row
    max_nnz = n_dof * 4 * Np
    
    row_ind = np.empty(max_nnz, dtype=np.int64)
    col_ind = np.empty(max_nnz, dtype=np.int64)
    data = np.empty(max_nnz, dtype=np.float64)
    nnz = 0
    
    # Mass matrix inverse (diagonal for nodal basis on triangle)
    # For simplicity, we use the Jacobian-scaled mass inverse
    # M^{-1} ≈ 1/J * V*V^T type approximation or exact
    # Here we use the fact that for DG, M is block diagonal per element
    
    # Reference element mass matrix (modal) or nodal
    # For this implementation, we work in strong form
    
    for k in range(K):
        Jk = J[k, 0]
        rx_k, ry_k = rx[k, 0], ry[k, 0]
        sx_k, sy_k = sx[k, 0], sy[k, 0]
        
        # Element local matrix: volume + face contributions
        # A_k[i,j] = contribution to dof i from dof j
        A_local = np.zeros((Np, Np), dtype=np.float64)
        
        # --- Volume term: ∫∇φ_j·∇ψ_i dx ---
        for i in range(Np):
            for j in range(Np):
                # Compute ∇φ_j at node i (via differentiation matrix)
                # This is approximate - better to integrate properly
                # For now, use Galerkin projection
                
                # ∂φ_j/∂x at all nodes via Dr, Ds
                dphi_j_dr = Dr[:, j]  # column j of Dr gives ∂φ_j/∂r at all nodes
                dphi_j_ds = Ds[:, j]
                
                dpsi_i_dr = Dr[:, i]
                dpsi_i_ds = Ds[:, i]
                
                # At node i: ∇φ_j · ∇ψ_i
                dphi_dx = rx_k * dphi_j_dr[i] + sx_k * dphi_j_ds[i]
                dphi_dy = ry_k * dphi_j_dr[i] + sy_k * dphi_j_ds[i]
                
                dpsi_dx = rx_k * dpsi_i_dr[i] + sx_k * dpsi_i_ds[i]
                dpsi_dy = ry_k * dpsi_i_dr[i] + sy_k * dpsi_i_ds[i]
                
                # Contribution to A_local (scaled by Jacobian)
                A_local[i, j] += (dphi_dx * dpsi_dx + dphi_dy * dpsi_dy) * Jk
        
        # --- Face terms (SIPG) ---
        for fpt in range(n_fp):
            vol_idx = Fmask_flat[fpt]
            pk = vmapP_k[k, fpt]
            pn = vmapP_n[fpt] if is_boundary[k, fpt] else vmapP_n[k, fpt]
            
            nx_f, ny_f = nx[k, fpt], ny[k, fpt]
            fs = Fscale[k, fpt]
            hk = 1.0 / fs  # characteristic length
            
            # Penalty parameter
            tau = sigma_ip / hk
            
            # For boundary (Neumann), exterior is same as interior
            # {{∇φ}} = ∇φ, [[φ]] = 0 (for Neumann BC ∂φ/∂n = 0)
            
            if is_boundary[k, fpt]:
                # Boundary face with Neumann BC
                # Only penalty term applies to [[φ]][[ψ]] = 0 since [[φ]]=0
                # Actually for Neumann: F̂·n = {{∇φ}}·n - σ[[φ]] = ∇φ·n
                # So the jump terms vanish, only volume term contributes
                # BUT we need to handle the boundary correctly for consistency
                pass  # Natural BC: no additional terms
            else:
                # Interior face - couple with neighbor
                # This is complex in local assembly - better to do global
                pass
        
        # Add local matrix to global (for now, just diagonal blocks)
        for i in range(Np):
            global_i = k * Np + i
            for j in range(Np):
                global_j = k * Np + j
                row_ind[nnz] = global_i
                col_ind[nnz] = global_j
                data[nnz] = A_local[i, j]
                nnz += 1
    
    return row_ind[:nnz], col_ind[:nnz], data[:nnz]


def assemble_sipg_laplacian(solver, sigma_ip=None):
    """
    Assemble global SIPG Laplacian matrix for pressure Poisson equation.
    
    Uses weak form DG discretization of -∇²φ.
    
    Parameters
    ----------
    solver : ProjectionDGSolver
    sigma_ip : float, optional
        Interior penalty parameter. Default: C*(p+1)*(p+2)
    
    Returns
    -------
    A : scipy.sparse.csr_matrix
        The Laplacian matrix (n_dof × n_dof)
    """
    p = solver.p
    K, Np = solver.K, solver.Np
    n_dof = K * Np
    
    if sigma_ip is None:
        # Shahbazi (2005) penalty parameter
        sigma_ip = 4.0 * (p + 1) * (p + 2) / 2.0
    
    # Get geometric factors
    rx, ry, sx, sy = solver.rx, solver.ry, solver.sx, solver.sy
    J = solver.J
    Fscale = solver.Fscale
    nx, ny = solver.nx, solver.ny
    
    # Reference element matrices
    V = solver.ref.V
    Vinv = solver.ref.Vinv
    Dr = solver.ref.Dr
    Ds = solver.ref.Ds
    LIFT = solver.ref.LIFT
    
    # Mass matrix and stiffness matrix on reference element
    # M = V^{-T} V^{-1}
    M = np.linalg.inv(V @ V.T)
    
    # Build global matrix in COO format
    rows, cols, vals = [], [], []
    
    # Face node mask
    Fmask_flat = solver.Fmask_flat
    n_fp = 3 * solver.Nfp
    
    # Element-to-global map
    def global_idx(k, i):
        return k * Np + i
    
    for k in range(K):
        Jk = J[k, 0]
        rx_k, ry_k = rx[k, 0], ry[k, 0]
        sx_k, sy_k = sx[k, 0], sy[k, 0]
        
        # Transform differentiation matrices to physical
        # ∂/∂x = rx * ∂/∂r + sx * ∂/∂s
        Dx = rx_k * Dr + sx_k * Ds
        Dy = ry_k * Dr + sy_k * Ds
        
        # Element stiffness matrix: ∫∇φ·∇ψ dx
        # K_loc = Jk * (Dx^T @ M @ Dx + Dy^T @ M @ Dy)
        K_loc = Jk * (Dx.T @ M @ Dx + Dy.T @ M @ Dy)
        
        # Add volume terms
        for i in range(Np):
            gi = global_idx(k, i)
            for j in range(Np):
                gj = global_idx(k, j)
                rows.append(gi)
                cols.append(gj)
                vals.append(K_loc[i, j])
        
        # Face contributions (interior faces only - contribute twice)
        for fpt in range(n_fp):
            vol_i = Fmask_flat[fpt]
            
            pk = solver.vmapP_k[k, fpt]
            pn = solver.vmapP_n[k, fpt] if not solver.is_boundary[k, fpt] else Fmask_flat[fpt]
            
            is_bdry = solver.is_boundary[k, fpt]
            
            # Face normal
            nx_f, ny_f = nx[k, fpt], ny[k, fpt]
            fs = Fscale[k, fpt]
            hk = 1.0 / fs
            tau = sigma_ip / hk
            
            # Compute gradients of basis functions at face nodes
            # ∇φ_j at face node
            grad_phi = np.zeros((Np, 2))  # j -> (dx, dy)
            for j in range(Np):
                grad_phi[j, 0] = rx_k * Dr[vol_i, j] + sx_k * Ds[vol_i, j]
                grad_phi[j, 1] = ry_k * Dr[vol_i, j] + sy_k * Ds[vol_i, j]
            
            if is_bdry:
                # Neumann BC: ∂φ/∂n = 0
                # Natural BC - no penalty, no jump
                # The boundary term is: ∫∂φ/∂n * ψ ds = 0
                pass
            else:
                # Interior face
                # Contributions to both elements k and pk
                
                # Face integral weights (simplified - use LIFT weights)
                # Actually need proper face quadrature
                w_face = 1.0 / fs * solver.sJ[k, fpt % solver.Nfp] / 3.0  # approximate
                
                # Average and jump operators
                # {{∇φ}} = 0.5*(∇φ_k + ∇φ_pk)
                # [[φ]] = φ_k - φ_pk
                
                # For interior faces, we need neighbor's gradient
                # This creates off-diagonal blocks
                
                # Skip for now - will add simplified version
                pass
    
    # Build sparse matrix
    A = sparse.coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof))
    A = A.tocsr()
    
    # Symmetrize: (A + A^T)/2 to ensure SPD
    A = 0.5 * (A + A.T)
    
    return A


def assemble_sipg_laplacian_full(solver, sigma_ip=None):
    """
    Assemble full SIPG Laplacian matrix for pressure Poisson equation.
    
    Uses weak form DG discretization of -∇²φ with full inter-element coupling.
    
    Parameters
    ----------
    solver : ProjectionDGSolver
    sigma_ip : float, optional
        Interior penalty parameter. Default: C*(p+1)*(p+2)
    
    Returns
    -------
    A : scipy.sparse.csr_matrix
        The Laplacian matrix (n_dof × n_dof)
    """
    p = solver.p
    K, Np = solver.K, solver.Np
    n_dof = K * Np
    
    if sigma_ip is None:
        # Shahbazi (2005) penalty parameter
        sigma_ip = 4.0 * (p + 1) * (p + 2) / 2.0
    
    # Get geometric factors
    rx, ry, sx, sy = solver.rx, solver.ry, solver.sx, solver.sy
    J = solver.J
    Fscale = solver.Fscale
    nx, ny = solver.nx, solver.ny
    
    # Reference element matrices
    V = solver.ref.V
    Dr = solver.ref.Dr
    Ds = solver.ref.Ds
    
    # Mass matrix and differentiation matrices on reference element
    M = np.linalg.inv(V @ V.T)
    
    # Face node mask
    Fmask_flat = solver.Fmask_flat
    Nfp = solver.Nfp
    n_fp = 3 * Nfp
    
    # Element-to-global map
    def global_idx(k, i):
        return k * Np + i
    
    # Build global matrix in COO format
    rows, cols, vals = [], [], []
    
    # Precompute element stiffness matrices and face operators
    for k in range(K):
        Jk = J[k, 0]
        rx_k, ry_k = rx[k, 0], ry[k, 0]
        sx_k, sy_k = sx[k, 0], sy[k, 0]
        
        # Physical differentiation matrices
        Dx = rx_k * Dr + sx_k * Ds
        Dy = ry_k * Dr + sy_k * Ds
        
        # Element stiffness matrix: ∫∇φ·∇ψ dx = Jk * (Dx^T @ M @ Dx + Dy^T @ M @ Dy)
        K_stiff = Jk * (Dx.T @ M @ Dx + Dy.T @ M @ Dy)
        
        # Add volume terms
        for i in range(Np):
            gi = global_idx(k, i)
            for j in range(Np):
                gj = global_idx(k, j)
                rows.append(gi)
                cols.append(gj)
                vals.append(K_stiff[i, j])
        
        # Face contributions (interior faces only - contribute twice)
        for f in range(3):
            for i_face in range(Nfp):
                fpt = f * Nfp + i_face
                vol_i = Fmask_flat[fpt]
                
                is_bdry = solver.is_boundary[k, fpt]
                
                if is_bdry:
                    # Neumann BC: natural boundary condition
                    continue
                
                # Interior face - add penalty coupling
                # Simplified: just add penalty between matching nodes
                pk = solver.vmapP_k[k, fpt]
                pn = solver.vmapP_n[k, fpt]
                
                fs = Fscale[k, fpt]
                hk = 1.0 / fs
                
                # Penalty parameter: σ/h
                tau = sigma_ip / hk
                
                # Penalty term: τ * [[φ]] [[ψ]] = τ * (φ_k - φ_pk) * (ψ_k - ψ_pk)
                # This gives:
                #   τ * φ_k * ψ_k on diagonal (k,k)
                #  -τ * φ_k * ψ_pk on off-diagonal (k,pk)
                
                gi = global_idx(k, vol_i)
                gpn = global_idx(pk, pn)
                
                # Diagonal contribution
                rows.append(gi)
                cols.append(gi)
                vals.append(tau)
                
                # Off-diagonal contribution
                rows.append(gi)
                cols.append(gpn)
                vals.append(-tau)
    
    # Build sparse matrix
    A = sparse.coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof))
    A = A.tocsr()
    
    return A


def assemble_simplified_laplacian(solver, sigma_ip=None):
    """
    Assemble simplified block-diagonal Laplacian for projection.
    
    This is a simplified version that only uses volume terms.
    For true projection, we need the full SIPG coupling.
    
    Parameters
    ----------
    solver : ProjectionDGSolver
    sigma_ip : float, optional
        Penalty parameter (unused in simplified version)
    
    Returns
    -------
    A : scipy.sparse.csr_matrix
        Block diagonal Laplacian approximation
    """
    p = solver.p
    K, Np = solver.K, solver.Np
    n_dof = K * Np
    
    # Reference element matrices
    V = solver.ref.V
    Dr = solver.ref.Dr
    Ds = solver.ref.Ds
    
    # Mass matrix
    M = np.linalg.inv(V @ V.T)
    
    # Build block-diagonal matrix
    rows, cols, vals = [], [], []
    
    for k in range(K):
        rx_k, ry_k = solver.rx[k, 0], solver.ry[k, 0]
        sx_k, sy_k = solver.sx[k, 0], solver.sy[k, 0]
        Jk = solver.J[k, 0]
        
        # Physical differentiation matrices
        Dx = rx_k * Dr + sx_k * Ds
        Dy = ry_k * Dr + sy_k * Ds
        
        # Element stiffness
        K_loc = Jk * (Dx.T @ M @ Dx + Dy.T @ M @ Dy)
        
        # Add small regularization for positive definiteness
        for i in range(Np):
            K_loc[i, i] += 1e-12
        
        for i in range(Np):
            gi = k * Np + i
            for j in range(Np):
                gj = k * Np + j
                rows.append(gi)
                cols.append(gj)
                vals.append(K_loc[i, j])
    
    A = sparse.coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof))
    return A.tocsr()


# =============================================================================
# Main solver class
# =============================================================================

class ProjectionDGSolver:
    """
    DG projection method solver for incompressible Navier-Stokes.
    
    Solves:
        ∂u/∂t + (u·∇)u = -∇p + (1/Re)∇²u
        ∇·u = 0
    
    Using Chorin's projection method with DG spatial discretization.
    
    Parameters
    ----------
    mesh : TriangleMesh
    p : int
        Polynomial degree
    Re : float
        Reynolds number
    lid_velocity : float
        Lid velocity for cavity flow
    """
    
    def __init__(self, mesh: TriangleMesh, p: int, Re: float, 
                 lid_velocity: float = 1.0, bc_tags=None):
        self.mesh = mesh
        self.p = p
        self.Re = Re
        self.nu = 1.0 / Re
        self.lid_velocity = lid_velocity
        self.K = mesh.n_elem
        self.bc_tags = bc_tags
        
        # Reference element
        self.ref = RefTriangle(p)
        self.Np = self.ref.Np
        self.Nfp = self.ref.Nfp
        
        # Geometric factors
        self.x, self.y, self.rx, self.ry, self.sx, self.sy, self.J = \
            geometric_factors(mesh, self.ref, p)
        
        self.nx, self.ny, self.sJ, self.Fscale = \
            face_normals(mesh, self.ref, self.x, self.y, self.J)
        
        self.Fmask_flat = np.concatenate(self.ref.Fmask).astype(np.int64)
        
        # Connectivity
        self.EToE, self.EToF = build_face_connectivity(
            mesh, periodic_x=False, periodic_y=False,
            x_range=(0.0, 1.0), y_range=(0.0, 1.0),
        )
        
        self._build_maps()
        
        # Store contiguous arrays for Numba
        self.Dr = np.ascontiguousarray(self.ref.Dr)
        self.Ds = np.ascontiguousarray(self.ref.Ds)
        self.LIFT = np.ascontiguousarray(self.ref.LIFT)
        
        # Precompute operators
        self._precompute_operators()
        
        # Build pressure Poisson matrix (simplified for now)
        self._build_pressure_matrix()
        
        # Warmup
        self._warmup()
    
    def _build_maps(self):
        """Build face-to-neighbor maps."""
        K, Nfp = self.K, self.Nfp
        Fmask = self.ref.Fmask
        EToE, EToF = self.EToE, self.EToF
        n_fp = 3 * Nfp
        
        self.vmapP_k = np.zeros((K, n_fp), dtype=np.int64)
        self.vmapP_n = np.zeros((K, n_fp), dtype=np.int64)
        self.is_boundary = np.zeros((K, n_fp), dtype=np.bool_)
        
        for k in range(K):
            for f in range(3):
                k2 = EToE[k, f]
                f2 = EToF[k, f]
                for i in range(Nfp):
                    idx = f * Nfp + i
                    if k2 == k and f2 == f:
                        self.vmapP_k[k, idx] = k
                        self.vmapP_n[k, idx] = Fmask[f][i]
                        self.is_boundary[k, idx] = True
                    else:
                        self.vmapP_k[k, idx] = k2
                        self.vmapP_n[k, idx] = Fmask[f2][Nfp - 1 - i]
                        self.is_boundary[k, idx] = False
    
    def _precompute_operators(self):
        """Precompute mass matrix inverse and projection operators."""
        # Mass matrix inverse (modal)
        V = self.ref.V
        self.M_inv = np.linalg.inv(V @ V.T)
        
        # For strong-form DG, we use differentiation matrices directly
        
    def _build_pressure_matrix(self):
        """Build and factorize the pressure Poisson matrix."""
        # Use full SIPG Laplacian with element coupling
        sigma_ip = 100.0  # Large penalty for stability
        A = assemble_sipg_laplacian_full(self, sigma_ip=sigma_ip)
        
        # Pin one DOF to make system non-singular
        # For Neumann BCs, solution is unique up to constant
        self.pinned_dof = 0
        A = A.tolil()
        # Zero out row and column for pinned DOF
        A[self.pinned_dof, :] = 0
        A[:, self.pinned_dof] = 0
        A[self.pinned_dof, self.pinned_dof] = 1.0
        self.A_pressure = A.tocsr()
    
    def _warmup(self):
        """Warm up Numba kernels."""
        K, Np = self.K, self.Np
        u = np.zeros((K, Np))
        v = np.zeros((K, Np))
        
        # Warmup divergence
        compute_divergence(u, v, self.Dr, self.Ds, 
                          self.rx, self.sx, self.ry, self.sy, K, Np)
        
        # Warmup gradient
        phi = np.zeros((K, Np))
        compute_gradient(phi, self.Dr, self.Ds,
                        self.rx, self.sx, self.ry, self.sy, K, Np)
        
        # Warmup convection fluxes
        conv_flux_x(u, v)
        conv_flux_y(u, v)
    
    def extract_face_values(self, u, v):
        """Extract interior and exterior face values."""
        K, Nfp = self.K, self.Nfp
        n_fp = 3 * Nfp
        Fmask = self.Fmask_flat
        
        u_int = u[:, Fmask]
        v_int = v[:, Fmask]
        
        u_ext = np.empty((K, n_fp), dtype=np.float64)
        v_ext = np.empty((K, n_fp), dtype=np.float64)
        
        for k in range(K):
            for idx in range(n_fp):
                if self.is_boundary[k, idx]:
                    # Apply BCs
                    bc = self._get_bc_tag(k, idx)
                    vol_idx = Fmask[idx]
                    
                    if bc == 1:  # Wall
                        u_ext[k, idx] = -u[k, vol_idx]
                        v_ext[k, idx] = -v[k, vol_idx]
                    elif bc == 2:  # Lid
                        u_ext[k, idx] = 2.0 * self.lid_velocity - u[k, vol_idx]
                        v_ext[k, idx] = -v[k, vol_idx]
                    else:  # Should not happen
                        u_ext[k, idx] = u[k, vol_idx]
                        v_ext[k, idx] = v[k, vol_idx]
                else:
                    pk = self.vmapP_k[k, idx]
                    pn = self.vmapP_n[k, idx]
                    u_ext[k, idx] = u[pk, pn]
                    v_ext[k, idx] = v[pk, pn]
        
        return u_int, v_int, u_ext, v_ext
    
    def _get_bc_tag(self, k, idx):
        """Get BC tag for face point."""
        if self.bc_tags is None:
            return 0
        f = idx // self.Nfp
        return self.bc_tags[k, f]
    
    def compute_convection_rhs(self, u, v):
        """Compute convective term -(u·∇)u using Rusanov flux."""
        K, Np, Nfp = self.K, self.Np, self.Nfp
        n_fp = 3 * Nfp
        
        # Physical fluxes
        Fx = conv_flux_x(u, v)
        Fy = conv_flux_y(u, v)
        
        # Face values
        u_int, v_int, u_ext, v_ext = self.extract_face_values(u, v)
        
        # Numerical flux
        f_num_u, f_num_v = conv_numerical_flux(u_int, v_int, u_ext, v_ext,
                                               self.nx, self.ny)
        
        # RHS kernel
        rhs_u = np.zeros((K, Np), dtype=np.float64)
        rhs_v = np.zeros((K, Np), dtype=np.float64)
        
        for k in range(K):
            rx_k, ry_k = self.rx[k, 0], self.ry[k, 0]
            sx_k, sy_k = self.sx[k, 0], self.sy[k, 0]
            
            for i in range(Np):
                # Volume term: -∂F/∂x - ∂G/∂y
                dfx_dr = 0.0
                dfx_ds = 0.0
                dfy_dr = 0.0
                dfy_ds = 0.0
                
                for j in range(Np):
                    dfx_dr += self.Dr[i, j] * Fx[k, j, 0]
                    dfx_ds += self.Ds[i, j] * Fx[k, j, 0]
                    dfy_dr += self.Dr[i, j] * Fy[k, j, 0]
                    dfy_ds += self.Ds[i, j] * Fy[k, j, 0]
                
                dfx_dx = rx_k * dfx_dr + sx_k * dfx_ds
                dfy_dy = ry_k * dfy_dr + sy_k * dfy_ds
                rhs_u[k, i] = -(dfx_dx + dfy_dy)
                
                dfx_dr = 0.0
                dfx_ds = 0.0
                dfy_dr = 0.0
                dfy_ds = 0.0
                
                for j in range(Np):
                    dfx_dr += self.Dr[i, j] * Fx[k, j, 1]
                    dfx_ds += self.Ds[i, j] * Fx[k, j, 1]
                    dfy_dr += self.Dr[i, j] * Fy[k, j, 1]
                    dfy_ds += self.Ds[i, j] * Fy[k, j, 1]
                
                dfx_dx = rx_k * dfx_dr + sx_k * dfx_ds
                dfy_dy = ry_k * dfy_dr + sy_k * dfy_ds
                rhs_v[k, i] = -(dfx_dx + dfy_dy)
            
            # Surface correction
            for fpt in range(n_fp):
                vol_idx = self.Fmask_flat[fpt]
                fx_phys_n = Fx[k, vol_idx, 0] * self.nx[k, fpt] + Fy[k, vol_idx, 0] * self.ny[k, fpt]
                fy_phys_n = Fx[k, vol_idx, 1] * self.nx[k, fpt] + Fy[k, vol_idx, 1] * self.ny[k, fpt]
                
                du_surf = self.Fscale[k, fpt] * (fx_phys_n - f_num_u[k, fpt])
                dv_surf = self.Fscale[k, fpt] * (fy_phys_n - f_num_v[k, fpt])
                
                for i in range(Np):
                    rhs_u[k, i] += self.LIFT[i, fpt] * du_surf
                    rhs_v[k, i] += self.LIFT[i, fpt] * dv_surf
        
        return rhs_u, rhs_v
    
    def compute_viscous_rhs(self, u, v):
        """Compute viscous term ν∇²u using SIPG."""
        K, Np, Nfp = self.K, self.Np, self.Nfp
        n_fp = 3 * Nfp
        nu = self.nu
        
        # SIPG penalty parameter
        C_ip = 4.0
        sigma_ip = C_ip * (self.p + 1) * (self.p + 2) * nu / 2.0
        
        rhs_u = np.zeros((K, Np), dtype=np.float64)
        rhs_v = np.zeros((K, Np), dtype=np.float64)
        
        for k in range(K):
            rx_k, ry_k = self.rx[k, 0], self.ry[k, 0]
            sx_k, sy_k = self.sx[k, 0], self.sy[k, 0]
            
            # Compute gradients
            ux = np.zeros(Np)
            uy = np.zeros(Np)
            vx = np.zeros(Np)
            vy = np.zeros(Np)
            
            for i in range(Np):
                for j in range(Np):
                    dudr = self.Dr[i, j] * u[k, j]
                    duds = self.Ds[i, j] * u[k, j]
                    ux[i] = rx_k * dudr + sx_k * duds
                    uy[i] = ry_k * dudr + sy_k * duds
                    
                    dvdr = self.Dr[i, j] * v[k, j]
                    dvds = self.Ds[i, j] * v[k, j]
                    vx[i] = rx_k * dvdr + sx_k * dvds
                    vy[i] = ry_k * dvdr + sy_k * dvds
            
            # Volume term: ∇·(ν∇u) = ν∇²u
            for i in range(Np):
                dux_dr = 0.0
                dux_ds = 0.0
                duy_dr = 0.0
                duy_ds = 0.0
                
                for j in range(Np):
                    dux_dr += self.Dr[i, j] * ux[j]
                    dux_ds += self.Ds[i, j] * ux[j]
                    duy_dr += self.Dr[i, j] * uy[j]
                    duy_ds += self.Ds[i, j] * uy[j]
                
                rhs_u[k, i] = nu * (rx_k * dux_dr + sx_k * dux_ds + 
                                    ry_k * duy_dr + sy_k * duy_ds)
                
                dvx_dr = 0.0
                dvx_ds = 0.0
                dvy_dr = 0.0
                dvy_ds = 0.0
                
                for j in range(Np):
                    dvx_dr += self.Dr[i, j] * vx[j]
                    dvx_ds += self.Ds[i, j] * vx[j]
                    dvy_dr += self.Dr[i, j] * vy[j]
                    dvy_ds += self.Ds[i, j] * vy[j]
                
                rhs_v[k, i] = nu * (rx_k * dvx_dr + sx_k * dvx_ds + 
                                    ry_k * dvy_dr + sy_k * dvy_ds)
            
            # Surface correction (SIPG)
            for fpt in range(n_fp):
                vol_idx = self.Fmask_flat[fpt]
                
                u_int = u[k, vol_idx]
                v_int = v[k, vol_idx]
                grad_n_u_int = ux[vol_idx] * self.nx[k, fpt] + uy[vol_idx] * self.ny[k, fpt]
                grad_n_v_int = vx[vol_idx] * self.nx[k, fpt] + vy[vol_idx] * self.ny[k, fpt]
                
                if self.is_boundary[k, fpt]:
                    bc = self._get_bc_tag(k, fpt)
                    if bc == 1:  # Wall
                        u_ext = -u_int
                        v_ext = -v_int
                    elif bc == 2:  # Lid
                        u_ext = 2.0 * self.lid_velocity - u_int
                        v_ext = -v_int
                    else:
                        u_ext = u_int
                        v_ext = v_int
                    grad_n_u_ext = -grad_n_u_int
                    grad_n_v_ext = -grad_n_v_int
                else:
                    pk = self.vmapP_k[k, fpt]
                    pn = self.vmapP_n[k, fpt]
                    u_ext = u[pk, pn]
                    v_ext = v[pk, pn]
                    
                    # Need neighbor's gradient - simplified: assume similar
                    grad_n_u_ext = grad_n_u_int
                    grad_n_v_ext = grad_n_v_int
                
                jump_u = u_int - u_ext
                jump_v = v_int - v_ext
                
                avg_grad_n_u = 0.5 * (grad_n_u_int + grad_n_u_ext)
                avg_grad_n_v = 0.5 * (grad_n_v_int + grad_n_v_ext)
                
                fs = self.Fscale[k, fpt]
                
                surf_u = fs * (nu * (avg_grad_n_u - grad_n_u_int) - sigma_ip * jump_u)
                surf_v = fs * (nu * (avg_grad_n_v - grad_n_v_int) - sigma_ip * jump_v)
                
                for i in range(Np):
                    rhs_u[k, i] += self.LIFT[i, fpt] * surf_u
                    rhs_v[k, i] += self.LIFT[i, fpt] * surf_v
        
        return rhs_u, rhs_v
    
    def solve_pressure_poisson(self, div_u_star, dt):
        """
        Solve ∇²φ = (1/dt)∇·u* for pressure correction.
        
        Returns φ as (K, Np) array.
        """
        K, Np = self.K, self.Np
        n_dof = K * Np
        
        # RHS: (1/dt) * divergence
        rhs = div_u_star.flatten() / dt
        
        # Pin one DOF
        rhs[self.pinned_dof] = 0.0
        
        # Solve using CG with higher tolerance and more iterations
        phi_flat, info = cg(self.A_pressure, rhs, rtol=1e-8, maxiter=5000, M=None)
        
        if info != 0:
            # If CG fails, try using a direct solver
            from scipy.sparse.linalg import spsolve
            phi_flat = spsolve(self.A_pressure, rhs)
        
        return phi_flat.reshape((K, Np))
    
    def step(self, u, v, p, dt):
        """
        Take one projection timestep.
        
        Parameters
        ----------
        u, v, p : ndarray (K, Np)
            Current velocity and pressure
        dt : float
            Timestep
        
        Returns
        -------
        u_new, v_new, p_new : ndarray (K, Np)
            Updated velocity and pressure
        """
        # Step 1: Predictor (explicit)
        rhs_conv_u, rhs_conv_v = self.compute_convection_rhs(u, v)
        rhs_visc_u, rhs_visc_v = self.compute_viscous_rhs(u, v)
        
        u_star = u + dt * (rhs_conv_u + rhs_visc_u)
        v_star = v + dt * (rhs_conv_v + rhs_visc_v)
        
        # Apply BCs to predictor (strong enforcement)
        self._apply_bc_strong(u_star, v_star)
        
        # Step 2: Pressure Poisson
        div_u_star = compute_divergence(u_star, v_star, self.Dr, self.Ds,
                                       self.rx, self.sx, self.ry, self.sy,
                                       self.K, self.Np)
        
        phi = self.solve_pressure_poisson(div_u_star, dt)
        
        # Step 3: Correction
        grad_phi_x, grad_phi_y = compute_gradient(phi, self.Dr, self.Ds,
                                                  self.rx, self.sx, self.ry, self.sy,
                                                  self.K, self.Np)
        
        u_new = u_star - dt * grad_phi_x
        v_new = v_star - dt * grad_phi_y
        
        # Pressure update
        p_new = p + phi
        
        # Apply BCs (strong enforcement)
        self._apply_bc_strong(u_new, v_new)
        
        return u_new, v_new, p_new
    
    def _apply_bc_strong(self, u, v):
        """Apply no-slip BCs strongly at boundary nodes."""
        if self.bc_tags is None:
            return
        
        K, Nfp = self.K, self.Nfp
        n_fp = 3 * Nfp
        
        for k in range(K):
            for f in range(3):
                bc = self.bc_tags[k, f]
                if bc == 0:
                    continue
                
                for i in range(Nfp):
                    idx = f * Nfp + i
                    vol_idx = self.Fmask_flat[idx]
                    
                    if bc == 1:  # Wall
                        u[k, vol_idx] = 0.0
                        v[k, vol_idx] = 0.0
                    elif bc == 2:  # Lid
                        u[k, vol_idx] = self.lid_velocity
                        v[k, vol_idx] = 0.0
    
    def compute_cfl_dt(self, u, v, cfl=0.5):
        """Compute timestep based on CFL condition."""
        vel_max = conv_max_wavespeed(u, v) + 1e-14
        
        # Minimum element size
        h_min = np.min(2.0 * np.abs(self.J[:, 0]) / 
                       np.max(self.sJ.reshape(self.K, 3, self.Nfp), axis=2).max(axis=1))
        
        # DG CFL: dt <= CFL * h / ((2p+1) * vel_max)
        dt = cfl * h_min / ((2 * self.p + 1) * vel_max)
        
        return dt
    
    def solve(self, u0, v0, p0, t_final, cfl=0.5, callback=None, max_steps=100000):
        """
        Time-march to t_final using projection method.
        
        Parameters
        ----------
        u0, v0, p0 : ndarray (K, Np)
            Initial conditions
        t_final : float
            Final time
        cfl : float
            CFL number
        callback : callable, optional
            Called as callback(u, v, p, t, step, dt)
        
        Returns
        -------
        u, v, p : ndarray (K, Np)
            Final solution
        t : float
            Final time
        n_steps : int
            Number of steps taken
        """
        u, v, p = u0.copy(), v0.copy(), p0.copy()
        t = 0.0
        step = 0
        
        while t < t_final - 1e-14 and step < max_steps:
            dt = self.compute_cfl_dt(u, v, cfl)
            dt = min(dt, t_final - t)
            
            u, v, p = self.step(u, v, p, dt)
            
            t += dt
            step += 1
            
            if callback is not None:
                callback(u, v, p, t, step, dt)
            
            if step % 1000 == 0:
                div = compute_divergence(u, v, self.Dr, self.Ds,
                                        self.rx, self.sx, self.ry, self.sy,
                                        self.K, self.Np)
                div_norm = np.max(np.abs(div))
                print(f"Step {step}, t={t:.4f}, dt={dt:.6f}, max|div|={div_norm:.6e}")
        
        return u, v, p, t, step


def run_cavity(nx=16, p=2, Re=100, t_final=10.0, cfl=0.5, lid_velocity=1.0):
    """
    Run lid-driven cavity flow with projection method.
    
    Parameters
    ----------
    nx : int
        Number of elements in each direction (mesh is nx × nx)
    p : int
        Polynomial degree
    Re : float
        Reynolds number
    t_final : float
        Final time
    cfl : float
        CFL number
    lid_velocity : float
        Lid velocity
    
    Returns
    -------
    solver : ProjectionDGSolver
    u, v, p : solution arrays
    """
    from .mesh.cavity import cavity_mesh
    
    print(f"Projection DG Cavity: nx={nx}, p={p}, Re={Re}")
    print(f"Domain: [0,1]², t_final={t_final}, CFL={cfl}")
    
    # Create mesh
    mesh, bc_tags = cavity_mesh(nx, nx)
    
    # Create solver
    solver = ProjectionDGSolver(mesh, p, Re, lid_velocity, bc_tags)
    
    # Initial condition: zero velocity, zero pressure
    K, Np = solver.K, solver.Np
    u0 = np.zeros((K, Np))
    v0 = np.zeros((K, Np))
    p0 = np.zeros((K, Np))
    
    # Apply BCs to IC
    solver._apply_bc_strong(u0, v0)
    
    print(f"Mesh: {K} elements, {K*Np} DOFs")
    print(f"Starting time integration...")
    
    start = time.time()
    u, v, p, t, n_steps = solver.solve(u0, v0, p0, t_final, cfl)
    elapsed = time.time() - start
    
    print(f"Completed {n_steps} steps in {elapsed:.2f}s ({n_steps/elapsed:.1f} steps/s)")
    
    # Compute final divergence
    div = compute_divergence(u, v, solver.Dr, solver.Ds,
                            solver.rx, solver.sx, solver.ry, solver.sy,
                            K, Np)
    print(f"Final max|div(u)| = {np.max(np.abs(div)):.6e}")
    
    return solver, u, v, p
