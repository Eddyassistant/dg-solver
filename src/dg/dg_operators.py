"""
Reusable DG operator matrices for mixed formulation solvers.

Provides sparse matrix construction for:
- Mass matrix (block-diagonal, element-local)
- SIPG Laplacian (global sparse matrix)
- Divergence operator (global sparse matrix)
- Gradient operator (global sparse matrix)

All operators use scipy.sparse for global solves.

References
----------
Hesthaven & Warburton (2008), Nodal DG Methods, Ch. 7.
Arnold et al. (2002), Unified analysis of DG for elliptic problems.
"""

import numpy as np
from scipy import sparse
from numba import njit, prange


# =============================================================================
# Mass Matrix (Block-diagonal, local per element)
# =============================================================================

def build_mass_matrix(solver):
    """
    Build block-diagonal DG mass matrix M.
    
    M is block-diagonal with K blocks of size Np×Np, where each block is:
        M_k = V^{-T} V^{-1} * |J_k|
    
    For modal basis, M is diagonal. For nodal basis, M is full but small.
    
    Parameters
    ----------
    solver : DG2DSystem or similar
        Must have attributes: K, Np, ref.Vinv, J
    
    Returns
    -------
    M : scipy.sparse.block_diag
        Block-diagonal mass matrix of shape (K*Np, K*Np)
    
    Notes
    -----
    M is cheap to invert per element: M^{-1} = V V^T / |J|
    """
    K, Np = solver.K, solver.Np
    Vinv = solver.ref.Vinv
    
    # Element-local mass matrix in modal space: M_ref = I (orthonormal basis)
    # In nodal space: M = V^{-T} V^{-1}
    M_ref = Vinv.T @ Vinv  # Reference element mass matrix
    
    blocks = []
    for k in range(K):
        M_k = M_ref * np.abs(solver.J[k, 0])
        blocks.append(M_k)
    
    M = sparse.block_diag(blocks, format='csr')
    return M


def apply_mass_inv(solver, rhs):
    """
    Apply M^{-1} to RHS efficiently (element-local, vectorized).
    
    For PKD modal basis: M_k = |J_k| * I (identity in modal space)
    So M_k^{-1} @ u = V @ (Vinv @ u) / |J_k|
    
    Parameters
    ----------
    solver : DG2DSystem
    rhs : ndarray, shape (K, Np) or (K, Np, n_vars)
        Right-hand side to invert against mass matrix
    
    Returns
    -------
    result : ndarray, same shape as rhs
        M^{-1} @ rhs
    """
    K, Np = solver.K, solver.Np
    V = solver.ref.V
    Vinv = solver.ref.Vinv
    
    # M^{-1} in nodal space = V @ V^T / |J|
    Minv_ref = V @ V.T  # This is the inverse of M_ref = V^{-T} @ V^{-1}
    
    if rhs.ndim == 2:
        # Single variable: (K, Np)
        # M^{-1} @ rhs = rhs @ Minv_ref.T / |J| for each element
        result = rhs @ Minv_ref.T
        result = result / np.abs(solver.J)
        return result
    else:
        # Multiple variables: (K, Np, n_vars)
        result = np.empty_like(rhs)
        for v in range(rhs.shape[2]):
            temp = rhs[:, :, v] @ Minv_ref.T
            temp = temp / np.abs(solver.J)
            result[:, :, v] = temp
        return result


# =============================================================================
# SIPG Laplacian Operator
# =============================================================================

def build_sipg_laplacian(solver, nu, sigma_scale=None):
    """
    Build SIPG Laplacian matrix L such that L @ u ≈ -ν∇²u.
    
    Uses Interior Penalty formulation:
        (∇u, ∇v) - <{{∇u}}·n, [[v]]> - <[[u]], {{∇v}}·n> + <σ/h [[u]], [[v]]>
    
    Parameters
    ----------
    solver : DG2DSystem
    nu : float
        Viscosity coefficient
    sigma_scale : float, optional
        Penalty scaling. Default: C*(p+1)*(p+2)/2 with C=4.
    
    Returns
    -------
    L : scipy.sparse.csr_matrix
        Sparse Laplacian matrix of shape (K*Np, K*Np)
    
    References
    ----------
    Arnold et al. (2002), SIAM J. Numer. Anal. 39(5), pp. 1749-1779.
    Shahbazi (2005), JCP 205, pp. 401-407 (penalty parameter).
    """
    K, Np = solver.K, solver.Np
    p = solver.p
    
    if sigma_scale is None:
        sigma_scale = 4.0 * (p + 1) * (p + 2) / 2.0
    
    n_dof = K * Np
    
    # Build in COO format, then convert to CSR
    row_indices = []
    col_indices = []
    data = []
    
    Dr, Ds = solver.Dr, solver.Ds
    Vinv = solver.ref.Vinv
    
    # Reference element mass matrix (for projection)
    M_ref_inv = Vinv @ Vinv.T
    
    # --- Volume contributions: (∇φ_i, ∇φ_j) ---
    for k in range(K):
        rx_k = solver.rx[k, 0]
        ry_k = solver.ry[k, 0]
        sx_k = solver.sx[k, 0]
        sy_k = solver.sy[k, 0]
        J_k = solver.J[k, 0]
        
        # Gradient in physical space: ∇φ = (rx*Dr + sx*Ds, ry*Dr + sy*Ds) φ
        # Build local stiffness matrix
        for i in range(Np):
            for j in range(Np):
                # ∇φ_i · ∇φ_j
                dphi_i_dr = Dr[i, :]
                dphi_i_ds = Ds[i, :]
                dphi_j_dr = Dr[j, :]
                dphi_j_ds = Ds[j, :]
                
                # ∂φ/∂x = rx * ∂φ/∂r + sx * ∂φ/∂s
                # ∂φ/∂y = ry * ∂φ/∂r + sy * ∂φ/∂s
                grad_i_x = rx_k * dphi_i_dr + sx_k * dphi_i_ds
                grad_i_y = ry_k * dphi_i_dr + sy_k * dphi_i_ds
                grad_j_x = rx_k * dphi_j_dr + sx_k * dphi_j_ds
                grad_j_y = ry_k * dphi_j_dr + sy_k * dphi_j_ds
                
                # Integrate: (∇φ_i, ∇φ_j) = Σ w_l ∇φ_i(r_l) · ∇φ_j(r_l) |J|
                # For nodal basis with GLL-type quadrature, this is approximated
                # Use mass matrix weighted sum
                integrand = grad_i_x * grad_j_x + grad_i_y * grad_j_y
                
                # Mass matrix quadrature
                val = np.sum(integrand * np.abs(J_k) * M_ref_inv.diagonal())
                
                if abs(val) > 1e-14:
                    row_indices.append(k * Np + i)
                    col_indices.append(k * Np + j)
                    data.append(nu * val)
    
    # --- Surface contributions ---
    Fmask = solver.ref.Fmask
    Nfp = solver.ref.Nfp
    
    for k in range(K):
        for f in range(3):
            k2 = solver.EToE[k, f]
            f2 = solver.EToF[k, f]
            
            fids = Fmask[f]
            
            for i_local, i_vol in enumerate(fids):
                # Face normal
                nx = solver.nx[k, f * Nfp + i_local]
                ny = solver.ny[k, f * Nfp + i_local]
                Fscale_kf = solver.Fscale[k, f * Nfp + i_local]
                
                # h_f ≈ 1/Fscale (face length / element volume factor)
                h_f = 1.0 / Fscale_kf if Fscale_kf > 0 else 1.0
                sigma_f = nu * sigma_scale / h_f
                
                if k2 == k and f2 == f:
                    # Boundary face - apply Dirichlet BC (u=0 for cavity walls)
                    # Penalty term only: σ/h * u_i * v_i
                    for i_local2, i_vol2 in enumerate(fids):
                        val = sigma_f * Fscale_kf  # σ/h * face_weight
                        if abs(val) > 1e-14:
                            row_indices.append(k * Np + i_vol)
                            col_indices.append(k * Np + i_vol2)
                            data.append(val)
                else:
                    # Interior face - coupling with neighbor
                    fids2 = Fmask[f2]
                    
                    # Penalty term: σ/h * (u_k - u_k2) * (v_k - v_k2)
                    # Contribution to self: +σ/h
                    # Contribution to neighbor: -σ/h
                    for i_local2, i_vol2 in enumerate(fids):
                        val = sigma_f * Fscale_kf
                        if abs(val) > 1e-14:
                            # Self contribution
                            row_indices.append(k * Np + i_vol)
                            col_indices.append(k * Np + i_vol2)
                            data.append(val)
                            
                            # Neighbor contribution (negative)
                            j_vol2 = fids2[Nfp - 1 - i_local2]
                            row_indices.append(k * Np + i_vol)
                            col_indices.append(k2 * Np + j_vol2)
                            data.append(-val)
    
    L = sparse.coo_matrix((data, (row_indices, col_indices)), shape=(n_dof, n_dof))
    return L.tocsr()


# =============================================================================
# Divergence Operator (velocity -> scalar)
# =============================================================================

def build_divergence_operator(solver):
    """
    Build divergence operator DIV such that DIV @ [u; v] = ∇·u.
    
    Returns a block matrix [DIV_x, DIV_y] where:
        DIV_x @ u = ∂u/∂x
        DIV_y @ v = ∂v/∂y
    
    Parameters
    ----------
    solver : DG2DSystem
    
    Returns
    -------
    DIV_x, DIV_y : scipy.sparse.csr_matrix
        Each of shape (K*Np, K*Np)
        Total divergence: div_u = DIV_x @ u + DIV_y @ v
    """
    K, Np = solver.K, solver.Np
    n_dof = K * Np
    
    row_indices_x, col_indices_x, data_x = [], [], []
    row_indices_y, col_indices_y, data_y = [], [], []
    
    Dr, Ds = solver.Dr, solver.Ds
    
    # --- Volume contribution ---
    for k in range(K):
        rx_k = solver.rx[k, 0]
        ry_k = solver.ry[k, 0]
        sx_k = solver.sx[k, 0]
        sy_k = solver.sy[k, 0]
        J_k = solver.J[k, 0]
        
        for i in range(Np):
            # Test function i, integrate against ∂φ_j/∂x
            for j in range(Np):
                dphi_j_dr = Dr[j, i]  # Note: transpose
                dphi_j_ds = Ds[j, i]
                
                grad_j_x = rx_k * dphi_j_dr + sx_k * dphi_j_ds
                grad_j_y = ry_k * dphi_j_dr + sy_k * dphi_j_ds
                
                # ∫ φ_i ∂φ_j/∂x dΩ ≈ Σ_l φ_i(r_l) ∂φ_j/∂x(r_l) w_l |J|
                # Nodal: φ_i(r_l) = δ_il
                val_x = grad_j_x * np.abs(J_k)
                val_y = grad_j_y * np.abs(J_k)
                
                if abs(val_x) > 1e-14:
                    row_indices_x.append(k * Np + i)
                    col_indices_x.append(k * Np + j)
                    data_x.append(val_x)
                if abs(val_y) > 1e-14:
                    row_indices_y.append(k * Np + i)
                    col_indices_y.append(k * Np + j)
                    data_y.append(val_y)
    
    DIV_x = sparse.coo_matrix((data_x, (row_indices_x, col_indices_x)), shape=(n_dof, n_dof)).tocsr()
    DIV_y = sparse.coo_matrix((data_y, (row_indices_y, col_indices_y)), shape=(n_dof, n_dof)).tocsr()
    
    return DIV_x, DIV_y


@njit(cache=True, parallel=True)
def compute_divergence_numba(u, v, Dr, Ds, rx, ry, sx, sy, J, K, Np):
    """
    Compute divergence ∇·u = ∂u/∂x + ∂v/∂y using differentiation matrices.
    
    Parameters
    ----------
    u, v : ndarray, shape (K, Np)
    
    Returns
    -------
    div : ndarray, shape (K, Np)
    """
    div = np.empty((K, Np), dtype=np.float64)
    
    for k in prange(K):
        rx_k = rx[k, 0]
        ry_k = ry[k, 0]
        sx_k = sx[k, 0]
        sy_k = sy[k, 0]
        
        for i in range(Np):
            dudr = 0.0
            duds = 0.0
            dvdr = 0.0
            dvds = 0.0
            
            for j in range(Np):
                dudr += Dr[i, j] * u[k, j]
                duds += Ds[i, j] * u[k, j]
                dvdr += Dr[i, j] * v[k, j]
                dvds += Ds[i, j] * v[k, j]
            
            dudx = rx_k * dudr + sx_k * duds
            dvdy = ry_k * dvdr + sy_k * dvds
            
            div[k, i] = dudx + dvdy
    
    return div


def compute_divergence(solver, u, v):
    """
    Compute element-wise divergence using Numba-accelerated kernel.
    
    Parameters
    ----------
    solver : DG2DSystem
    u, v : ndarray, shape (K, Np)
    
    Returns
    -------
    div : ndarray, shape (K, Np)
    """
    return compute_divergence_numba(
        u, v, solver.Dr, solver.Ds, 
        solver.rx, solver.ry, solver.sx, solver.sy,
        solver.J, solver.K, solver.Np
    )


# =============================================================================
# Gradient Operator (scalar -> vector)
# =============================================================================

def build_gradient_operator(solver):
    """
    Build gradient operator GRAD such that GRAD @ p = [∂p/∂x; ∂p/∂y].
    
    Returns two matrices GRAD_x, GRAD_y where:
        GRAD_x @ p = ∂p/∂x
        GRAD_y @ p = ∂p/∂y
    
    Parameters
    ----------
    solver : DG2DSystem
    
    Returns
    -------
    GRAD_x, GRAD_y : scipy.sparse.csr_matrix
        Each of shape (K*Np, K*Np)
    """
    K, Np = solver.K, solver.Np
    n_dof = K * Np
    
    row_indices_x, col_indices_x, data_x = [], [], []
    row_indices_y, col_indices_y, data_y = [], [], []
    
    Dr, Ds = solver.Dr, solver.Ds
    
    for k in range(K):
        rx_k = solver.rx[k, 0]
        ry_k = solver.ry[k, 0]
        sx_k = solver.sx[k, 0]
        sy_k = solver.sy[k, 0]
        J_k = solver.J[k, 0]
        
        for i in range(Np):
            for j in range(Np):
                dphi_j_dr = Dr[i, j]
                dphi_j_ds = Ds[i, j]
                
                grad_j_x = rx_k * dphi_j_dr + sx_k * dphi_j_ds
                grad_j_y = ry_k * dphi_j_dr + sy_k * dphi_j_ds
                
                val_x = grad_j_x * np.abs(J_k)
                val_y = grad_j_y * np.abs(J_k)
                
                if abs(val_x) > 1e-14:
                    row_indices_x.append(k * Np + i)
                    col_indices_x.append(k * Np + j)
                    data_x.append(val_x)
                if abs(val_y) > 1e-14:
                    row_indices_y.append(k * Np + i)
                    col_indices_y.append(k * Np + j)
                    data_y.append(val_y)
    
    GRAD_x = sparse.coo_matrix((data_x, (row_indices_x, col_indices_x)), shape=(n_dof, n_dof)).tocsr()
    GRAD_y = sparse.coo_matrix((data_y, (row_indices_y, col_indices_y)), shape=(n_dof, n_dof)).tocsr()
    
    return GRAD_x, GRAD_y


@njit(cache=True, parallel=True)
def compute_gradient_numba(p, Dr, Ds, rx, ry, sx, sy, K, Np):
    """
    Compute gradient ∇p = (∂p/∂x, ∂p/∂y) using differentiation matrices.
    
    Parameters
    ----------
    p : ndarray, shape (K, Np)
    
    Returns
    -------
    grad_x, grad_y : ndarray, shape (K, Np)
    """
    grad_x = np.empty((K, Np), dtype=np.float64)
    grad_y = np.empty((K, Np), dtype=np.float64)
    
    for k in prange(K):
        rx_k = rx[k, 0]
        ry_k = ry[k, 0]
        sx_k = sx[k, 0]
        sy_k = sy[k, 0]
        
        for i in range(Np):
            dpdr = 0.0
            dpds = 0.0
            
            for j in range(Np):
                dpdr += Dr[i, j] * p[k, j]
                dpds += Ds[i, j] * p[k, j]
            
            grad_x[k, i] = rx_k * dpdr + sx_k * dpds
            grad_y[k, i] = ry_k * dpdr + sy_k * dpds
    
    return grad_x, grad_y


def compute_gradient(solver, p):
    """
    Compute element-wise gradient using Numba-accelerated kernel.
    
    Parameters
    ----------
    solver : DG2DSystem
    p : ndarray, shape (K, Np)
    
    Returns
    -------
    grad_x, grad_y : ndarray, shape (K, Np)
    """
    return compute_gradient_numba(
        p, solver.Dr, solver.Ds,
        solver.rx, solver.ry, solver.sx, solver.sy,
        solver.K, solver.Np
    )


# =============================================================================
# Weak Laplacian via application of DIV @ GRAD
# =============================================================================

def build_weak_laplacian(solver):
    """
    Build weak Laplacian L = -DIV @ GRAD (consistent with Poisson equation).
    
    The weak form of -∇²p = f is:
        (∇p, ∇v) = (f, v)
    
    This builds the left-hand side operator.
    
    Parameters
    ----------
    solver : DG2DSystem
    
    Returns
    -------
    L_weak : scipy.sparse.csr_matrix
        Weak Laplacian of shape (K*Np, K*Np)
    """
    GRAD_x, GRAD_y = build_gradient_operator(solver)
    
    # Build mass matrix for integration
    K, Np = solver.K, solver.Np
    n_dof = K * Np
    
    # Mass-weighted gradient (for proper L2 inner product)
    M_data = []
    M_row = []
    M_col = []
    
    for k in range(K):
        J_k = np.abs(solver.J[k, 0])
        for i in range(Np):
            M_row.append(k * Np + i)
            M_col.append(k * Np + i)
            M_data.append(J_k)
    
    M_diag = sparse.coo_matrix((M_data, (M_row, M_col)), shape=(n_dof, n_dof)).tocsr()
    
    # Weak Laplacian: L = GRAD^T @ M @ GRAD
    L_weak = GRAD_x.T @ M_diag @ GRAD_x + GRAD_y.T @ M_diag @ GRAD_y
    
    return L_weak


# =============================================================================
# Pressure stabilization (Brezzi-Pitkäranta style)
# =============================================================================

def build_pressure_stabilization(solver, gamma_p=1.0):
    """
    Build pressure jump stabilization matrix.
    
    Adds penalty on pressure jumps across interior faces:
        J(p, q) = γ_p * h * [[p]] [[q]]
    
    This allows using equal-order spaces for velocity and pressure.
    
    Parameters
    ----------
    solver : DG2DSystem
    gamma_p : float
        Stabilization parameter (typical: 0.1 to 1.0)
    
    Returns
    -------
    J : scipy.sparse.csr_matrix
        Stabilization matrix of shape (K*Np, K*Np)
    
    References
    ----------
    Brezzi & Pitkäranta (1984), On the stabilization of finite element 
        approximations of the Stokes equations.
    """
    K, Np = solver.K, solver.Np
    Nfp = solver.ref.Nfp
    Fmask = solver.ref.Fmask
    n_dof = K * Np
    
    row_indices = []
    col_indices = []
    data = []
    
    for k in range(K):
        for f in range(3):
            k2 = solver.EToE[k, f]
            f2 = solver.EToF[k, f]
            
            if k2 == k and f2 == f:
                continue  # Skip boundary faces
            
            fids = Fmask[f]
            fids2 = Fmask[f2]
            
            for i_local, i_vol in enumerate(fids):
                # Face weight
                Fscale_kf = solver.Fscale[k, f * Nfp + i_local]
                h_f = 1.0 / Fscale_kf if Fscale_kf > 0 else 1.0
                
                # Jump: u_k - u_k2
                # Self term: +γ*h
                val = gamma_p * h_f * Fscale_kf
                
                row_indices.append(k * Np + i_vol)
                col_indices.append(k * Np + i_vol)
                data.append(val)
                
                # Cross term: -γ*h (neighbor)
                j_vol = fids2[Nfp - 1 - i_local]
                row_indices.append(k * Np + i_vol)
                col_indices.append(k2 * Np + j_vol)
                data.append(-val)
    
    J = sparse.coo_matrix((data, (row_indices, col_indices)), shape=(n_dof, n_dof))
    return J.tocsr()
