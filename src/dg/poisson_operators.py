"""
Simple and robust Poisson operator for mixed DG formulation.

Uses the weak Laplacian formulation with Dirichlet fix at one node
to make the matrix invertible.
"""

import numpy as np
from scipy import sparse


def build_weak_laplacian_simple(solver, gamma_p=0.1):
    """
    Build weak Laplacian with proper volume integration.
    
    For Neumann BCs everywhere (pressure Poisson), the operator has a 
    constant null space. We fix this by setting Dirichlet BC at one node.
    
    Parameters
    ----------
    solver : DG2DSystem
    gamma_p : float
        Pressure stabilization parameter
    
    Returns
    -------
    L : scipy.sparse.csr_matrix
        Weak Laplacian matrix (SPD)
    fixed_dof : int
        Index of the fixed DOF (for setting BC)
    """
    K, Np = solver.K, solver.Np
    n_dof = K * Np
    
    Dr, Ds = solver.Dr, solver.Ds
    
    # Get quadrature weights from mass matrix
    Vinv = solver.ref.Vinv
    M_ref = Vinv.T @ Vinv
    
    row_indices = []
    col_indices = []
    data = []
    
    # Volume contributions
    for k in range(K):
        rx_k = solver.rx[k, 0]
        ry_k = solver.ry[k, 0]
        sx_k = solver.sx[k, 0]
        sy_k = solver.sy[k, 0]
        J_k = np.abs(solver.J[k, 0])
        
        for i in range(Np):
            for j in range(Np):
                # Compute ∫_K ∇φ_i · ∇φ_j dV
                val = 0.0
                for q in range(Np):
                    # ∇φ_i at node q
                    dphi_i_dr = Dr[i, q]
                    dphi_i_ds = Ds[i, q]
                    gi_x = rx_k * dphi_i_dr + sx_k * dphi_i_ds
                    gi_y = ry_k * dphi_i_dr + sy_k * dphi_i_ds
                    
                    # ∇φ_j at node q  
                    dphi_j_dr = Dr[j, q]
                    dphi_j_ds = Ds[j, q]
                    gj_x = rx_k * dphi_j_dr + sx_k * dphi_j_ds
                    gj_y = ry_k * dphi_j_dr + sy_k * dphi_j_ds
                    
                    # Weight includes Jacobian and quadrature weight
                    w_q = M_ref[q, q]
                    val += (gi_x * gj_x + gi_y * gj_y) * w_q * J_k
                
                if abs(val) > 1e-15:
                    row_indices.append(k * Np + i)
                    col_indices.append(k * Np + j)
                    data.append(val)
    
    L = sparse.coo_matrix((data, (row_indices, col_indices)), shape=(n_dof, n_dof))
    L = L.tocsr()
    
    # Make symmetric
    L = 0.5 * (L + L.T)
    
    # Add pressure jump stabilization for equal-order spaces
    L = L + _build_jump_stabilization(solver, gamma_p)
    
    # Fix one DOF to make matrix invertible (Dirichlet at first node)
    fixed_dof = 0
    L = L.tolil()
    L[fixed_dof, :] = 0
    L[:, fixed_dof] = 0
    L[fixed_dof, fixed_dof] = 1.0
    L = L.tocsr()
    
    return L, fixed_dof


def _build_jump_stabilization(solver, gamma_p):
    """Build interior jump stabilization matrix."""
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
            
            # Skip boundary faces (Neumann BC)
            if k2 == k and f2 == f:
                continue
            
            fids = Fmask[f]
            fids2 = Fmask[f2]
            
            for i_local, i_vol in enumerate(fids):
                Fscale_kf = solver.Fscale[k, f * Nfp + i_local]
                h_f = 1.0 / Fscale_kf if Fscale_kf > 1e-10 else 1.0
                sJ_k = Fscale_kf * np.abs(solver.J[k, 0])
                
                stab_val = gamma_p * h_f * sJ_k
                j_vol2 = fids2[Nfp - 1 - i_local]
                
                # Self: +γh
                row_indices.append(k * Np + i_vol)
                col_indices.append(k * Np + i_vol)
                data.append(stab_val)
                
                # Cross: -γh
                row_indices.append(k * Np + i_vol)
                col_indices.append(k2 * Np + j_vol2)
                data.append(-stab_val)
    
    J = sparse.coo_matrix((data, (row_indices, col_indices)), shape=(n_dof, n_dof))
    J = J.tocsr()
    J = 0.5 * (J + J.T)
    return J
