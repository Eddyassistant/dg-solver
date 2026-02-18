"""
Boundary conditions for artificial compressibility incompressible Navier-Stokes.

The state vector is q = [u, v, p] where:
    u : x-velocity
    v : y-velocity  
    p : pressure

Boundary conditions are applied weakly via the exterior state q_ext
in the numerical flux computation.

References
----------
Karniadakis & Sherwin (2005), Spectral/hp Element Methods for CFD, Ch. 8.
"""

import numpy as np
from numba import njit


# BC type constants
BC_INTERIOR = 0
BC_WALL = 1
BC_LID = 2


def apply_bc(u_int: np.ndarray, bc_type: np.ndarray,
             lid_velocity: float = 1.0) -> np.ndarray:
    """
    Apply boundary conditions by computing exterior state.

    For artificial compressibility INS with q = [u, v, p]:

    Wall BC (no-slip):
        u_ext = -u_int   (mirror for velocity)
        v_ext = -v_int   (mirror for velocity)
        p_ext = p_int    (Neumann for pressure)

    Lid BC (moving lid with velocity U_lid):
        u_ext = 2*U_lid - u_int  (prescribed velocity at wall)
        v_ext = -v_int           (mirror for wall-normal velocity)
        p_ext = p_int            (Neumann for pressure)

    Parameters
    ----------
    u_int : ndarray, shape (n_bc_faces, n_vars)
        Interior state at boundary faces [u, v, p].
        n_vars = 3 for [u, v, p].
    bc_type : ndarray, shape (n_bc_faces,), int
        Boundary condition type for each face:
        - BC_WALL (1): stationary wall (no-slip)
        - BC_LID (2): moving lid
    lid_velocity : float, optional
        Velocity of the moving lid (default: 1.0).

    Returns
    -------
    u_ext : ndarray, shape (n_bc_faces, n_vars)
        Exterior state at boundary faces.

    Examples
    --------
    >>> u_int = np.array([[0.5, 0.3, 1.0]])  # [u, v, p]
    >>> bc_type = np.array([BC_WALL])
    >>> u_ext = apply_bc(u_int, bc_type)
    >>> print(u_ext)  # [-0.5, -0.3, 1.0]
    """
    u_int = np.asarray(u_int, dtype=np.float64)
    bc_type = np.asarray(bc_type, dtype=np.int64)

    if u_int.ndim != 2 or u_int.shape[1] != 3:
        raise ValueError(f"u_int must have shape (n_faces, 3), got {u_int.shape}")
    if bc_type.shape[0] != u_int.shape[0]:
        raise ValueError(f"bc_type length {bc_type.shape[0]} != n_faces {u_int.shape[0]}")

    return _apply_bc_kernel(u_int, bc_type, float(lid_velocity))


@njit(cache=True)
def _apply_bc_kernel(u_int: np.ndarray, bc_type: np.ndarray,
                     lid_velocity: float) -> np.ndarray:
    """
    Numba kernel for BC application.

    Parameters
    ----------
    u_int : ndarray, shape (n, 3), float64
    bc_type : ndarray, shape (n,), int64
    lid_velocity : float64

    Returns
    -------
    u_ext : ndarray, shape (n, 3), float64
    """
    n_faces = u_int.shape[0]
    u_ext = np.empty_like(u_int)

    for i in range(n_faces):
        u = u_int[i, 0]
        v = u_int[i, 1]
        p = u_int[i, 2]

        if bc_type[i] == BC_WALL:
            # No-slip wall: mirror velocities, keep pressure
            u_ext[i, 0] = -u
            u_ext[i, 1] = -v
            u_ext[i, 2] = p
        elif bc_type[i] == BC_LID:
            # Moving lid: prescribed u, mirror v, keep pressure
            u_ext[i, 0] = 2.0 * lid_velocity - u
            u_ext[i, 1] = -v
            u_ext[i, 2] = p
        else:
            # Interior or unknown: keep same (should not happen for BC faces)
            u_ext[i, 0] = u
            u_ext[i, 1] = v
            u_ext[i, 2] = p

    return u_ext


def apply_bc_vectorized(u_int: np.ndarray, bc_tags: np.ndarray,
                        lid_velocity: float = 1.0) -> np.ndarray:
    """
    Apply boundary conditions to full element-face array.

    Parameters
    ----------
    u_int : ndarray, shape (K, 3, Nfp, n_vars)
        Interior state at all face nodes for all elements.
        K = n_elements, Nfp = nodes per face, n_vars = 3.
    bc_tags : ndarray, shape (K, 3), int
        BC type tag for each face (0=interior, 1=wall, 2=lid).
    lid_velocity : float, optional
        Velocity of the moving lid (default: 1.0).

    Returns
    -------
    u_ext : ndarray, shape (K, 3, Nfp, n_vars)
        Exterior state at all face nodes.
    """
    u_int = np.asarray(u_int, dtype=np.float64)
    bc_tags = np.asarray(bc_tags, dtype=np.int64)

    K, n_faces, Nfp, n_vars = u_int.shape
    u_ext = np.empty_like(u_int)

    # Vectorized BC application
    for k in range(K):
        for f in range(n_faces):
            tag = bc_tags[k, f]
            if tag == BC_INTERIOR:
                # Interior: keep same (will be overwritten by neighbor)
                u_ext[k, f, :, :] = u_int[k, f, :, :]
            elif tag == BC_WALL:
                # Wall: mirror u,v; keep p
                u_ext[k, f, :, 0] = -u_int[k, f, :, 0]  # u
                u_ext[k, f, :, 1] = -u_int[k, f, :, 1]  # v
                u_ext[k, f, :, 2] = u_int[k, f, :, 2]   # p
            elif tag == BC_LID:
                # Lid: prescribed u, mirror v, keep p
                u_ext[k, f, :, 0] = 2.0 * lid_velocity - u_int[k, f, :, 0]  # u
                u_ext[k, f, :, 1] = -u_int[k, f, :, 1]                      # v
                u_ext[k, f, :, 2] = u_int[k, f, :, 2]                       # p

    return u_ext


def get_face_state(u: np.ndarray, mesh, ref, bc_tags: np.ndarray,
                   lid_velocity: float = 1.0) -> tuple:
    """
    Extract interior and exterior states at all face nodes.

    Parameters
    ----------
    u : ndarray, shape (K, Np, n_vars)
        Volume solution at nodal points.
    mesh : TriangleMesh
        The mesh object.
    ref : RefTriangle
        Reference element with Fmask for face node extraction.
    bc_tags : ndarray, shape (K, 3), int
        BC tags from cavity_mesh().
    lid_velocity : float, optional
        Lid velocity (default: 1.0).

    Returns
    -------
    uM : ndarray, shape (K, 3*Nfp, n_vars)
        Interior trace at all face nodes (flattened faces).
    uP : ndarray, shape (K, 3*Nfp, n_vars)
        Exterior trace at all face nodes.
    """
    K, Np, n_vars = u.shape
    Nfp = ref.Nfp
    Fmask = ref.Fmask  # List of 3 arrays, each of length Nfp

    # Extract interior face values
    uM = np.zeros((K, 3 * Nfp, n_vars), dtype=np.float64)
    for f in range(3):
        for i in range(Nfp):
            idx = f * Nfp + i
            node = Fmask[f][i]
            uM[:, idx, :] = u[:, node, :]

    # Compute exterior values (apply BCs where needed)
    # Reshape to (K, 3, Nfp, n_vars) for BC application
    uM_reshaped = uM.reshape(K, 3, Nfp, n_vars)
    uP_reshaped = apply_bc_vectorized(uM_reshaped, bc_tags, lid_velocity)

    # For interior faces, we need to get neighbor values
    # This requires connectivity information (EToE, EToF)
    # For now, return BC-applied values (caller handles interior faces)
    uP = uP_reshaped.reshape(K, 3 * Nfp, n_vars)

    return uM, uP
