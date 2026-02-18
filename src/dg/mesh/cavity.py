"""
Cavity mesh generation with boundary condition tags.

Lid-driven cavity mesh on [0,1]^2 with tagged boundary faces.
"""

import numpy as np
from .triangle_mesh import TriangleMesh


def cavity_mesh(nx: int, ny: int) -> tuple[TriangleMesh, np.ndarray]:
    """
    Create a structured triangular mesh for the lid-driven cavity on [0,1]^2.

    Each quad cell is split into 2 triangles. Boundary faces are tagged
    according to their type:
        - 0: interior face (neighbor exists)
        - 1: wall face (bottom, left, right boundaries)
        - 2: lid face (top boundary, y=1)

    Parameters
    ----------
    nx, ny : int
        Number of quad cells in x and y directions.
        Total triangles = 2 * nx * ny.

    Returns
    -------
    mesh : TriangleMesh
        The triangular mesh on [0,1]^2.
    bc_tags : ndarray, shape (K, 3), int
        Boundary condition tags for each face of each element:
        - 0: interior (has neighbor)
        - 1: wall (bottom y=0, left x=0, right x=1)
        - 2: lid (top y=1)

    References
    ----------
    Ghia et al. (1982), J. Comput. Phys., 48:387-411.
    """
    # Create structured mesh on [0,1]^2
    mesh = TriangleMesh.rectangle((0.0, 1.0), (0.0, 1.0), nx, ny)

    K = mesh.n_elem
    EToV = mesh.EToV

    # Face vertex indices (counterclockwise ordering)
    face_verts = [(0, 1), (1, 2), (2, 0)]

    # Initialize: all faces assumed boundary (will be updated for interior)
    bc_tags = np.ones((K, 3), dtype=np.int64)  # default to wall (1)

    # Build midpoint dictionary to find matching faces
    def _quantize(x, y, tol=1e-8):
        return (round(x / tol) * tol, round(y / tol) * tol)

    face_dict = {}

    for k in range(K):
        for f in range(3):
            va, vb = face_verts[f]
            i0, i1 = EToV[k, va], EToV[k, vb]

            # Midpoint coordinates
            mx = 0.5 * (mesh.VX[i0] + mesh.VX[i1])
            my = 0.5 * (mesh.VY[i0] + mesh.VY[i1])

            key = _quantize(mx, my)

            if key in face_dict:
                # Found a matching face - mark both as interior
                k2, f2 = face_dict[key]
                bc_tags[k, f] = 0  # interior
                bc_tags[k2, f2] = 0  # interior
                del face_dict[key]
            else:
                face_dict[key] = (k, f)

    # Tag remaining boundary faces
    # Unmatched faces in face_dict are boundary faces
    for key, (k, f) in face_dict.items():
        va, vb = face_verts[f]
        i0, i1 = EToV[k, va], EToV[k, vb]

        # Midpoint coordinates
        mx = 0.5 * (mesh.VX[i0] + mesh.VX[i1])
        my = 0.5 * (mesh.VY[i0] + mesh.VY[i1])

        # Check if on lid (top boundary, y=1)
        # Use tolerance for floating point comparison
        if abs(my - 1.0) < 1e-8:
            bc_tags[k, f] = 2  # lid
        else:
            bc_tags[k, f] = 1  # wall (bottom, left, or right)

    return mesh, bc_tags


def get_boundary_faces(bc_tags: np.ndarray) -> dict:
    """
    Get indices of boundary faces by type.

    Parameters
    ----------
    bc_tags : ndarray, shape (K, 3), int
        Boundary condition tags from cavity_mesh().

    Returns
    -------
    faces : dict
        Dictionary with keys 'interior', 'wall', 'lid' containing
        lists of (element_idx, face_idx) tuples.
    """
    K = bc_tags.shape[0]
    faces = {'interior': [], 'wall': [], 'lid': []}

    for k in range(K):
        for f in range(3):
            tag = bc_tags[k, f]
            if tag == 0:
                faces['interior'].append((k, f))
            elif tag == 1:
                faces['wall'].append((k, f))
            elif tag == 2:
                faces['lid'].append((k, f))

    return faces
