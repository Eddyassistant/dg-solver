"""
2D triangular mesh generation and connectivity.

Generates structured triangular meshes on rectangular domains by
splitting each quad cell into 2 triangles. Supports periodic BCs.

References
----------
Hesthaven & Warburton (2008), Ch. 6.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class TriangleMesh:
    """
    2D triangular mesh.

    Attributes
    ----------
    VX, VY : ndarray, shape (n_verts,)
        Vertex coordinates.
    EToV : ndarray, shape (n_elem, 3), int
        Element-to-vertex connectivity (counterclockwise).
    n_elem : int
        Number of elements.
    n_verts : int
        Number of vertices.
    """
    VX: np.ndarray
    VY: np.ndarray
    EToV: np.ndarray
    n_elem: int
    n_verts: int

    @staticmethod
    def rectangle(x_range: tuple, y_range: tuple,
                  nx: int, ny: int) -> "TriangleMesh":
        """
        Create a structured triangular mesh on a rectangle.

        Each quad cell is split into 2 triangles by the diagonal.

        Parameters
        ----------
        x_range : (x_min, x_max)
        y_range : (y_min, y_max)
        nx, ny : int
            Number of quad cells in each direction.
            Total triangles = 2 * nx * ny.
        """
        x = np.linspace(x_range[0], x_range[1], nx + 1)
        y = np.linspace(y_range[0], y_range[1], ny + 1)
        X, Y = np.meshgrid(x, y, indexing='ij')
        VX = X.ravel()
        VY = Y.ravel()
        n_verts = len(VX)

        def vid(i, j):
            return i * (ny + 1) + j

        triangles = []
        for i in range(nx):
            for j in range(ny):
                v0 = vid(i, j)
                v1 = vid(i + 1, j)
                v2 = vid(i + 1, j + 1)
                v3 = vid(i, j + 1)
                # Split quad into 2 triangles
                triangles.append([v0, v1, v3])
                triangles.append([v1, v2, v3])

        EToV = np.array(triangles, dtype=np.int64)
        return TriangleMesh(VX=VX, VY=VY, EToV=EToV,
                            n_elem=len(triangles), n_verts=n_verts)


def _quantize(x, y, tol=1e-8):
    """Quantize coordinates for hash-based matching."""
    return (round(x / tol) * tol, round(y / tol) * tol)


def build_face_connectivity(mesh: TriangleMesh,
                            periodic_x: bool = False,
                            periodic_y: bool = False,
                            x_range: tuple = None,
                            y_range: tuple = None,
                            tol: float = 1e-8):
    """
    Build face-to-face connectivity using hash-based midpoint matching. O(K).

    Face numbering (must match reference_triangle.face_mask):
        Face 0 (s=-1 on ref): edge v0-v1
        Face 1 (r+s=0 on ref): edge v1-v2
        Face 2 (r=-1 on ref): edge v2-v0

    Returns
    -------
    EToE : ndarray, shape (n_elem, 3), int
    EToF : ndarray, shape (n_elem, 3), int
    """
    K = mesh.n_elem
    EToV = mesh.EToV
    face_verts = [(0, 1), (1, 2), (2, 0)]

    # Initialize: self-reference (boundary faces)
    EToE = np.tile(np.arange(K)[:, None], (1, 3))
    EToF = np.tile(np.arange(3)[None, :], (K, 1))

    # Build midpoint → (element, face) dictionary
    face_dict = {}
    for k in range(K):
        for f in range(3):
            va, vb = face_verts[f]
            i0, i1 = EToV[k, va], EToV[k, vb]
            mx = 0.5 * (mesh.VX[i0] + mesh.VX[i1])
            my = 0.5 * (mesh.VY[i0] + mesh.VY[i1])
            key = _quantize(mx, my, tol)
            if key in face_dict:
                k2, f2 = face_dict[key]
                EToE[k, f] = k2
                EToF[k, f] = f2
                EToE[k2, f2] = k
                EToF[k2, f2] = f
                del face_dict[key]
            else:
                face_dict[key] = (k, f)

    # Handle periodic BCs for unmatched boundary faces
    if (periodic_x or periodic_y) and face_dict:
        Lx = (x_range[1] - x_range[0]) if (periodic_x and x_range) else 0.0
        Ly = (y_range[1] - y_range[0]) if (periodic_y and y_range) else 0.0

        offsets = []
        if periodic_x:
            offsets.extend([(Lx, 0.0), (-Lx, 0.0)])
        if periodic_y:
            offsets.extend([(0.0, Ly), (0.0, -Ly)])
        if periodic_x and periodic_y:
            offsets.extend([(Lx, Ly), (Lx, -Ly), (-Lx, Ly), (-Lx, -Ly)])

        # Collect all unmatched faces
        unmatched = list(face_dict.items())
        for dx, dy in offsets:
            for key, (k, f) in unmatched:
                if EToE[k, f] != k:
                    continue  # already matched
                shifted = _quantize(key[0] + dx, key[1] + dy, tol)
                if shifted in face_dict:
                    k2, f2 = face_dict[shifted]
                    if k2 != k or f2 != f:
                        EToE[k, f] = k2
                        EToF[k, f] = f2
                        EToE[k2, f2] = k
                        EToF[k2, f2] = f

    return EToE, EToF


def geometric_factors(mesh: TriangleMesh, ref, p: int):
    """
    Compute geometric factors for all elements (vectorized).

    x = ½[-(r+s)*v0x + (1+r)*v1x + (1+s)*v2x]

    Returns
    -------
    x, y : ndarray, shape (K, Np)
    rx, ry, sx, sy : ndarray, shape (K, 1)
    J : ndarray, shape (K, 1)
    """
    K = mesh.n_elem
    r, s = ref.r, ref.s
    EToV = mesh.EToV

    # Vectorized coordinate computation
    v0x = mesh.VX[EToV[:, 0]]  # (K,)
    v1x = mesh.VX[EToV[:, 1]]
    v2x = mesh.VX[EToV[:, 2]]
    v0y = mesh.VY[EToV[:, 0]]
    v1y = mesh.VY[EToV[:, 1]]
    v2y = mesh.VY[EToV[:, 2]]

    x = 0.5 * (np.outer(-(r + s), np.ones(K)).T * v0x[:, None]
               + np.outer(1 + r, np.ones(K)).T * v1x[:, None]
               + np.outer(1 + s, np.ones(K)).T * v2x[:, None])
    # Simpler:
    x = (v0x[:, None] * (-(r + s)[None, :] * 0.5)
         + v1x[:, None] * ((1 + r)[None, :] * 0.5)
         + v2x[:, None] * ((1 + s)[None, :] * 0.5))
    y = (v0y[:, None] * (-(r + s)[None, :] * 0.5)
         + v1y[:, None] * ((1 + r)[None, :] * 0.5)
         + v2y[:, None] * ((1 + s)[None, :] * 0.5))

    # Geometric factors (constant per element)
    xr = (0.5 * (-v0x + v1x))[:, None]  # (K, 1)
    xs = (0.5 * (-v0x + v2x))[:, None]
    yr = (0.5 * (-v0y + v1y))[:, None]
    ys = (0.5 * (-v0y + v2y))[:, None]

    J = xr * ys - xs * yr
    rx = ys / J
    ry = -xs / J
    sx = -yr / J
    sy = xr / J

    return x, y, rx, ry, sx, sy, J


def face_normals(mesh: TriangleMesh, ref, x, y, J):
    """
    Compute outward unit normals and face Jacobians (vectorized).
    """
    K = mesh.n_elem
    Nfp = ref.Nfp
    v = mesh.EToV
    face_verts_local = [(0, 1), (1, 2), (2, 0)]

    nx = np.zeros((K, 3 * Nfp))
    ny = np.zeros((K, 3 * Nfp))
    sJ = np.zeros((K, 3 * Nfp))

    for f in range(3):
        va, vb = face_verts_local[f]
        opp = 3 - va - vb

        # Tangent vectors (K,)
        tx = mesh.VX[v[:, vb]] - mesh.VX[v[:, va]]
        ty = mesh.VY[v[:, vb]] - mesh.VY[v[:, va]]

        # Candidate outward normal
        nnx = ty.copy()
        nny = -tx.copy()

        # Check orientation: normal should point away from opposite vertex
        xmid = 0.5 * (mesh.VX[v[:, va]] + mesh.VX[v[:, vb]])
        ymid = 0.5 * (mesh.VY[v[:, va]] + mesh.VY[v[:, vb]])
        to_opp_x = mesh.VX[v[:, opp]] - xmid
        to_opp_y = mesh.VY[v[:, opp]] - ymid
        flip = (nnx * to_opp_x + nny * to_opp_y) > 0
        nnx[flip] *= -1
        nny[flip] *= -1

        face_len = np.sqrt(tx**2 + ty**2)
        nnx /= face_len
        nny /= face_len

        sl = slice(f * Nfp, (f + 1) * Nfp)
        nx[:, sl] = nnx[:, None]
        ny[:, sl] = nny[:, None]
        sJ[:, sl] = (face_len / 2.0)[:, None]  # ref face length = 2

    Fscale = sJ / np.abs(J)
    return nx, ny, sJ, Fscale
