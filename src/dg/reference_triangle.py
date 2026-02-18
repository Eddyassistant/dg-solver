"""
Reference triangle operations for nodal DG.

Reference element: vertices at (-1,-1), (1,-1), (-1,1).

Orthonormal Proriol-Koornwinder-Dubiner (PKD) basis on the triangle,
Warp & Blend high-order interpolation nodes, and precomputed
differentiation / lift matrices.

References
----------
Hesthaven & Warburton (2008), Nodal DG Methods, Ch. 3–4 & Appendix A.
Karniadakis & Sherwin (2005), Ch. 3.
Warburton (2006), J. Eng. Math. 56, pp. 307-326 (Warp & Blend nodes).
"""

import numpy as np
from scipy.special import gamma as gamma_func
from functools import lru_cache
from numba import njit


# =============================================================================
# Jacobi polynomials
# =============================================================================

def jacobi_p(x: np.ndarray, alpha: float, beta: float, n: int) -> np.ndarray:
    """
    Evaluate the normalized Jacobi polynomial P_n^{α,β}(x).

    Uses the three-term recurrence. Normalization:
        ∫₋₁¹ [P_n^{α,β}(x)]² (1-x)^α (1+x)^β dx = 1

    Parameters
    ----------
    x : ndarray, shape (N,)
    alpha, beta : float
        Jacobi parameters (α, β > -1).
    n : int
        Polynomial degree.

    Returns
    -------
    P : ndarray, shape (N,)
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    N = len(x)

    # Normalization constant for degree 0
    gamma0 = (2.0**(alpha + beta + 1) / (alpha + beta + 1.0)
              * gamma_func(alpha + 1) * gamma_func(beta + 1)
              / gamma_func(alpha + beta + 1))
    P_prev = 1.0 / np.sqrt(gamma0) * np.ones(N)

    if n == 0:
        return P_prev

    gamma1 = ((alpha + 1) * (beta + 1) / (alpha + beta + 3)) * gamma0
    P_curr = ((alpha + beta + 2) * x / 2.0
              + (alpha - beta) / 2.0) / np.sqrt(gamma1)

    if n == 1:
        return P_curr

    # Three-term recurrence
    a_old = (2.0 / (2.0 + alpha + beta)
             * np.sqrt((alpha + 1) * (beta + 1) / (alpha + beta + 3)))

    for i in range(1, n):
        h1 = 2.0 * i + alpha + beta
        a_new = (2.0 / (h1 + 2.0)
                 * np.sqrt((i + 1) * (i + 1 + alpha + beta)
                           * (i + 1 + alpha) * (i + 1 + beta)
                           / ((h1 + 1) * (h1 + 3))))
        b_new = -(alpha**2 - beta**2) / (h1 * (h1 + 2.0))
        P_next = (1.0 / a_new) * (-a_old * P_prev + (x - b_new) * P_curr)
        P_prev = P_curr
        P_curr = P_next
        a_old = a_new

    return P_curr


def grad_jacobi_p(x: np.ndarray, alpha: float, beta: float, n: int) -> np.ndarray:
    """
    Evaluate the derivative of the normalized Jacobi polynomial.

    dP_n^{α,β}/dx = sqrt(n(n+α+β+1)) * P_{n-1}^{α+1,β+1}(x)
    """
    if n == 0:
        return np.zeros_like(x, dtype=np.float64)
    return (np.sqrt(n * (n + alpha + beta + 1.0))
            * jacobi_p(x, alpha + 1.0, beta + 1.0, n - 1))


# =============================================================================
# PKD basis on triangle
# =============================================================================

def _rs_to_ab(r: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Map from (r,s) on reference triangle to collapsed coordinates (a,b).

    a = 2(1+r)/(1-s) - 1  if s ≠ 1
    b = s
    """
    a = np.where(np.abs(1.0 - s) > 1e-14,
                 2.0 * (1.0 + r) / (1.0 - s) - 1.0,
                 -1.0)
    b = s.copy()
    return a, b


def simplex2d_p(r: np.ndarray, s: np.ndarray, i: int, j: int) -> np.ndarray:
    """
    Evaluate 2D orthonormal polynomial on the reference triangle.

    ψ_{ij}(r,s) = P_i^{0,0}(a) · P_j^{2i+1,0}(b) · (1-b)^i · √2

    where (a,b) are collapsed coordinates.
    """
    a, b = _rs_to_ab(r, s)
    h1 = jacobi_p(a, 0, 0, i)
    h2 = jacobi_p(b, 2 * i + 1, 0, j)
    P = np.sqrt(2.0) * h1 * h2 * (1.0 - b)**i
    return P


def grad_simplex2d_p(r: np.ndarray, s: np.ndarray,
                     i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate gradient of 2D orthonormal polynomial ψ_{ij}(r,s).

    ψ_{ij} = √2 · P_i^{0,0}(a) · P_j^{2i+1,0}(b) · (1-b)^i

    Collapsed coordinates: a = 2(1+r)/(1-s)-1, b = s
    Chain rule: da/dr = 2/(1-b), da/ds = (1+a)/(1-b), db/dr = 0, db/ds = 1

    Returns (dψ/dr, dψ/ds).
    """
    a, b = _rs_to_ab(r, s)

    fa = jacobi_p(a, 0, 0, i)       # P_i^{0,0}(a)
    dfa = grad_jacobi_p(a, 0, 0, i)  # dP_i/da
    gb = jacobi_p(b, 2 * i + 1, 0, j)  # P_j^{2i+1,0}(b)
    dgb = grad_jacobi_p(b, 2 * i + 1, 0, j)  # dP_j/db

    # --- dψ/dr ---
    # dψ/dr = √2 · dP_i/da · (da/dr) · P_j(b) · (1-b)^i
    #       = √2 · dfa · 2/(1-b) · gb · (1-b)^i
    #       = √2 · 2 · dfa · gb · (1-b)^{i-1}
    if i == 0:
        # (1-b)^{-1} * dfa: but dfa=0 for i=0, so the whole thing is 0
        dmdr = np.zeros_like(r)
    else:
        dmdr = np.sqrt(2.0) * 2.0 * dfa * gb * (1.0 - b)**(i - 1)

    # --- dψ/ds ---
    # Three terms from product rule:
    # T1: √2 · dP_i/da · (da/ds) · P_j(b) · (1-b)^i
    #   = √2 · dfa · (1+a)/(1-b) · gb · (1-b)^i
    #   = √2 · dfa · (1+a) · gb · (1-b)^{i-1}
    # T2: √2 · P_i(a) · dP_j/db · (1-b)^i
    # T3: √2 · P_i(a) · P_j(b) · (-i) · (1-b)^{i-1}

    if i == 0:
        # T1 = 0 (dfa=0), T3 = 0 (i=0)
        # T2 = √2 · fa · dgb
        dmds = np.sqrt(2.0) * fa * dgb
    else:
        T1 = dfa * (1.0 + a) * gb * (1.0 - b)**(i - 1)
        T2 = fa * dgb * (1.0 - b)**i
        T3 = fa * gb * (-i) * (1.0 - b)**(i - 1)
        dmds = np.sqrt(2.0) * (T1 + T2 + T3)

    return dmdr, dmds


# =============================================================================
# Vandermonde matrices
# =============================================================================

def _mode_indices(p: int) -> list[tuple[int, int]]:
    """Return (i, j) mode indices for total degree ≤ p."""
    modes = []
    for i in range(p + 1):
        for j in range(p + 1 - i):
            modes.append((i, j))
    return modes


def n_modes_2d(p: int) -> int:
    """Number of modes = (p+1)(p+2)/2."""
    return (p + 1) * (p + 2) // 2


def vandermonde_2d(p: int, r: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    2D Vandermonde matrix: V[n, m] = ψ_m(r_n, s_n).

    Parameters
    ----------
    p : int
        Polynomial degree.
    r, s : ndarray, shape (Np,)
        Node coordinates on the reference triangle.

    Returns
    -------
    V : ndarray, shape (Np, Nmodes)
    """
    modes = _mode_indices(p)
    Np = len(r)
    Nm = len(modes)
    V = np.empty((Np, Nm), dtype=np.float64)
    for m, (i, j) in enumerate(modes):
        V[:, m] = simplex2d_p(r, s, i, j)
    return V


def grad_vandermonde_2d(p: int, r: np.ndarray,
                        s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Gradient of 2D Vandermonde: V2Dr[n,m] = dψ_m/dr(r_n, s_n), etc.
    """
    modes = _mode_indices(p)
    Np = len(r)
    Nm = len(modes)
    V2Dr = np.empty((Np, Nm), dtype=np.float64)
    V2Ds = np.empty((Np, Nm), dtype=np.float64)
    for m, (i, j) in enumerate(modes):
        V2Dr[:, m], V2Ds[:, m] = grad_simplex2d_p(r, s, i, j)
    return V2Dr, V2Ds


# =============================================================================
# Warp & Blend nodes on reference triangle
# =============================================================================

def _warp_factor(p: int, rout: np.ndarray) -> np.ndarray:
    """
    Compute warp function for Warp & Blend nodes.

    Based on the difference between GLL nodes and equidistant nodes,
    interpolated via Lagrange basis.
    """
    # Equidistant and GLL nodes on [-1, 1]
    eq = np.linspace(-1, 1, p + 1)

    from .quadrature.gauss import gauss_lobatto
    gll, _ = gauss_lobatto(p + 1)

    # Build Lagrange interpolation at equidistant points, evaluate at rout
    Np = len(rout)
    warp = np.zeros(Np)
    for i in range(p + 1):
        # Lagrange basis polynomial L_i evaluated at rout
        Li = np.ones(Np)
        for j in range(p + 1):
            if i != j:
                Li *= (rout - eq[j]) / (eq[i] - eq[j])
        warp += Li * (gll[i] - eq[i])

    # Zero warp at boundaries
    # Scale to maintain interpolation accuracy
    zerof = np.abs(rout) < 1.0 - 1e-10
    sf = 1.0 - (zerof * rout)**2
    warp = warp / sf + warp * (1.0 - zerof)

    return warp


def nodes_triangle(p: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute interpolation nodes on the reference triangle.

    Uses equidistant nodes in barycentric coordinates, mapped to the
    reference triangle [(-1,-1), (1,-1), (-1,1)].

    For high orders (p > 8), consider replacing with Warp & Blend nodes
    for better Lebesgue constants.

    Parameters
    ----------
    p : int
        Polynomial degree.

    Returns
    -------
    r, s : ndarray, shape (Np,)
        Node coordinates on reference triangle [(-1,-1), (1,-1), (-1,1)].
    """
    Np = n_modes_2d(p)

    # Barycentric coordinates for equidistant nodes
    L1 = np.zeros(Np)
    L2 = np.zeros(Np)
    L3 = np.zeros(Np)

    idx = 0
    for i in range(p + 1):
        for j in range(p + 1 - i):
            k = p - i - j
            L1[idx] = i / p if p > 0 else 1.0 / 3.0
            L2[idx] = j / p if p > 0 else 1.0 / 3.0
            L3[idx] = k / p if p > 0 else 1.0 / 3.0
            idx += 1

    # Map barycentric to reference triangle:
    # (r,s) = L1*(-1,-1) + L2*(1,-1) + L3*(-1,1)
    r = -L1 + L2 - L3
    s = -L1 - L2 + L3

    return r, s


# =============================================================================
# Differentiation and Lift matrices
# =============================================================================

def diff_matrices_2d(p: int, r: np.ndarray,
                     s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute differentiation matrices Dr, Ds on the reference triangle.

    Dr = Vr @ V^{-1},  Ds = Vs @ V^{-1}

    Returns
    -------
    Dr, Ds : ndarray, shape (Np, Np)
    """
    V = vandermonde_2d(p, r, s)
    Vr, Vs = grad_vandermonde_2d(p, r, s)
    Vinv = np.linalg.inv(V)
    Dr = Vr @ Vinv
    Ds = Vs @ Vinv
    return Dr, Ds


def face_mask(p: int, r: np.ndarray, s: np.ndarray) -> list[np.ndarray]:
    """
    Find indices of nodes on each face of the reference triangle.

    Face 0: s = -1 (bottom, from (-1,-1) to (1,-1))
    Face 1: r + s = 0 (hypotenuse, from (1,-1) to (-1,1))
    Face 2: r = -1 (left, from (-1,1) to (-1,-1))

    Returns
    -------
    Fmask : list of 3 ndarray, each shape (p+1,)
    """
    tol = 1e-10
    f0 = np.where(np.abs(s + 1.0) < tol)[0]
    f1 = np.where(np.abs(r + s) < tol)[0]
    f2 = np.where(np.abs(r + 1.0) < tol)[0]

    # Sort by coordinate along the face
    f0 = f0[np.argsort(r[f0])]
    f1 = f1[np.argsort(-r[f1])]  # from (1,-1) to (-1,1)
    f2 = f2[np.argsort(-s[f2])]  # from (-1,1) to (-1,-1)

    return [f0, f1, f2]


def lift_matrix(p: int, r: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    Compute the surface-to-volume lift matrix: LIFT = M^{-1} @ Emat.

    Emat integrates a function defined on the faces against volume
    test functions. LIFT maps face data to volume RHS contributions.

    Returns
    -------
    LIFT : ndarray, shape (Np, 3*(p+1))
    """
    V = vandermonde_2d(p, r, s)
    Fmask = face_mask(p, r, s)
    Nfp = p + 1  # nodes per face
    Np = n_modes_2d(p)

    Emat = np.zeros((Np, 3 * Nfp))

    # For each face, build 1D mass matrix and map to volume
    for f in range(3):
        fids = Fmask[f]

        if f == 0:
            # Face 0: s=-1, parameterized by r
            face_r = r[fids]
        elif f == 1:
            # Face 1: r+s=0, parameterized by r (descending)
            face_r = r[fids]
        else:
            # Face 2: r=-1, parameterized by s
            face_r = s[fids]

        # 1D Vandermonde on face
        V1D = np.zeros((Nfp, Nfp))
        for j in range(Nfp):
            V1D[:, j] = jacobi_p(face_r, 0, 0, j)

        Mface = np.linalg.inv(V1D @ V1D.T)

        # Map face mass matrix to volume
        for i_local in range(Nfp):
            i_vol = fids[i_local]
            for j_local in range(Nfp):
                Emat[i_vol, f * Nfp + j_local] += Mface[i_local, j_local]

    LIFT = V @ (V.T @ Emat)
    return LIFT


# =============================================================================
# Precomputed reference element data
# =============================================================================

class RefTriangle:
    """
    Precomputed reference triangle data for DG order p.

    Attributes
    ----------
    p : int
        Polynomial degree.
    Np : int
        Number of DOFs per element = (p+1)(p+2)/2.
    Nfp : int
        Number of DOFs per face = p+1.
    r, s : ndarray, shape (Np,)
        Node coordinates.
    V, Vinv : ndarray, shape (Np, Np)
        Vandermonde and its inverse.
    Dr, Ds : ndarray, shape (Np, Np)
        Differentiation matrices.
    LIFT : ndarray, shape (Np, 3*Nfp)
        Surface-to-volume lift matrix.
    Fmask : list of 3 ndarray
        Face node indices.
    """

    def __init__(self, p: int):
        self.p = p
        self.Np = n_modes_2d(p)
        self.Nfp = p + 1

        self.r, self.s = nodes_triangle(p)
        self.V = vandermonde_2d(p, self.r, self.s)
        self.Vinv = np.linalg.inv(self.V)
        self.Dr, self.Ds = diff_matrices_2d(p, self.r, self.s)
        self.LIFT = lift_matrix(p, self.r, self.s)
        self.Fmask = face_mask(p, self.r, self.s)

        # Mass matrix (for diagnostics)
        self.M = np.linalg.inv(self.V @ self.V.T)
