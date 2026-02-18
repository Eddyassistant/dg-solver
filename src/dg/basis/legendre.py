"""
Legendre polynomial basis on the reference element [-1, 1].

Modal DG basis using orthogonal Legendre polynomials P_k(ξ), k = 0, ..., p.
Orthogonality: ∫₋₁¹ Pᵢ(ξ) Pⱼ(ξ) dξ = 2/(2i+1) δᵢⱼ

The mass matrix in the modal basis is diagonal:
    M_ij = 2/(2i+1) δ_ij

References
----------
Hesthaven & Warburton (2008), Ch. 3.
Karniadakis & Sherwin (2005), Ch. 2.
"""

import numpy as np
from functools import lru_cache


def legendre_poly(x: np.ndarray, p: int) -> np.ndarray:
    """
    Evaluate Legendre polynomials P_0(x) through P_p(x).

    Uses the three-term recurrence:
        (k+1) P_{k+1}(x) = (2k+1) x P_k(x) - k P_{k-1}(x)

    Parameters
    ----------
    x : ndarray, shape (n_pts,), float64
        Evaluation points.
    p : int
        Maximum polynomial degree (p ≥ 0).

    Returns
    -------
    P : ndarray, shape (p+1, n_pts), float64
        P[k, :] = P_k(x) for k = 0, ..., p.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    P = np.empty((p + 1, n), dtype=np.float64)
    P[0, :] = 1.0
    if p == 0:
        return P
    P[1, :] = x
    for k in range(1, p):
        P[k + 1, :] = ((2 * k + 1) * x * P[k, :] - k * P[k - 1, :]) / (k + 1)
    return P


def legendre_poly_deriv(x: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate Legendre polynomials and their derivatives up to degree p.

    Derivative recurrence:
        P'_{k+1}(x) = (2k+1) P_k(x) + P'_{k-1}(x)

    Parameters
    ----------
    x : ndarray, shape (n_pts,), float64
    p : int

    Returns
    -------
    P : ndarray, shape (p+1, n_pts), float64
        P[k, :] = P_k(x)
    dP : ndarray, shape (p+1, n_pts), float64
        dP[k, :] = P'_k(x)
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    P = np.empty((p + 1, n), dtype=np.float64)
    dP = np.empty((p + 1, n), dtype=np.float64)

    P[0, :] = 1.0
    dP[0, :] = 0.0
    if p == 0:
        return P, dP

    P[1, :] = x
    dP[1, :] = 1.0
    for k in range(1, p):
        P[k + 1, :] = ((2 * k + 1) * x * P[k, :] - k * P[k - 1, :]) / (k + 1)
        dP[k + 1, :] = (2 * k + 1) * P[k, :] + dP[k - 1, :]
    return P, dP


def modal_mass_matrix_diag(p: int) -> np.ndarray:
    """
    Diagonal of the mass matrix for Legendre modal basis on [-1, 1].

    M_kk = 2 / (2k + 1)

    Parameters
    ----------
    p : int
        Polynomial degree.

    Returns
    -------
    M_diag : ndarray, shape (p+1,), float64
    """
    k = np.arange(p + 1, dtype=np.float64)
    return 2.0 / (2.0 * k + 1.0)


def vandermonde(x: np.ndarray, p: int) -> np.ndarray:
    """
    Vandermonde matrix: V[i, j] = P_j(x_i).

    Maps modal coefficients to nodal values: u_nodal = V @ u_modal

    Parameters
    ----------
    x : ndarray, shape (n_pts,), float64
        Evaluation points.
    p : int
        Polynomial degree.

    Returns
    -------
    V : ndarray, shape (n_pts, p+1), float64
    """
    return legendre_poly(x, p).T  # (p+1, n_pts).T → (n_pts, p+1)


def grad_vandermonde(x: np.ndarray, p: int) -> np.ndarray:
    """
    Gradient Vandermonde matrix: Vr[i, j] = P'_j(x_i).

    Parameters
    ----------
    x : ndarray, shape (n_pts,), float64
    p : int

    Returns
    -------
    Vr : ndarray, shape (n_pts, p+1), float64
    """
    _, dP = legendre_poly_deriv(x, p)
    return dP.T


def differentiation_matrix(x: np.ndarray, p: int) -> np.ndarray:
    """
    Differentiation matrix D such that (df/dξ)_i = Σ_j D_ij f_j.

    D = Vr @ V^{-1}

    Parameters
    ----------
    x : ndarray, shape (n_pts,), float64
        Nodal points (e.g. Gauss-Legendre or Gauss-Lobatto).
    p : int
        Polynomial degree.

    Returns
    -------
    D : ndarray, shape (n_pts, n_pts), float64
    """
    V = vandermonde(x, p)
    Vr = grad_vandermonde(x, p)
    return Vr @ np.linalg.inv(V)
