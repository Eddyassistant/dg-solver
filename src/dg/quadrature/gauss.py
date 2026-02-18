"""
Gauss-Legendre and Gauss-Lobatto quadrature on the reference interval [-1, 1].

Uses numpy's polynomial module for node/weight computation.
Exactness: n-point Gauss-Legendre integrates polynomials of degree ≤ 2n-1 exactly.

References
----------
Hesthaven & Warburton (2008), Appendix A.
Abramowitz & Stegun, Ch. 25.
"""

import numpy as np
from functools import lru_cache


@lru_cache(maxsize=32)
def gauss_legendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return n-point Gauss-Legendre nodes and weights on [-1, 1].

    Parameters
    ----------
    n : int
        Number of quadrature points (n ≥ 1).

    Returns
    -------
    nodes : ndarray, shape (n,), float64
        Quadrature nodes in ascending order.
    weights : ndarray, shape (n,), float64
        Corresponding quadrature weights (sum = 2).
    """
    if n < 1:
        raise ValueError(f"Need n ≥ 1, got {n}")
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return np.ascontiguousarray(nodes), np.ascontiguousarray(weights)


@lru_cache(maxsize=32)
def gauss_lobatto(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return n-point Gauss-Lobatto-Legendre nodes and weights on [-1, 1].

    Includes endpoints -1 and +1. Exactness: degree ≤ 2n-3.

    Parameters
    ----------
    n : int
        Number of quadrature points (n ≥ 2).

    Returns
    -------
    nodes : ndarray, shape (n,), float64
    weights : ndarray, shape (n,), float64
    """
    if n < 2:
        raise ValueError(f"Need n ≥ 2, got {n}")
    if n == 2:
        return np.array([-1.0, 1.0]), np.array([1.0, 1.0])

    # GLL nodes: -1, +1, and roots of P'_{n-1}(x).
    # P'_{n-1}(x) has degree n-2, so n-2 interior roots.
    # Find them by computing the derivative polynomial and finding its roots.
    # Legendre coefficients for P_{n-1}: all zeros except index n-1 = 1
    coeffs_pnm1 = np.zeros(n)
    coeffs_pnm1[n - 1] = 1.0
    # Derivative in Legendre basis
    deriv_coeffs = np.polynomial.legendre.legder(coeffs_pnm1)
    # Find roots
    interior_nodes = np.sort(np.polynomial.legendre.legroots(deriv_coeffs))

    nodes = np.empty(n, dtype=np.float64)
    nodes[0] = -1.0
    nodes[-1] = 1.0
    nodes[1:-1] = interior_nodes

    # Weights: w_j = 2 / (n*(n-1)*[P_{n-1}(x_j)]^2)
    P_vals = np.polynomial.legendre.legval(nodes, coeffs_pnm1)
    weights = 2.0 / (n * (n - 1) * P_vals**2)

    return np.ascontiguousarray(nodes), np.ascontiguousarray(weights)
