"""
TVB (Total Variation Bounded) minmod limiter for modal DG.

Limits the slope (and higher modes) to prevent spurious oscillations
near discontinuities while preserving high-order accuracy in smooth regions.

The TVB modification introduces a parameter M that avoids limiting near
smooth extrema: if |ũ₁| ≤ M·h², don't limit.

References
----------
Cockburn & Shu (1989), Math. Comp. 52(186), pp. 411-435.
Cockburn & Shu (2001), J. Sci. Comput. 16(3), pp. 173-261.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def minmod(a: float, b: float, c: float) -> float:
    """
    Three-argument minmod function.

    minmod(a, b, c) = s · min(|a|, |b|, |c|)  if sign(a) = sign(b) = sign(c) = s
                    = 0                          otherwise
    """
    if a > 0.0 and b > 0.0 and c > 0.0:
        return min(a, min(b, c))
    elif a < 0.0 and b < 0.0 and c < 0.0:
        return max(a, max(b, c))
    else:
        return 0.0


@njit(cache=True)
def tvb_minmod(a: float, b: float, c: float, M: float, h: float) -> float:
    """
    TVB-modified minmod: skip limiting if |a| ≤ M·h².

    Parameters
    ----------
    a : float
        The slope to potentially limit (modal coefficient ũ₁).
    b, c : float
        Neighboring slopes for comparison.
    M : float
        TVB constant (problem-dependent, M = 0 gives strict minmod).
    h : float
        Element size.
    """
    if abs(a) <= M * h * h:
        return a
    return minmod(a, b, c)


@njit(cache=True)
def apply_tvb_limiter(u_modal: np.ndarray, dx: np.ndarray,
                      M: float = 0.0) -> np.ndarray:
    """
    Apply TVB minmod limiter to 1D DG solution in modal (Legendre) basis.

    For each element, check if the linear mode (ũ₁) is "too steep" compared
    to neighbor cell averages. If so, replace ũ₁ with the limited value and
    zero out all higher modes.

    Parameters
    ----------
    u_modal : ndarray, shape (n_elem, n_modes), float64
        Modal coefficients. u_modal[:, 0] = cell averages,
        u_modal[:, 1] = linear slopes (scaled by basis normalization).
    dx : ndarray, shape (n_elem,), float64
        Element sizes.
    M : float
        TVB constant (M = 0 → standard minmod).

    Returns
    -------
    u_limited : ndarray, same shape as u_modal, float64
        Limited modal coefficients.
    """
    n_elem, n_modes = u_modal.shape
    u_out = u_modal.copy()

    for i in range(n_elem):
        if n_modes < 2:
            continue

        # Cell averages of neighbors (periodic BCs)
        avg_L = u_modal[(i - 1) % n_elem, 0]
        avg_C = u_modal[i, 0]
        avg_R = u_modal[(i + 1) % n_elem, 0]

        # The linear mode represents the slope.
        # At the element boundary, the jump from cell average is ũ₁
        # (since P_1(±1) = ±1 and the contribution is ũ₁·P_1(ξ)).
        u_tilde = u_modal[i, 1]

        # Compare with differences of cell averages
        delta_minus = avg_C - avg_L
        delta_plus = avg_R - avg_C

        u_limited = tvb_minmod(u_tilde, delta_plus, delta_minus, M, dx[i])

        if abs(u_limited - u_tilde) > 1e-14:
            # Limiting was applied: replace slope and kill higher modes
            u_out[i, 1] = u_limited
            for k in range(2, n_modes):
                u_out[i, k] = 0.0

    return u_out
