"""
Explicit Runge-Kutta time integrators for DG.

SSP (Strong Stability Preserving) variants maintain TVD/TVB properties
when combined with appropriate spatial discretizations.

References
----------
Gottlieb, Shu & Tadmor (2001), SIAM Review 43(1), pp. 89-112.
Shu & Osher (1988), J. Comput. Phys. 77, pp. 439-471.
"""

import numpy as np
from typing import Callable


RHSFunc = Callable[[np.ndarray, float], np.ndarray]


def ssp_rk3(u: np.ndarray, dt: float, rhs: RHSFunc, t: float) -> np.ndarray:
    """
    3rd-order SSP Runge-Kutta (Shu-Osher form).

    u⁽¹⁾ = uⁿ + Δt L(uⁿ)
    u⁽²⁾ = ¾ uⁿ + ¼ [u⁽¹⁾ + Δt L(u⁽¹⁾)]
    uⁿ⁺¹ = ⅓ uⁿ + ⅔ [u⁽²⁾ + Δt L(u⁽²⁾)]

    CFL coefficient: c = 1 (same as forward Euler).

    Parameters
    ----------
    u : ndarray, shape (n_elements, n_dofs), float64
        Current solution.
    dt : float
        Time step.
    rhs : callable (u, t) -> ndarray
        Spatial discretization operator L(u).
    t : float
        Current time.

    Returns
    -------
    u_new : ndarray, same shape as u, float64
    """
    u1 = u + dt * rhs(u, t)
    u2 = 0.75 * u + 0.25 * (u1 + dt * rhs(u1, t + dt))
    return (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * rhs(u2, t + 0.5 * dt))


def ssp_rk2(u: np.ndarray, dt: float, rhs: RHSFunc, t: float) -> np.ndarray:
    """
    2nd-order SSP Runge-Kutta (Heun's method / modified Euler).

    u⁽¹⁾ = uⁿ + Δt L(uⁿ)
    uⁿ⁺¹ = ½ uⁿ + ½ [u⁽¹⁾ + Δt L(u⁽¹⁾)]

    CFL coefficient: c = 1.
    """
    u1 = u + dt * rhs(u, t)
    return 0.5 * u + 0.5 * (u1 + dt * rhs(u1, t + dt))


def rk4_classic(u: np.ndarray, dt: float, rhs: RHSFunc, t: float) -> np.ndarray:
    """
    Classical 4th-order Runge-Kutta. NOT SSP — do not use for shocks.
    """
    k1 = rhs(u, t)
    k2 = rhs(u + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = rhs(u + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = rhs(u + dt * k3, t + dt)
    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def euler_forward(u: np.ndarray, dt: float, rhs: RHSFunc, t: float) -> np.ndarray:
    """Forward Euler. First-order, SSP. Mainly for debugging."""
    return u + dt * rhs(u, t)
