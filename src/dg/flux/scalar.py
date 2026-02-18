"""
Numerical fluxes for 1D scalar conservation laws: u_t + f(u)_x = 0

All fluxes satisfy the consistency condition: F̂(u, u) = f(u).

References
----------
Toro (2009), Riemann Solvers, Ch. 6 & 10.
LeVeque (2002), Finite Volume Methods, Ch. 12.
"""

import numpy as np
from numba import njit


# --- Physical flux functions ---

@njit(cache=True)
def linear_advection_flux(u: np.ndarray, a: float) -> np.ndarray:
    """f(u) = a * u"""
    return a * u


@njit(cache=True)
def burgers_flux(u: np.ndarray) -> np.ndarray:
    """f(u) = u² / 2"""
    return 0.5 * u * u


# --- Numerical fluxes ---

@njit(cache=True)
def rusanov(f_L: np.ndarray, f_R: np.ndarray,
            u_L: np.ndarray, u_R: np.ndarray,
            max_wavespeed: np.ndarray) -> np.ndarray:
    """
    Rusanov (local Lax-Friedrichs) numerical flux.

    F̂(u⁻, u⁺) = ½[f(u⁻) + f(u⁺)] - ½λ(u⁺ - u⁻)

    Parameters
    ----------
    f_L, f_R : ndarray, shape (n_faces,), float64
        Physical flux evaluated at left/right states.
    u_L, u_R : ndarray, shape (n_faces,), float64
        Left and right solution states.
    max_wavespeed : ndarray, shape (n_faces,), float64
        Maximum wave speed λ = max|f'(u)| at each face.

    Returns
    -------
    flux : ndarray, shape (n_faces,), float64

    References
    ----------
    Toro (2009), Eq. (10.55).
    """
    return 0.5 * (f_L + f_R) - 0.5 * max_wavespeed * (u_R - u_L)


@njit(cache=True)
def lax_friedrichs_global(f_L: np.ndarray, f_R: np.ndarray,
                          u_L: np.ndarray, u_R: np.ndarray,
                          alpha: float) -> np.ndarray:
    """
    Global Lax-Friedrichs flux with a single global dissipation coefficient.

    F̂ = ½[f(u⁻) + f(u⁺)] - ½α(u⁺ - u⁻)

    Parameters
    ----------
    alpha : float
        Global max wavespeed.
    """
    return 0.5 * (f_L + f_R) - 0.5 * alpha * (u_R - u_L)


@njit(cache=True)
def godunov_linear(u_L: np.ndarray, u_R: np.ndarray, a: float) -> np.ndarray:
    """
    Exact Godunov flux for linear advection: f(u) = a*u.

    F̂ = a * u⁻  if a ≥ 0
       = a * u⁺  if a < 0
    """
    if a >= 0.0:
        return a * u_L
    else:
        return a * u_R


@njit(cache=True)
def godunov_burgers(u_L: np.ndarray, u_R: np.ndarray) -> np.ndarray:
    """
    Exact Godunov flux for Burgers' equation: f(u) = u²/2.

    Handles the sonic case (rarefaction fan passing through u=0).

    References
    ----------
    LeVeque (2002), Eq. (12.4).
    """
    n = u_L.shape[0]
    flux = np.empty(n, dtype=np.float64)
    for i in range(n):
        uL = u_L[i]
        uR = u_R[i]
        if uL >= uR:
            # Shock: use Rankine-Hugoniot
            s = 0.5 * (uL + uR)
            if s >= 0.0:
                flux[i] = 0.5 * uL * uL
            else:
                flux[i] = 0.5 * uR * uR
        else:
            # Rarefaction
            if uL >= 0.0:
                flux[i] = 0.5 * uL * uL
            elif uR <= 0.0:
                flux[i] = 0.5 * uR * uR
            else:
                flux[i] = 0.0  # Sonic point in rarefaction
    return flux


@njit(cache=True)
def engquist_osher_burgers(u_L: np.ndarray, u_R: np.ndarray) -> np.ndarray:
    """
    Engquist-Osher flux for Burgers' equation.

    F̂(u⁻, u⁺) = ½max(u⁻,0)² + ½min(u⁺,0)²

    Smooth, entropy-satisfying, monotone.
    """
    return 0.5 * np.maximum(u_L, 0.0)**2 + 0.5 * np.minimum(u_R, 0.0)**2
