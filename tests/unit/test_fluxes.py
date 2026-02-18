"""Tests for numerical flux functions."""

import numpy as np
import numpy.testing as npt
import pytest

from src.dg.flux.scalar import (
    rusanov, lax_friedrichs_global, godunov_burgers,
    engquist_osher_burgers, burgers_flux, linear_advection_flux,
)


class TestFluxConsistency:
    """Every numerical flux must satisfy F̂(u, u) = f(u)."""

    @pytest.mark.parametrize("u_val", [-2.0, -0.5, 0.0, 0.5, 2.0])
    def test_rusanov_consistency_linear(self, u_val):
        a = 1.5
        u = np.array([u_val])
        f_u = linear_advection_flux(u, a)
        ws = np.array([abs(a)])
        F = rusanov(f_u, f_u, u, u, ws)
        npt.assert_allclose(F, f_u, rtol=1e-14)

    @pytest.mark.parametrize("u_val", [-2.0, -0.5, 0.0, 0.5, 2.0])
    def test_rusanov_consistency_burgers(self, u_val):
        u = np.array([u_val])
        f_u = burgers_flux(u)
        ws = np.abs(u)
        F = rusanov(f_u, f_u, u, u, ws)
        npt.assert_allclose(F, f_u, rtol=1e-14)

    @pytest.mark.parametrize("u_val", [-1.0, 0.0, 1.0, 3.0])
    def test_godunov_burgers_consistency(self, u_val):
        u = np.array([u_val])
        F = godunov_burgers(u, u)
        npt.assert_allclose(F, burgers_flux(u), rtol=1e-14)

    @pytest.mark.parametrize("u_val", [-1.0, 0.0, 1.0, 3.0])
    def test_engquist_osher_consistency(self, u_val):
        u = np.array([u_val])
        F = engquist_osher_burgers(u, u)
        npt.assert_allclose(F, burgers_flux(u), rtol=1e-14)


class TestRusanovProperties:
    def test_vectorized(self):
        """Should handle arrays."""
        u_L = np.array([1.0, 2.0, -1.0])
        u_R = np.array([0.5, 1.5, -0.5])
        f_L = burgers_flux(u_L)
        f_R = burgers_flux(u_R)
        ws = np.maximum(np.abs(u_L), np.abs(u_R))
        F = rusanov(f_L, f_R, u_L, u_R, ws)
        assert F.shape == (3,)

    def test_symmetry_dissipation(self):
        """Rusanov adds dissipation: F̂(u_L, u_R) ≠ F̂(u_R, u_L) in general."""
        u_L = np.array([1.0])
        u_R = np.array([0.0])
        f_L = burgers_flux(u_L)
        f_R = burgers_flux(u_R)
        ws = np.array([1.0])
        F_lr = rusanov(f_L, f_R, u_L, u_R, ws)
        F_rl = rusanov(f_R, f_L, u_R, u_L, ws)
        # F̂(a,b) ≠ F̂(b,a) for Rusanov (different dissipation direction)
        # Actually Rusanov IS symmetric: F̂(u_L,u_R) = F̂(u_L,u_R) by definition
        # but F̂(u_L,u_R) ≠ F̂(u_R,u_L) because dissipation sign changes
        # Wait: ½(f_R+f_L) - ½λ(u_R-u_L) vs ½(f_L+f_R) - ½λ(u_L-u_R)
        # These ARE different. But the flux is designed so the correct one is used
        # based on orientation.
        assert not np.allclose(F_lr, F_rl)


class TestGodunovBurgers:
    def test_shock(self):
        """Shock: u_L > u_R, both positive → f(u_L)."""
        u_L = np.array([2.0])
        u_R = np.array([1.0])
        # Shock speed s = (u_L+u_R)/2 = 1.5 > 0, so flux = f(u_L)
        F = godunov_burgers(u_L, u_R)
        npt.assert_allclose(F, burgers_flux(u_L), rtol=1e-14)

    def test_rarefaction_sonic(self):
        """Rarefaction spanning u=0: flux = 0."""
        u_L = np.array([-1.0])
        u_R = np.array([1.0])
        F = godunov_burgers(u_L, u_R)
        npt.assert_allclose(F, 0.0, atol=1e-14)
