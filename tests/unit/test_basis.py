"""Tests for Legendre basis functions."""

import numpy as np
import numpy.testing as npt
import pytest

from src.dg.basis.legendre import (
    legendre_poly, legendre_poly_deriv, modal_mass_matrix_diag,
    vandermonde, differentiation_matrix,
)
from src.dg.quadrature.gauss import gauss_legendre


class TestLegendrePoly:
    def test_values_at_endpoints(self):
        """P_k(1) = 1 and P_k(-1) = (-1)^k for all k."""
        p = 8
        P = legendre_poly(np.array([1.0, -1.0]), p)
        for k in range(p + 1):
            npt.assert_allclose(P[k, 0], 1.0, atol=1e-14)
            npt.assert_allclose(P[k, 1], (-1.0)**k, atol=1e-14)

    def test_orthogonality(self):
        """∫₋₁¹ Pᵢ Pⱼ dξ = 2/(2i+1) δᵢⱼ."""
        p = 6
        n_quad = p + 2  # enough for exact integration of P_i * P_j (degree 2p)
        xi, w = gauss_legendre(n_quad)
        P = legendre_poly(xi, p)  # (p+1, n_quad)
        # Gram matrix: G[i,j] = Σ_q w_q P_i(ξ_q) P_j(ξ_q)
        G = P @ np.diag(w) @ P.T
        expected = np.diag(2.0 / (2.0 * np.arange(p + 1) + 1.0))
        npt.assert_allclose(G, expected, atol=1e-13)

    def test_p0_is_constant(self):
        x = np.linspace(-1, 1, 20)
        P = legendre_poly(x, 0)
        npt.assert_allclose(P[0], 1.0, atol=1e-15)

    def test_p1_is_x(self):
        x = np.linspace(-1, 1, 20)
        P = legendre_poly(x, 1)
        npt.assert_allclose(P[1], x, atol=1e-15)


class TestLegendreDerivative:
    def test_p0_deriv_zero(self):
        x = np.linspace(-1, 1, 10)
        _, dP = legendre_poly_deriv(x, 0)
        npt.assert_allclose(dP[0], 0.0, atol=1e-15)

    def test_p1_deriv_one(self):
        x = np.linspace(-1, 1, 10)
        _, dP = legendre_poly_deriv(x, 1)
        npt.assert_allclose(dP[1], 1.0, atol=1e-15)

    def test_p2_deriv(self):
        """P_2(x) = (3x²-1)/2, P'_2(x) = 3x."""
        x = np.linspace(-1, 1, 15)
        _, dP = legendre_poly_deriv(x, 2)
        npt.assert_allclose(dP[2], 3.0 * x, atol=1e-13)


class TestMassMatrix:
    def test_values(self):
        M = modal_mass_matrix_diag(4)
        expected = 2.0 / (2.0 * np.arange(5) + 1.0)
        npt.assert_allclose(M, expected, atol=1e-15)


class TestVandermonde:
    def test_square_invertible(self):
        """V at p+1 GL points should be invertible."""
        for p in range(1, 6):
            xi, _ = gauss_legendre(p + 1)
            V = vandermonde(xi, p)
            assert V.shape == (p + 1, p + 1)
            assert np.linalg.cond(V) < 1e10

    def test_reconstruction(self):
        """V @ coeffs should give nodal values."""
        p = 3
        xi, _ = gauss_legendre(p + 1)
        V = vandermonde(xi, p)
        # Pure P_2: coeffs = [0, 0, 1, 0]
        coeffs = np.array([0.0, 0.0, 1.0, 0.0])
        nodal = V @ coeffs
        P = legendre_poly(xi, p)
        npt.assert_allclose(nodal, P[2], atol=1e-14)


class TestDiffMatrix:
    def test_differentiates_x(self):
        """D should differentiate f(ξ) = ξ exactly: f' = 1."""
        p = 3
        xi, _ = gauss_legendre(p + 1)
        D = differentiation_matrix(xi, p)
        f = xi
        npt.assert_allclose(D @ f, np.ones_like(xi), atol=1e-13)

    def test_differentiates_x_squared(self):
        """D should differentiate f(ξ) = ξ² exactly: f' = 2ξ."""
        p = 3
        xi, _ = gauss_legendre(p + 1)
        D = differentiation_matrix(xi, p)
        npt.assert_allclose(D @ (xi**2), 2.0 * xi, atol=1e-12)
