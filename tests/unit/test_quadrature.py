"""Tests for Gauss quadrature rules."""

import numpy as np
import numpy.testing as npt
import pytest

from src.dg.quadrature.gauss import gauss_legendre, gauss_lobatto


class TestGaussLegendre:
    def test_weights_sum(self):
        """Weights must sum to 2 (length of [-1,1])."""
        for n in range(1, 10):
            _, w = gauss_legendre(n)
            npt.assert_allclose(np.sum(w), 2.0, rtol=1e-14)

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8])
    def test_exactness(self, n):
        """n-point rule must integrate x^k exactly for k ≤ 2n-1."""
        xi, w = gauss_legendre(n)
        max_degree = 2 * n - 1
        for k in range(max_degree + 1):
            computed = np.sum(w * xi**k)
            # ∫₋₁¹ x^k dx = [x^{k+1}/(k+1)]₋₁¹ = (1 - (-1)^{k+1})/(k+1)
            exact = (1.0 - (-1.0)**(k + 1)) / (k + 1)
            npt.assert_allclose(computed, exact, atol=1e-13,
                                err_msg=f"Failed for n={n}, k={k}")

    def test_symmetry(self):
        """Nodes and weights should be symmetric about 0."""
        for n in range(1, 8):
            xi, w = gauss_legendre(n)
            npt.assert_allclose(xi + xi[::-1], 0.0, atol=1e-14)
            npt.assert_allclose(w - w[::-1], 0.0, atol=1e-14)

    def test_invalid(self):
        with pytest.raises(ValueError):
            gauss_legendre(0)


class TestGaussLobatto:
    def test_endpoints(self):
        """GLL nodes must include -1 and +1."""
        for n in range(2, 8):
            xi, _ = gauss_lobatto(n)
            npt.assert_allclose(xi[0], -1.0, atol=1e-14)
            npt.assert_allclose(xi[-1], 1.0, atol=1e-14)

    def test_weights_sum(self):
        for n in range(2, 8):
            _, w = gauss_lobatto(n)
            npt.assert_allclose(np.sum(w), 2.0, rtol=1e-14)

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
    def test_exactness(self, n):
        """n-point GLL integrates up to degree 2n-3 exactly."""
        xi, w = gauss_lobatto(n)
        max_degree = 2 * n - 3
        for k in range(max_degree + 1):
            computed = np.sum(w * xi**k)
            exact = (1.0 - (-1.0)**(k + 1)) / (k + 1)
            npt.assert_allclose(computed, exact, atol=1e-12,
                                err_msg=f"Failed for n={n}, k={k}")

    def test_invalid(self):
        with pytest.raises(ValueError):
            gauss_lobatto(1)
