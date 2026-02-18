"""
Tests for the artificial compressibility INS solver.

Test: flux consistency (F̂(q,q) = F(q)·n) for the AC system.
"""

import numpy as np
import numpy.testing as npt
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.dg.ins_ac import (
    ac_flux_x, ac_flux_y, ac_numerical_flux, ac_max_wavespeed,
)


class TestACFluxConsistency:
    """Numerical flux must satisfy F̂(q,q)·n = F(q)·n."""

    @pytest.mark.parametrize("beta", [1.0, 5.0, 10.0])
    def test_consistency(self, beta):
        K, Nfp, nv = 4, 6, 3
        # Random state
        rng = np.random.default_rng(42)
        q = rng.standard_normal((K, Nfp, nv))
        q[:, :, 2] = np.abs(q[:, :, 2])  # positive pressure

        # Random normals (unit)
        theta = rng.uniform(0, 2 * np.pi, (K, Nfp))
        nx = np.cos(theta)
        ny = np.sin(theta)

        # Numerical flux with identical states
        f_num = ac_numerical_flux(q, q, nx, ny, beta=beta)

        # Physical flux · n
        u, v, p = q[:, :, 0], q[:, :, 1], q[:, :, 2]
        fn_exact = np.empty((K, Nfp, nv))
        fn_exact[:, :, 0] = (u**2 + p) * nx + (u * v) * ny
        fn_exact[:, :, 1] = (u * v) * nx + (v**2 + p) * ny
        fn_exact[:, :, 2] = beta**2 * (u * nx + v * ny)

        npt.assert_allclose(f_num, fn_exact, rtol=1e-13)


class TestACWavespeed:
    def test_positive(self):
        q = np.zeros((2, 3, 3))
        q[:, :, 0] = 0.5  # u
        q[:, :, 1] = 0.3  # v
        ws = ac_max_wavespeed(q, beta=5.0)
        assert ws > 0
        # Should be ≈ |V| + sqrt(|V|² + β²) ≈ 0.583 + sqrt(0.34 + 25) ≈ 5.62
        assert ws > 5.0


class TestACConstantPreserving:
    def test_uniform_flow_rhs_zero(self):
        """Uniform flow q = (U, 0, P) should have zero inviscid RHS."""
        from src.dg.solver2d_system import DG2DSystem
        from src.dg.mesh.triangle_mesh import TriangleMesh

        beta = 5.0
        mesh = TriangleMesh.rectangle((0.0, 1.0), (0.0, 1.0), 4, 4)
        solver = DG2DSystem(
            mesh, p=2, n_vars=3,
            flux_x=lambda q: ac_flux_x(q, beta),
            flux_y=lambda q: ac_flux_y(q, beta),
            numerical_flux=lambda qi, qe, nx, ny: ac_numerical_flux(qi, qe, nx, ny, beta),
            max_wavespeed=lambda q: ac_max_wavespeed(q, beta),
            periodic=True, x_range=(0.0, 1.0), y_range=(0.0, 1.0),
        )

        # Uniform state
        q = np.zeros((solver.K, solver.Np, 3))
        q[:, :, 0] = 1.0   # u = 1
        q[:, :, 2] = 0.5   # p = 0.5

        rhs = solver.compute_rhs(q, 0.0)
        npt.assert_allclose(rhs, 0.0, atol=1e-12)
