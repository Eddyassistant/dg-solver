"""
Tests for the 2D system DG solver.

Test case: decoupled advection of 2 variables with different speeds.
    q = [u, v], u_t + a*u_x = 0, v_t + b*v_y = 0
"""

import numpy as np
import numpy.testing as npt
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.dg.solver2d_system import DG2DSystem
from src.dg.mesh.triangle_mesh import TriangleMesh


def make_advection_system(nx, p, a=1.0, b=0.5):
    """Create a 2-variable decoupled advection system."""
    mesh = TriangleMesh.rectangle((0.0, 1.0), (0.0, 1.0), nx, nx)

    def flux_x(q):
        Fx = np.zeros_like(q)
        Fx[:, :, 0] = a * q[:, :, 0]  # u advects in x
        return Fx

    def flux_y(q):
        Fy = np.zeros_like(q)
        Fy[:, :, 1] = b * q[:, :, 1]  # v advects in y
        return Fy

    def numerical_flux(q_int, q_ext, nx_arr, ny_arr):
        """Rusanov flux for decoupled system."""
        K, n_fp, nv = q_int.shape
        f_num = np.zeros((K, n_fp, nv))

        # Variable 0: flux = a*u in x-direction
        fn_int_0 = a * q_int[:, :, 0] * nx_arr
        fn_ext_0 = a * q_ext[:, :, 0] * nx_arr
        lam0 = abs(a)
        f_num[:, :, 0] = 0.5 * (fn_int_0 + fn_ext_0) - 0.5 * lam0 * (q_ext[:, :, 0] - q_int[:, :, 0])

        # Variable 1: flux = b*v in y-direction
        fn_int_1 = b * q_int[:, :, 1] * ny_arr
        fn_ext_1 = b * q_ext[:, :, 1] * ny_arr
        lam1 = abs(b)
        f_num[:, :, 1] = 0.5 * (fn_int_1 + fn_ext_1) - 0.5 * lam1 * (q_ext[:, :, 1] - q_int[:, :, 1])

        return f_num

    def max_wavespeed(q):
        return max(abs(a), abs(b))

    return DG2DSystem(
        mesh, p, n_vars=2,
        flux_x=flux_x, flux_y=flux_y,
        numerical_flux=numerical_flux,
        max_wavespeed=max_wavespeed,
        periodic=True, x_range=(0.0, 1.0), y_range=(0.0, 1.0),
    )


class TestSystemDGConstant:
    def test_constant_preserving(self):
        solver = make_advection_system(4, 2)
        q = np.ones((solver.K, solver.Np, 2))
        rhs = solver.compute_rhs(q, 0.0)
        npt.assert_allclose(rhs, 0.0, atol=1e-13)


class TestSystemDGConvergence:
    @pytest.mark.parametrize("p", [1, 2, 3])
    def test_advection_system_convergence(self, p):
        a, b = 1.0, 0.5
        tf = 0.1

        def q0(x, y):
            q = np.zeros((*x.shape, 2))
            q[:, :, 0] = np.sin(2 * np.pi * x)
            q[:, :, 1] = np.cos(2 * np.pi * y)
            return q

        def q_exact(x, y):
            q = np.zeros((*x.shape, 2))
            q[:, :, 0] = np.sin(2 * np.pi * (x - a * tf))
            q[:, :, 1] = np.cos(2 * np.pi * (y - b * tf))
            return q

        errors = []
        for nx in [4, 8, 16]:
            solver = make_advection_system(nx, p, a, b)
            cfl = min(0.1, 0.5 / (2 * p + 1)**2)
            q, t, _ = solver.solve(q0, tf, cfl=cfl)
            errs = solver.l2_error(q, q_exact, t)
            errors.append(np.max(errs))

        # Check convergence rate
        rate = np.log(errors[-2] / errors[-1]) / np.log(2)
        expected = p + 1
        assert rate > expected - 0.5, \
            f"P{p} system: expected rate ~{expected}, got {rate:.2f}"
