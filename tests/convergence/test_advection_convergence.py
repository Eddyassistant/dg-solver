"""
MMS convergence tests for DG solver on linear advection: u_t + a u_x = 0.

Exact solution: u(x,t) = sin(2π(x - at)) on [0, 1] periodic.

Expected convergence rate: O(h^{p+1}) in L2 norm for smooth solutions.

References
----------
Hesthaven & Warburton (2008), Ch. 5, Table 5.1.
"""

import numpy as np
import numpy.testing as npt
import pytest

from src.dg.solver import DG1DScalar
from src.dg.mesh.mesh1d import Mesh1D
from src.dg.flux.scalar import rusanov, linear_advection_flux


def make_advection_solver(n_elem: int, p: int, a: float = 1.0) -> DG1DScalar:
    mesh = Mesh1D.uniform(0.0, 1.0, n_elem, periodic=True)

    def phys_flux(u):
        return a * u

    def max_ws(u_L, u_R):
        return np.full_like(u_L, abs(a))

    # Use enough quadrature points
    n_quad = max(p + 1, (3 * p + 1) // 2 + 1)

    return DG1DScalar(
        mesh=mesh, p=p,
        physical_flux=phys_flux,
        max_wavespeed_func=max_ws,
        n_quad=n_quad,
    )


def run_advection(n_elem: int, p: int, a: float = 1.0,
                  t_final: float = 1.0, cfl: float = 0.1) -> float:
    """Run advection and return L2 error."""
    solver = make_advection_solver(n_elem, p, a)

    def u0(x):
        return np.sin(2.0 * np.pi * x)

    def u_exact(x):
        return np.sin(2.0 * np.pi * (x - a * t_final))

    u_modal, t = solver.solve(u0, t_final, cfl=cfl)
    return solver.l2_error(u_modal, u_exact, t)


@pytest.mark.parametrize("p", [0, 1, 2, 3, 4])
def test_advection_convergence_rate(p):
    """
    Verify O(h^{p+1}) convergence in L2 for DG-P{p} on smooth advection.
    """
    n_elems = [8, 16, 32, 64]
    # Use smaller CFL for high-order to avoid temporal error domination
    cfl = 0.01 if p >= 4 else 0.1
    errors = [run_advection(n, p, cfl=cfl) for n in n_elems]

    # Compute convergence rates between successive refinements
    h = np.array([1.0 / n for n in n_elems])
    rates = np.log(np.array(errors[:-1]) / np.array(errors[1:])) / np.log(2.0)

    expected_rate = p + 1
    # Use the finest rate (most reliable)
    best_rate = rates[-1]

    print(f"\nDG-P{p} advection convergence:")
    for i, n in enumerate(n_elems):
        print(f"  N={n:4d}  h={h[i]:.4f}  error={errors[i]:.4e}"
              + (f"  rate={rates[i-1]:.2f}" if i > 0 else ""))

    assert best_rate > expected_rate - 0.3, \
        f"DG-P{p}: expected rate ~{expected_rate}, got {best_rate:.2f}"
