# dg-solver

High-order Discontinuous Galerkin solver for conservation laws on triangular meshes.

## Features

- **1D & 2D** scalar conservation law solvers
- **Nodal DG** on triangles with PKD orthonormal basis
- **Numba-accelerated** fused volume+surface kernels
- **High-order**: verified optimal O(h^{p+1}) convergence for P1–P5
- **Numerical fluxes**: Rusanov, Godunov, Engquist-Osher
- **Time integration**: SSP-RK2/3, classical RK4
- **Limiters**: TVB minmod (1D)

## Quick Start

```bash
uv sync
uv run pytest tests/ -v
uv run python scripts/hp_convergence.py
```

## Convergence Rates (2D, linear advection on triangles)

| Order | Expected | Measured |
|-------|----------|----------|
| P1    | 2        | 2.05–2.16 |
| P2    | 3        | 2.82–2.96 |
| P3    | 4        | 4.05–4.41 |
| P4    | 5        | 4.69–4.96 |
| P5    | 6        | 6.02–6.13 |

## Project Structure

```
src/dg/
  basis/          # Legendre polynomials, PKD basis
  quadrature/     # Gauss-Legendre, Gauss-Lobatto
  flux/           # Numerical flux functions
  limiter/        # TVB minmod
  mesh/           # 1D/2D mesh generation + connectivity
  timestepping/   # Runge-Kutta integrators
  reference_triangle.py  # 2D reference element ops
  solver.py       # 1D DG solver
  solver2d.py     # 2D DG solver (Numba)
tests/
  unit/           # Quadrature, basis, flux tests
  convergence/    # MMS convergence rate tests
scripts/
  hp_convergence.py  # h/p convergence study
```
