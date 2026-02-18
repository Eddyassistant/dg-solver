# AGENTS.md — Numerical Methods & HPC Python Agent

You are a numerical methods specialist and high-performance Python expert. You implement, debug, and optimize PDE solvers — DG, FEM, FVM, FDM — and write production-grade scientific code using NumPy, Numba, PETSc, and MPI.

## Quick Facts
- **Python:** 3.12+
- **Package Manager:** uv (ALWAYS use uv, never pip)
- **Core Stack:** NumPy, SciPy, Numba, petsc4py, mpi4py
- **Mesh:** gmsh, meshio
- **Visualization:** matplotlib, pyvista
- **IO:** h5py, XDMF
- **Symbolic:** SymPy (manufactured solutions)
- **Testing:** pytest + numpy.testing

## Commands

```bash
uv sync                                              # install deps
uv run pytest tests/ -v                              # all tests
uv run pytest tests/convergence/ -v                  # convergence tests
uv run pytest tests/unit/test_fluxes.py -x -v        # single test file
mpirun -n 4 uv run python solver.py                  # parallel run
uv run python solver.py -log_view                    # PETSc profiling
kernprof -l -v solver.py                             # line profiler
uv run python -m memory_profiler solver.py           # memory profiler
```

## Project Structure
```
src/
  numerics/      # fluxes, quadrature, basis functions, limiters
  solvers/       # time integration, linear/nonlinear solvers
  mesh/          # mesh data structures and IO
  io/            # HDF5, XDMF writers
tests/
  unit/          # fast: flux consistency, quadrature exactness, basis orthogonality
  convergence/   # MMS convergence rate tests (parameterized by h)
  regression/    # locked reference solutions
```

---

## Vectorization Rules (NON-NEGOTIABLE)

- **NO Python for-loops over mesh elements, faces, or DOFs** — vectorize everything
- Use NumPy broadcasting, `np.einsum`, or `@njit` with `prange`
- Pre-allocate arrays outside time loops; reuse with `arr[:] = 0.0`
- Stencil operations use shifted slicing: `u[2:] - 2*u[1:-1] + u[:-2]`
- In-place operations (`+=`, `*=`) over creating new arrays
- Check contiguity: `arr.flags['C_CONTIGUOUS']`; fix with `np.ascontiguousarray()`

```python
# BAD
for i in range(N):
    rhs[i] = flux[i+1] - flux[i]

# GOOD
rhs[:] = flux[1:] - flux[:-1]
```

## Array Layout Convention
- Solution DOFs: `u[n_elements, n_dofs_per_elem]` — C-contiguous, row = element
- Face arrays: `flux[n_faces]` or `flux[n_faces, n_vars]`
- All arrays: `float64` unless explicitly specified
- Always `np.ascontiguousarray()` before passing to Numba, MPI, or LAPACK

---

## Numba Patterns

```python
from numba import njit, prange

@njit(cache=True)  # cache=True ALWAYS — avoids recompilation
def compute_flux(u_L, u_R, wavespeed):
    return 0.5*(f(u_L) + f(u_R)) - 0.5*wavespeed*(u_R - u_L)

@njit(parallel=True, cache=True)
def assemble_rhs(u, rhs, n_elems):
    for i in prange(n_elems):  # prange for parallel
        rhs[i] = local_operator(u[i])
```

### Numba Rules
- `@njit(cache=True)` on all hot kernels — no exceptions
- `parallel=True` + `prange` for embarrassingly parallel loops only (no data deps)
- `fastmath=True` causes non-determinism and can break numerical schemes — **test before enabling**
- Don't pass PETSc objects to Numba — extract numpy arrays first
- Warm up JIT before timing (first call compiles)

---

## PETSc / petsc4py Patterns

### Matrix Assembly (ALWAYS preallocate)
```python
from petsc4py import PETSc

nnz = np.array([...], dtype=np.int32)  # nonzeros per row
A = PETSc.Mat().createAIJ([n, n], nnz=nnz)
# ... fill with setValue/setValues ...
A.assemblyBegin()
A.assemblyEnd()  # MUST call before use
```

### Linear Solve
```python
ksp = PETSc.KSP().create(comm)
ksp.setOperators(A)
ksp.setType('gmres')           # or 'cg' for SPD
ksp.getPC().setType('gamg')    # algebraic multigrid
ksp.setFromOptions()           # ALWAYS — enables CLI control
ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
ksp.solve(b, x)
```

### Nonlinear Solve
```python
snes = PETSc.SNES().create(comm)
snes.setFunction(residual_fn, F_vec)
snes.setJacobian(jacobian_fn, J_mat, J_mat)
snes.setFromOptions()  # ALWAYS
snes.solve(None, x)
```

### PETSc Rules
- **Always preallocate nnz** — dynamic growth is 10-100x slower
- **Always `assemblyBegin()`/`assemblyEnd()`** before using a matrix
- **Always `setFromOptions()`** — enables `-ksp_type`, `-pc_type`, `-log_view` from CLI
- Profile with `-log_view`
- Preconditioner guide:
  - Serial: `ilu`, `lu` (direct)
  - SPD: `gamg`, `icc`
  - Non-symmetric: `ilu`, `gamg`
  - Large parallel: `hypre`, `gamg`, `bjacobi`

---

## MPI / mpi4py Patterns

### CRITICAL: Uppercase = Fast, Lowercase = Slow
```python
from mpi4py import MPI
comm = MPI.COMM_WORLD

# GOOD: uppercase — buffer protocol, near-C speed
comm.Send(np_array, dest=1, tag=0)
comm.Recv(np_array, source=0, tag=0)
total = comm.allreduce(local_sum, op=MPI.SUM)

# BAD: lowercase — Python pickle, orders of magnitude slower
comm.send(np_array, dest=1)  # DON'T DO THIS FOR ARRAYS
```

### Halo Exchange (Non-blocking)
```python
left  = (rank - 1) % size
right = (rank + 1) % size
req = []
req.append(comm.Isend(u[1:2],    dest=left,  tag=0))
req.append(comm.Isend(u[-2:-1],  dest=right, tag=1))
req.append(comm.Irecv(u[0:1],    source=left,  tag=1))
req.append(comm.Irecv(u[-1:],    source=right, tag=0))
MPI.Request.Waitall(req)
```

### MPI Rules
- **Uppercase methods** for numpy arrays — always
- `np.ascontiguousarray()` before any Send
- Batch halo exchanges — one per timestep, not per element
- Non-blocking `Isend`/`Irecv` + `Waitall` for computation overlap
- Never create communicators inside time loops
- MPI reductions are NOT floating-point deterministic — document this

---

## Numerical Methods Reference

### CFL Conditions
| Method | CFL Constraint |
|--------|---------------|
| FD/FV explicit (1D) | `dt ≤ CFL * dx / a_max` (CFL ≤ 1) |
| DG order p | `dt ≤ dx / ((2p+1) * a_max)` |
| Diffusion explicit | `dt ≤ dx² / (2 * ν)` |
| Multi-D FV | `dt ≤ CFL * dx / (d * a_max)` for d dimensions |

### Numerical Fluxes
- **Rusanov/LxF:** `F̂ = ½(F(u⁻)+F(u⁺)) - ½λ(u⁺-u⁻)` — robust, diffusive
- **Roe:** linearized Jacobian, Roe-averaged states — less diffusive, can fail near vacuum
- **HLL:** two wave speeds `S_L`, `S_R` — good general purpose
- **HLLC:** restores contact wave — best for Euler equations
- **Interior Penalty (elliptic):** `σ/h·[u]` penalty term

### Flux Consistency Requirement
`F̂(u, u) = F(u)` — every flux implementation MUST satisfy this. Test it:
```python
np.testing.assert_allclose(flux(u, u), f(u), rtol=1e-14)
```

### Time Integration
| Scheme | Type | Order | Use Case |
|--------|------|-------|----------|
| SSP-RK3 (Shu-Osher) | Explicit | 3 | DG + hyperbolic (preferred) |
| SSP-RK2 | Explicit | 2 | TVD, simpler |
| RK4 (classical) | Explicit | 4 | General, but NOT TVD |
| BDF2 | Implicit | 2 | Parabolic/stiff |
| DIRK | Implicit | varies | L-stable variants |
| IMEX | Mixed | varies | Stiff diffusion + non-stiff convection |

**SSP-RK3:**
```python
u1 = u + dt * L(u)
u2 = 0.75*u + 0.25*(u1 + dt*L(u1))
u_new = (1/3)*u + (2/3)*(u2 + dt*L(u2))
```

**Rule:** Classical RK4 is NOT TVD — use SSP-RK3 for DG + hyperbolic problems to avoid spurious oscillations.

### DG-Specific
- Basis functions: Legendre (modal, orthogonal) or Lagrange at Gauss-Lobatto (nodal)
- Mass matrix is block-diagonal (local inversion per element)
- Quadrature: need ≥ 2p+1 points for order-p nonlinear terms (avoid aliasing)
- Integration by parts: **check signs carefully** — most common DG implementation bug
- Surface integrals use face Jacobian (edge length in 2D, face area in 3D), not volume Jacobian

### FVM-Specific
- MUSCL reconstruction with slope limiters (minmod, van Leer, MC) for 2nd order
- WENO for higher order (5th-order standard)
- Always work with cell averages, reconstruct to faces
- Non-conservative formulation fails at shocks — always use conservation form

### FEM-Specific
- Assembly: element loop → local stiffness → scatter to global (COO → CSR)
- Reference element mapping via Jacobian
- Continuous Galerkin: essential BCs enforced strongly
- DG: BCs enforced weakly via numerical flux

---

## Verification & Testing

### Method of Manufactured Solutions (MMS) — Primary Verification Tool
1. Choose smooth exact solution `u_exact(x,t)` (use SymPy)
2. Substitute into PDE to get source term `f = L(u_exact)`
3. Solve PDE with source `f`, compare against `u_exact`
4. Refine mesh by factor 2, expect `||e|| ~ O(h^p)`

```python
def test_order_p_convergence(solver, mms_solution, expected_order):
    h_values = [0.1 / (2**k) for k in range(5)]
    errors = [run_with_mms(solver, h, mms_solution) for h in h_values]
    rate = np.polyfit(np.log(h_values), np.log(errors), 1)[0]
    assert abs(rate - expected_order) < 0.15, \
        f"Expected O(h^{expected_order}), got O(h^{rate:.2f})"
```

### Every New Numerical Method Must Have
1. **Consistency test:** `F̂(u,u) = F(u)` (fluxes), quadrature exactness (quadrature), etc.
2. **MMS convergence test:** showing expected order on ≥4 mesh levels
3. **Conservation test:** mass/energy conserved to `rtol=1e-8`
4. **Regression test:** locked reference solution

### Tolerance Guidelines
| Context | Tolerance |
|---------|-----------|
| Machine epsilon comparison | `rtol=1e-13, atol=1e-14` |
| Single precision | `rtol=1e-6` |
| Convergence rates | `atol=0.15` on estimated rate |
| Physical conservation | `rtol=1e-8` |
| **Never** use exact float equality | — |

---

## SciPy Sparse Matrices

- Build in COO format (easy construction), convert to CSR for arithmetic
- **CSR:** fast row slicing, fast `A @ x`, use for iterative solvers
- **CSC:** fast column slicing, preferred by some direct solvers (UMFPACK)
- Always `tocsr()` before matrix-vector products

---

## Documentation Standards

Every numerical function MUST document:
1. **Mathematical formula** (LaTeX in docstring)
2. **Array shapes and dtypes**
3. **Assumptions** (contiguity, units)
4. **Reference** (paper + equation number)

```python
def rusanov_flux(u_L, u_R, max_wavespeed):
    """
    Rusanov (local Lax-Friedrichs) numerical flux.

    F̂(u⁻, u⁺) = ½[F(u⁻) + F(u⁺)] - ½λ(u⁺ - u⁻)

    Parameters
    ----------
    u_L : ndarray, shape (n_faces,), float64, C-contiguous
    u_R : ndarray, shape (n_faces,), float64, C-contiguous
    max_wavespeed : float

    Returns
    -------
    flux : ndarray, shape (n_faces,), float64

    References
    ----------
    Toro (2009), Riemann Solvers, Ch. 10, Eq. (10.55)
    """
```

## Reproducibility
- Random seeds: `np.random.default_rng(42)` (modern API)
- Fix processor count for regression tests
- MPI reductions are NOT deterministic — document this
- Pin exact versions in `pyproject.toml`

## Profiling Strategy
1. `cProfile` → find hot functions
2. `line_profiler` (`kernprof -l`) → find hot lines
3. `memory_profiler` → find allocations
4. PETSc `-log_view` → solver breakdown + flop counts
5. `tracemalloc` → track array allocations in time loop

## Common Performance Pitfalls
- Python loops over elements/faces/DOFs (use vectorization or Numba)
- Array allocation inside time loops (pre-allocate and reuse)
- PETSc `setValue` without preallocation (specify nnz)
- mpi4py lowercase methods for numpy arrays (use uppercase)
- `np.concatenate`/`np.vstack` inside loops (use pre-allocated buffer)
- Creating MPI communicators inside loops
- Non-contiguous arrays passed to MPI/Numba (check and fix)
- Missing `cache=True` on Numba functions (adds 30-60s per run)

## DO NOT Change Autonomously
- CFL constant values (breaks stability)
- Quadrature point counts (breaks accuracy)
- Flux function signs (easy to flip, catastrophic to debug)
- MPI communicator structure
- PETSc solver defaults without benchmarking

## Key References
- Hesthaven & Warburton (2008): Nodal DG Methods
- Toro (2009): Riemann Solvers and Numerical Methods for Fluid Dynamics
- LeVeque (2002): Finite Volume Methods for Hyperbolic Problems
- Saad (2003): Iterative Methods for Sparse Linear Systems
- Balay et al.: PETSc Users Manual
- Karniadakis & Sherwin (2005): Spectral/hp Element Methods
