"""
h/p convergence study for the 2D DG solver on triangular meshes.

Test problem: linear advection u_t + a·u_x + b·u_y = 0
on [0,1]² with periodic BCs.
Exact solution: u(x,y,t) = sin(2π(x - at)) · sin(2π(y - bt))
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

from src.dg.solver2d import DG2DScalar
from src.dg.mesh.triangle_mesh import TriangleMesh

# Problem parameters
A, B = 1.0, 0.5
TF = 0.2
SPEED = np.sqrt(A**2 + B**2)

def make_solver(nx, p):
    mesh = TriangleMesh.rectangle((0.0, 1.0), (0.0, 1.0), nx, nx)
    return DG2DScalar(
        mesh, p,
        flux_x=lambda u: A * u,
        flux_y=lambda u: B * u,
        max_wavespeed=lambda uL, uR: np.full_like(uL, SPEED),
        periodic=True, x_range=(0.0, 1.0), y_range=(0.0, 1.0),
    )

def u0(x, y):
    return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

def u_exact(x, y):
    return np.sin(2 * np.pi * (x - A * TF)) * np.sin(2 * np.pi * (y - B * TF))


def run(nx, p):
    t0 = time.time()
    solver = make_solver(nx, p)
    cfl = min(0.1, 0.5 / (2 * p + 1)**2)
    u, t = solver.solve(u0, TF, cfl=cfl)
    err = solver.l2_error(u, u_exact, t)
    dofs = solver.K * solver.Np
    wall = time.time() - t0
    return err, dofs, wall


# ================================================================
# h-convergence
# ================================================================
print("=" * 65)
print("h-CONVERGENCE (fixed p, refine mesh)")
print("=" * 65)

p_list = [1, 2, 3, 4, 5]
nx_list = [4, 8, 16, 32]
h_results = {}

for p in p_list:
    errs, dofs_list = [], []
    for nx in nx_list:
        try:
            err, dofs, wall = run(nx, p)
            errs.append(err)
            dofs_list.append(dofs)
            print(f"  P{p}  nx={nx:3d}  K={2*nx*nx:5d}  DOFs={dofs:6d}  "
                  f"err={err:.3e}  wall={wall:.1f}s")
        except Exception as e:
            print(f"  P{p}  nx={nx}: FAILED ({e})")
            errs.append(np.nan)
            dofs_list.append(0)
    h_results[p] = (errs, dofs_list)
    # Rates
    valid = [e for e in errs if not np.isnan(e)]
    if len(valid) >= 2:
        rates = [np.log(valid[i]/valid[i+1])/np.log(2) for i in range(len(valid)-1)]
        print(f"  → rates: {['%.2f' % r for r in rates]}  (expected ≈{p+1})")

# ================================================================
# p-convergence
# ================================================================
print("\n" + "=" * 65)
print("p-CONVERGENCE (fixed mesh, increase p)")
print("=" * 65)

nx_fixed = [4, 8, 16]
p_range = list(range(1, 7))
p_results = {}

for nx in nx_fixed:
    errs = []
    for p in p_range:
        try:
            err, dofs, wall = run(nx, p)
            errs.append(err)
            print(f"  nx={nx:2d}  P{p}  DOFs={dofs:6d}  err={err:.3e}  wall={wall:.1f}s")
        except Exception as e:
            print(f"  nx={nx}  P{p}: FAILED ({e})")
            errs.append(np.nan)
    p_results[nx] = errs

# ================================================================
# Plot
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
markers = ['o', 's', 'D', '^', 'v', 'P']

# --- Panel 1: h-convergence (error vs h) ---
ax = axes[0]
h_vals = np.array([1.0 / nx for nx in nx_list])
for idx, p in enumerate(p_list):
    errs = np.array(h_results[p][0])
    valid = ~np.isnan(errs)
    if valid.sum() >= 2:
        ax.loglog(h_vals[valid], errs[valid], f'{markers[idx]}-',
                  color=colors[idx], label=f'P{p}', lw=2, ms=8)
        # Reference slope
        e0 = errs[valid][0]
        h0 = h_vals[valid][0]
        ref_h = h_vals[valid]
        ax.loglog(ref_h, e0 * (ref_h / h0)**(p + 1), '--',
                  color=colors[idx], alpha=0.35, lw=1)

ax.set_xlabel('h (element size)', fontsize=13)
ax.set_ylabel('L² error', fontsize=13)
ax.set_title('h-Convergence', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, which='both', alpha=0.25)
ax.set_ylim(bottom=1e-12)

# --- Panel 2: h-convergence (error vs DOFs) ---
ax = axes[1]
for idx, p in enumerate(p_list):
    errs = np.array(h_results[p][0])
    dofs = np.array(h_results[p][1])
    valid = ~np.isnan(errs) & (dofs > 0)
    if valid.sum() >= 2:
        ax.loglog(dofs[valid], errs[valid], f'{markers[idx]}-',
                  color=colors[idx], label=f'P{p}', lw=2, ms=8)

ax.set_xlabel('Degrees of freedom', fontsize=13)
ax.set_ylabel('L² error', fontsize=13)
ax.set_title('Error vs DOFs', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='lower left')
ax.grid(True, which='both', alpha=0.25)
ax.set_ylim(bottom=1e-12)

# --- Panel 3: p-convergence ---
ax = axes[2]
for idx, nx in enumerate(nx_fixed):
    errs = np.array(p_results[nx])
    valid = ~np.isnan(errs) & (errs > 0)
    ps = np.array(p_range)[valid]
    es = errs[valid]
    if len(ps) >= 2:
        ax.semilogy(ps, es, f'{markers[idx]}-', color=colors[idx],
                    label=f'{2*nx*nx} triangles (nx={nx})', lw=2, ms=8)

ax.set_xlabel('Polynomial degree p', fontsize=13)
ax.set_ylabel('L² error', fontsize=13)
ax.set_title('p-Convergence', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, which='both', alpha=0.25)
ax.set_xticks(p_range)

plt.suptitle('2D DG on Triangles — Linear Advection h/p Convergence',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
outpath = os.path.join(os.path.dirname(__file__), '..', 'hp_convergence.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to {outpath}")
