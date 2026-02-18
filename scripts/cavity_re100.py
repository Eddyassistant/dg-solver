"""
Lid-driven cavity flow at Re=100 using DG with artificial compressibility.

Solves the incompressible Navier-Stokes equations:
    ∂u/∂t + (u·∇)u = -∇p + (1/Re)∇²u
    ∇·u = 0

via the artificial compressibility perturbation (Chorin 1967):
    (1/β²)∂p/∂t + ∇·u = 0

Time-marched to steady state with SSP-RK3.
Viscous terms via simplified interior penalty (BR1-like).
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.dg.solver2d_system import DG2DSystem
from src.dg.mesh.triangle_mesh import TriangleMesh, build_face_connectivity
from src.dg.ins_ac import (
    ac_flux_x, ac_flux_y, ac_numerical_flux,
    ac_max_wavespeed, cavity_bc, make_viscous_rhs,
)
from src.dg.reference_triangle import RefTriangle


def build_cavity_bc_tags(mesh, solver, tol=1e-8):
    """
    Tag boundary faces: 0=interior, 1=wall, 2=lid.
    """
    K = solver.K
    Nfp = solver.Nfp
    Fmask = solver.ref.Fmask
    bc_tags = np.zeros((K, 3), dtype=np.int32)

    for k in range(K):
        for f in range(3):
            if solver.EToE[k, f] == k and solver.EToF[k, f] == f:
                # This is a boundary face
                fids = Fmask[f]
                face_y = np.mean(solver.y[k, fids])
                if face_y > 1.0 - tol:
                    bc_tags[k, f] = 2  # lid
                else:
                    bc_tags[k, f] = 1  # wall

    return bc_tags


def run_cavity(nx=16, p=3, Re=100.0, beta=10.0, cfl=0.05,
               t_final=50.0, print_every=500):
    """Run the lid-driven cavity simulation."""

    print(f"Lid-Driven Cavity: Re={Re}, β={beta}, nx={nx}, P{p}")
    print(f"  Mesh: {2*nx*nx} triangles, DOFs = {2*nx*nx * (p+1)*(p+2)//2 * 3}")

    mesh = TriangleMesh.rectangle((0.0, 1.0), (0.0, 1.0), nx, nx)

    # Create solver (no periodic BCs)
    solver = DG2DSystem(
        mesh, p, n_vars=3,
        flux_x=lambda q: ac_flux_x(q, beta),
        flux_y=lambda q: ac_flux_y(q, beta),
        numerical_flux=lambda qi, qe, nx_a, ny_a: ac_numerical_flux(qi, qe, nx_a, ny_a, beta),
        max_wavespeed=lambda q: ac_max_wavespeed(q, beta),
        periodic=False,
    )

    # Build BC tags
    bc_tags = build_cavity_bc_tags(mesh, solver)
    solver.bc_tags = bc_tags

    # BC function
    def bc_func(q_int, q_ext, bc_per_node, face_nx, face_ny):
        return cavity_bc(q_int, q_ext, bc_per_node, face_nx, face_ny,
                         lid_velocity=1.0)
    solver.bc_func = bc_func

    # Viscous operator
    viscous_func = make_viscous_rhs(solver, Re)
    solver.viscous_rhs = viscous_func

    # Initial condition: quiescent
    def q0(x, y):
        q = np.zeros((*x.shape, 3))
        return q

    print("  Solving...", flush=True)
    t0_wall = time.perf_counter()

    residuals = []

    def callback(q, t, step, dt):
        if step % print_every == 0:
            # Compute residual (max |dq/dt|)
            rhs = solver.compute_rhs(q, t)
            res = np.max(np.abs(rhs))
            residuals.append((t, res))
            elapsed = time.perf_counter() - t0_wall
            div = compute_divergence(solver, q)
            print(f"  step={step:6d}  t={t:8.3f}  res={res:.4e}  "
                  f"max|div|={div:.4e}  wall={elapsed:.1f}s", flush=True)

    q, t, n_steps = solver.solve(q0, t_final, cfl=cfl, callback=callback,
                                  max_steps=100000)

    elapsed = time.perf_counter() - t0_wall
    print(f"\n  Done: {n_steps} steps in {elapsed:.1f}s")
    print(f"  Final residual: {residuals[-1][1]:.4e}" if residuals else "")

    return solver, q, residuals


def compute_divergence(solver, q):
    """Compute max |∇·u| as a measure of incompressibility."""
    u = q[:, :, 0]
    v = q[:, :, 1]
    Dr, Ds = solver.Dr, solver.Ds
    rx, sx = solver.rx, solver.sx
    ry, sy = solver.ry, solver.sy

    dudr = u @ Dr.T
    duds = u @ Ds.T
    dudx = rx[:, 0:1] * dudr + sx[:, 0:1] * duds

    dvdr = v @ Dr.T
    dvds = v @ Ds.T
    dvdy = ry[:, 0:1] * dvdr + sy[:, 0:1] * dvds

    div = dudx + dvdy
    return float(np.max(np.abs(div)))


def plot_results(solver, q, residuals, outdir='.'):
    """Generate diagnostic plots."""
    u = q[:, :, 0]
    v = q[:, :, 1]
    p = q[:, :, 2]

    x = solver.x.ravel()
    y = solver.y.ravel()
    u_flat = u.ravel()
    v_flat = v.ravel()
    p_flat = p.ravel()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Velocity magnitude
    ax = axes[0, 0]
    speed = np.sqrt(u_flat**2 + v_flat**2)
    sc = ax.tricontourf(x, y, speed, levels=20, cmap='viridis')
    plt.colorbar(sc, ax=ax)
    ax.set_title('Velocity magnitude |V|')
    ax.set_aspect('equal')

    # Pressure
    ax = axes[0, 1]
    sc = ax.tricontourf(x, y, p_flat, levels=20, cmap='RdBu_r')
    plt.colorbar(sc, ax=ax)
    ax.set_title('Pressure p')
    ax.set_aspect('equal')

    # Streamlines (interpolated to grid)
    ax = axes[1, 0]
    ngrid = 50
    xi = np.linspace(0, 1, ngrid)
    yi = np.linspace(0, 1, ngrid)
    Xi, Yi = np.meshgrid(xi, yi)
    from scipy.interpolate import griddata
    Ui = griddata((x, y), u_flat, (Xi, Yi), method='cubic', fill_value=0)
    Vi = griddata((x, y), v_flat, (Xi, Yi), method='cubic', fill_value=0)
    ax.streamplot(xi, yi, Ui, Vi, density=2, linewidth=0.8, color='k')
    speed_grid = np.sqrt(Ui**2 + Vi**2)
    ax.contourf(Xi, Yi, speed_grid, levels=20, cmap='viridis', alpha=0.5)
    ax.set_title('Streamlines')
    ax.set_aspect('equal')

    # Residual history
    ax = axes[1, 1]
    if residuals:
        ts, res = zip(*residuals)
        ax.semilogy(ts, res, 'b-', linewidth=1.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Max |residual|')
        ax.set_title('Convergence history')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Lid-Driven Cavity Re=100 (DG P3)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    outpath = os.path.join(outdir, 'cavity_re100.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {outpath}")
    return outpath


def centerline_comparison(solver, q, outdir='.'):
    """Compare centerline profiles with Ghia et al. (1982) Re=100 data."""
    # Ghia et al. (1982) reference data for Re=100
    # u-velocity along vertical centerline (x=0.5)
    ghia_y = np.array([0.0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719,
                        0.2813, 0.4531, 0.5, 0.6172, 0.7344, 0.8516,
                        0.9531, 0.9609, 0.9688, 0.9766, 1.0])
    ghia_u = np.array([0.0, -0.03717, -0.04192, -0.04775, -0.06434,
                        -0.10150, -0.15662, -0.21090, -0.20581, -0.13641,
                        0.00332, 0.23151, 0.68717, 0.73722, 0.78871,
                        0.84123, 1.0])

    # v-velocity along horizontal centerline (y=0.5)
    ghia_x = np.array([0.0, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563,
                        0.2266, 0.2344, 0.5, 0.8047, 0.8594, 0.9063,
                        0.9453, 0.9531, 0.9609, 0.9688, 1.0])
    ghia_v = np.array([0.0, 0.09233, 0.10091, 0.10890, 0.12317,
                        0.16077, 0.17507, 0.17527, 0.05454, -0.24533,
                        -0.22445, -0.16914, -0.10313, -0.08864,
                        -0.07391, -0.05906, 0.0])

    # Extract DG solution along centerlines
    x_all = solver.x.ravel()
    y_all = solver.y.ravel()
    u_all = q[:, :, 0].ravel()
    v_all = q[:, :, 1].ravel()

    from scipy.interpolate import griddata

    # Vertical centerline x=0.5
    y_line = np.linspace(0, 1, 100)
    x_line = np.full_like(y_line, 0.5)
    u_line = griddata((x_all, y_all), u_all, (x_line, y_line),
                       method='cubic', fill_value=0)

    # Horizontal centerline y=0.5
    x_line2 = np.linspace(0, 1, 100)
    y_line2 = np.full_like(x_line2, 0.5)
    v_line = griddata((x_all, y_all), v_all, (x_line2, y_line2),
                       method='cubic', fill_value=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(u_line, y_line, 'b-', linewidth=2, label='DG')
    ax1.plot(ghia_u, ghia_y, 'ro', markersize=6, label='Ghia et al.')
    ax1.set_xlabel('u', fontsize=13)
    ax1.set_ylabel('y', fontsize=13)
    ax1.set_title('u-velocity at x=0.5', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2.plot(x_line2, v_line, 'b-', linewidth=2, label='DG')
    ax2.plot(ghia_x, ghia_v, 'ro', markersize=6, label='Ghia et al.')
    ax2.set_xlabel('x', fontsize=13)
    ax2.set_ylabel('v', fontsize=13)
    ax2.set_title('v-velocity at y=0.5', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Lid-Driven Cavity Re=100 — Comparison with Ghia et al. (1982)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    outpath = os.path.join(outdir, 'cavity_ghia_comparison.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"Ghia comparison saved to {outpath}")
    return outpath


if __name__ == '__main__':
    solver, q, residuals = run_cavity(
        nx=12, p=3, Re=100.0, beta=10.0,
        cfl=0.01, t_final=20.0, print_every=200,
    )
    plot_results(solver, q, residuals, outdir=os.path.dirname(__file__) + '/..')
    centerline_comparison(solver, q, outdir=os.path.dirname(__file__) + '/..')
