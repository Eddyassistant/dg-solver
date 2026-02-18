"""
Lid-driven cavity flow at Re=100 using mixed DG formulation.

Solves the incompressible Navier-Stokes equations:
    ∂u/∂t + (u·∇)u = -∇p + (1/Re)∇²u
    ∇·u = 0

via pressure-Poisson projection method:
    1. Predictor: u* = u^n + dt * [-conv(u^n) + ν∇²u^n]
    2. Pressure Poisson: ∇²p^{n+1} = (1/dt) ∇·u*
    3. Velocity corrector: u^{n+1} = u* - dt * ∇p^{n+1}

Key features:
- Equal-order P_p spaces for velocity and pressure
- Pressure stabilization via interior jump penalty
- Sharp lid BC (constant velocity, no regularization)
- SIPG for viscous terms

References
----------
Karniadakis & Sherwin (2005), Spectral/hp Element Methods.
Cockburn, Kanschat & Schötzau (2005), Math. Comp. 74(249).
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.dg.mesh.triangle_mesh import TriangleMesh
from src.dg.ins_mixed import MixedDGCavitySolver


def compute_divergence_field(solver, u, v):
    """Compute max |∇·u| as a measure of incompressibility."""
    div = solver._compute_divergence(u, v)
    return float(np.max(np.abs(div)))


def run_cavity(nx=16, p=2, Re=100.0, gamma_p=0.1, cfl=0.1,
               t_final=30.0, print_every=500):
    """Run the lid-driven cavity simulation with mixed DG."""
    
    K = 2 * nx * nx
    Np = (p + 1) * (p + 2) // 2
    print(f"Mixed DG Lid-Driven Cavity: Re={Re}, nx={nx}, P{p}")
    print(f"  Mesh: {K} triangles, {Np} nodes/elem, {K*Np*3} total DOFs")
    print(f"  Pressure stabilization: γ_p={gamma_p}")
    
    mesh = TriangleMesh.rectangle((0.0, 1.0), (0.0, 1.0), nx, nx)
    
    solver = MixedDGCavitySolver(mesh, p=p, Re=Re, gamma_p=gamma_p)
    
    print("  Solving...", flush=True)
    
    residuals = []
    div_history = []
    
    def callback(q, t, step, dt):
        if step % print_every == 0:
            u = q[:, :, 0]
            v = q[:, :, 1]
            div = compute_divergence_field(solver, u, v)
            div_history.append((t, div))
    
    u, v, p, t, n_steps = solver.solve(
        t_final=t_final, cfl=cfl, print_every=print_every, callback=callback
    )
    
    return solver, u, v, p, t, n_steps, div_history


def plot_results(solver, u, v, p, div_history=None, outdir='.'):
    """Generate diagnostic plots."""
    x = solver.solver.x.ravel()
    y = solver.solver.y.ravel()
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
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Pressure
    ax = axes[0, 1]
    # Normalize pressure to zero mean
    p_mean = np.mean(p_flat)
    sc = ax.tricontourf(x, y, p_flat - p_mean, levels=20, cmap='RdBu_r')
    plt.colorbar(sc, ax=ax)
    ax.set_title('Pressure p (zero-mean)')
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Streamlines
    ax = axes[1, 0]
    ngrid = 60
    xi = np.linspace(0, 1, ngrid)
    yi = np.linspace(0, 1, ngrid)
    Xi, Yi = np.meshgrid(xi, yi)
    from scipy.interpolate import griddata
    Ui = griddata((x, y), u_flat, (Xi, Yi), method='cubic', fill_value=0)
    Vi = griddata((x, y), v_flat, (Xi, Yi), method='cubic', fill_value=0)
    speed_grid = np.sqrt(Ui**2 + Vi**2)
    ax.contourf(Xi, Yi, speed_grid, levels=20, cmap='viridis', alpha=0.5)
    ax.streamplot(xi, yi, Ui, Vi, density=2, linewidth=0.8, color='k')
    ax.set_title('Streamlines')
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Divergence history
    ax = axes[1, 1]
    if div_history:
        ts, divs = zip(*div_history)
        ax.semilogy(ts, divs, 'b-', linewidth=1.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Max |∇·u|')
        ax.set_title('Divergence evolution')
        ax.grid(True, alpha=0.3)
    else:
        # Vorticity instead
        ax.text(0.5, 0.5, 'No divergence history', ha='center', va='center',
                transform=ax.transAxes)
    
    plt.suptitle(f'Mixed DG Cavity Re=100 (P{solver.p}, γ_p={solver.gamma_p})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    outpath = os.path.join(outdir, 'cavity_mixed.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"  Plot saved to {outpath}")
    plt.close()
    
    return outpath


def centerline_comparison(solver, u, v, outdir='.'):
    """Compare centerline profiles with Ghia et al. (1982) Re=100 data."""
    # Ghia et al. (1982) reference data for Re=100
    ghia_y = np.array([0.0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719,
                        0.2813, 0.4531, 0.5, 0.6172, 0.7344, 0.8516,
                        0.9531, 0.9609, 0.9688, 0.9766, 1.0])
    ghia_u = np.array([0.0, -0.03717, -0.04192, -0.04775, -0.06434,
                        -0.10150, -0.15662, -0.21090, -0.20581, -0.13641,
                        0.00332, 0.23151, 0.68717, 0.73722, 0.78871,
                        0.84123, 1.0])

    ghia_x = np.array([0.0, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563,
                        0.2266, 0.2344, 0.5, 0.8047, 0.8594, 0.9063,
                        0.9453, 0.9531, 0.9609, 0.9688, 1.0])
    ghia_v = np.array([0.0, 0.09233, 0.10091, 0.10890, 0.12317,
                        0.16077, 0.17507, 0.17527, 0.05454, -0.24533,
                        -0.22445, -0.16914, -0.10313, -0.08864,
                        -0.07391, -0.05906, 0.0])

    x_all = solver.solver.x.ravel()
    y_all = solver.solver.y.ravel()
    u_all = u.ravel()
    v_all = v.ravel()

    from scipy.interpolate import griddata

    # Vertical centerline x=0.5
    y_line = np.linspace(0, 1, 200)
    x_line = np.full_like(y_line, 0.5)
    u_line = griddata((x_all, y_all), u_all, (x_line, y_line),
                       method='cubic', fill_value=0)

    # Horizontal centerline y=0.5
    x_line2 = np.linspace(0, 1, 200)
    y_line2 = np.full_like(x_line2, 0.5)
    v_line = griddata((x_all, y_all), v_all, (x_line2, y_line2),
                       method='cubic', fill_value=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(u_line, y_line, 'b-', linewidth=2, label='Mixed DG (this work)')
    ax1.plot(ghia_u, ghia_y, 'ro', markersize=6, label='Ghia et al. (1982)')
    ax1.set_xlabel('u', fontsize=13)
    ax1.set_ylabel('y', fontsize=13)
    ax1.set_title('u-velocity at x = 0.5', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2.plot(x_line2, v_line, 'b-', linewidth=2, label='Mixed DG (this work)')
    ax2.plot(ghia_x, ghia_v, 'ro', markersize=6, label='Ghia et al. (1982)')
    ax2.set_xlabel('x', fontsize=13)
    ax2.set_ylabel('v', fontsize=13)
    ax2.set_title('v-velocity at y = 0.5', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Mixed DG Cavity Re=100 P{solver.p} — Comparison with Ghia et al. (1982)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    outpath = os.path.join(outdir, 'cavity_mixed_ghia.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"  Ghia comparison saved to {outpath}")
    plt.close()
    
    return outpath


def vorticity_plot(solver, u, v, outdir='.'):
    """Plot vorticity field."""
    x = solver.solver.x
    y = solver.solver.y
    Dr, Ds = solver.solver.Dr, solver.solver.Ds
    rx, ry = solver.solver.rx, solver.solver.ry
    sx, sy = solver.solver.sx, solver.solver.sy
    
    K, Np = solver.solver.K, solver.solver.Np
    
    # Compute vorticity: ω = ∂v/∂x - ∂u/∂y
    omega = np.zeros((K, Np))
    
    for k in range(K):
        rx_k = rx[k, 0]
        ry_k = ry[k, 0]
        sx_k = sx[k, 0]
        sy_k = sy[k, 0]
        
        dudr = u[k] @ Dr.T
        duds = u[k] @ Ds.T
        dvdr = v[k] @ Dr.T
        dvds = v[k] @ Ds.T
        
        dudx = rx_k * dudr + sx_k * duds
        dudy = ry_k * dudr + sy_k * duds
        dvdx = rx_k * dvdr + sx_k * dvds
        dvdy = ry_k * dvdr + sy_k * dvds
        
        omega[k] = dvdx - dudy
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    x_flat = x.ravel()
    y_flat = y.ravel()
    omega_flat = omega.ravel()
    
    sc = ax.tricontourf(x_flat, y_flat, omega_flat, levels=20, cmap='RdBu_r')
    plt.colorbar(sc, ax=ax)
    ax.set_title('Vorticity ω = ∂v/∂x - ∂u/∂y')
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    outpath = os.path.join(outdir, 'cavity_mixed_vorticity.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"  Vorticity plot saved to {outpath}")
    plt.close()
    
    return outpath


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Lid-driven cavity with mixed DG formulation'
    )
    parser.add_argument('--nx', type=int, default=12,
                        help='Number of elements in each direction')
    parser.add_argument('--p', type=int, default=2,
                        help='Polynomial degree')
    parser.add_argument('--Re', type=float, default=100.0,
                        help='Reynolds number')
    parser.add_argument('--gamma-p', type=float, default=0.1,
                        help='Pressure stabilization parameter')
    parser.add_argument('--cfl', type=float, default=0.1,
                        help='CFL number')
    parser.add_argument('--t-final', type=float, default=30.0,
                        help='Final time')
    parser.add_argument('--print-every', type=int, default=500,
                        help='Print frequency')
    args = parser.parse_args()

    outdir = os.path.join(os.path.dirname(__file__), '..')

    solver, u, v, p, t, n_steps, div_history = run_cavity(
        nx=args.nx, p=args.p, Re=args.Re, gamma_p=args.gamma_p,
        cfl=args.cfl, t_final=args.t_final, print_every=args.print_every,
    )
    
    plot_results(solver, u, v, p, div_history, outdir=outdir)
    centerline_comparison(solver, u, v, outdir=outdir)
    vorticity_plot(solver, u, v, outdir=outdir)
    
    # Compute final divergence
    final_div = compute_divergence_field(solver, u, v)
    print(f"\n  Final max|div|: {final_div:.4e}")
    
    # Compute velocity extrema
    u_max = np.max(u)
    v_min = np.min(v)
    print(f"  u_max (center): {u_max:.4f}")
    print(f"  v_min (center): {v_min:.4f}")
