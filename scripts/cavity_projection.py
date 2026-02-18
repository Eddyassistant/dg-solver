#!/usr/bin/env python3
"""
Driver script for lid-driven cavity flow using projection method DG solver.

Compares results with Ghia et al. (1982) reference data.
"""

import numpy as np
import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dg.ins_projection import run_cavity, ProjectionDGSolver, compute_divergence
from dg.mesh.cavity import cavity_mesh


# Ghia et al. (1982) reference data for Re=100
GHIA_RE100_U = {
    'y': np.array([1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172,
                   0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0000]),
    'u': np.array([1.00000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, -0.13641,
                   -0.20581, -0.21090, -0.15662, -0.10150, -0.06434, -0.04775, -0.04192, 0.00000])
}

GHIA_RE100_V = {
    'x': np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047,
                   0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000]),
    'v': np.array([0.00000, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445, -0.24533,
                   0.05454, 0.17527, 0.17507, 0.16077, 0.12317, 0.10890, 0.10091, 0.09233, 0.00000])
}


def interpolate_to_line(solver, u, v, p, x_val=None, y_val=None):
    """
    Interpolate solution to a vertical (x=const) or horizontal (y=const) line.
    
    Parameters
    ----------
    solver : ProjectionDGSolver
    u, v, p : solution arrays (K, Np)
    x_val : float, optional
        If given, interpolate along vertical line x=x_val
    y_val : float, optional
        If given, interpolate along horizontal line y=y_val
    
    Returns
    -------
    coords, u_interp, v_interp : arrays
        Coordinates and interpolated values
    """
    x = solver.x.flatten()
    y = solver.y.flatten()
    u_flat = u.flatten()
    v_flat = v.flatten()
    
    if x_val is not None:
        # Vertical line: find points near x=x_val
        mask = np.abs(x - x_val) < 0.1
        coords = y[mask]
        u_vals = u_flat[mask]
        v_vals = v_flat[mask]
        # Sort by y
        idx = np.argsort(coords)
        return coords[idx], u_vals[idx], v_vals[idx]
    
    if y_val is not None:
        # Horizontal line: find points near y=y_val
        mask = np.abs(y - y_val) < 0.1
        coords = x[mask]
        u_vals = u_flat[mask]
        v_vals = v_flat[mask]
        # Sort by x
        idx = np.argsort(coords)
        return coords[idx], u_vals[idx], v_vals[idx]
    
    raise ValueError("Must specify x_val or y_val")


def sample_center_line(solver, u, v, n_sample=100):
    """
    Sample u-velocity on vertical centerline (x=0.5) 
    and v-velocity on horizontal centerline (y=0.5).
    
    Returns sampled data matching Ghia et al. format.
    """
    # Sample u at x=0.5 (vertical line through center)
    y_vals, u_vals, _ = interpolate_to_line(solver, u, v, None, x_val=0.5)
    
    # Sample v at y=0.5 (horizontal line through center)
    x_vals, _, v_vals = interpolate_to_line(solver, u, v, None, y_val=0.5)
    
    return y_vals, u_vals, x_vals, v_vals


def compute_ghia_error(solver, u, v):
    """Compute L2 error against Ghia reference data."""
    # Sample our solution
    y_vals, u_vals, x_vals, v_vals = sample_center_line(solver, u, v)
    
    # Interpolate Ghia data to our sample points
    u_ghia_interp = np.interp(y_vals, GHIA_RE100_U['y'], GHIA_RE100_U['u'])
    v_ghia_interp = np.interp(x_vals, GHIA_RE100_V['x'], GHIA_RE100_V['v'])
    
    # Compute errors
    u_error = np.sqrt(np.mean((u_vals - u_ghia_interp)**2))
    v_error = np.sqrt(np.mean((v_vals - v_ghia_interp)**2))
    
    return u_error, v_error


def main():
    parser = argparse.ArgumentParser(
        description='Lid-driven cavity flow with projection method DG')
    
    parser.add_argument('--nx', type=int, default=16,
                        help='Number of elements in each direction (default: 16)')
    parser.add_argument('-p', '--degree', type=int, default=2,
                        help='Polynomial degree (default: 2)')
    parser.add_argument('--Re', type=float, default=100.0,
                        help='Reynolds number (default: 100)')
    parser.add_argument('--t-final', type=float, default=10.0,
                        help='Final time (default: 10)')
    parser.add_argument('--cfl', type=float, default=0.5,
                        help='CFL number (default: 0.5)')
    parser.add_argument('--lid-velocity', type=float, default=1.0,
                        help='Lid velocity (default: 1.0)')
    parser.add_argument('--compare-ghia', action='store_true',
                        help='Compare with Ghia et al. reference data')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for solution (npz format)')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    # Run simulation
    solver, u, v, p = run_cavity(
        nx=args.nx,
        p=args.degree,
        Re=args.Re,
        t_final=args.t_final,
        cfl=args.cfl,
        lid_velocity=args.lid_velocity
    )
    
    # Compute diagnostics
    div = compute_divergence(u, v, solver.Dr, solver.Ds,
                            solver.rx, solver.sx, solver.ry, solver.sy,
                            solver.K, solver.Np)
    
    max_div = np.max(np.abs(div))
    l2_div = np.sqrt(np.sum(div**2 * np.abs(solver.J[:, 0:1]))) / solver.K
    
    print(f"\nDiagnostics:")
    print(f"  max|div(u)| = {max_div:.6e}")
    print(f"  L2|div(u)|  = {l2_div:.6e}")
    
    # Compare with Ghia if requested and Re=100
    if args.compare_ghia and abs(args.Re - 100) < 1:
        u_err, v_err = compute_ghia_error(solver, u, v)
        print(f"\nGhia et al. (1982) comparison (Re=100):")
        print(f"  u-velocity error (centerline): {u_err:.6f}")
        print(f"  v-velocity error (centerline): {v_err:.6f}")
    
    # Save output if requested
    if args.output:
        np.savez(args.output,
                 x=solver.x, y=solver.y,
                 u=u, v=v, p=p,
                 Re=args.Re, degree=args.degree, nx=args.nx)
        print(f"\nSolution saved to {args.output}")
    
    # Print summary
    print(f"\nSimulation complete.")
    print(f"  Mesh: {args.nx}×{args.nx} quads → {solver.K} triangles")
    print(f"  Polynomial degree: {args.degree}")
    print(f"  Total DOFs: {solver.K * solver.Np}")


if __name__ == '__main__':
    main()
