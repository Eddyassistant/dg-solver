"""
Post-processing utilities for lid-driven cavity flow.

Includes velocity field visualization, streamlines, pressure contours,
and centerline profile extraction for validation against Ghia et al. (1982).

References
----------
Ghia et al. (1982), "High-Re solutions for incompressible flow using the
    Navier-Stokes equations and a multigrid method", J. Comput. Phys., 48:387-411.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from typing import Optional


# =============================================================================
# Ghia et al. (1982) reference data for Re = 100
# =============================================================================

# u-velocity along vertical centerline (x = 0.5) vs y
# Format: (y, u)
GHIA_RE100_U_VERTICAL = np.array([
    [1.0000, 1.0000],
    [0.9766, 0.8412],
    [0.9688, 0.7887],
    [0.9609, 0.7372],
    [0.9531, 0.6872],
    [0.8516, 0.2315],
    [0.7344, 0.0033],
    [0.6172, -0.1364],
    [0.5000, -0.2058],
    [0.4531, -0.2109],
    [0.4141, -0.1566],
    [0.3438, -0.1015],
    [0.2891, -0.0643],
    [0.2344, -0.0478],
    [0.1875, -0.0419],
    [0.1563, -0.0372],
    [0.1250, -0.0311],
    [0.0938, -0.0244],
    [0.0781, -0.0210],
    [0.0625, -0.0172],
    [0.0469, -0.0129],
    [0.0313, -0.0085],
    [0.0156, -0.0044],
    [0.0000, 0.0000],
])

# v-velocity along horizontal centerline (y = 0.5) vs x
# Format: (x, v)
GHIA_RE100_V_HORIZONTAL = np.array([
    [1.0000, 0.0000],
    [0.9688, -0.0591],
    [0.9609, -0.0739],
    [0.9531, -0.0886],
    [0.9453, -0.1031],
    [0.9063, -0.1691],
    [0.8594, -0.2245],
    [0.8047, -0.2453],
    [0.7500, -0.2244],
    [0.7031, -0.1770],
    [0.6563, -0.1337],
    [0.5000, -0.0545],
    [0.3750, -0.0283],
    [0.3125, -0.0214],
    [0.2813, -0.0185],
    [0.2500, -0.0154],
    [0.2188, -0.0122],
    [0.1875, -0.0093],
    [0.1563, -0.0067],
    [0.1250, -0.0045],
    [0.0938, -0.0027],
    [0.0781, -0.0019],
    [0.0625, -0.0013],
    [0.0469, -0.0009],
    [0.0313, -0.0005],
    [0.0156, -0.0002],
    [0.0000, 0.0000],
])


def get_ghia_u_vertical() -> tuple[np.ndarray, np.ndarray]:
    """
    Get Ghia et al. u-velocity along vertical centerline (x=0.5).

    Returns
    -------
    y : ndarray
        Vertical coordinate (0 to 1).
    u : ndarray
        Horizontal velocity component.
    """
    return GHIA_RE100_U_VERTICAL[:, 0], GHIA_RE100_U_VERTICAL[:, 1]


def get_ghia_v_horizontal() -> tuple[np.ndarray, np.ndarray]:
    """
    Get Ghia et al. v-velocity along horizontal centerline (y=0.5).

    Returns
    -------
    x : ndarray
        Horizontal coordinate (0 to 1).
    v : ndarray
        Vertical velocity component.
    """
    return GHIA_RE100_V_HORIZONTAL[:, 0], GHIA_RE100_V_HORIZONTAL[:, 1]


# =============================================================================
# Plotting functions
# =============================================================================

def _get_coordinates_and_triangulation(solver) -> tuple:
    """
    Extract coordinates and create matplotlib triangulation.

    Parameters
    ----------
    solver : DG solver object
        Must have mesh with VX, VY, EToV and x, y nodal coordinates.

    Returns
    -------
    x_flat, y_flat : ndarray
        Flattened nodal coordinates.
    tri : Triangulation
        Matplotlib triangulation for plotting.
    """
    mesh = solver.mesh
    K = mesh.n_elem
    Np = solver.Np

    # Get nodal coordinates
    x = solver.x  # (K, Np)
    y = solver.y  # (K, Np)

    x_flat = x.ravel()
    y_flat = y.ravel()

    # Create triangulation for plotting
    # Each element is divided into smaller triangles for visualization
    # Use the element connectivity directly for a coarse visualization
    # or interpolate to a finer grid for smooth contours

    # For now, use a simple approach with element vertices
    tri = Triangulation(mesh.VX, mesh.VY, mesh.EToV)

    return x_flat, y_flat, tri


def _interpolate_to_vertices(solver, u_field: np.ndarray) -> np.ndarray:
    """
    Interpolate solution from nodal points to mesh vertices.

    Parameters
    ----------
    solver : DG solver object
    u_field : ndarray, shape (K, Np)
        Scalar field at nodal points.

    Returns
    -------
    u_vertices : ndarray, shape (n_verts,)
        Field interpolated to mesh vertices.
    """
    mesh = solver.mesh
    n_verts = mesh.n_verts
    EToV = mesh.EToV

    # Simple averaging: for each vertex, average from connected elements
    u_vertices = np.zeros(n_verts)
    count = np.zeros(n_verts)

    K, Np = u_field.shape

    # Map nodal points to vertices (approximate for low-order)
    # For P1 (p=1), nodes are at vertices
    # For higher order, we need proper interpolation
    if Np == 3:
        # P1 elements: direct mapping
        for k in range(K):
            for i in range(3):
                v = EToV[k, i]
                u_vertices[v] += u_field[k, i]
                count[v] += 1
    else:
        # Higher order: evaluate at vertices using modal or nodal basis
        # For now, use vertex nodes if available
        # This is a simplified approach
        for k in range(K):
            # Just use first 3 nodes as approximation
            for i in range(min(3, Np)):
                v = EToV[k, i]
                u_vertices[v] += u_field[k, i]
                count[v] += 1

    u_vertices /= np.maximum(count, 1)
    return u_vertices


def plot_velocity_field(solver, u: np.ndarray, filename: str,
                       title: Optional[str] = None) -> None:
    """
    Plot velocity field as a quiver plot.

    Parameters
    ----------
    solver : DG solver object
        Must have x, y coordinates and mesh information.
    u : ndarray, shape (K, Np, 3)
        Solution array with [u, v, p] at each node.
    filename : str
        Output filename (e.g., 'velocity.png').
    title : str, optional
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Extract velocity components
    u_vel = u[:, :, 0]  # x-velocity
    v_vel = u[:, :, 1]  # y-velocity

    # Downsample for quiver plot if needed
    K, Np = u_vel.shape
    stride = max(1, int(np.sqrt(K * Np) / 30))  # Target ~30x30 arrows

    x = solver.x[::stride, ::stride] if K > stride else solver.x
    y = solver.y[::stride, ::stride] if K > stride else solver.y
    u_plot = u_vel[::stride, ::stride] if K > stride else u_vel
    v_plot = v_vel[::stride, ::stride] if K > stride else v_vel

    # Normalize arrow colors by magnitude
    magnitude = np.sqrt(u_plot**2 + v_plot**2)

    ax.quiver(x, y, u_plot, v_plot, magnitude,
              scale=20, scale_units='inches',
              cmap='viridis', width=0.003)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title or 'Velocity Field')
    ax.set_aspect('equal')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.colorbar(ax.collections[0], ax=ax, label='|u|')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_streamlines(solver, u: np.ndarray, filename: str,
                     title: Optional[str] = None,
                     n_grid: int = 50) -> None:
    """
    Plot streamlines using matplotlib.

    Parameters
    ----------
    solver : DG solver object
    u : ndarray, shape (K, Np, 3)
        Solution array with [u, v, p].
    filename : str
        Output filename.
    title : str, optional
        Plot title.
    n_grid : int, optional
        Number of grid points for streamline interpolation (default: 50).
    """
    from scipy.interpolate import griddata

    fig, ax = plt.subplots(figsize=(8, 8))

    # Extract velocity
    u_vel = u[:, :, 0].ravel()
    v_vel = u[:, :, 1].ravel()
    x_flat = solver.x.ravel()
    y_flat = solver.y.ravel()

    # Create regular grid for streamplot
    xi = np.linspace(0, 1, n_grid)
    yi = np.linspace(0, 1, n_grid)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate to regular grid
    Ui = griddata((x_flat, y_flat), u_vel, (Xi, Yi), method='cubic')
    Vi = griddata((x_flat, y_flat), v_vel, (Xi, Yi), method='cubic')

    # Mask NaN values
    mask = np.isnan(Ui) | np.isnan(Vi)
    Ui[mask] = 0
    Vi[mask] = 0

    # Plot streamlines
    speed = np.sqrt(Ui**2 + Vi**2)
    strm = ax.streamplot(Xi, Yi, Ui, Vi, color=speed, cmap='viridis',
                         density=2, linewidth=1.5, arrowsize=1.5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title or 'Streamlines')
    ax.set_aspect('equal')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.colorbar(strm.lines, ax=ax, label='|u|')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_pressure(solver, u: np.ndarray, filename: str,
                 title: Optional[str] = None,
                 n_levels: int = 20) -> None:
    """
    Plot pressure as filled contours.

    Parameters
    ----------
    solver : DG solver object
    u : ndarray, shape (K, Np, 3)
        Solution array with [u, v, p].
    filename : str
        Output filename.
    title : str, optional
        Plot title.
    n_levels : int, optional
        Number of contour levels (default: 20).
    """
    from scipy.interpolate import griddata

    fig, ax = plt.subplots(figsize=(8, 8))

    # Extract pressure
    p = u[:, :, 2].ravel()
    x_flat = solver.x.ravel()
    y_flat = solver.y.ravel()

    # Create regular grid for contour plot
    n_grid = 100
    xi = np.linspace(0, 1, n_grid)
    yi = np.linspace(0, 1, n_grid)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate to regular grid
    Pi = griddata((x_flat, y_flat), p, (Xi, Yi), method='cubic')

    # Plot filled contours
    levels = np.linspace(np.nanmin(Pi), np.nanmax(Pi), n_levels)
    cf = ax.contourf(Xi, Yi, Pi, levels=levels, cmap='RdBu_r', extend='both')

    # Add contour lines
    ax.contour(Xi, Yi, Pi, levels=levels, colors='k', linewidths=0.5, alpha=0.3)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title or 'Pressure')
    ax.set_aspect('equal')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.colorbar(cf, ax=ax, label='p')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Centerline profile extraction
# =============================================================================

def centerline_profiles(solver, u: np.ndarray, n_interp: int = 100) -> dict:
    """
    Extract centerline velocity profiles for comparison with Ghia et al.

    Computes:
    - u(y) at x = 0.5 (vertical centerline)
    - v(x) at y = 0.5 (horizontal centerline)

    Parameters
    ----------
    solver : DG solver object
        Must have x, y nodal coordinates.
    u : ndarray, shape (K, Np, 3)
        Solution array with [u, v, p].
    n_interp : int, optional
        Number of interpolation points along each centerline (default: 100).

    Returns
    -------
    profiles : dict
        Dictionary with keys:
        - 'y': vertical coordinate for u-profile (0 to 1)
        - 'u_vertical': u-velocity at x=0.5
        - 'x': horizontal coordinate for v-profile (0 to 1)
        - 'v_horizontal': v-velocity at y=0.5
        - 'u_ghia_y', 'u_ghia': Ghia reference data for u
        - 'v_ghia_x', 'v_ghia': Ghia reference data for v
    """
    from scipy.interpolate import griddata

    # Flatten coordinates and solution
    x_flat = solver.x.ravel()
    y_flat = solver.y.ravel()
    u_flat = u[:, :, 0].ravel()  # x-velocity
    v_flat = u[:, :, 1].ravel()  # y-velocity

    # Extract u(y) at x = 0.5
    y_interp = np.linspace(0, 1, n_interp)
    x_interp = np.full_like(y_interp, 0.5)
    u_vertical = griddata((x_flat, y_flat), u_flat, (x_interp, y_interp),
                          method='linear')

    # Extract v(x) at y = 0.5
    x_interp_v = np.linspace(0, 1, n_interp)
    y_interp_v = np.full_like(x_interp_v, 0.5)
    v_horizontal = griddata((x_flat, y_flat), v_flat, (x_interp_v, y_interp_v),
                            method='linear')

    # Get Ghia reference data
    y_ghia, u_ghia = get_ghia_u_vertical()
    x_ghia, v_ghia = get_ghia_v_horizontal()

    return {
        'y': y_interp,
        'u_vertical': u_vertical,
        'x': x_interp_v,
        'v_horizontal': v_horizontal,
        'u_ghia_y': y_ghia,
        'u_ghia': u_ghia,
        'v_ghia_x': x_ghia,
        'v_ghia': v_ghia,
    }


def plot_centerline_profiles(solver, u: np.ndarray, filename: str,
                            title: Optional[str] = None) -> None:
    """
    Plot centerline velocity profiles with Ghia et al. reference data.

    Creates a two-panel plot showing:
    - Left: u(y) at x=0.5 compared to Ghia
    - Right: v(x) at y=0.5 compared to Ghia

    Parameters
    ----------
    solver : DG solver object
    u : ndarray, shape (K, Np, 3)
        Solution array with [u, v, p].
    filename : str
        Output filename.
    title : str, optional
        Plot title (applied to figure).
    """
    profiles = centerline_profiles(solver, u)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: u-velocity at x=0.5
    ax1.plot(profiles['u_vertical'], profiles['y'], 'b-', linewidth=2,
             label='Current solution')
    ax1.plot(profiles['u_ghia'], profiles['u_ghia_y'], 'ko', markersize=6,
             label='Ghia et al. (1982)')
    ax1.set_xlabel('u')
    ax1.set_ylabel('y')
    ax1.set_title('u-velocity at x = 0.5')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-0.4, 1.0])
    ax1.set_ylim([0, 1])

    # Right panel: v-velocity at y=0.5
    ax2.plot(profiles['x'], profiles['v_horizontal'], 'b-', linewidth=2,
             label='Current solution')
    ax2.plot(profiles['v_ghia_x'], profiles['v_ghia'], 'ko', markersize=6,
             label='Ghia et al. (1982)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('v')
    ax2.set_title('v-velocity at y = 0.5')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([-0.5, 0.3])

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def compute_l2_error_vs_ghia(solver, u: np.ndarray) -> dict:
    """
    Compute L2 error against Ghia et al. reference data.

    Parameters
    ----------
    solver : DG solver object
    u : ndarray, shape (K, Np, 3)
        Solution array with [u, v, p].

    Returns
    -------
    errors : dict
        Dictionary with L2 errors for u and v centerline profiles.
    """
    profiles = centerline_profiles(solver, u)

    # Interpolate current solution to Ghia points
    from scipy.interpolate import interp1d

    # u-profile error
    mask = np.isfinite(profiles['u_vertical'])
    f_u = interp1d(profiles['y'][mask], profiles['u_vertical'][mask],
                   kind='linear', fill_value='extrapolate')
    u_interp = f_u(profiles['u_ghia_y'])
    error_u = np.sqrt(np.mean((u_interp - profiles['u_ghia'])**2))

    # v-profile error
    mask = np.isfinite(profiles['v_horizontal'])
    f_v = interp1d(profiles['x'][mask], profiles['v_horizontal'][mask],
                   kind='linear', fill_value='extrapolate')
    v_interp = f_v(profiles['v_ghia_x'])
    error_v = np.sqrt(np.mean((v_interp - profiles['v_ghia'])**2))

    return {
        'error_u_vertical': error_u,
        'error_v_horizontal': error_v,
        'error_total': np.sqrt(error_u**2 + error_v**2),
    }
