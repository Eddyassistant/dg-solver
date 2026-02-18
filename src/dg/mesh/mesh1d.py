"""
1D uniform mesh with periodic or Dirichlet boundary conditions.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class Mesh1D:
    """
    Uniform 1D mesh on [x_left, x_right].

    Attributes
    ----------
    x_left, x_right : float
        Domain bounds.
    n_elem : int
        Number of elements.
    dx : float
        Element size (uniform).
    x_centers : ndarray, shape (n_elem,)
        Cell center coordinates.
    x_faces : ndarray, shape (n_elem+1,)
        Face coordinates (including domain boundaries).
    periodic : bool
        Whether the mesh is periodic.
    """
    x_left: float
    x_right: float
    n_elem: int
    dx: float
    x_centers: np.ndarray
    x_faces: np.ndarray
    periodic: bool = True

    @staticmethod
    def uniform(x_left: float, x_right: float, n_elem: int,
                periodic: bool = True) -> "Mesh1D":
        """Create a uniform mesh."""
        dx = (x_right - x_left) / n_elem
        x_faces = np.linspace(x_left, x_right, n_elem + 1)
        x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])
        return Mesh1D(
            x_left=x_left, x_right=x_right, n_elem=n_elem,
            dx=dx, x_centers=x_centers, x_faces=x_faces,
            periodic=periodic,
        )

    def physical_coords(self, xi: np.ndarray) -> np.ndarray:
        """
        Map reference coordinates ξ ∈ [-1,1] to physical coordinates
        for all elements.

        x = x_center + (dx/2) * ξ

        Parameters
        ----------
        xi : ndarray, shape (n_pts,)
            Reference coordinates.

        Returns
        -------
        x : ndarray, shape (n_elem, n_pts)
        """
        return self.x_centers[:, None] + 0.5 * self.dx * xi[None, :]
