"""
2D Discontinuous Galerkin solver for SYSTEMS of conservation laws on triangles.

    ∂q/∂t + ∂F(q)/∂x + ∂G(q)/∂y = S(q, x, y, t)

where q ∈ ℝ^{n_vars} is the state vector.

Extends solver2d.py to handle vector-valued solutions. Each variable
is stored at the same nodal points. The Numba kernel processes all
variables in a fused loop for cache efficiency.

Strong form DG:
    dq/dt = -(rx·Dr + sx·Ds)·F - (ry·Dr + sy·Ds)·G
            + LIFT·(Fscale·(F_phys·n - F̂·n))

References
----------
Hesthaven & Warburton (2008), Ch. 7-9.
"""

import numpy as np
from numba import njit, prange
from typing import Callable, Optional

from .reference_triangle import RefTriangle
from .mesh.triangle_mesh import (
    TriangleMesh, build_face_connectivity,
    geometric_factors, face_normals,
)
from .timestepping.runge_kutta import ssp_rk3


@njit(cache=True, parallel=True)
def _system_rhs_kernel(q, Fx, Fy, Dr, Ds, rx, ry, sx, sy,
                       LIFT, Fmask_flat, nx, ny, Fscale,
                       vmapP_k, vmapP_n, f_num,
                       K, Np, Nfp, n_vars):
    """
    Fused volume + surface RHS for a system of n_vars equations.

    q: (K, Np, n_vars) — nodal solution
    Fx, Fy: (K, Np, n_vars) — physical fluxes
    f_num: (K, 3*Nfp, n_vars) — numerical flux · n at face nodes
    """
    rhs = np.empty((K, Np, n_vars), dtype=np.float64)
    n_face_pts = 3 * Nfp

    for k in prange(K):
        rx_k = rx[k, 0]
        ry_k = ry[k, 0]
        sx_k = sx[k, 0]
        sy_k = sy[k, 0]

        for v in range(n_vars):
            # --- Volume: -(rx*Dr + sx*Ds)*Fx - (ry*Dr + sy*Ds)*Fy ---
            vol = np.empty(Np, dtype=np.float64)
            for i in range(Np):
                dfdr = 0.0
                dfds = 0.0
                dgdr = 0.0
                dgds = 0.0
                for j in range(Np):
                    dfdr += Dr[i, j] * Fx[k, j, v]
                    dfds += Ds[i, j] * Fx[k, j, v]
                    dgdr += Dr[i, j] * Fy[k, j, v]
                    dgds += Ds[i, j] * Fy[k, j, v]
                dfdx = rx_k * dfdr + sx_k * dfds
                dgdy = ry_k * dgdr + sy_k * dgds
                vol[i] = -(dfdx + dgdy)

            # --- Surface: LIFT @ (Fscale * (F_phys·n - F_num·n)) ---
            du_surf = np.empty(n_face_pts, dtype=np.float64)
            for i in range(n_face_pts):
                vol_idx = Fmask_flat[i]
                f_phys_n = (Fx[k, vol_idx, v] * nx[k, i]
                            + Fy[k, vol_idx, v] * ny[k, i])
                du_surf[i] = Fscale[k, i] * (f_phys_n - f_num[k, i, v])

            for i in range(Np):
                val = vol[i]
                for j in range(n_face_pts):
                    val += LIFT[i, j] * du_surf[j]
                rhs[k, i, v] = val

    return rhs


class DG2DSystem:
    """
    2D DG solver for systems of conservation laws on triangles.

    Parameters
    ----------
    mesh : TriangleMesh
    p : int
        Polynomial degree.
    n_vars : int
        Number of equations/variables in the system.
    flux_x : callable (q) -> ndarray, shape (K, Np, n_vars)
        Physical x-flux F(q).
    flux_y : callable (q) -> ndarray, shape (K, Np, n_vars)
        Physical y-flux G(q).
    numerical_flux : callable (q_int, q_ext, nx, ny) -> ndarray
        Numerical flux function F̂·n at faces.
        Input: q_int, q_ext of shape (K, 3*Nfp, n_vars), nx/ny of shape (K, 3*Nfp)
        Output: shape (K, 3*Nfp, n_vars)
    max_wavespeed : callable (q) -> float
        Returns global max wavespeed for CFL.
    source : callable (q, x, y, t) -> ndarray, optional
    viscous_rhs : callable (q, t) -> ndarray, optional
        Viscous contribution to RHS (e.g., from BR2 operator).
    bc_func : callable (q_int, bc_tags, face_nx, face_ny) -> q_ext, optional
        Applies boundary conditions by returning exterior state.
    bc_tags : ndarray, shape (K, 3), optional
        Boundary condition tags per face (0=interior).
    periodic : bool
    x_range, y_range : tuple
    """

    def __init__(
        self,
        mesh: TriangleMesh,
        p: int,
        n_vars: int,
        flux_x: Callable,
        flux_y: Callable,
        numerical_flux: Callable,
        max_wavespeed: Callable,
        source: Optional[Callable] = None,
        viscous_rhs: Optional[Callable] = None,
        bc_func: Optional[Callable] = None,
        bc_tags: Optional[np.ndarray] = None,
        periodic: bool = False,
        x_range: tuple = (0.0, 1.0),
        y_range: tuple = (0.0, 1.0),
    ):
        self.mesh = mesh
        self.p = p
        self.n_vars = n_vars
        self.K = mesh.n_elem
        self._flux_x = flux_x
        self._flux_y = flux_y
        self._numerical_flux = numerical_flux
        self._max_wavespeed = max_wavespeed
        self.source = source
        self.viscous_rhs = viscous_rhs
        self.bc_func = bc_func
        self.bc_tags = bc_tags

        # Reference element
        self.ref = RefTriangle(p)
        self.Np = self.ref.Np
        self.Nfp = self.ref.Nfp

        # Connectivity
        self.EToE, self.EToF = build_face_connectivity(
            mesh, periodic_x=periodic, periodic_y=periodic,
            x_range=x_range, y_range=y_range,
        )

        # Geometric factors
        self.x, self.y, self.rx, self.ry, self.sx, self.sy, self.J = \
            geometric_factors(mesh, self.ref, p)

        self.nx, self.ny, self.sJ, self.Fscale = \
            face_normals(mesh, self.ref, self.x, self.y, self.J)

        self.Fmask_flat = np.concatenate(self.ref.Fmask).astype(np.int64)

        self._build_maps()

        self.Dr = np.ascontiguousarray(self.ref.Dr)
        self.Ds = np.ascontiguousarray(self.ref.Ds)
        self.LIFT = np.ascontiguousarray(self.ref.LIFT)

        # Warmup
        self._warmup()

    def _build_maps(self):
        """Build face-to-neighbor maps."""
        K, Nfp = self.K, self.Nfp
        Fmask = self.ref.Fmask
        EToE, EToF = self.EToE, self.EToF
        n_fp = 3 * Nfp

        self.vmapP_k = np.zeros((K, n_fp), dtype=np.int64)
        self.vmapP_n = np.zeros((K, n_fp), dtype=np.int64)
        self.is_boundary = np.zeros((K, n_fp), dtype=np.bool_)

        for k in range(K):
            for f in range(3):
                k2 = EToE[k, f]
                f2 = EToF[k, f]
                for i in range(Nfp):
                    idx = f * Nfp + i
                    if k2 == k and f2 == f:
                        self.vmapP_k[k, idx] = k
                        self.vmapP_n[k, idx] = Fmask[f][i]
                        self.is_boundary[k, idx] = True
                    else:
                        self.vmapP_k[k, idx] = k2
                        self.vmapP_n[k, idx] = Fmask[f2][Nfp - 1 - i]
                        self.is_boundary[k, idx] = False

    def _warmup(self):
        """JIT warmup."""
        K, Np, Nfp, nv = self.K, self.Np, self.Nfp, self.n_vars
        q = np.zeros((K, Np, nv))
        Fx = np.zeros((K, Np, nv))
        Fy = np.zeros((K, Np, nv))
        fn = np.zeros((K, 3 * Nfp, nv))
        _system_rhs_kernel(
            q, Fx, Fy, self.Dr, self.Ds,
            self.rx, self.ry, self.sx, self.sy,
            self.LIFT, self.Fmask_flat, self.nx, self.ny, self.Fscale,
            self.vmapP_k, self.vmapP_n, fn,
            K, Np, Nfp, nv,
        )

    def _extract_face_values(self, q: np.ndarray):
        """
        Extract interior and exterior face values from q.

        Parameters
        ----------
        q : ndarray, shape (K, Np, n_vars)

        Returns
        -------
        q_int, q_ext : ndarray, shape (K, 3*Nfp, n_vars)
        """
        K, Nfp, nv = self.K, self.Nfp, self.n_vars
        n_fp = 3 * Nfp
        Fmask = self.Fmask_flat

        q_int = np.empty((K, n_fp, nv))
        q_ext = np.empty((K, n_fp, nv))

        for idx in range(n_fp):
            vol_idx = Fmask[idx]
            q_int[:, idx, :] = q[:, vol_idx, :]
            pk = self.vmapP_k[:, idx]
            pn = self.vmapP_n[:, idx]
            q_ext[:, idx, :] = q[pk, pn, :]

        # Apply BCs for boundary faces
        if self.bc_func is not None and self.bc_tags is not None:
            # Expand bc_tags to per-face-node
            bc_per_node = np.zeros((K, n_fp), dtype=np.int32)
            for f in range(3):
                bc_per_node[:, f * Nfp:(f + 1) * Nfp] = self.bc_tags[:, f:f + 1]

            # Face normals at face nodes
            face_nx = self.nx
            face_ny = self.ny

            q_ext = self.bc_func(q_int, q_ext, bc_per_node, face_nx, face_ny)

        return q_int, q_ext

    def compute_rhs(self, q: np.ndarray, t: float) -> np.ndarray:
        """
        Compute RHS of the semi-discrete system.

        Parameters
        ----------
        q : ndarray, shape (K, Np, n_vars)
        t : float

        Returns
        -------
        rhs : ndarray, shape (K, Np, n_vars)
        """
        # Physical fluxes
        Fx = self._flux_x(q)
        Fy = self._flux_y(q)

        # Face values
        q_int, q_ext = self._extract_face_values(q)

        # Numerical flux at faces
        f_num = self._numerical_flux(q_int, q_ext, self.nx, self.ny)

        # Inviscid RHS (Numba)
        rhs = _system_rhs_kernel(
            q, Fx, Fy, self.Dr, self.Ds,
            self.rx, self.ry, self.sx, self.sy,
            self.LIFT, self.Fmask_flat, self.nx, self.ny, self.Fscale,
            self.vmapP_k, self.vmapP_n, f_num,
            self.K, self.Np, self.Nfp, self.n_vars,
        )

        # Viscous contribution
        if self.viscous_rhs is not None:
            rhs += self.viscous_rhs(q, t)

        # Source term
        if self.source is not None:
            rhs += self.source(q, self.x, self.y, t)

        return rhs

    def project_ic(self, q0_func: Callable) -> np.ndarray:
        """Set IC: q0(x, y) → (K, Np, n_vars)."""
        return q0_func(self.x, self.y)

    def solve(
        self,
        q0_func: Callable,
        t_final: float,
        cfl: float = 0.1,
        time_integrator: Optional[Callable] = None,
        callback: Optional[Callable] = None,
        max_steps: int = 10_000_000,
    ) -> tuple[np.ndarray, float, int]:
        """
        Time-march from t=0 to t_final.

        Returns
        -------
        q : ndarray, shape (K, Np, n_vars)
        t : float
        n_steps : int
        """
        integrator = time_integrator or ssp_rk3
        q = self.project_ic(q0_func)
        t = 0.0

        h_min = np.min(2.0 * np.abs(self.J[:, 0]) /
                       np.max(self.sJ.reshape(self.K, 3, self.Nfp), axis=2).max(axis=1))

        step = 0
        while t < t_final - 1e-14 and step < max_steps:
            a_max = self._max_wavespeed(q) + 1e-14
            dt = cfl * h_min / ((2 * self.p + 1) * a_max)
            dt = min(dt, t_final - t)

            q = integrator(q, dt, self.compute_rhs, t)
            t += dt
            step += 1

            if callback is not None:
                callback(q, t, step, dt)

        return q, t, step

    def l2_error(self, q: np.ndarray, exact_func: Callable, t: float) -> np.ndarray:
        """
        L2 error per variable.

        Returns
        -------
        errors : ndarray, shape (n_vars,)
        """
        q_exact = exact_func(self.x, self.y)
        err = q - q_exact
        Vinv = self.ref.Vinv
        errors = np.zeros(self.n_vars)
        for v in range(self.n_vars):
            err_modal = err[:, :, v] @ Vinv.T
            errors[v] = np.sqrt(np.sum(err_modal**2 * np.abs(self.J)))
        return errors
