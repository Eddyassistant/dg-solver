"""
2D Discontinuous Galerkin solver for scalar conservation laws on triangles.

    u_t + ∂f/∂x + ∂g/∂y = s(x, y, t)

Nodal DG with strong form:

    du/dt = -(rx·Dr + sx·Ds)·f - (ry·Dr + sy·Ds)·g
            + LIFT·(Fscale·(F·n - F̂·n))

References
----------
Hesthaven & Warburton (2008), Ch. 6.
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


# =============================================================================
# Numba kernels
# =============================================================================

@njit(cache=True, parallel=True)
def _compute_rhs_kernel(u, fx, fy, Dr, Ds, rx, ry, sx, sy,
                        LIFT, Fmask_flat, nx, ny, Fscale,
                        vmapM_k, vmapM_n, vmapP_k, vmapP_n,
                        max_speed_val, K, Np, Nfp):
    """
    Fused volume + surface RHS computation. Single pass over elements.
    """
    rhs = np.empty((K, Np), dtype=np.float64)
    n_face_pts = 3 * Nfp

    for k in prange(K):
        # --- Volume: -(rx*Dr + sx*Ds)*fx - (ry*Dr + sy*Ds)*fy ---
        vol = np.empty(Np, dtype=np.float64)
        rx_k = rx[k, 0]
        ry_k = ry[k, 0]
        sx_k = sx[k, 0]
        sy_k = sy[k, 0]

        for i in range(Np):
            dfdr = 0.0
            dfds = 0.0
            dgdr = 0.0
            dgds = 0.0
            for j in range(Np):
                dfdr += Dr[i, j] * fx[k, j]
                dfds += Ds[i, j] * fx[k, j]
                dgdr += Dr[i, j] * fy[k, j]
                dgds += Ds[i, j] * fy[k, j]
            dfdx = rx_k * dfdr + sx_k * dfds
            dgdy = ry_k * dgdr + sy_k * dgds
            vol[i] = -(dfdx + dgdy)

        # --- Surface: LIFT @ (Fscale * (F_phys·n - F_num·n)) ---
        du_surf = np.empty(n_face_pts, dtype=np.float64)
        for i in range(n_face_pts):
            vol_idx = Fmask_flat[i]
            # Interior trace
            f_phys_n = fx[k, vol_idx] * nx[k, i] + fy[k, vol_idx] * ny[k, i]
            # Exterior trace
            pk = vmapP_k[k, i]
            pn = vmapP_n[k, i]
            u_int = u[k, vol_idx]
            u_ext = u[pk, pn]
            fx_ext = fx[pk, pn]
            fy_ext = fy[pk, pn]
            fn_ext = fx_ext * nx[k, i] + fy_ext * ny[k, i]
            # Rusanov: F̂·n = ½(F⁻·n + F⁺·n) - ½λ(u⁺ - u⁻)
            f_num = 0.5 * (f_phys_n + fn_ext) - 0.5 * max_speed_val * (u_ext - u_int)
            du_surf[i] = Fscale[k, i] * (f_phys_n - f_num)

        # LIFT @ du_surf + volume
        for i in range(Np):
            val = vol[i]
            for j in range(n_face_pts):
                val += LIFT[i, j] * du_surf[j]
            rhs[k, i] = val

    return rhs


class DG2DScalar:
    """
    2D DG solver for scalar conservation laws on triangular meshes.
    """

    def __init__(
        self,
        mesh: TriangleMesh,
        p: int,
        flux_x: Callable,
        flux_y: Callable,
        max_wavespeed: Callable,
        source: Optional[Callable] = None,
        periodic: bool = True,
        x_range: tuple = (0.0, 1.0),
        y_range: tuple = (0.0, 1.0),
    ):
        self.mesh = mesh
        self.p = p
        self.K = mesh.n_elem
        self.source = source
        self._flux_x = flux_x
        self._flux_y = flux_y
        self._max_wavespeed = max_wavespeed

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

        # Flatten Fmask
        self.Fmask_flat = np.concatenate(self.ref.Fmask).astype(np.int64)

        # Build maps
        self._build_maps()

        # Contiguous matrices for Numba
        self.Dr = np.ascontiguousarray(self.ref.Dr)
        self.Ds = np.ascontiguousarray(self.ref.Ds)
        self.LIFT = np.ascontiguousarray(self.ref.LIFT)

        # JIT warmup
        self._warmup()

    def _build_maps(self):
        """Build face node maps using reversed ordering for neighbor faces."""
        K, Nfp = self.K, self.Nfp
        Fmask = self.ref.Fmask
        EToE, EToF = self.EToE, self.EToF
        n_fp = 3 * Nfp

        vmapP_k = np.zeros((K, n_fp), dtype=np.int64)
        vmapP_n = np.zeros((K, n_fp), dtype=np.int64)

        for k in range(K):
            for f in range(3):
                k2 = EToE[k, f]
                f2 = EToF[k, f]
                for i in range(Nfp):
                    idx = f * Nfp + i
                    if k2 == k and f2 == f:
                        # Boundary: self
                        vmapP_k[k, idx] = k
                        vmapP_n[k, idx] = Fmask[f][i]
                    else:
                        # Neighbor face nodes are in reverse order
                        vmapP_k[k, idx] = k2
                        vmapP_n[k, idx] = Fmask[f2][Nfp - 1 - i]

        self.vmapP_k = vmapP_k
        self.vmapP_n = vmapP_n

    def _warmup(self):
        """JIT compile kernels."""
        K, Np, Nfp = self.K, self.Np, self.Nfp
        u = np.zeros((K, Np))
        _compute_rhs_kernel(
            u, u, u, self.Dr, self.Ds,
            self.rx, self.ry, self.sx, self.sy,
            self.LIFT, self.Fmask_flat, self.nx, self.ny, self.Fscale,
            self.vmapP_k, self.vmapP_n, self.vmapP_k, self.vmapP_n,
            1.0, K, Np, Nfp,
        )

    def compute_rhs(self, u: np.ndarray, t: float) -> np.ndarray:
        """Compute RHS of semi-discrete DG system."""
        fx = self._flux_x(u)
        fy = self._flux_y(u)
        a_max = float(np.max(np.abs(self._max_wavespeed(u, u)))) + 1e-14

        rhs = _compute_rhs_kernel(
            u, fx, fy, self.Dr, self.Ds,
            self.rx, self.ry, self.sx, self.sy,
            self.LIFT, self.Fmask_flat, self.nx, self.ny, self.Fscale,
            self.vmapP_k, self.vmapP_n, self.vmapP_k, self.vmapP_n,
            a_max, self.K, self.Np, self.Nfp,
        )

        if self.source is not None:
            rhs += self.source(self.x, self.y, t)

        return rhs

    def project_ic(self, u0_func: Callable) -> np.ndarray:
        """Set IC by nodal evaluation."""
        return u0_func(self.x, self.y)

    def solve(
        self,
        u0_func: Callable,
        t_final: float,
        cfl: float = 0.1,
        time_integrator: Optional[Callable] = None,
    ) -> tuple[np.ndarray, float]:
        """Solve from t=0 to t_final."""
        integrator = time_integrator or ssp_rk3
        u = self.project_ic(u0_func)
        t = 0.0

        # Characteristic element size
        h_min = np.min(2.0 * np.abs(self.J[:, 0]) /
                       np.max(self.sJ.reshape(self.K, 3, self.Nfp), axis=2).max(axis=1))

        # Max wavespeed (constant for linear advection)
        a_max = float(np.max(np.abs(self._max_wavespeed(u, u)))) + 1e-14
        dt_base = cfl * h_min / ((2 * self.p + 1) * a_max)

        step = 0
        while t < t_final - 1e-14:
            dt = min(dt_base, t_final - t)
            u = integrator(u, dt, self.compute_rhs, t)
            t += dt
            step += 1

        return u, t

    def l2_error(self, u: np.ndarray, exact_func: Callable, t: float) -> float:
        """L2 error via mass matrix: ||e||² = Σ_k |J_k| ||V⁻¹ e||²."""
        u_exact = exact_func(self.x, self.y)
        err = u - u_exact
        err_modal = err @ self.ref.Vinv.T
        return np.sqrt(np.sum(err_modal**2 * np.abs(self.J)))
