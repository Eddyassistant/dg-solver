"""
1D Discontinuous Galerkin solver for scalar conservation laws.

    u_t + f(u)_x = s(x, t)

Uses modal Legendre basis, Gauss-Legendre quadrature, and
pluggable numerical fluxes and time integrators.

The DG weak form (after integration by parts):

  M · dû/dt = ∫ f(u) · dφ/dx dΩ - [F̂ · φ]_{∂K} + ∫ s · φ dΩ

where û are the modal coefficients, M is the (diagonal) mass matrix,
φ are the test functions (Legendre polynomials), and F̂ is the numerical flux.

References
----------
Hesthaven & Warburton (2008), Nodal DG Methods, Ch. 5.
Cockburn & Shu (2001), J. Sci. Comput. 16(3), pp. 173-261.
"""

import numpy as np
from typing import Callable, Optional

from .basis.legendre import (
    legendre_poly, modal_mass_matrix_diag, vandermonde, grad_vandermonde,
)
from .quadrature.gauss import gauss_legendre
from .mesh.mesh1d import Mesh1D
from .flux.scalar import rusanov
from .timestepping.runge_kutta import ssp_rk3


class DG1DScalar:
    """
    1D DG solver for scalar conservation laws with periodic BCs.

    Parameters
    ----------
    mesh : Mesh1D
        The computational mesh.
    p : int
        Polynomial degree of the approximation space.
    physical_flux : callable (u: ndarray) -> ndarray
        Physical flux function f(u).
    max_wavespeed_func : callable (u_L: ndarray, u_R: ndarray) -> ndarray
        Returns max |f'(u)| for Rusanov flux at each face.
    source : callable (x: ndarray, t: float) -> ndarray, optional
        Source term s(x, t). Default: no source.
    numerical_flux : callable, optional
        Custom numerical flux. Default: Rusanov.
    n_quad : int, optional
        Number of quadrature points. Default: p + 1 (exact for linear fluxes;
        use ≥ ceil((3p+1)/2) for nonlinear fluxes to avoid aliasing).
    """

    def __init__(
        self,
        mesh: Mesh1D,
        p: int,
        physical_flux: Callable,
        max_wavespeed_func: Callable,
        source: Optional[Callable] = None,
        numerical_flux: Optional[Callable] = None,
        n_quad: Optional[int] = None,
    ):
        self.mesh = mesh
        self.p = p
        self.n_modes = p + 1
        self.physical_flux = physical_flux
        self.max_wavespeed_func = max_wavespeed_func
        self.source = source
        self.numerical_flux = numerical_flux or rusanov
        self.n_quad = n_quad or (p + 1)

        # Precompute quadrature
        self.xi_q, self.w_q = gauss_legendre(self.n_quad)

        # Precompute basis values at quadrature points: shape (n_modes, n_quad)
        self.phi_q = legendre_poly(self.xi_q, p)  # (n_modes, n_quad)

        # Basis derivatives at quadrature points
        from .basis.legendre import legendre_poly_deriv
        _, self.dphi_q = legendre_poly_deriv(self.xi_q, p)  # (n_modes, n_quad)

        # Basis values at element boundaries ξ = -1, +1
        self.phi_left = legendre_poly(np.array([-1.0]), p)[:, 0]   # (n_modes,)
        self.phi_right = legendre_poly(np.array([1.0]), p)[:, 0]   # (n_modes,)

        # Mass matrix diagonal (reference element)
        self.M_diag = modal_mass_matrix_diag(p)  # (n_modes,)

        # Jacobian: dx/dξ = dx/2 for uniform mesh
        self.jac = mesh.dx / 2.0
        self.inv_jac = 2.0 / mesh.dx

        # Physical coordinates of quadrature points: (n_elem, n_quad)
        self.x_quad = mesh.physical_coords(self.xi_q)

    def modal_to_nodal(self, u_modal: np.ndarray) -> np.ndarray:
        """
        Convert modal coefficients to nodal values at quadrature points.

        u_nodal[e, q] = Σ_k u_modal[e, k] · φ_k(ξ_q)

        Parameters
        ----------
        u_modal : ndarray, shape (n_elem, n_modes)

        Returns
        -------
        u_nodal : ndarray, shape (n_elem, n_quad)
        """
        return u_modal @ self.phi_q  # (n_elem, n_modes) @ (n_modes, n_quad)

    def boundary_values(self, u_modal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate solution at element boundaries.

        Parameters
        ----------
        u_modal : ndarray, shape (n_elem, n_modes)

        Returns
        -------
        u_left : ndarray, shape (n_elem,)
            u(ξ=-1) for each element (left boundary).
        u_right : ndarray, shape (n_elem,)
            u(ξ=+1) for each element (right boundary).
        """
        u_left = u_modal @ self.phi_left     # (n_elem,)
        u_right = u_modal @ self.phi_right   # (n_elem,)
        return u_left, u_right

    def compute_rhs(self, u_modal: np.ndarray, t: float) -> np.ndarray:
        """
        Compute the RHS of the semi-discrete DG scheme: M⁻¹ · R(u).

        The residual for each element K and test function φ_k:

        R_k = ∫_K f(u) · dφ_k/dx dx - [F̂ · φ_k]_{∂K} + ∫_K s · φ_k dx

        Transformed to reference element [-1, 1] with x = x_c + (h/2)ξ:
        - Volume integral: Σ_q f(u(ξ_q)) · dφ_k/dξ · w_q  (note: no Jacobian since dx cancels)
        - Surface terms: F̂_R · φ_k(+1) - F̂_L · φ_k(-1)
        - Source: Σ_q s(x_q, t) · φ_k(ξ_q) · w_q · (h/2)

        Final: dû_k/dt = (1/M_kk) · [volume - surface + source] · (2/h)

        Wait, let me be precise about the Jacobian treatment:

        ∫_{x_L}^{x_R} f(u) φ'_k(x) dx = ∫_{-1}^{1} f(u(ξ)) · (dφ_k/dξ) · (dξ/dx) · (dx/dξ) dξ
                                          = ∫_{-1}^{1} f(u(ξ)) · (dφ_k/dξ) · dξ   [Jacobians cancel!]

        Surface terms remain in physical space:
            F̂_R φ_k(1) - F̂_L φ_k(-1)

        Source: ∫ s φ_k dx = ∫ s φ_k (h/2) dξ

        So: M̃ dû/dt = vol_integral - surface_terms + source_integral
        where M̃_kk = M_kk · (h/2)

        => dû/dt = (2/h) / M_kk · [vol - surface + source*(h/2)]

        Parameters
        ----------
        u_modal : ndarray, shape (n_elem, n_modes), float64
        t : float

        Returns
        -------
        rhs : ndarray, shape (n_elem, n_modes), float64
        """
        n_elem = self.mesh.n_elem
        n_modes = self.n_modes

        # 1. Evaluate solution at quadrature points
        u_q = self.modal_to_nodal(u_modal)  # (n_elem, n_quad)

        # 2. Evaluate physical flux at quadrature points
        f_q = self.physical_flux(u_q)  # (n_elem, n_quad)

        # 3. Volume integral: Σ_q f(u_q) · dφ_k/dξ_q · w_q
        #    Shape: (n_elem, n_quad) · (n_modes, n_quad).T weighted by w_q
        #    = (n_elem, n_quad) * w_q @ dphi_q.T → (n_elem, n_modes)
        vol = (f_q * self.w_q[None, :]) @ self.dphi_q.T  # (n_elem, n_modes)

        # 4. Compute interface fluxes
        u_left, u_right = self.boundary_values(u_modal)  # each (n_elem,)

        # Interior state at each face:
        # Face i sits between element i-1 (right) and element i (left)
        # For n_elem elements with periodic BCs, there are n_elem faces
        # u_minus[face] = u_right of left element
        # u_plus[face] = u_left of right element
        u_minus = u_right  # (n_elem,) — right boundary of each element = left state at face i+1
        u_plus = np.roll(u_left, -1)  # left boundary of next element = right state at face i+1

        # Actually let's think about this carefully.
        # Face i is at x_faces[i]. Element i has faces i (left) and i+1 (right).
        # For periodic: face 0 = face n_elem.
        # At face i: u⁻ = u_right[i-1], u⁺ = u_left[i]

        # We need n_elem faces for periodic (face 0 through n_elem-1)
        # Face i: left state = right boundary of element (i-1) mod n_elem
        #         right state = left boundary of element i
        face_u_minus = np.roll(u_right, 1)  # u_right[(i-1) % n_elem] for face i
        face_u_plus = u_left                 # u_left[i] for face i

        # Physical flux at face states
        face_f_minus = self.physical_flux(face_u_minus)
        face_f_plus = self.physical_flux(face_u_plus)

        # Max wavespeed at each face
        face_wavespeed = self.max_wavespeed_func(face_u_minus, face_u_plus)

        # Numerical flux at each face
        F_hat = self.numerical_flux(
            face_f_minus, face_f_plus,
            face_u_minus, face_u_plus,
            face_wavespeed,
        )  # (n_elem,) — one flux per face

        # 5. Surface terms for each element
        # Element i has left face = face i, right face = face (i+1) % n_elem
        F_left = F_hat                   # flux at left face of element i
        F_right = np.roll(F_hat, -1)     # flux at right face of element i

        # Surface contribution: F̂_R · φ_k(+1) - F̂_L · φ_k(-1)
        # Shape: (n_elem, n_modes)
        surface = (F_right[:, None] * self.phi_right[None, :]
                   - F_left[:, None] * self.phi_left[None, :])

        # 6. Source term
        if self.source is not None:
            s_q = self.source(self.x_quad, t)  # (n_elem, n_quad)
            source_integral = (s_q * self.w_q[None, :]) @ self.phi_q.T * self.jac
        else:
            source_integral = 0.0

        # 7. Assemble: dû/dt = (2/h) / M_kk · [vol - surface + source]
        rhs = (vol - surface + source_integral) * self.inv_jac / self.M_diag[None, :]

        return rhs

    def project_ic(self, u0_func: Callable) -> np.ndarray:
        """
        L2-project an initial condition function onto the DG space.

        û_k = (1/M_kk) ∫ u0(x) φ_k(ξ) (h/2) dξ
            = (1/M_kk) · (h/2) · Σ_q u0(x_q) φ_k(ξ_q) w_q

        Parameters
        ----------
        u0_func : callable (x: ndarray) -> ndarray
            Initial condition function.

        Returns
        -------
        u_modal : ndarray, shape (n_elem, n_modes), float64
        """
        u0_q = u0_func(self.x_quad)  # (n_elem, n_quad)
        # Project: û_k = (1/M_kk) * jac * Σ_q u0_q * φ_k(ξ_q) * w_q
        # û_k = (jac * Σ_q u0 w_q P_k) / (M_kk * jac) = Σ_q u0 w_q P_k / M_kk
        u_modal = ((u0_q * self.w_q[None, :]) @ self.phi_q.T) / self.M_diag[None, :]
        return u_modal

    def solve(
        self,
        u0_func: Callable,
        t_final: float,
        cfl: float = 0.1,
        time_integrator: Optional[Callable] = None,
        callback: Optional[Callable] = None,
    ) -> tuple[np.ndarray, float]:
        """
        Solve the conservation law from t=0 to t=t_final.

        Parameters
        ----------
        u0_func : callable
            Initial condition function.
        t_final : float
            Final time.
        cfl : float
            CFL number for adaptive time stepping.
        time_integrator : callable, optional
            Time integrator function. Default: SSP-RK3.
        callback : callable (u_modal, t, step) -> None, optional
            Called each time step for monitoring.

        Returns
        -------
        u_modal : ndarray, shape (n_elem, n_modes)
            Solution at t_final.
        t : float
            Actual final time reached.
        """
        integrator = time_integrator or ssp_rk3
        u_modal = self.project_ic(u0_func)
        t = 0.0
        step = 0

        while t < t_final - 1e-14:
            # Adaptive dt from CFL
            u_left, u_right = self.boundary_values(u_modal)
            u_all = np.concatenate([u_left, u_right])
            # max wavespeed across all faces
            a_max = np.max(np.abs(self.max_wavespeed_func(u_all, u_all))) + 1e-14
            dt = cfl * self.mesh.dx / ((2 * self.p + 1) * a_max)
            dt = min(dt, t_final - t)

            u_modal = integrator(u_modal, dt, self.compute_rhs, t)
            t += dt
            step += 1

            if callback is not None:
                callback(u_modal, t, step)

        return u_modal, t

    def evaluate(self, u_modal: np.ndarray, xi: np.ndarray) -> np.ndarray:
        """
        Evaluate the DG solution at reference points ξ within each element.

        Parameters
        ----------
        u_modal : ndarray, shape (n_elem, n_modes)
        xi : ndarray, shape (n_pts,)

        Returns
        -------
        u : ndarray, shape (n_elem, n_pts)
        """
        phi = legendre_poly(xi, self.p)  # (n_modes, n_pts)
        return u_modal @ phi  # (n_elem, n_pts)

    def cell_averages(self, u_modal: np.ndarray) -> np.ndarray:
        """Cell averages (= zeroth modal coefficient, since P_0 = 1)."""
        return u_modal[:, 0].copy()

    def l2_error(self, u_modal: np.ndarray, exact_func: Callable, t: float) -> float:
        """
        Compute the L2 error ||u_h - u_exact||_L2.

        Uses the same quadrature as the solver (may need more points for
        highly oscillatory exact solutions).
        """
        u_q = self.modal_to_nodal(u_modal)
        u_exact = exact_func(self.x_quad)
        err_sq = np.sum((u_q - u_exact)**2 * self.w_q[None, :]) * self.jac
        return np.sqrt(err_sq)
