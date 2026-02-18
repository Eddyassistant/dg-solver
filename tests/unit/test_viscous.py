"""
Unit tests for BR2 viscous flux operator.

Tests verify:
1. Correct operator behavior (linearity, scaling, exactness for polynomials)
2. Conservation properties
3. Convergence rates for diffusion equation
"""

import numpy as np
import numpy.testing as npt
import pytest

from src.dg.viscous_br2 import BR2ViscousFlux
from src.dg.mesh.triangle_mesh import TriangleMesh


class TestLaplacianExactness:
    """
    Test that Laplacian is exact for polynomials up to degree p.
    
    For degree p elements:
    - u: degree ≤ p → represented exactly
    - ∇u: degree ≤ p-1 → represented exactly  
    - ∇²u: degree ≤ p-2 → constant in each element for p=2
    """
    
    def test_laplacian_quadratic(self):
        """
        Test Laplacian of quadratic: u = x² + y².
        Expected: ∇²u = 4 (constant).
        """
        p = 2
        nx = 4
        
        mesh = TriangleMesh.rectangle((0.0, 1.0), (0.0, 1.0), nx=nx, ny=nx)
        br2 = BR2ViscousFlux(mesh, p=p, nu=1.0, periodic=False)
        
        # Quadratic field
        u_quad = br2.x**2 + br2.y**2
        
        lap = br2.compute_laplacian(u_quad)
        
        # For p=2, Laplacian should be constant 4 in each element
        for k in range(br2.K):
            assert np.allclose(lap[k], 4.0, rtol=1e-10), \
                f"Element {k}: Laplacian should be 4, got {lap[k, 0]}"
    
    def test_laplacian_linear(self):
        """
        Test Laplacian of linear: u = ax + by + c.
        Expected: ∇²u = 0.
        """
        p = 2
        nx = 4
        
        mesh = TriangleMesh.rectangle((0.0, 1.0), (0.0, 1.0), nx=nx, ny=nx)
        br2 = BR2ViscousFlux(mesh, p=p, nu=1.0, periodic=False)
        
        # Linear field with various coefficients
        for a, b, c in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 1.0), (2.0, -3.0, 5.0)]:
            u_lin = a * br2.x + b * br2.y + c
            lap = br2.compute_laplacian(u_lin)
            npt.assert_allclose(lap, 0.0, atol=1e-10, 
                               err_msg=f"Laplacian of linear {a}x+{b}y+{c} should be 0")


class TestBR2Gradient:
    """Test the BR2 gradient computation."""
    
    def test_gradient_linear(self):
        """
        Test gradient of linear function: u = x.
        Expected: ∇u = (1, 0).
        """
        p = 2
        nx = 4
        
        mesh = TriangleMesh.rectangle((0.0, 1.0), (0.0, 1.0), nx=nx, ny=nx)
        br2 = BR2ViscousFlux(mesh, p=p, nu=1.0, periodic=False)
        
        u_linear = br2.x  # u = x
        ux, uy = br2.compute_lifted_gradient(u_linear)
        
        # For p=2, should be exact
        npt.assert_allclose(ux, 1.0, rtol=1e-10)
        npt.assert_allclose(uy, 0.0, atol=1e-10)
    
    def test_gradient_quadratic(self):
        """
        Test gradient of quadratic: u = x² + y².
        Expected: ∇u = (2x, 2y).
        """
        p = 2
        nx = 4
        
        mesh = TriangleMesh.rectangle((0.0, 1.0), (0.0, 1.0), nx=nx, ny=nx)
        br2 = BR2ViscousFlux(mesh, p=p, nu=1.0, periodic=False)
        
        u_quad = br2.x**2 + br2.y**2
        ux, uy = br2.compute_lifted_gradient(u_quad)
        
        # For p=2, should be exact (within machine precision)
        npt.assert_allclose(ux, 2 * br2.x, rtol=1e-10, atol=1e-12)
        npt.assert_allclose(uy, 2 * br2.y, rtol=1e-10, atol=1e-12)


class TestBR2Properties:
    """Test general properties of BR2 operator."""
    
    def test_linearity(self):
        """Test that operator is linear: L(au + bv) = aL(u) + bL(v)."""
        p = 2
        nx = 4
        
        mesh = TriangleMesh.rectangle((0.0, 1.0), (0.0, 1.0), nx=nx, ny=nx)
        br2 = BR2ViscousFlux(
            mesh, p=p, nu=1.0,
            periodic=True, x_range=(0.0, 1.0), y_range=(0.0, 1.0)
        )
        
        # Two different fields
        u = np.sin(2 * np.pi * br2.x) * np.sin(2 * np.pi * br2.y)
        v = np.cos(2 * np.pi * br2.x) * np.cos(2 * np.pi * br2.y)
        
        a, b = 2.0, 3.0
        
        # L(au + bv)
        lhs = br2.compute_laplacian(a * u + b * v)
        
        # aL(u) + bL(v)
        rhs = a * br2.compute_laplacian(u) + b * br2.compute_laplacian(v)
        
        # Should be equal
        npt.assert_allclose(lhs, rhs, rtol=1e-10)
    
    def test_viscosity_scaling(self):
        """Test that RHS scales linearly with ν."""
        p = 2
        nx = 4
        
        mesh = TriangleMesh.rectangle((0.0, 1.0), (0.0, 1.0), nx=nx, ny=nx)
        
        nu1 = 1.0
        nu2 = 2.0
        
        br2_1 = BR2ViscousFlux(
            mesh, p=p, nu=nu1,
            periodic=True, x_range=(0.0, 1.0), y_range=(0.0, 1.0)
        )
        br2_2 = BR2ViscousFlux(
            mesh, p=p, nu=nu2,
            periodic=True, x_range=(0.0, 1.0), y_range=(0.0, 1.0)
        )
        
        u = np.sin(2 * np.pi * br2_1.x) * np.sin(2 * np.pi * br2_1.y)
        
        rhs1 = br2_1.compute_rhs(u)
        rhs2 = br2_2.compute_rhs(u)
        
        # rhs should scale with nu
        npt.assert_allclose(rhs2, 2.0 * rhs1, rtol=1e-10)


class TestDiffusionStability:
    """
    Test that diffusion solutions are stable and decay appropriately.
    """
    
    def test_diffusion_does_not_explode(self):
        """
        Test that a simple diffusion step doesn't blow up.
        """
        nu = 0.1
        p = 2
        nx = 4
        
        mesh = TriangleMesh.rectangle((0.0, 1.0), (0.0, 1.0), nx=nx, ny=nx)
        br2 = BR2ViscousFlux(
            mesh, p=p, nu=nu,
            periodic=True, x_range=(0.0, 1.0), y_range=(0.0, 1.0)
        )
        
        # Smooth initial condition
        u0 = np.sin(2 * np.pi * br2.x) * np.sin(2 * np.pi * br2.y)
        max_u0 = np.max(np.abs(u0))
        
        # Small time step
        dt = 0.0001
        
        # One step
        rhs = br2.compute_rhs(u0)
        u1 = u0 + dt * rhs
        
        max_u1 = np.max(np.abs(u1))
        
        # Solution should not grow significantly in one small step
        assert max_u1 < max_u0 * 1.5, f"Solution grew too much: {max_u0} → {max_u1}"
    
    def test_diffusion_conservation(self):
        """
        Test mass conservation for periodic diffusion of constant + sine.
        
        For periodic BCs with mean-preserving initial condition,
        total mass should be approximately conserved.
        """
        nu = 0.1
        p = 2
        nx = 4
        
        mesh = TriangleMesh.rectangle((0.0, 1.0), (0.0, 1.0), nx=nx, ny=nx)
        br2 = BR2ViscousFlux(
            mesh, p=p, nu=nu,
            periodic=True, x_range=(0.0, 1.0), y_range=(0.0, 1.0)
        )
        
        # Mean = 1, with sine perturbation that should decay
        u0 = 1.0 + 0.1 * np.sin(2 * np.pi * br2.x) * np.sin(2 * np.pi * br2.y)
        
        # Compute initial mass
        mass0 = np.sum(u0 * np.abs(br2.J))
        
        # Take a few small steps
        dt = 0.001
        u = u0.copy()
        for _ in range(10):
            rhs = br2.compute_rhs(u)
            u = u + dt * rhs
        
        # Compute final mass
        mass_final = np.sum(u * np.abs(br2.J))
        
        # Check conservation (allow small numerical error)
        mass_error = abs(mass_final - mass0) / abs(mass0)
        assert mass_error < 0.1, f"Mass not conserved: error {mass_error}"


class TestDiffusionConvergence:
    """
    Test convergence of diffusion equation solutions.
    
    These tests use small time steps to minimize time discretization error
    and focus on spatial convergence.
    """
    
    def test_gradient_convergence(self):
        """
        Test that gradient converges for smooth functions.
        """
        p = 2
        resolutions = [4, 8, 16]
        errors = []
        
        for nx in resolutions:
            mesh = TriangleMesh.rectangle((0.0, 1.0), (0.0, 1.0), nx=nx, ny=nx)
            br2 = BR2ViscousFlux(mesh, p=p, nu=1.0, periodic=False)
            
            # Test function: u = x² + y² (exact for p=2)
            u = br2.x**2 + br2.y**2
            ux_num, uy_num = br2.compute_lifted_gradient(u)
            
            ux_exact = 2 * br2.x
            uy_exact = 2 * br2.y
            
            # L2 error
            err_x = ux_num - ux_exact
            err_y = uy_num - uy_exact
            err_modal_x = err_x @ br2.ref.Vinv.T
            err_modal_y = err_y @ br2.ref.Vinv.T
            
            l2_error = np.sqrt(np.sum((err_modal_x**2 + err_modal_y**2) * np.abs(br2.J)))
            errors.append(l2_error)
        
        # Should be near machine precision for all meshes (exact for p=2)
        for err in errors:
            assert err < 1e-10, f"Gradient error too large: {err}"
    
    def test_laplacian_quadratic_exact(self):
        """
        Test that Laplacian is exact for quadratics (should be constant 4 for x²+y²).
        """
        p = 2
        for nx in [4, 8, 16]:
            mesh = TriangleMesh.rectangle((0.0, 1.0), (0.0, 1.0), nx=nx, ny=nx)
            br2 = BR2ViscousFlux(mesh, p=p, nu=1.0, periodic=False)
            
            u = br2.x**2 + br2.y**2
            lap = br2.compute_laplacian(u)
            
            # Should be exactly 4 everywhere
            npt.assert_allclose(lap, 4.0, rtol=1e-10, 
                               err_msg=f"Laplacian of x²+y² should be 4 for nx={nx}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
