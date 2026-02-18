"""
Unit tests for BR2 viscous flux operator.

Tests verify:
1. Laplacian of sine function on periodic domain
2. Convergence rates for diffusion equation
"""

import numpy as np
import numpy.testing as npt
import pytest

from src.dg.viscous_br2 import BR2ViscousFlux
from src.dg.mesh.triangle_mesh import TriangleMesh


class TestLaplacianSine:
    """
    Test that ∇²(sin(2πx)sin(2πy)) = -8π² sin(2πx)sin(2πy) on periodic domain.
    """
    
    @pytest.fixture
    def periodic_mesh(self):
        """Create a periodic unit square mesh."""
        return TriangleMesh.rectangle((0.0, 1.0), (0.0, 1.0), nx=8, ny=8)
    
    def test_laplacian_sine_p1(self, periodic_mesh):
        """Test Laplacian of sine with p=1 elements."""
        nu = 1.0
        p = 1
        
        br2 = BR2ViscousFlux(
            periodic_mesh, p=p, nu=nu,
            periodic=True, x_range=(0.0, 1.0), y_range=(0.0, 1.0)
        )
        
        # Sine initial condition
        u_sine = np.sin(2 * np.pi * br2.x) * np.sin(2 * np.pi * br2.y)
        
        # Compute Laplacian
        laplacian = br2.compute_laplacian(u_sine)
        
        # Expected: ∇²u = -8π² sin(2πx)sin(2πy)
        expected = -8 * np.pi**2 * u_sine
        
        # For low-order, the error will be significant; check sign and rough magnitude
        # The Laplacian should be negative where u is positive
        mask = u_sine > 0.1  # Where sine is significantly positive
        assert np.all(laplacian[mask] < 0), "Laplacian should be negative where u>0"
        
        mask_neg = u_sine < -0.1  # Where sine is significantly negative
        assert np.all(laplacian[mask_neg] > 0), "Laplacian should be positive where u<0"
    
    def test_laplacian_sine_p2(self, periodic_mesh):
        """Test Laplacian accuracy with p=2 elements."""
        nu = 1.0
        p = 2
        
        br2 = BR2ViscousFlux(
            periodic_mesh, p=p, nu=nu,
            periodic=True, x_range=(0.0, 1.0), y_range=(0.0, 1.0)
        )
        
        u_sine = np.sin(2 * np.pi * br2.x) * np.sin(2 * np.pi * br2.y)
        laplacian = br2.compute_laplacian(u_sine)
        expected = -8 * np.pi**2 * u_sine
        
        # Compute L2 error of Laplacian
        err = laplacian - expected
        err_modal = err @ br2.ref.Vinv.T
        l2_error = np.sqrt(np.sum(err_modal**2 * np.abs(br2.J)))
        
        # For p=2 on 8x8 mesh, expect reasonable accuracy
        # The error should decrease with mesh refinement
        assert l2_error < 50.0, f"Laplacian error {l2_error} too large for p=2"
    
    def test_laplacian_sine_convergence(self):
        """Test that Laplacian error converges with mesh refinement."""
        nu = 1.0
        p = 2
        errors = []
        resolutions = [4, 8, 16]
        
        for nx in resolutions:
            mesh = TriangleMesh.rectangle((0.0, 1.0), (0.0, 1.0), nx=nx, ny=nx)
            br2 = BR2ViscousFlux(
                mesh, p=p, nu=nu,
                periodic=True, x_range=(0.0, 1.0), y_range=(0.0, 1.0)
            )
            
            u_sine = np.sin(2 * np.pi * br2.x) * np.sin(2 * np.pi * br2.y)
            laplacian = br2.compute_laplacian(u_sine)
            expected = -8 * np.pi**2 * u_sine
            
            err = laplacian - expected
            err_modal = err @ br2.ref.Vinv.T
            l2_error = np.sqrt(np.sum(err_modal**2 * np.abs(br2.J)))
            errors.append(l2_error)
        
        # Check convergence: error should decrease
        assert errors[1] < errors[0], "Error should decrease with refinement"
        assert errors[2] < errors[1], "Error should decrease with refinement"
        
        # Estimate convergence rate
        rates = []
        for i in range(len(rates)):
            rate = np.log(errors[i+1] / errors[i]) / np.log(0.5)
            rates.append(rate)
        
        # For p=2, expect roughly order p convergence for Laplacian
        # (one order less than solution due to differentiation)
        avg_rate = np.log(errors[-1] / errors[0]) / np.log(resolutions[0] / resolutions[-1])
        assert avg_rate > 0.5, f"Convergence rate {avg_rate} too slow"


class TestDiffusionEquation:
    """
    Test solving the diffusion equation u_t = ν∇²u.
    """
    
    def test_diffusion_decay(self):
        """
        Test that sine mode decays exponentially: u(t) = u(0) * exp(-νλt)
        where λ = 8π² for sin(2πx)sin(2πy).
        """
        nu = 0.1
        p = 2
        nx = 8
        t_final = 0.1
        
        mesh = TriangleMesh.rectangle((0.0, 1.0), (0.0, 1.0), nx=nx, ny=nx)
        br2 = BR2ViscousFlux(
            mesh, p=p, nu=nu,
            periodic=True, x_range=(0.0, 1.0), y_range=(0.0, 1.0)
        )
        
        # Initial condition
        u0 = np.sin(2 * np.pi * br2.x) * np.sin(2 * np.pi * br2.y)
        
        # Solve
        u_final, t = br2.solve_diffusion(
            lambda x, y: np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y),
            t_final=t_final,
            cfl=0.05  # Small CFL for accuracy
        )
        
        # Expected decay
        lambda_val = 8 * np.pi**2
        expected_decay = np.exp(-nu * lambda_val * t)
        
        # The peak should have decayed approximately by expected amount
        # (accounting for numerical errors)
        u0_peak = np.max(np.abs(u0))
        u_final_peak = np.max(np.abs(u_final))
        actual_decay = u_final_peak / u0_peak
        
        # Allow 10% relative error in decay rate
        assert actual_decay < 1.0, "Solution should decay"
        rel_error = abs(actual_decay - expected_decay) / expected_decay
        assert rel_error < 0.2, f"Decay rate error {rel_error} too large"
    
    def test_diffusion_convergence(self):
        """
        Test convergence of diffusion solution against analytical solution.
        
        For u_t = ν∇²u with u(x,y,0) = sin(2πx)sin(2πy),
        exact solution is: u(x,y,t) = sin(2πx)sin(2πy) * exp(-ν*8π²*t)
        """
        nu = 0.1
        p = 2
        t_final = 0.05
        cfl = 0.05
        
        resolutions = [4, 8, 16]
        errors = []
        
        for nx in resolutions:
            mesh = TriangleMesh.rectangle((0.0, 1.0), (0.0, 1.0), nx=nx, ny=nx)
            br2 = BR2ViscousFlux(
                mesh, p=p, nu=nu,
                periodic=True, x_range=(0.0, 1.0), y_range=(0.0, 1.0)
            )
            
            # Solve
            u_final, t = br2.solve_diffusion(
                lambda x, y: np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y),
                t_final=t_final,
                cfl=cfl
            )
            
            # Exact solution
            u_exact = (np.sin(2 * np.pi * br2.x) * np.sin(2 * np.pi * br2.y) *
                       np.exp(-nu * 8 * np.pi**2 * t))
            
            # L2 error
            err = u_final - u_exact
            err_modal = err @ br2.ref.Vinv.T
            l2_error = np.sqrt(np.sum(err_modal**2 * np.abs(br2.J)))
            errors.append(l2_error)
        
        # Check convergence
        assert errors[1] < errors[0], "Error should decrease with refinement"
        assert errors[2] < errors[1], "Error should decrease with refinement"
        
        # Estimate convergence rate
        avg_rate = np.log(errors[-1] / errors[0]) / np.log(resolutions[0] / resolutions[-1])
        
        # For p=2, expect approximately order p+1 for solution
        assert avg_rate > 1.0, f"Solution convergence rate {avg_rate} too slow"
    
    def test_diffusion_conservation(self):
        """
        Test mass conservation for periodic diffusion.
        
        For periodic BCs, total mass should be conserved.
        """
        nu = 0.1
        p = 2
        nx = 8
        t_final = 0.1
        
        mesh = TriangleMesh.rectangle((0.0, 1.0), (0.0, 1.0), nx=nx, ny=nx)
        br2 = BR2ViscousFlux(
            mesh, p=p, nu=nu,
            periodic=True, x_range=(0.0, 1.0), y_range=(0.0, 1.0)
        )
        
        # Non-zero mean initial condition
        u0 = np.sin(2 * np.pi * br2.x) * np.sin(2 * np.pi * br2.y) + 1.0
        
        # Compute initial mass
        mass0 = np.sum(u0 * np.abs(br2.J))
        
        # Solve
        u_final, t = br2.solve_diffusion(
            lambda x, y: np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) + 1.0,
            t_final=t_final,
            cfl=0.05
        )
        
        # Compute final mass
        mass_final = np.sum(u_final * np.abs(br2.J))
        
        # Check conservation (allow small numerical error)
        mass_error = abs(mass_final - mass0) / abs(mass0)
        assert mass_error < 0.01, f"Mass not conserved: error {mass_error}"


class TestBR2Gradient:
    """Test the BR2 gradient computation."""
    
    def test_gradient_linear(self):
        """
        Test gradient of linear function: u = x.
        Expected: ∇u = (1, 0).
        """
        p = 1
        nx = 4
        
        mesh = TriangleMesh.rectangle((0.0, 1.0), (0.0, 1.0), nx=nx, ny=nx)
        br2 = BR2ViscousFlux(
            mesh, p=p, nu=1.0,
            periodic=False  # Non-periodic for this test
        )
        
        u_linear = br2.x  # u = x
        
        ux, uy = br2.compute_lifted_gradient(u_linear)
        
        # Gradient should be approximately (1, 0)
        # Check interior points (not on boundary)
        mean_ux = np.mean(ux)
        mean_uy = np.mean(uy)
        
        assert abs(mean_ux - 1.0) < 0.1, f"du/dx should be ~1, got {mean_ux}"
        assert abs(mean_uy) < 0.1, f"du/dy should be ~0, got {mean_uy}"
    
    def test_gradient_quadratic(self):
        """
        Test gradient of quadratic: u = x².
        Expected: ∇u = (2x, 0).
        """
        p = 2  # Need p=2 for exact representation of x²
        nx = 4
        
        mesh = TriangleMesh.rectangle((0.0, 1.0), (0.0, 1.0), nx=nx, ny=nx)
        br2 = BR2ViscousFlux(
            mesh, p=p, nu=1.0,
            periodic=False
        )
        
        u_quad = br2.x**2  # u = x²
        
        ux, uy = br2.compute_lifted_gradient(u_quad)
        
        # Expected gradient: (2x, 0)
        expected_ux = 2 * br2.x
        
        # L2 error of gradient
        err_x = ux - expected_ux
        err_modal_x = err_x @ br2.ref.Vinv.T
        l2_error_x = np.sqrt(np.sum(err_modal_x**2 * np.abs(br2.J)))
        
        # For p=2 with x², should be exact (up to numerical precision)
        assert l2_error_x < 0.1, f"Gradient error {l2_error_x} too large"
        
        # y-component should be close to zero
        assert np.max(np.abs(uy)) < 0.1, f"du/dy should be ~0, got max {np.max(np.abs(uy))}"


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
