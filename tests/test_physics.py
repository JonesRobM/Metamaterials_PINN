"""
Unit tests for physics module components.

Tests Maxwell equations, metamaterial properties, and boundary conditions
for correctness and numerical stability.
"""

import pytest
import torch
import numpy as np
from src.physics.maxwell_equations import MaxwellEquations
from src.physics.metamaterial import MetamaterialProperties
from src.physics.boundary_conditions import BoundaryConditions


class TestMaxwellEquations:
    """Test suite for Maxwell equations implementation."""
    
    @pytest.fixture
    def maxwell_solver(self):
        """Create Maxwell equations solver for testing."""
        omega = 2 * np.pi * 1e15  # 1 PHz
        return MaxwellEquations(omega)
    
    @pytest.fixture
    def sample_fields(self):
        """Create sample E and H fields for testing."""
        torch.manual_seed(42)
        coords = torch.randn(10, 3, requires_grad=True)
        E_field = torch.complex(torch.randn(10, 3), torch.randn(10, 3))
        H_field = torch.complex(torch.randn(10, 3), torch.randn(10, 3))
        return coords, E_field, H_field
    
    def test_initialization(self, maxwell_solver):
        """Test Maxwell solver initialization."""
        assert maxwell_solver.omega > 0
        assert maxwell_solver.k0 > 0
        assert maxwell_solver.c == pytest.approx(3e8, rel=1e-3)
    
    def test_curl_operator_shape(self, maxwell_solver, sample_fields):
        """Test curl operator returns correct shape."""
        coords, E_field, _ = sample_fields
        curl_E = maxwell_solver.curl_operator(E_field, coords)
        assert curl_E.shape == E_field.shape
    
    def test_curl_identity(self, maxwell_solver):
        """Test curl of gradient is zero (within numerical precision)."""
        # Create a scalar potential and compute its gradient
        coords = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True, dtype=torch.float32)
        phi = coords[:, 0]**2 + coords[:, 1]**2 + coords[:, 2]**2
        
        # Compute gradient (this would be E = -∇φ in electrostatics)
        grad_phi = torch.autograd.grad(phi, coords, create_graph=True, retain_graph=True)[0]
        grad_phi_complex = torch.complex(grad_phi, torch.zeros_like(grad_phi))
        
        # Curl of gradient should be zero
        curl_grad = maxwell_solver.curl_operator(grad_phi_complex, coords)
        assert torch.allclose(curl_grad, torch.zeros_like(curl_grad), atol=1e-6)
    
    def test_maxwell_residual_shapes(self, maxwell_solver, sample_fields):
        """Test Maxwell equation residuals return correct shapes."""
        coords, E_field, H_field = sample_fields
        epsilon_tensor = torch.eye(3).unsqueeze(0).expand(10, -1, -1).to(dtype=torch.complex64)
        
        curl_E_res = maxwell_solver.curl_E_residual(E_field, H_field, coords)
        curl_H_res = maxwell_solver.curl_H_residual(E_field, H_field, coords, epsilon_tensor)
        div_E_res = maxwell_solver.divergence_E_residual(E_field, coords, epsilon_tensor)
        div_B_res = maxwell_solver.divergence_B_residual(H_field, coords)
        
        assert curl_E_res.shape == (10, 6)  # Real + imaginary parts
        assert curl_H_res.shape == (10, 6)
        assert div_E_res.shape == (10, 2)
        assert div_B_res.shape == (10, 2)
    
    def test_total_residual_shape(self, maxwell_solver, sample_fields):
        """Test total residual returns correct shape."""
        coords, E_field, H_field = sample_fields
        epsilon_tensor = torch.eye(3).unsqueeze(0).expand(10, -1, -1).to(dtype=torch.complex64)
        
        total_res = maxwell_solver.total_residual(E_field, H_field, coords, epsilon_tensor)
        assert total_res.shape == (10, 16)  # 4 equations × 2 (real/imag) × 2 components avg
    
    def test_poynting_vector_shape(self, maxwell_solver, sample_fields):
        """Test Poynting vector calculation."""
        _, E_field, H_field = sample_fields
        S = maxwell_solver.poynting_vector(E_field, H_field)
        assert S.shape == E_field.shape
        assert torch.allclose(S.imag, torch.zeros_like(S.imag))  # Should be real
    
    def test_plane_wave_solution(self, maxwell_solver):
        """Test Maxwell equations with known plane wave solution."""
        # Create plane wave: E = E₀ exp(ik·r), H = (k×E)/(ωμ₀)
        k = torch.tensor([1e7, 0, 0], dtype=torch.float32)  # Wavevector
        E0 = torch.tensor([0, 1, 0], dtype=torch.complex64)  # Polarisation
        
        coords = torch.tensor([[0, 0, 0], [1e-6, 0, 0]], requires_grad=True, dtype=torch.float32)
        
        # Plane wave fields
        phase = torch.sum(k.unsqueeze(0) * coords, dim=1)
        exp_factor = torch.exp(1j * phase).unsqueeze(1)
        E_field = E0.unsqueeze(0) * exp_factor
        
        # H field from Faraday's law
        k_cross_E = torch.cross(k.unsqueeze(0).expand(2, -1), E_field.real) + 1j * torch.cross(k.unsqueeze(0).expand(2, -1), E_field.imag)
        H_field = k_cross_E / (maxwell_solver.omega * maxwell_solver.mu0)
        
        epsilon_tensor = torch.eye(3).unsqueeze(0).expand(2, -1, -1).to(dtype=torch.complex64)
        
        # Check curl E residual (should be small for plane wave)
        curl_E_res = maxwell_solver.curl_E_residual(E_field, H_field, coords)
        assert torch.allclose(curl_E_res, torch.zeros_like(curl_E_res), atol=1e-3)


class TestMetamaterialProperties:
    """Test suite for metamaterial properties."""
    
    @pytest.fixture
    def uniaxial_metamaterial(self):
        """Create uniaxial metamaterial for testing."""
        eps_par = -2.0 + 0.1j
        eps_perp = 4.0 + 0.05j
        return MetamaterialProperties(eps_par, eps_perp, optical_axis='z')
    
    def test_initialization(self, uniaxial_metamaterial):
        """Test metamaterial initialization."""
        assert uniaxial_metamaterial.eps_par == (-2.0 + 0.1j)
        assert uniaxial_metamaterial.eps_perp == (4.0 + 0.05j)
        assert uniaxial_metamaterial.optical_axis == 'z'
    
    def test_invalid_optical_axis(self):
        """Test invalid optical axis raises error."""
        with pytest.raises(ValueError):
            MetamaterialProperties(-2.0, 4.0, optical_axis='w')
    
    def test_permittivity_tensor_shape(self, uniaxial_metamaterial):
        """Test permittivity tensor returns correct shape."""
        coords = torch.randn(5, 3)
        eps_tensor = uniaxial_metamaterial.permittivity_tensor(coords)
        assert eps_tensor.shape == (5, 3, 3)
    
    def test_permittivity_tensor_uniaxial_z(self, uniaxial_metamaterial):
        """Test permittivity tensor for z-axis uniaxial material."""
        coords = torch.randn(3, 3)
        eps_tensor = uniaxial_metamaterial.permittivity_tensor(coords)
        
        # Check diagonal elements
        assert torch.allclose(eps_tensor[:, 0, 0], torch.full((3,), 4.0 + 0.05j))
        assert torch.allclose(eps_tensor[:, 1, 1], torch.full((3,), 4.0 + 0.05j))
        assert torch.allclose(eps_tensor[:, 2, 2], torch.full((3,), -2.0 + 0.1j))
        
        # Check off-diagonal elements are zero
        assert torch.allclose(eps_tensor[:, 0, 1], torch.zeros(3, dtype=torch.complex64))
        assert torch.allclose(eps_tensor[:, 1, 2], torch.zeros(3, dtype=torch.complex64))
    
    def test_permittivity_tensor_uniaxial_x(self):
        """Test permittivity tensor for x-axis uniaxial material."""
        metamaterial = MetamaterialProperties(-2.0 + 0.1j, 4.0 + 0.05j, optical_axis='x')
        coords = torch.randn(2, 3)
        eps_tensor = metamaterial.permittivity_tensor(coords)
        
        # Check diagonal elements
        assert torch.allclose(eps_tensor[:, 0, 0], torch.full((2,), -2.0 + 0.1j))
        assert torch.allclose(eps_tensor[:, 1, 1], torch.full((2,), 4.0 + 0.05j))
        assert torch.allclose(eps_tensor[:, 2, 2], torch.full((2,), 4.0 + 0.05j))
    
    def test_spp_dispersion_relation(self, uniaxial_metamaterial):
        """Test SPP dispersion relation calculation."""
        omega = 2 * np.pi * 1e15
        k_real, k_imag = uniaxial_metamaterial.spp_dispersion_relation(omega)
        
        # SPP should have real propagation constant
        assert k_real > 0
        # Some damping expected for lossy metamaterial
        assert k_imag >= 0
    
    def test_propagation_length(self, uniaxial_metamaterial):
        """Test SPP propagation length calculation."""
        omega = 2 * np.pi * 1e15
        L_prop = uniaxial_metamaterial.propagation_length(omega)
        assert L_prop > 0  # Should be finite for lossy material
    
    def test_penetration_depths(self, uniaxial_metamaterial):
        """Test penetration depth calculations."""
        omega = 2 * np.pi * 1e15
        depth_meta = uniaxial_metamaterial.penetration_depth_metamaterial(omega)
        depth_diel = uniaxial_metamaterial.penetration_depth_dielectric(omega)
        
        assert depth_meta > 0
        assert depth_diel > 0
    
    def test_spp_support_condition(self, uniaxial_metamaterial):
        """Test SPP existence condition."""
        # Should support SPPs with negative real permittivity
        assert uniaxial_metamaterial.is_spp_supported(eps_dielectric=1.0)
        
        # Should not support SPPs if both permittivities positive
        positive_metamaterial = MetamaterialProperties(2.0, 3.0, optical_axis='z')
        assert not positive_metamaterial.is_spp_supported(eps_dielectric=1.0)
    
    def test_field_enhancement(self, uniaxial_metamaterial):
        """Test field enhancement factor calculation."""
        omega = 2 * np.pi * 1e15
        enhancement = uniaxial_metamaterial.field_enhancement_factor(omega)
        assert enhancement > 0  # Should be finite and positive


class TestBoundaryConditions:
    """Test suite for boundary conditions."""
    
    @pytest.fixture
    def boundary_conditions(self):
        """Create boundary conditions for testing."""
        return BoundaryConditions(interface_normal=(0, 0, 1))
    
    @pytest.fixture
    def sample_boundary_fields(self):
        """Create sample fields at boundary for testing."""
        torch.manual_seed(42)
        E1 = torch.complex(torch.randn(5, 3), torch.randn(5, 3))
        E2 = torch.complex(torch.randn(5, 3), torch.randn(5, 3))
        H1 = torch.complex(torch.randn(5, 3), torch.randn(5, 3))
        H2 = torch.complex(torch.randn(5, 3), torch.randn(5, 3))
        return E1, E2, H1, H2
    
    def test_initialization(self, boundary_conditions):
        """Test boundary conditions initialization."""
        assert torch.allclose(boundary_conditions.interface_normal, torch.tensor([0, 0, 1], dtype=torch.float32))
    
    def test_normal_vector_normalisation(self):
        """Test that interface normal is normalised."""
        bc = BoundaryConditions(interface_normal=(3, 4, 0))
        expected_normal = torch.tensor([0.6, 0.8, 0], dtype=torch.float32)
        assert torch.allclose(bc.interface_normal, expected_normal)
    
    def test_cross_product(self, boundary_conditions):
        """Test cross product implementation."""
        a = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32)
        b = torch.tensor([[0, 1, 0], [0, 0, 1]], dtype=torch.float32)
        cross = boundary_conditions.cross_product(a, b)
        
        expected = torch.tensor([[0, 0, 1], [1, 0, 0]], dtype=torch.float32)
        assert torch.allclose(cross, expected)
    
    def test_tangential_E_continuity_shape(self, boundary_conditions, sample_boundary_fields):
        """Test tangential E continuity returns correct shape."""
        E1, E2, _, _ = sample_boundary_fields
        residual = boundary_conditions.tangential_E_continuity(E1, E2)
        assert residual.shape == (5, 6)  # Real + imaginary parts
    
    def test_tangential_H_continuity_shape(self, boundary_conditions, sample_boundary_fields):
        """Test tangential H continuity returns correct shape."""
        _, _, H1, H2 = sample_boundary_fields
        residual = boundary_conditions.tangential_H_continuity(H1, H2)
        assert residual.shape == (5, 6)  # Real + imaginary parts
    
    def test_normal_D_continuity_shape(self, boundary_conditions, sample_boundary_fields):
        """Test normal D continuity returns correct shape."""
        E1, E2, _, _ = sample_boundary_fields
        eps1 = torch.eye(3).unsqueeze(0).expand(5, -1, -1).to(dtype=torch.complex64)
        eps2 = 2.0 * torch.eye(3).unsqueeze(0).expand(5, -1, -1).to(dtype=torch.complex64)
        
        residual = boundary_conditions.normal_D_continuity(E1, E2, eps1, eps2)
        assert residual.shape == (5, 2)  # Real + imaginary parts
    
    def test_normal_B_continuity_shape(self, boundary_conditions, sample_boundary_fields):
        """Test normal B continuity returns correct shape."""
        _, _, H1, H2 = sample_boundary_fields
        residual = boundary_conditions.normal_B_continuity(H1, H2)
        assert residual.shape == (5, 2)  # Real + imaginary parts
    
    def test_perfect_continuity(self, boundary_conditions):
        """Test that identical fields satisfy boundary conditions."""
        # Create identical fields on both sides
        E_field = torch.complex(torch.ones(3, 3), torch.zeros(3, 3))
        H_field = torch.complex(torch.ones(3, 3), torch.zeros(3, 3))
        eps_tensor = torch.eye(3).unsqueeze(0).expand(3, -1, -1).to(dtype=torch.complex64)
        
        tang_E_res = boundary_conditions.tangential_E_continuity(E_field, E_field)
        tang_H_res = boundary_conditions.tangential_H_continuity(H_field, H_field)
        norm_D_res = boundary_conditions.normal_D_continuity(E_field, E_field, eps_tensor, eps_tensor)
        norm_B_res = boundary_conditions.normal_B_continuity(H_field, H_field)
        
        assert torch.allclose(tang_E_res, torch.zeros_like(tang_E_res), atol=1e-6)
        assert torch.allclose(tang_H_res, torch.zeros_like(tang_H_res), atol=1e-6)
        assert torch.allclose(norm_D_res, torch.zeros_like(norm_D_res), atol=1e-6)
        assert torch.allclose(norm_B_res, torch.zeros_like(norm_B_res), atol=1e-6)
    
    def test_spp_boundary_conditions_shape(self, boundary_conditions, sample_boundary_fields):
        """Test combined SPP boundary conditions shape."""
        E1, E2, H1, H2 = sample_boundary_fields
        eps_metamaterial = torch.eye(3).unsqueeze(0).expand(5, -1, -1).to(dtype=torch.complex64)
        
        combined_res = boundary_conditions.spp_boundary_conditions(
            E1, H1, E2, H2, eps_metamaterial, eps_dielectric=1.0
        )
        assert combined_res.shape == (5, 16)  # 4 boundary conditions × 4 components avg
    
    def test_perfect_conductor_boundary(self, boundary_conditions):
        
        """Test perfect conductor boundary condition."""
        # Tangential E field should be zero at PEC
        E_tangential = torch.tensor([[1, 1, 0], [0, 1, 1]], dtype=torch.complex64)
        residual = boundary_conditions.perfect_conductor_boundary(E_tangential)
        
        # For PEC with normal in z-direction, Ex and Ey should be zero
        assert residual.shape == (2, 6)  # Real + imaginary parts
    
    def test_impedance_boundary_condition_shape(self, boundary_conditions):
        """Test impedance boundary condition shape."""
        E_tan = torch.complex(torch.randn(3, 3), torch.randn(3, 3))
        H_tan = torch.complex(torch.randn(3, 3), torch.randn(3, 3))
        Z_s = 377.0 + 10j  # Surface impedance
        
        residual = boundary_conditions.impedance_boundary_condition(E_tan, H_tan, Z_s)
        assert residual.shape == (3, 6)  # Real + imaginary parts


if __name__ == "__main__":
    pytest.main([__file__])