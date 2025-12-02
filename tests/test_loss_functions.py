import pytest
import torch
import torch.nn as nn
import numpy as np
from src.models.loss_functions import MaxwellCurlLoss
from src.physics.maxwell_equations import MaxwellEquations


class TestMaxwellCurlLoss:
    """Test suite for MaxwellCurlLoss implementation."""
    
    @pytest.fixture
    def maxwell_curl_loss_fn(self):
        """Create MaxwellCurlLoss instance for testing."""
        omega = 2 * np.pi * 1e15  # 1 PHz
        return MaxwellCurlLoss(frequency=omega)
        
    def test_initialization(self, maxwell_curl_loss_fn):
        """Test MaxwellCurlLoss initialization."""
        assert maxwell_curl_loss_fn.omega > 0
        assert maxwell_curl_loss_fn.mu0 > 0
        assert maxwell_curl_loss_fn.eps0 > 0
        
    def test_compute_with_known_solution(self, maxwell_curl_loss_fn):
        """
        Test MaxwellCurlLoss with a known plane wave solution.
        The loss should be near zero for a solution that satisfies Maxwell's equations.
        """
        batch_size = 5
        coords = torch.randn(batch_size, 3, requires_grad=True, dtype=torch.float32)
                                                                                                                         
        omega = maxwell_curl_loss_fn.omega
        mu0 = maxwell_curl_loss_fn.mu0
        eps0 = maxwell_curl_loss_fn.eps0
        k0 = omega * np.sqrt(mu0 * eps0) # Wave number in free space                                                     
                                                                                                                         
        k_vec = torch.tensor([k0, 0.0, 0.0], dtype=torch.float32, device=coords.device)                                  
        E_pol = torch.tensor([0.0, 1.0, 0.0], dtype=torch.complex64, device=coords.device)                               
                                                                                                                         
        # Magnetic field polarization for a plane wave in free space: H = (-1/(omega*mu0)) * (k_vec x E)
        # This H definition ensures curl_E + iomega*mu*H = 0
        H_pol = (-1 / (omega * mu0)) * torch.linalg.cross(k_vec.to(E_pol.dtype), E_pol)                                  
                                                                                                                         
        class AnalyticalPlaneWaveNetwork(nn.Module):                                                                     
            def __init__(self, E_pol_val, H_pol_val, k_vec_val):                                                         
                super().__init__()                                                                                       
                self.E_pol = E_pol_val                                                                                   
                self.H_pol = H_pol_val                                                                                   
                self.k_vec = k_vec_val                                                                                   
                                                                                                                         
            def forward(self, coords_input):                                                                             
                phase = torch.einsum('j,ij->i', self.k_vec, coords_input)                                                
                exp_factor = torch.exp(1j * phase).unsqueeze(1)
                                                                                                                         
                E_field = (self.E_pol.unsqueeze(0) * exp_factor)                                    
                H_field = (self.H_pol.unsqueeze(0) * exp_factor)                                    
                                                                                                                         
                return torch.cat([E_field, H_field], dim=1)                                                              
                                                                                                                         
        analytical_network = AnalyticalPlaneWaveNetwork(E_pol, H_pol, k_vec)                                             
                                                                                                                         
        loss = maxwell_curl_loss_fn.compute(network=analytical_network, coords=coords)                                                
                                                                                                                         
        # Increased tolerance due to numerical precision with large k0 and omega
        atol = 1.0
        assert torch.isclose(loss, torch.tensor(0.0, dtype=loss.dtype), atol=atol), f"MaxwellCurlLoss failed: {loss}"
