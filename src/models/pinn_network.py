"""
Electromagnetic Physics-Informed Neural Network architectures

Implements PINN architectures specifically designed for electromagnetic
problems, including complex-valued fields, metamaterial systems, and
Surface Plasmon Polariton (SPP) modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Union, Callable, Dict
import math


class ComplexLinear(nn.Module):
    """
    Complex-valued linear layer for electromagnetic fields.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        bias: Whether to include bias term
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Real and imaginary weight matrices
        self.weight_real = nn.Parameter(torch.randn(out_features, in_features))
        self.weight_imag = nn.Parameter(torch.randn(out_features, in_features))
        
        if bias:
            self.bias_real = nn.Parameter(torch.randn(out_features))
            self.bias_imag = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)
        
        self._initialise_weights()
    
    def _initialise_weights(self):
        """Xavier initialisation for complex weights."""
        bound = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.uniform_(self.weight_real, -bound, bound)
        nn.init.uniform_(self.weight_imag, -bound, bound)
        
        if self.bias_real is not None:
            nn.init.uniform_(self.bias_real, -bound, bound)
            nn.init.uniform_(self.bias_imag, -bound, bound)
    
    def forward(self, input_complex: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for complex input.
        
        Args:
            input_complex: Complex input [batch_size, in_features, 2] (real, imag)
            
        Returns:
            Complex output [batch_size, out_features, 2]
        """
        input_real = input_complex[..., 0]
        input_imag = input_complex[..., 1]
        
        # Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        output_real = F.linear(input_real, self.weight_real) - F.linear(input_imag, self.weight_imag)
        output_imag = F.linear(input_real, self.weight_imag) + F.linear(input_imag, self.weight_real)
        
        if self.bias_real is not None:
            output_real += self.bias_real
            output_imag += self.bias_imag
        
        return torch.stack([output_real, output_imag], dim=-1)


class ElectromagneticActivation(nn.Module):
    """
    Activation functions suitable for electromagnetic fields.
    
    Args:
        activation_type: Type of activation ('complex_tanh', 'modulus', 'split')
    """
    
    def __init__(self, activation_type: str = 'complex_tanh'):
        super().__init__()
        self.activation_type = activation_type
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply electromagnetic-specific activation."""
        if self.activation_type == 'complex_tanh':
            # Apply tanh to both real and imaginary parts
            return torch.tanh(x)
        
        elif self.activation_type == 'modulus':
            # Preserve magnitude, apply activation to phase
            magnitude = torch.norm(x, dim=-1, keepdim=True)
            phase = torch.atan2(x[..., 1], x[..., 0]).unsqueeze(-1)
            activated_phase = torch.tanh(phase)
            
            new_real = magnitude.squeeze(-1) * torch.cos(activated_phase).squeeze(-1)
            new_imag = magnitude.squeeze(-1) * torch.sin(activated_phase).squeeze(-1)
            return torch.stack([new_real, new_imag], dim=-1)
        
        elif self.activation_type == 'split':
            # Different activations for real and imaginary parts
            real_part = torch.tanh(x[..., 0])
            imag_part = torch.sin(x[..., 1])  # Oscillatory for phase
            return torch.stack([real_part, imag_part], dim=-1)
        
        else:
            return torch.tanh(x)


class FourierEMFeatures(nn.Module):
    """
    Fourier features optimised for electromagnetic problems.
    
    Args:
        input_dim: Spatial input dimension
        encoding_size: Number of Fourier modes
        frequency_range: Frequency range for sampling (k0 units)
        include_dc: Whether to include DC component
    """
    
    def __init__(self, 
                 input_dim: int,
                 encoding_size: int = 128,
                 frequency_range: Tuple[float, float] = (0.1, 10.0),
                 include_dc: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.encoding_size = encoding_size
        self.include_dc = include_dc
        
        # Sample frequencies for electromagnetic problems
        k_min, k_max = frequency_range
        k_values = torch.logspace(np.log10(k_min), np.log10(k_max), encoding_size//2)
        
        # Random directions for 2D/3D problems
        if input_dim == 2:
            angles = torch.rand(encoding_size//2) * 2 * np.pi
            k_vectors = torch.stack([k_values * torch.cos(angles),
                                   k_values * torch.sin(angles)], dim=1)
        elif input_dim == 3:
            # Spherical sampling
            theta = torch.rand(encoding_size//2) * np.pi
            phi = torch.rand(encoding_size//2) * 2 * np.pi
            k_vectors = torch.stack([
                k_values * torch.sin(theta) * torch.cos(phi),
                k_values * torch.sin(theta) * torch.sin(phi),
                k_values * torch.cos(theta)
            ], dim=1)
        else:
            k_vectors = k_values.unsqueeze(1)
        
        self.register_buffer('k_vectors', k_vectors)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier encoding.
        
        Args:
            coords: Spatial coordinates [batch_size, input_dim]
            
        Returns:
            Fourier features [batch_size, feature_dim]
        """
        # k·r
        phases = torch.matmul(coords, self.k_vectors.T)
        
        # Sine and cosine features
        features = torch.cat([torch.sin(phases), torch.cos(phases)], dim=1)
        
        if self.include_dc:
            features = torch.cat([coords, features], dim=1)
        
        return features


class ElectromagneticPINN(nn.Module):
    """
    PINN for electromagnetic problems with complex-valued outputs.
    
    Args:
        spatial_dim: Spatial dimension (2 or 3)
        field_components: Number of field components (3 or 6)
        hidden_dims: Hidden layer dimensions
        complex_valued: Whether outputs are complex
        frequency: Operating frequency (for frequency-dependent problems)
        use_fourier: Whether to use Fourier features
        activation_type: Type of activation function
    """
    
    def __init__(self,
                 spatial_dim: int = 3,
                 field_components: int = 6,  # Ex, Ey, Ez, Hx, Hy, Hz
                 hidden_dims: List[int] = [128, 128, 128, 128],
                 complex_valued: bool = True,
                 frequency: Optional[float] = None,
                 use_fourier: bool = True,
                 fourier_modes: int = 128,
                 activation_type: str = 'complex_tanh',
                 **kwargs):
        super().__init__()
        
        self.spatial_dim = spatial_dim
        self.field_components = field_components
        self.complex_valued = complex_valued
        self.frequency = frequency
        
        # Input dimension
        input_dim = spatial_dim
        if frequency is None:  # Time-dependent problem
            input_dim += 1
        
        # Fourier feature encoding
        if use_fourier:
            self.fourier_encoder = FourierEMFeatures(
                input_dim, fourier_modes, (0.1, 20.0)
            )
            first_layer_input = input_dim + fourier_modes
        else:
            self.fourier_encoder = None
            first_layer_input = input_dim
        
        # Build network
        if complex_valued:
            self._build_complex_network(first_layer_input, hidden_dims, activation_type)
        else:
            self._build_real_network(first_layer_input, hidden_dims)
        
        self._initialise_weights()
    
    def _build_complex_network(self, input_dim: int, hidden_dims: List[int], activation_type: str):
        """Build complex-valued network."""
        # Project real input to complex first hidden layer
        self.input_projection = nn.Linear(input_dim, hidden_dims[0])
        
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(ComplexLinear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(ElectromagneticActivation(activation_type))
        
        # Output layer
        output_features = self.field_components
        layers.append(ComplexLinear(hidden_dims[-1], output_features, bias=True))
        
        self.complex_network = nn.ModuleList(layers)
    
    def _build_real_network(self, input_dim: int, hidden_dims: List[int]):
        """Build real-valued network with separated real/imaginary outputs."""
        layers = []
        dims = [input_dim] + hidden_dims + [self.field_components * 2]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        
        self.real_network = nn.Sequential(*layers)
    
    def _initialise_weights(self):
        """Electromagnetic-specific weight initialisation."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, ComplexLinear)):
                # Smaller initialisation for EM problems
                if isinstance(module, ComplexLinear):
                    nn.init.xavier_normal_(module.weight_real, gain=0.5)
                    nn.init.xavier_normal_(module.weight_imag, gain=0.5)
                    if module.bias_real is not None:
                        nn.init.zeros_(module.bias_real)
                        nn.init.zeros_(module.bias_imag)
                else: # nn.Linear
                    nn.init.xavier_normal_(module.weight, gain=0.5)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            coords: Input coordinates [batch_size, spatial_dim] or [..., spatial_dim+1]
            
        Returns:
            Electromagnetic fields [batch_size, field_components] or [..., field_components, 2]
        """
        # Apply Fourier encoding
        if self.fourier_encoder is not None:
            x = self.fourier_encoder(coords)
        else:
            x = coords
        
        if self.complex_valued:
            # Project to first hidden layer
            x_projected = self.input_projection(x)
            # Convert to complex format: [batch, features, 2] with zero imaginary part initially
            x_complex = torch.stack([x_projected, torch.zeros_like(x_projected)], dim=-1)
            
            # Forward through complex network
            for layer in self.complex_network:
                x_complex = layer(x_complex)
            
            return x_complex
        else:
            # Real-valued network
            output = self.real_network(x)
            # Reshape to [batch, components, 2] for real/imag
            return output.view(output.shape[0], self.field_components, 2)
    
    def get_fields(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get E and H fields separately.
        
        Returns:
            E_field: Electric field [batch_size, 3] or [batch_size, 3, 2]
            H_field: Magnetic field [batch_size, 3] or [batch_size, 3, 2]
        """
        fields = self(coords)
        
        if self.field_components == 6:
            E_field = fields[:, :3]
            H_field = fields[:, 3:]
        elif self.field_components == 3:
            # Assume only E field
            E_field = fields
            H_field = torch.zeros_like(E_field)
        else:
            raise ValueError(f"Unsupported field_components: {self.field_components}")
        
        return E_field, H_field


class SPPNetwork(ElectromagneticPINN):
    """
    Specialised network for Surface Plasmon Polariton modeling.
    
    Args:
        interface_position: z-coordinate of metal-dielectric interface
        metal_permittivity: Complex permittivity of metal
        dielectric_permittivity: Permittivity of dielectric
        frequency: Operating frequency
        **kwargs: Additional arguments for ElectromagneticPINN
    """
    
    def __init__(self,
                 interface_position: float = 0.0,
                 metal_permittivity: complex = -20 + 1j,
                 dielectric_permittivity: float = 2.25,
                 frequency: float = 1e15,
                 spatial_dim: int = 3,
                 **kwargs):
        super().__init__(
            spatial_dim=spatial_dim,
            field_components=6,
            complex_valued=True,
            frequency=frequency,
            **kwargs
        )
        
        self.interface_z = interface_position
        self.eps_metal = metal_permittivity
        self.eps_dielectric = dielectric_permittivity
        self.omega = frequency
        
        # SPP-specific parameters
        self.k0 = frequency / 299792458  # Free space wavevector
        self.k_spp = self._calculate_spp_wavevector()
        
        # Add SPP-specific layers
        self.spp_modulation = nn.Sequential(
            nn.Linear(spatial_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 6),
            nn.Tanh()
        )
    
    def _calculate_spp_wavevector(self) -> complex:
        """Calculate SPP wavevector from material properties."""
        eps_m = torch.tensor(self.eps_metal, dtype=torch.complex64)
        eps_d = torch.tensor(self.eps_dielectric, dtype=torch.complex64)
        
        k_spp = self.k0 * torch.sqrt((eps_m * eps_d) / (eps_m + eps_d))
        return k_spp
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward pass with SPP-specific modifications."""
        # Base electromagnetic network
        base_fields = super().forward(coords)
        
        # SPP-specific modulation
        spp_mod = self.spp_modulation(coords)
        
        # Apply interface-dependent modulation
        z_coords = coords[:, 2] if coords.shape[1] > 2 else coords[:, 1]
        interface_mask = torch.tanh(10 * z_coords)  # Smooth transition
        
        # Modulate fields based on distance from interface
        if self.complex_valued:
            # Apply modulation to both real and imaginary parts
            modulated_fields = base_fields.clone()
            for i in range(self.field_components):
                decay_factor = torch.exp(-torch.abs(z_coords) / 1e-6)  # SPP decay
                modulated_fields[:, i, 0] *= decay_factor * (1 + 0.1 * spp_mod[:, i])
                modulated_fields[:, i, 1] *= decay_factor * (1 + 0.1 * spp_mod[:, i])
        else:
            modulated_fields = base_fields * (1 + 0.1 * spp_mod.unsqueeze(-1))
        
        return modulated_fields


class MetamaterialDeepONet(nn.Module):
    """
    DeepONet for metamaterial operator learning.
    
    Args:
        material_param_dim: Dimension of material parameter space
        spatial_dim: Spatial dimension
        field_components: Number of electromagnetic field components
        **kwargs: Additional DeepONet parameters
    """
    
    def __init__(self,
                 material_param_dim: int = 9,  # 3x3 permittivity tensor
                 spatial_dim: int = 3,
                 field_components: int = 6,
                 branch_hidden: List[int] = [128, 128, 128],
                 trunk_hidden: List[int] = [128, 128, 128],
                 latent_dim: int = 128,
                 **kwargs):
        super().__init__()
        
        self.material_param_dim = material_param_dim
        self.spatial_dim = spatial_dim
        self.field_components = field_components
        self.latent_dim = latent_dim
        
        # Branch network: metamaterial parameters → latent space
        branch_dims = [material_param_dim] + branch_hidden + [latent_dim]
        branch_layers = []
        for i in range(len(branch_dims) - 1):
            branch_layers.append(nn.Linear(branch_dims[i], branch_dims[i+1]))
            if i < len(branch_dims) - 2:
                branch_layers.append(nn.Tanh())
        
        self.branch_net = nn.Sequential(*branch_layers)
        
        # Trunk network: spatial coordinates → latent space  
        trunk_input_dim = spatial_dim + 1  # Include frequency
        trunk_dims = [trunk_input_dim] + trunk_hidden + [latent_dim * field_components]
        trunk_layers = []
        for i in range(len(trunk_dims) - 1):
            trunk_layers.append(nn.Linear(trunk_dims[i], trunk_dims[i+1]))
            if i < len(trunk_dims) - 2:
                trunk_layers.append(nn.Tanh())
        
        self.trunk_net = nn.Sequential(*trunk_layers)
        
        # Output bias
        self.bias = nn.Parameter(torch.zeros(field_components))
    
    def forward(self, 
                material_params: torch.Tensor,
                spatial_coords: torch.Tensor,
                frequency: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through metamaterial DeepONet.
        
        Args:
            material_params: Material parameters [batch_size, material_param_dim]
            spatial_coords: Spatial coordinates [batch_size, spatial_dim]
            frequency: Frequency values [batch_size, 1]
            
        Returns:
            Electromagnetic fields [batch_size, field_components]
        """
        # Branch network: encode material properties
        branch_output = self.branch_net(material_params)  # [batch, latent_dim]
        
        # Trunk network: encode spatiotemporal coordinates
        trunk_input = torch.cat([spatial_coords, frequency], dim=1)
        trunk_output = self.trunk_net(trunk_input)  # [batch, latent_dim * field_components]
        
        # Reshape trunk output
        trunk_reshaped = trunk_output.view(-1, self.latent_dim, self.field_components)
        
        # Compute dot product for each field component
        fields = torch.zeros(material_params.shape[0], self.field_components, 
                           device=material_params.device)
        
        for i in range(self.field_components):
            fields[:, i] = torch.sum(branch_output * trunk_reshaped[:, :, i], dim=1)
        
        return fields + self.bias


class MultiFrequencyPINN(nn.Module):
    """
    Multi-frequency PINN for broadband electromagnetic problems.
    
    Args:
        frequency_range: (min_freq, max_freq) in Hz
        num_frequency_modes: Number of frequency sampling points
        **kwargs: Additional PINN parameters
    """
    
    def __init__(self,
                 frequency_range: Tuple[float, float] = (1e14, 1e16),
                 num_frequency_modes: int = 10,
                 spatial_dim: int = 3,
                 **kwargs):
        super().__init__()
        
        self.freq_min, self.freq_max = frequency_range
        self.num_modes = num_frequency_modes
        
        # Sample frequencies
        frequencies = torch.logspace(
            np.log10(self.freq_min), np.log10(self.freq_max), num_frequency_modes
        )
        self.register_buffer('frequencies', frequencies)
        
        # Individual networks for each frequency
        self.frequency_networks = nn.ModuleList([
            ElectromagneticPINN(spatial_dim=spatial_dim, frequency=freq.item(), **kwargs)
            for freq in frequencies
        ])
        
        # Frequency interpolation network
        self.freq_interpolator = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, num_frequency_modes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, coords: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with frequency interpolation.
        
        Args:
            coords: Spatial coordinates [batch_size, spatial_dim]
            frequency: Frequency values [batch_size, 1]
            
        Returns:
            Interpolated electromagnetic fields
        """
        # Normalise frequency
        freq_norm = (torch.log10(frequency) - np.log10(self.freq_min)) / \
                   (np.log10(self.freq_max) - np.log10(self.freq_min))
        
        # Get interpolation weights
        weights = self.freq_interpolator(freq_norm)
        
        # Evaluate all frequency networks
        outputs = []
        for net in self.frequency_networks:
            outputs.append(net(coords))
        
        # Weighted combination
        weighted_output = torch.zeros_like(outputs[0])
        for i, output in enumerate(outputs):
            weighted_output += weights[:, i:i+1].unsqueeze(-1) * output
        
        return weighted_output


class ComplexPINN(ElectromagneticPINN):
    """
    Simplified interface for complex-valued electromagnetic PINNs.
    """
    
    def __init__(self, **kwargs):
        kwargs.setdefault('complex_valued', True)
        kwargs.setdefault('field_components', 6)
        kwargs.setdefault('activation_type', 'complex_tanh')
        super().__init__(**kwargs)
    
    def compute_em_derivatives(self, 
                              coords: torch.Tensor,
                              field_component: int,
                              spatial_derivative: int) -> torch.Tensor:
        """
        Compute electromagnetic field derivatives.
        
        Args:
            coords: Input coordinates
            field_component: Which field component (0-5 for Ex,Ey,Ez,Hx,Hy,Hz)
            spatial_derivative: Which spatial derivative (0,1,2 for x,y,z)
            
        Returns:
            Complex derivative [batch_size, 2] (real, imag)
        """
        coords.requires_grad_(True)
        fields = self(coords)
        
        # Extract specific field component
        field = fields[:, field_component, :]  # [batch, 2] for real/imag
        
        # Compute derivatives for both real and imaginary parts
        real_grad = torch.autograd.grad(
            outputs=field[:, 0].sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True
        )[0][:, spatial_derivative]
        
        imag_grad = torch.autograd.grad(
            outputs=field[:, 1].sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True
        )[0][:, spatial_derivative]
        
        return torch.stack([real_grad, imag_grad], dim=1)