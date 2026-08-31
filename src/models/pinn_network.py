"""
Electromagnetic Physics-Informed Neural Network architectures

Implements PINN architectures specifically designed for electromagnetic
problems, including complex-valued fields, metamaterial systems, and
Surface Plasmon Polariton (SPP) modeling.
"""

import cmath
import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import C0, ETA0


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

    ``encoding_size // 2`` wave-vectors are sampled; each contributes a sine and
    a cosine feature, so the encoding has ``2 * (encoding_size // 2)`` Fourier
    features (equal to ``encoding_size`` when it is even, one fewer when odd)
    plus ``input_dim`` raw coordinates if ``include_dc``. Use :attr:`output_dim`
    to size the following layer.

    Args:
        input_dim: Spatial input dimension
        encoding_size: Requested number of Fourier features
        frequency_range: (k_min, k_max) wavenumber range in absolute units (rad/m)
        include_dc: Whether to pass the raw coordinates through as well
    """

    def __init__(self,
                 input_dim: int,
                 encoding_size: int = 128,
                 frequency_range: Tuple[float, float] = (0.1, 10.0),
                 include_dc: bool = True,
                 dc_scale: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.include_dc = include_dc
        # The raw ("DC") coordinate channel is divided by this so that physical
        # coordinates of order 1e-6 m still enter the MLP as O(1) inputs.
        self.dc_scale = float(dc_scale)
        self.num_modes = encoding_size // 2
        self.encoding_size = 2 * self.num_modes

        # Sample frequencies for electromagnetic problems
        k_min, k_max = frequency_range
        k_values = torch.logspace(np.log10(k_min), np.log10(k_max), self.num_modes)

        # Random directions for 2D/3D problems
        if input_dim == 2:
            angles = torch.rand(self.num_modes) * 2 * np.pi
            k_vectors = torch.stack([k_values * torch.cos(angles),
                                   k_values * torch.sin(angles)], dim=1)
        elif input_dim == 3:
            # Spherical sampling
            theta = torch.rand(self.num_modes) * np.pi
            phi = torch.rand(self.num_modes) * 2 * np.pi
            k_vectors = torch.stack([
                k_values * torch.sin(theta) * torch.cos(phi),
                k_values * torch.sin(theta) * torch.sin(phi),
                k_values * torch.cos(theta)
            ], dim=1)
        else:
            # Generic dimension: random directions on the unit sphere
            directions = torch.randn(self.num_modes, input_dim)
            directions = directions / directions.norm(dim=1, keepdim=True)
            k_vectors = k_values.unsqueeze(1) * directions

        self.register_buffer('k_vectors', k_vectors)

    @property
    def output_dim(self) -> int:
        """Number of features produced by :meth:`forward`."""
        return self.encoding_size + (self.input_dim if self.include_dc else 0)

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
            features = torch.cat([coords / self.dc_scale, features], dim=1)

        return features


class ElectromagneticPINN(nn.Module):
    """
    PINN for electromagnetic problems with complex-valued outputs.

    Input layout: ``coords`` has shape ``(N, spatial_dim)`` when ``frequency`` is
    given (time-harmonic problem). When ``frequency is None`` the problem is
    treated as time-dependent and the network expects one extra trailing
    column, ``coords = [x, (y, z), t]`` of shape ``(N, spatial_dim + 1)``.

    Output layout: ``(N, field_components, 2)`` with ``[Re, Im]`` on the last
    axis and components ordered ``[Ex, Ey, Ez, Hx, Hy, Hz]``.

    Args:
        spatial_dim: Spatial dimension (2 or 3)
        field_components: Number of field components (3 or 6)
        hidden_dims: Hidden layer dimensions
        complex_valued: Whether the hidden layers are complex-valued
        frequency: Operating angular frequency; ``None`` selects the time-dependent input layout
        use_fourier: Whether to use Fourier features
        fourier_modes: Requested number of Fourier features (rounded down to even)
        activation_type: Type of activation function
    """

    def __init__(self,
                 spatial_dim: int = 3,
                 field_components: int = 6,  # Ex, Ey, Ez, Hx, Hy, Hz
                 hidden_dims: List[int] = None,
                 complex_valued: bool = True,
                 frequency: Optional[float] = None,
                 use_fourier: bool = True,
                 fourier_modes: int = 128,
                 activation_type: str = 'complex_tanh',
                 coord_scale: float = 1.0,
                 fourier_k_range: Tuple[float, float] = (0.1, 20.0),
                 **kwargs):
        if hidden_dims is None:
            hidden_dims = [128, 128, 128, 128]
        super().__init__()

        self.spatial_dim = spatial_dim
        self.field_components = field_components
        self.complex_valued = complex_valued
        self.frequency = frequency

        # Length scale dividing raw coordinates before the MLP (default 1: the
        # caller feeds O(1) coordinates, e.g. via :class:`NondimensionalPINN`).
        self.coord_scale = float(coord_scale)

        # Input dimension
        input_dim = spatial_dim
        if frequency is None:  # Time-dependent problem
            input_dim += 1

        # Fourier feature encoding. ``fourier_k_range`` is in rad per unit of the
        # *input* coordinates: with dimensionless inputs x/λ the target wavenumber
        # is 2π, inside the default (0.1, 20) band. For raw SI coordinates pass
        # e.g. ``fourier_k_range=(0.1 * k0, 20 * k0)`` or wrap the network in
        # :class:`NondimensionalPINN` (preferred).
        if use_fourier:
            self.fourier_encoder = FourierEMFeatures(
                input_dim, fourier_modes, fourier_k_range, dc_scale=self.coord_scale
            )
            first_layer_input = self.fourier_encoder.output_dim
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
            x = coords / self.coord_scale

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

    Sign convention ``exp(-iωt)``: the SPP wavevector is chosen with
    ``Im(k_spp) > 0`` so the mode decays along its propagation direction.

    Units contract: ``interface_position`` and ``decay_length`` must be expressed
    in the SAME units as the coordinates passed to ``forward``. With raw SI
    coordinates that is metres; when the network is driven with scaled
    coordinates (e.g. wrapped in :class:`NondimensionalPINN`, which feeds the
    core coords/length_scale), pass both in those scaled units and set
    ``coord_scale`` accordingly so the internal MLPs also see O(1) inputs.

    Args:
        interface_position: z-coordinate of metal-dielectric interface
            (units of the input coordinates)
        metal_permittivity: Complex permittivity of metal (``Im > 0`` for loss)
        dielectric_permittivity: Permittivity of dielectric
        frequency: Operating angular frequency (rad/s)
        spatial_dim: Spatial dimension
        decay_length: Envelope decay length imposed away from the interface
            (units of the input coordinates)
        **kwargs: Additional arguments for ElectromagneticPINN (notably
            ``coord_scale`` and ``fourier_k_range``)
    """

    def __init__(self,
                 interface_position: float = 0.0,
                 metal_permittivity: complex = -20 + 1j,
                 dielectric_permittivity: float = 2.25,
                 frequency: float = 1e15,
                 spatial_dim: int = 3,
                 decay_length: float = 1e-6,
                 **kwargs):
        super().__init__(
            spatial_dim=spatial_dim,
            field_components=6,
            complex_valued=True,
            frequency=frequency,
            **kwargs
        )

        self.interface_z = interface_position
        self.eps_metal = complex(metal_permittivity)
        self.eps_dielectric = complex(dielectric_permittivity)
        self.omega = frequency
        self.decay_length = decay_length

        # SPP-specific parameters
        self.k0 = frequency / C0  # Free space wavevector
        self.k_spp = self._calculate_spp_wavevector()

        # Add SPP-specific layers
        self.spp_modulation = nn.Sequential(
            nn.Linear(spatial_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 6),
            nn.Tanh()
        )

    def _calculate_spp_wavevector(self) -> torch.Tensor:
        """
        SPP wavevector ``k0 sqrt(ε_m ε_d / (ε_m + ε_d))`` as a 0-d complex tensor,
        on the branch with ``Im(k_spp) >= 0`` (``exp(-iωt)`` convention).
        """
        eps_m, eps_d = self.eps_metal, self.eps_dielectric
        k_spp = self.k0 * cmath.sqrt(eps_m * eps_d / (eps_m + eps_d))
        if k_spp.imag < 0 or (k_spp.imag == 0 and k_spp.real < 0):
            k_spp = -k_spp
        return torch.complex(
            torch.tensor(k_spp.real, dtype=torch.float32),
            torch.tensor(k_spp.imag, dtype=torch.float32),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward pass with SPP-specific modifications."""
        # Base electromagnetic network
        base_fields = super().forward(coords)

        # SPP-specific modulation (scaled coords so the tanh MLP sees O(1) inputs)
        spp_mod = self.spp_modulation(coords / self.coord_scale)

        # Distance from the interface
        z_coords = coords[:, 2] if coords.shape[1] > 2 else coords[:, 1]
        z_rel = z_coords - self.interface_z

        # Modulate fields based on distance from interface (out-of-place)
        decay_factor = torch.exp(-torch.abs(z_rel) / self.decay_length)  # (N,)
        envelope = decay_factor.unsqueeze(-1) * (1 + 0.1 * spp_mod)  # (N, 6)
        if self.complex_valued:
            modulated_fields = base_fields * envelope.unsqueeze(-1)
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
                 branch_hidden: List[int] = None,
                 trunk_hidden: List[int] = None,
                 latent_dim: int = 128,
                 **kwargs):
        if trunk_hidden is None:
            trunk_hidden = [128, 128, 128]
        if branch_hidden is None:
            branch_hidden = [128, 128, 128]
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



class NondimensionalPINN(nn.Module):
    """
    Wrap a field network so it can be trained and evaluated in SI units while the
    core MLP sees O(1) inputs and produces O(1) outputs.

    ``forward(coords_SI) = core(coords_SI / length_scale) * field_scale``

    ``length_scale`` is typically the free-space wavelength and ``field_scale`` a
    per-component amplitude, e.g. ``[E0]*3 + [E0 / ETA0]*3`` so that E and H are
    comparable inside the network. Exposes ``get_fields`` like the core.
    """

    def __init__(self, core: nn.Module, length_scale: float, field_scale):
        super().__init__()
        self.core = core
        self.length_scale = float(length_scale)
        scale = torch.as_tensor(field_scale, dtype=torch.float32).reshape(1, -1, 1)
        self.register_buffer("field_scale", scale)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.core(coords / self.length_scale) * self.field_scale

    def get_fields(self, coords: torch.Tensor):
        out = self.forward(coords)
        return out[:, :3], out[:, 3:6]

    @staticmethod
    def em_field_scale(e0: float = 1.0) -> list:
        """Field scale for a 6-component EM network: ``E ~ e0``, ``H ~ e0 / η0``."""
        return [e0] * 3 + [e0 / ETA0] * 3

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
