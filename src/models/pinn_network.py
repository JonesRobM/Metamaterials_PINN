"""
Physics-Informed Neural Network architectures

Implements PINN and DeepONet architectures with automatic differentiation
capabilities for solving PDEs and learning operators.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Union, Callable
import math


class FourierFeatures(nn.Module):
    """
    Fourier feature encoding for improving high-frequency learning.
    
    Args:
        input_dim: Input dimension
        encoding_size: Number of Fourier features
        scale: Scale parameter for frequency sampling
        learnable: Whether frequency parameters are learnable
    """
    
    def __init__(self, 
                 input_dim: int,
                 encoding_size: int = 256,
                 scale: float = 1.0,
                 learnable: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.encoding_size = encoding_size
        
        # Sample random frequencies
        if learnable:
            self.frequencies = nn.Parameter(
                torch.randn(encoding_size, input_dim) * scale
            )
        else:
            self.register_buffer(
                'frequencies',
                torch.randn(encoding_size, input_dim) * scale
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier feature encoding.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Encoded features [batch_size, 2 * encoding_size]
        """
        # x: [batch_size, input_dim]
        # frequencies: [encoding_size, input_dim]
        freq_proj = torch.matmul(x, self.frequencies.T)  # [batch_size, encoding_size]
        
        # Concatenate sine and cosine features
        return torch.cat([torch.sin(freq_proj), torch.cos(freq_proj)], dim=-1)


class ResidualBlock(nn.Module):
    """
    Residual block with layer normalisation and skip connections.
    
    Args:
        hidden_dim: Hidden layer dimension
        activation: Activation function
        dropout: Dropout probability
    """
    
    def __init__(self, 
                 hidden_dim: int,
                 activation: nn.Module = nn.Tanh(),
                 dropout: float = 0.0):
        super().__init__()
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x
        
        out = self.layer1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.layer2(out)
        out = self.norm2(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        return out + residual


class PINNNetwork(nn.Module):
    """
    Physics-Informed Neural Network with flexible architecture.
    
    Args:
        input_dim: Input dimension (spatial + temporal coordinates)
        output_dim: Output dimension (field components)
        hidden_dims: List of hidden layer dimensions
        activation: Activation function
        use_fourier: Whether to use Fourier feature encoding
        fourier_features: Number of Fourier features
        fourier_scale: Scale for Fourier frequencies
        use_residual: Whether to use residual blocks
        dropout: Dropout probability
        batch_norm: Whether to use batch normalisation
        final_activation: Final layer activation
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: List[int] = [128, 128, 128, 128],
                 activation: nn.Module = nn.Tanh(),
                 use_fourier: bool = False,
                 fourier_features: int = 256,
                 fourier_scale: float = 1.0,
                 use_residual: bool = False,
                 dropout: float = 0.0,
                 batch_norm: bool = False,
                 final_activation: Optional[nn.Module] = None):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_fourier = use_fourier
        self.use_residual = use_residual
        
        # Fourier feature encoding
        if use_fourier:
            self.fourier_encoder = FourierFeatures(
                input_dim, fourier_features, fourier_scale
            )
            first_dim = 2 * fourier_features
        else:
            self.fourier_encoder = None
            first_dim = input_dim
        
        # Build network layers
        layers = []
        dims = [first_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            if use_residual and i > 0 and i < len(dims) - 2 and dims[i] == dims[i+1]:
                # Use residual block for same-dimension hidden layers
                layers.append(ResidualBlock(dims[i], activation, dropout))
            else:
                # Standard linear layer
                layers.append(nn.Linear(dims[i], dims[i+1]))
                
                # Add normalisation (except for final layer)
                if i < len(dims) - 2:
                    if batch_norm:
                        layers.append(nn.BatchNorm1d(dims[i+1]))
                    
                    # Add activation (except for final layer)
                    layers.append(activation)
                    
                    # Add dropout
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))
        
        # Add final activation if specified
        if final_activation is not None:
            layers.append(final_activation)
        
        self.network = nn.Sequential(*layers)
        
        # Initialise weights
        self._initialise_weights()
    
    def _initialise_weights(self):
        """Xavier normal initialisation for better gradient flow."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input coordinates [batch_size, input_dim]
            
        Returns:
            Network output [batch_size, output_dim]
        """
        if self.use_fourier:
            x = self.fourier_encoder(x)
        
        return self.network(x)
    
    def compute_derivatives(self, 
                          x: torch.Tensor, 
                          order: int = 1,
                          dim: Optional[int] = None) -> torch.Tensor:
        """
        Compute derivatives using automatic differentiation.
        
        Args:
            x: Input coordinates [batch_size, input_dim]
            order: Derivative order
            dim: Specific dimension for partial derivative (None for all)
            
        Returns:
            Derivative tensor
        """
        x.requires_grad_(True)
        output = self(x)
        
        if order == 1:
            if dim is not None:
                # Partial derivative w.r.t. specific dimension
                return torch.autograd.grad(
                    outputs=output.sum(),
                    inputs=x,
                    create_graph=True,
                    retain_graph=True
                )[0][:, dim:dim+1]
            else:
                # Gradient w.r.t. all dimensions
                return torch.autograd.grad(
                    outputs=output.sum(),
                    inputs=x,
                    create_graph=True,
                    retain_graph=True
                )[0]
        else:
            # Higher-order derivatives (recursive)
            grad = self.compute_derivatives(x, order-1, dim)
            return torch.autograd.grad(
                outputs=grad.sum(),
                inputs=x,
                create_graph=True,
                retain_graph=True
            )[0]


class DeepONet(nn.Module):
    """
    Deep Operator Network for learning operators between function spaces.
    
    Args:
        branch_input_dim: Input dimension for branch network
        trunk_input_dim: Input dimension for trunk network  
        branch_hidden_dims: Hidden dimensions for branch network
        trunk_hidden_dims: Hidden dimensions for trunk network
        latent_dim: Latent space dimension
        activation: Activation function
        dropout: Dropout probability
    """
    
    def __init__(self,
                 branch_input_dim: int,
                 trunk_input_dim: int,
                 branch_hidden_dims: List[int] = [128, 128, 128],
                 trunk_hidden_dims: List[int] = [128, 128, 128],
                 latent_dim: int = 128,
                 activation: nn.Module = nn.Tanh(),
                 dropout: float = 0.0):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Branch network (encodes function input)
        branch_dims = [branch_input_dim] + branch_hidden_dims + [latent_dim]
        branch_layers = []
        for i in range(len(branch_dims) - 1):
            branch_layers.append(nn.Linear(branch_dims[i], branch_dims[i+1]))
            if i < len(branch_dims) - 2:
                branch_layers.append(activation)
                if dropout > 0:
                    branch_layers.append(nn.Dropout(dropout))
        
        self.branch_net = nn.Sequential(*branch_layers)
        
        # Trunk network (encodes evaluation points)
        trunk_dims = [trunk_input_dim] + trunk_hidden_dims + [latent_dim]
        trunk_layers = []
        for i in range(len(trunk_dims) - 1):
            trunk_layers.append(nn.Linear(trunk_dims[i], trunk_dims[i+1]))
            if i < len(trunk_dims) - 2:
                trunk_layers.append(activation)
                if dropout > 0:
                    trunk_layers.append(nn.Dropout(dropout))
        
        self.trunk_net = nn.Sequential(*trunk_layers)
        
        # Bias term
        self.bias = nn.Parameter(torch.zeros(1))
        
        # Initialise weights
        self._initialise_weights()
    
    def _initialise_weights(self):
        """Xavier normal initialisation."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, 
                branch_input: torch.Tensor,
                trunk_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DeepONet.
        
        Args:
            branch_input: Function input [batch_size, branch_input_dim]
            trunk_input: Evaluation points [batch_size, trunk_input_dim]
            
        Returns:
            Operator output [batch_size, 1]
        """
        # Encode inputs
        branch_output = self.branch_net(branch_input)  # [batch_size, latent_dim]
        trunk_output = self.trunk_net(trunk_input)     # [batch_size, latent_dim]
        
        # Compute dot product and add bias
        output = torch.sum(branch_output * trunk_output, dim=1, keepdim=True)
        output = output + self.bias
        
        return output


class MultiScalePINN(nn.Module):
    """
    Multi-scale PINN for handling problems with multiple characteristic scales.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension  
        scale_factors: List of scale factors for different sub-networks
        hidden_dims: Hidden dimensions for each sub-network
        activation: Activation function
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 scale_factors: List[float] = [1.0, 10.0, 100.0],
                 hidden_dims: List[int] = [64, 64, 64],
                 activation: nn.Module = nn.Tanh()):
        super().__init__()
        
        self.scale_factors = scale_factors
        self.num_scales = len(scale_factors)
        
        # Create sub-networks for different scales
        self.sub_networks = nn.ModuleList()
        for scale in scale_factors:
            net = PINNNetwork(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                use_fourier=True,
                fourier_scale=scale
            )
            self.sub_networks.append(net)
        
        # Learnable weights for combining scales
        self.scale_weights = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining multiple scales.
        
        Args:
            x: Input coordinates [batch_size, input_dim]
            
        Returns:
            Combined output [batch_size, output_dim]
        """
        outputs = []
        for net in self.sub_networks:
            outputs.append(net(x))
        
        # Weighted combination
        output = torch.zeros_like(outputs[0])
        weights = F.softmax(self.scale_weights, dim=0)
        
        for i, out in enumerate(outputs):
            output += weights[i] * out
        
        return output