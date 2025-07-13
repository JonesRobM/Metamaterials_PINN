"""
Loss functions for Physics-Informed Neural Networks (PINNs)

Implements various loss components commonly used in PINN training,
including PDE residuals, boundary conditions, and regularisation terms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod


class BaseLoss(ABC):
    """Abstract base class for PINN loss components."""
    
    def __init__(self, weight: float = 1.0):
        self.weight = weight
    
    @abstractmethod
    def compute(self, *args, **kwargs) -> torch.Tensor:
        """Compute the loss value."""
        pass
    
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.weight * self.compute(*args, **kwargs)


class PDELoss(BaseLoss):
    """
    PDE residual loss for enforcing physics constraints.
    
    Args:
        pde_func: Function that computes PDE residual given network outputs and inputs
        weight: Weighting factor for this loss component
    """
    
    def __init__(self, pde_func: Callable, weight: float = 1.0):
        super().__init__(weight)
        self.pde_func = pde_func
    
    def compute(self, 
                network: nn.Module, 
                coords: torch.Tensor, 
                **kwargs) -> torch.Tensor:
        """
        Compute PDE residual loss.
        
        Args:
            network: Neural network model
            coords: Input coordinates [batch_size, input_dim]
            
        Returns:
            PDE residual loss
        """
        coords.requires_grad_(True)
        output = network(coords)
        residual = self.pde_func(output, coords, **kwargs)
        return torch.mean(residual**2)


class BoundaryLoss(BaseLoss):
    """
    Boundary condition loss.
    
    Args:
        boundary_func: Function that computes boundary condition residual
        weight: Weighting factor for this loss component
    """
    
    def __init__(self, boundary_func: Callable, weight: float = 1.0):
        super().__init__(weight)
        self.boundary_func = boundary_func
    
    def compute(self, 
                network: nn.Module,
                boundary_coords: torch.Tensor,
                boundary_values: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """
        Compute boundary condition loss.
        
        Args:
            network: Neural network model
            boundary_coords: Boundary coordinates [n_boundary, input_dim]
            boundary_values: Target boundary values [n_boundary, output_dim]
            
        Returns:
            Boundary condition loss
        """
        boundary_coords.requires_grad_(True)
        output = network(boundary_coords)
        
        if boundary_values is not None:
            # Dirichlet boundary conditions
            return F.mse_loss(output, boundary_values)
        else:
            # General boundary condition via function
            residual = self.boundary_func(output, boundary_coords, **kwargs)
            return torch.mean(residual**2)


class InitialConditionLoss(BaseLoss):
    """
    Initial condition loss for time-dependent problems.
    
    Args:
        weight: Weighting factor for this loss component
    """
    
    def compute(self, 
                network: nn.Module,
                initial_coords: torch.Tensor,
                initial_values: torch.Tensor) -> torch.Tensor:
        """
        Compute initial condition loss.
        
        Args:
            network: Neural network model
            initial_coords: Initial time coordinates [n_initial, input_dim]
            initial_values: Initial values [n_initial, output_dim]
            
        Returns:
            Initial condition loss
        """
        output = network(initial_coords)
        return F.mse_loss(output, initial_values)


class DataLoss(BaseLoss):
    """
    Data fitting loss for supervised training components.
    
    Args:
        weight: Weighting factor for this loss component
    """
    
    def compute(self, 
                network: nn.Module,
                data_coords: torch.Tensor,
                data_values: torch.Tensor) -> torch.Tensor:
        """
        Compute data fitting loss.
        
        Args:
            network: Neural network model
            data_coords: Data coordinates [n_data, input_dim]
            data_values: Target data values [n_data, output_dim]
            
        Returns:
            Data fitting loss
        """
        output = network(data_coords)
        return F.mse_loss(output, data_values)


class GradientPenalty(BaseLoss):
    """
    Gradient penalty for network regularisation.
    
    Args:
        penalty_type: Type of gradient penalty ('l2', 'l1', 'lipschitz')
        target_norm: Target gradient norm (for Lipschitz penalty)
        weight: Weighting factor for this loss component
    """
    
    def __init__(self, 
                 penalty_type: str = 'l2',
                 target_norm: float = 1.0,
                 weight: float = 1e-4):
        super().__init__(weight)
        self.penalty_type = penalty_type
        self.target_norm = target_norm
    
    def compute(self, 
                network: nn.Module,
                coords: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient penalty.
        
        Args:
            network: Neural network model
            coords: Input coordinates [batch_size, input_dim]
            
        Returns:
            Gradient penalty loss
        """
        coords.requires_grad_(True)
        output = network(coords)
        
        gradients = torch.autograd.grad(
            outputs=output.sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True
        )[0]
        
        if self.penalty_type == 'l2':
            return torch.mean(gradients**2)
        elif self.penalty_type == 'l1':
            return torch.mean(torch.abs(gradients))
        elif self.penalty_type == 'lipschitz':
            grad_norm = torch.norm(gradients, dim=1)
            return torch.mean((grad_norm - self.target_norm)**2)
        else:
            raise ValueError(f"Unknown penalty type: {self.penalty_type}")


class CausalityLoss(BaseLoss):
    """
    Causality loss for time-dependent problems to enforce temporal ordering.
    
    Args:
        weight: Weighting factor for this loss component
    """
    
    def compute(self, 
                network: nn.Module,
                coords: torch.Tensor,
                time_dim: int = -1) -> torch.Tensor:
        """
        Compute causality loss.
        
        Args:
            network: Neural network model
            coords: Input coordinates [batch_size, input_dim]
            time_dim: Dimension index for time coordinate
            
        Returns:
            Causality loss
        """
        coords.requires_grad_(True)
        output = network(coords)
        
        # Compute time derivative
        dt = torch.autograd.grad(
            outputs=output.sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True
        )[0][:, time_dim]
        
        # Penalise negative time derivatives (anti-causal behaviour)
        causality_violation = F.relu(-dt)
        return torch.mean(causality_violation**2)


class CompositeLoss:
    """
    Composite loss function that combines multiple loss components.
    
    Args:
        losses: Dictionary of loss components with their names as keys
        adaptive_weights: Whether to use adaptive weighting
        weight_update_freq: Frequency of weight updates (in steps)
    """
    
    def __init__(self, 
                 losses: Dict[str, BaseLoss],
                 adaptive_weights: bool = False,
                 weight_update_freq: int = 100):
        self.losses = losses
        self.adaptive_weights = adaptive_weights
        self.weight_update_freq = weight_update_freq
        self.step_count = 0
        self.loss_history = {name: [] for name in losses.keys()}
        
        if adaptive_weights:
            self.alpha = 0.95  # Exponential moving average factor
            self.running_means = {name: 1.0 for name in losses.keys()}
    
    def compute(self, **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss and individual components.
        
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary of individual loss components
        """
        loss_dict = {}
        total_loss = 0.0
        
        for name, loss_fn in self.losses.items():
            component_loss = loss_fn(**kwargs)
            loss_dict[name] = component_loss
            total_loss += component_loss
            
            # Store for adaptive weighting
            if self.adaptive_weights:
                self.loss_history[name].append(component_loss.item())
        
        # Update adaptive weights
        if self.adaptive_weights and self.step_count % self.weight_update_freq == 0:
            self._update_adaptive_weights()
        
        self.step_count += 1
        return total_loss, loss_dict
    
    def _update_adaptive_weights(self):
        """Update loss weights based on relative magnitudes."""
        # Compute running means
        for name in self.losses.keys():
            if self.loss_history[name]:
                recent_loss = self.loss_history[name][-1]
                self.running_means[name] = (
                    self.alpha * self.running_means[name] + 
                    (1 - self.alpha) * recent_loss
                )
        
        # Compute mean of all running means
        mean_loss = sum(self.running_means.values()) / len(self.running_means)
        
        # Update weights to balance loss magnitudes
        for name, loss_fn in self.losses.items():
            if self.running_means[name] > 0:
                new_weight = mean_loss / self.running_means[name]
                loss_fn.weight = new_weight
    
    def get_weights(self) -> Dict[str, float]:
        """Get current loss weights."""
        return {name: loss_fn.weight for name, loss_fn in self.losses.items()}
    
    def set_weights(self, weights: Dict[str, float]):
        """Set loss weights manually."""
        for name, weight in weights.items():
            if name in self.losses:
                self.losses[name].weight = weight