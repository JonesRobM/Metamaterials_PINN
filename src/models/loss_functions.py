"""
Loss functions for Electromagnetic Physics-Informed Neural Networks (PINNs)

Implements loss components for Maxwell's equations in metamaterial systems,
specifically designed for Surface Plasmon Polariton (SPP) modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import numpy as np

# Import physics modules from the project
try:
    from ..physics.maxwell_equations import MaxwellSolver
    from ..physics.metamaterial import MetamaterialProperties
    from ..physics.boundary_conditions import BoundaryConditionSolver
except ImportError:
    # Fallback for testing
    MaxwellSolver = None
    MetamaterialProperties = None
    BoundaryConditionSolver = None


class BaseLoss(ABC):
    """Abstract base class for electromagnetic PINN loss components."""
    
    def __init__(self, weight: float = 1.0):
        self.weight = weight
    
    @abstractmethod
    def compute(self, *args, **kwargs) -> torch.Tensor:
        """Compute the loss value."""
        pass
    
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.weight * self.compute(*args, **kwargs)


class MaxwellCurlLoss(BaseLoss):
    """
    Maxwell curl equation loss: ∇×E = -iωB and ∇×H = iωD + J
    
    Args:
        frequency: Angular frequency ω
        mu0: Permeability of free space
        weight: Loss weight
    """
    
    def __init__(self, 
                 frequency: float,
                 mu0: float = 4e-7 * np.pi,
                 weight: float = 1.0):
        super().__init__(weight)
        self.omega = frequency
        self.mu0 = mu0
        self.maxwell_solver = MaxwellSolver() if MaxwellSolver else None
    
    def compute(self, 
                network: nn.Module,
                coords: torch.Tensor,
                material_props: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """
        Compute Maxwell curl equation residuals.
        
        Args:
            network: Neural network (outputs [Ex, Ey, Ez, Hx, Hy, Hz])
            coords: Spatial coordinates [batch_size, spatial_dim]
            material_props: Material properties at coordinates
            
        Returns:
            Curl equation loss
        """
        coords.requires_grad_(True)
        fields = network(coords)  # [batch_size, 6] for Ex,Ey,Ez,Hx,Hy,Hz
        
        # Split electromagnetic fields
        E = fields[:, :3]  # Electric field components
        H = fields[:, 3:]  # Magnetic field components
        
        # Compute curl of E and H using automatic differentiation
        curl_E = self._compute_curl(E, coords)
        curl_H = self._compute_curl(H, coords)
        
        # Maxwell's equations in frequency domain
        # ∇×E = -iωμH
        # ∇×H = iωεE (assuming no current density)
        
        if material_props is not None:
            mu_r = material_props[:, 0:1]  # Relative permeability
            eps_r = material_props[:, 1:4]  # Relative permittivity (tensor)
        else:
            mu_r = torch.ones_like(H[:, 0:1])
            eps_r = torch.ones_like(E)
        
        # Maxwell curl residuals
        residual_E = curl_E + 1j * self.omega * self.mu0 * mu_r * H
        residual_H = curl_H - 1j * self.omega * self.mu0 * eps_r * E
        
        # Combine residuals (take real part of squared magnitude)
        loss_E = torch.mean(torch.real(residual_E * torch.conj(residual_E)))
        loss_H = torch.mean(torch.real(residual_H * torch.conj(residual_H)))
        
        return loss_E + loss_H
    
    def _compute_curl(self, field: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Compute curl using automatic differentiation."""
        batch_size = field.shape[0]
        curl = torch.zeros_like(field)
        
        # Extract field components
        Fx, Fy, Fz = field[:, 0], field[:, 1], field[:, 2]
        
        # Compute partial derivatives
        # ∂Fz/∂y - ∂Fy/∂z for curl_x
        dFz_dy = torch.autograd.grad(
            outputs=Fz.sum(), inputs=coords,
            create_graph=True, retain_graph=True
        )[0][:, 1] if coords.shape[1] > 1 else torch.zeros_like(Fz)
        
        dFy_dz = torch.autograd.grad(
            outputs=Fy.sum(), inputs=coords,
            create_graph=True, retain_graph=True
        )[0][:, 2] if coords.shape[1] > 2 else torch.zeros_like(Fy)
        
        curl[:, 0] = dFz_dy - dFy_dz
        
        # ∂Fx/∂z - ∂Fz/∂x for curl_y
        dFx_dz = torch.autograd.grad(
            outputs=Fx.sum(), inputs=coords,
            create_graph=True, retain_graph=True
        )[0][:, 2] if coords.shape[1] > 2 else torch.zeros_like(Fx)
        
        dFz_dx = torch.autograd.grad(
            outputs=Fz.sum(), inputs=coords,
            create_graph=True, retain_graph=True
        )[0][:, 0]
        
        curl[:, 1] = dFx_dz - dFz_dx
        
        # ∂Fy/∂x - ∂Fx/∂y for curl_z
        dFy_dx = torch.autograd.grad(
            outputs=Fy.sum(), inputs=coords,
            create_graph=True, retain_graph=True
        )[0][:, 0]
        
        dFx_dy = torch.autograd.grad(
            outputs=Fx.sum(), inputs=coords,
            create_graph=True, retain_graph=True
        )[0][:, 1] if coords.shape[1] > 1 else torch.zeros_like(Fx)
        
        curl[:, 2] = dFy_dx - dFx_dy
        
        return curl


class MaxwellDivergenceLoss(BaseLoss):
    """
    Maxwell divergence constraints: ∇·D = ρ and ∇·B = 0
    
    Args:
        weight: Loss weight
    """
    
    def compute(self, 
                network: nn.Module,
                coords: torch.Tensor,
                material_props: Optional[torch.Tensor] = None,
                charge_density: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """Compute divergence constraint residuals."""
        coords.requires_grad_(True)
        fields = network(coords)
        
        E = fields[:, :3]
        H = fields[:, 3:]
        
        # Compute divergences
        div_E = self._compute_divergence(E, coords)
        div_H = self._compute_divergence(H, coords)
        
        # Apply material properties
        if material_props is not None:
            eps_r = material_props[:, 1:4]
            D = eps_r * E
            div_D = self._compute_divergence(D, coords)
        else:
            div_D = div_E
        
        # Divergence constraints
        if charge_density is not None:
            residual_D = div_D - charge_density
        else:
            residual_D = div_D  # Assume no free charges
            
        residual_B = div_H  # ∇·B = 0 always
        
        return torch.mean(residual_D**2) + torch.mean(residual_B**2)
    
    def _compute_divergence(self, field: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Compute divergence using automatic differentiation."""
        div = torch.zeros(field.shape[0], 1, device=field.device, dtype=field.dtype)
        
        for i in range(min(3, coords.shape[1])):
            if i < field.shape[1]:  # Make sure field has this component
                grad = torch.autograd.grad(
                    outputs=field[:, i].sum(),
                    inputs=coords,
                    create_graph=True,
                    retain_graph=True
                )[0][:, i]
                div[:, 0] += grad
        
        return div


class MetamaterialConstitutiveLoss(BaseLoss):
    """
    Enforce metamaterial constitutive relations: D = ε·E, B = μ·H
    
    Args:
        metamaterial_solver: Metamaterial properties calculator
        weight: Loss weight
    """
    
    def __init__(self, 
                 metamaterial_solver: Optional[object] = None,
                 weight: float = 1.0):
        super().__init__(weight)
        self.metamaterial = metamaterial_solver or MetamaterialProperties()
    
    def compute(self, 
                network: nn.Module,
                coords: torch.Tensor,
                frequency: float,
                **kwargs) -> torch.Tensor:
        """Enforce constitutive relations in metamaterial regions."""
        fields = network(coords)
        E = fields[:, :3]
        H = fields[:, 3:]
        
        # Get material properties from metamaterial solver
        if hasattr(self.metamaterial, 'get_permittivity_tensor'):
            eps_tensor = self.metamaterial.get_permittivity_tensor(coords, frequency)
            mu_tensor = self.metamaterial.get_permeability_tensor(coords, frequency)
        else:
            # Fallback: assume isotropic metamaterial
            eps_tensor = torch.eye(3).unsqueeze(0).repeat(coords.shape[0], 1, 1)
            mu_tensor = torch.eye(3).unsqueeze(0).repeat(coords.shape[0], 1, 1)
        
        # Apply constitutive relations
        D_expected = torch.bmm(eps_tensor, E.unsqueeze(-1)).squeeze(-1)
        B_expected = torch.bmm(mu_tensor, H.unsqueeze(-1)).squeeze(-1)
        
        # For complex metamaterials, D and B might be additional network outputs
        # or computed from auxiliary equations
        D_network = E  # Placeholder - modify based on network architecture
        B_network = H  # Placeholder - modify based on network architecture
        
        residual_D = D_network - D_expected
        residual_B = B_network - B_expected
        
        return torch.mean(torch.abs(residual_D)**2) + torch.mean(torch.abs(residual_B)**2)


class InterfaceBoundaryLoss(BaseLoss):
    """
    Enforce boundary conditions at metamaterial-dielectric interfaces.
    
    Args:
        boundary_solver: Boundary condition calculator
        interface_coords: Coordinates of interface points
        weight: Loss weight
    """
    
    def __init__(self, 
                 boundary_solver: Optional[object] = None,
                 interface_coords: Optional[torch.Tensor] = None,
                 weight: float = 1.0):
        super().__init__(weight)
        self.boundary_solver = boundary_solver or BoundaryConditionSolver()
        self.interface_coords = interface_coords
    
    def compute(self, 
                network: nn.Module,
                coords: torch.Tensor,
                **kwargs) -> torch.Tensor:
        """Enforce interface boundary conditions."""
        if self.interface_coords is None:
            return torch.tensor(0.0, device=coords.device)
        
        # Evaluate fields at interface
        fields_interface = network(self.interface_coords)
        E_interface = fields_interface[:, :3]
        H_interface = fields_interface[:, 3:]
        
        # Get interface normal vectors and material properties
        if hasattr(self.boundary_solver, 'get_interface_conditions'):
            residuals = self.boundary_solver.get_interface_conditions(
                E_interface, H_interface, self.interface_coords
            )
        else:
            # Simplified tangential continuity
            residuals = torch.zeros_like(E_interface)
        
        return torch.mean(torch.abs(residuals)**2)


class SPPBoundaryLoss(BaseLoss):
    """
    Specific boundary conditions for Surface Plasmon Polaritons.
    
    Args:
        spp_wavevector: Expected SPP wavevector
        decay_length: Expected decay length in z-direction
        weight: Loss weight
    """
    
    def __init__(self, 
                 spp_wavevector: float,
                 decay_length: float = 1e-6,
                 weight: float = 1.0):
        super().__init__(weight)
        self.k_spp = spp_wavevector
        self.decay_length = decay_length
    
    def compute(self, 
                network: nn.Module,
                coords: torch.Tensor,
                **kwargs) -> torch.Tensor:
        """Enforce SPP boundary conditions."""
        fields = network(coords)
        E = fields[:, :3]
        
        # SPP should decay exponentially away from interface
        z_coords = coords[:, 2] if coords.shape[1] > 2 else coords[:, 1]
        
        # Expected decay behavior |E| ∝ exp(-|z|/δ)
        expected_decay = torch.exp(-torch.abs(z_coords) / self.decay_length)
        field_magnitude = torch.norm(E, dim=1)
        
        # Normalize by maximum field to get relative decay
        max_field = torch.max(field_magnitude)
        if max_field > 0:
            normalized_field = field_magnitude / max_field
            decay_residual = normalized_field - expected_decay
        else:
            decay_residual = torch.zeros_like(field_magnitude)
        
        return torch.mean(decay_residual**2)


class TangentialContinuityLoss(BaseLoss):
    """
    Enforce tangential field continuity at interfaces: n × (E2 - E1) = 0
    """
    
    def compute(self, 
                network: nn.Module,
                interface_coords: torch.Tensor,
                normal_vectors: torch.Tensor,
                **kwargs) -> torch.Tensor:
        """Enforce tangential continuity."""
        # Evaluate fields on both sides of interface
        eps = 1e-6
        coords_plus = interface_coords + eps * normal_vectors
        coords_minus = interface_coords - eps * normal_vectors
        
        fields_plus = network(coords_plus)
        fields_minus = network(coords_minus)
        
        E_plus = fields_plus[:, :3]
        E_minus = fields_minus[:, :3]
        H_plus = fields_plus[:, 3:]
        H_minus = fields_minus[:, 3:]
        
        # Tangential continuity: n × (E2 - E1) = 0, n × (H2 - H1) = K_s
        E_jump = E_plus - E_minus
        H_jump = H_plus - H_minus
        
        # Cross product with normal (assuming 3D)
        n_cross_E = torch.cross(normal_vectors, E_jump, dim=1)
        n_cross_H = torch.cross(normal_vectors, H_jump, dim=1)
        
        # For perfect conductor interface, tangential E should be zero
        # For dielectric interface, tangential E should be continuous
        return torch.mean(torch.norm(n_cross_E, dim=1)**2) + \
               torch.mean(torch.norm(n_cross_H, dim=1)**2)


class PowerFlowLoss(BaseLoss):
    """
    Power flow constraint: ∇·S = 0 (Poynting vector divergence)
    """
    
    def compute(self, 
                network: nn.Module,
                coords: torch.Tensor,
                **kwargs) -> torch.Tensor:
        """Enforce power conservation."""
        coords.requires_grad_(True)
        fields = network(coords)
        
        E = fields[:, :3]
        H = fields[:, 3:]
        
        # Poynting vector S = E × H*
        S = torch.cross(E, torch.conj(H), dim=1)
        
        # Compute divergence of Poynting vector
        div_S = self._compute_divergence(S, coords)
        
        # In steady state, ∇·S = 0
        return torch.mean(torch.abs(div_S)**2)
    
    def _compute_divergence(self, field: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Compute divergence using automatic differentiation."""
        div = torch.zeros(field.shape[0], 1, device=field.device, dtype=field.dtype)
        
        for i in range(min(3, coords.shape[1])):
            if i < field.shape[1]:  # Make sure field has this component
                grad = torch.autograd.grad(
                    outputs=field[:, i].sum(),
                    inputs=coords,
                    create_graph=True,
                    retain_graph=True
                )[0][:, i]
                div[:, 0] += grad
        
        return div


class WaveguideLoss(BaseLoss):
    """
    Waveguide mode constraints for guided SPP modes.
    """
    
    def __init__(self, 
                 propagation_direction: int = 0,  # x-direction
                 weight: float = 1.0):
        super().__init__(weight)
        self.prop_dir = propagation_direction
    
    def compute(self, 
                network: nn.Module,
                coords: torch.Tensor,
                beta: float,  # Propagation constant
                **kwargs) -> torch.Tensor:
        """Enforce guided mode behavior."""
        coords.requires_grad_(True)
        fields = network(coords)
        
        # For guided modes, expect exp(iβx) dependence
        x_coords = coords[:, self.prop_dir]
        
        # Compute phase derivative
        phase_grad = torch.autograd.grad(
            outputs=torch.angle(fields[:, 0]).sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True
        )[0][:, self.prop_dir]
        
        # Expected phase gradient is β
        phase_residual = phase_grad - beta
        
        return torch.mean(phase_residual**2)


class RadiationLoss(BaseLoss):
    """
    Radiation boundary condition for open boundaries.
    """
    
    def compute(self, 
                network: nn.Module,
                boundary_coords: torch.Tensor,
                k0: float,  # Free space wavevector
                **kwargs) -> torch.Tensor:
        """Enforce radiation boundary conditions."""
        boundary_coords.requires_grad_(True)
        fields = network(boundary_coords)
        
        E = fields[:, :3]
        H = fields[:, 3:]
        
        # Sommerfeld radiation condition
        # ∂E/∂r - ik₀E = 0 at boundary
        r = torch.norm(boundary_coords, dim=1, keepdim=True)
        
        # Radial derivative
        radial_grad = torch.autograd.grad(
            outputs=torch.norm(E, dim=1).sum(),
            inputs=boundary_coords,
            create_graph=True,
            retain_graph=True
        )[0]
        
        radial_dir = boundary_coords / (r + 1e-8)
        radial_derivative = torch.sum(radial_grad * radial_dir, dim=1, keepdim=True)
        
        # Radiation condition residual
        E_magnitude = torch.norm(E, dim=1, keepdim=True)
        radiation_residual = radial_derivative - 1j * k0 * E_magnitude
        
        return torch.mean(torch.abs(radiation_residual)**2)


class EM_CompositeLoss:
    """
    Electromagnetic composite loss combining multiple physics constraints.
    
    Args:
        losses: Dictionary of electromagnetic loss components
        adaptive_weights: Whether to use adaptive weighting
        frequency_dependent: Whether to adjust weights based on frequency
    """
    
    def __init__(self, 
                 losses: Dict[str, BaseLoss],
                 adaptive_weights: bool = True,
                 frequency_dependent: bool = False):
        self.losses = losses
        self.adaptive_weights = adaptive_weights
        self.frequency_dependent = frequency_dependent
        self.step_count = 0
        self.loss_history = {name: [] for name in losses.keys()}
        
        # Electromagnetic-specific parameters
        self.field_balance_factor = 1.0  # Balance E and H field losses
        
        if adaptive_weights:
            self.alpha = 0.9  # More conservative for EM problems
            self.running_means = {name: 1.0 for name in losses.keys()}
    
    def compute(self, **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute electromagnetic composite loss."""
        loss_dict = {}
        total_loss = 0.0
        
        # Separate field and physics losses for better balancing
        field_losses = []
        physics_losses = []
        
        for name, loss_fn in self.losses.items():
            component_loss = loss_fn(**kwargs)
            loss_dict[name] = component_loss
            
            if 'curl' in name.lower() or 'divergence' in name.lower():
                physics_losses.append(component_loss)
            else:
                field_losses.append(component_loss)
            
            self.loss_history[name].append(component_loss.item())
        
        # Balance field and physics contributions
        if field_losses and physics_losses:
            field_total = sum(field_losses)
            physics_total = sum(physics_losses)
            
            # Adaptive balancing
            if physics_total > 0:
                balance_factor = field_total / (physics_total + 1e-8)
                physics_total *= min(balance_factor, 10.0)  # Cap the scaling
        
        total_loss = sum(loss_dict.values())
        
        # Update adaptive weights
        if self.adaptive_weights and self.step_count % 50 == 0:
            self._update_adaptive_weights()
        
        self.step_count += 1
        return total_loss, loss_dict
    
    def _update_adaptive_weights(self):
        """Update weights with electromagnetic-specific considerations."""
        # Similar to base class but with EM-specific adjustments
        for name in self.losses.keys():
            if self.loss_history[name]:
                recent_loss = np.mean(self.loss_history[name][-10:])
                self.running_means[name] = (
                    self.alpha * self.running_means[name] + 
                    (1 - self.alpha) * recent_loss
                )
        
        # Adjust weights to maintain physics-data balance
        mean_loss = np.mean(list(self.running_means.values()))
        
        for name, loss_fn in self.losses.items():
            if self.running_means[name] > 0:
                base_weight = mean_loss / self.running_means[name]
                
                # Give higher priority to Maxwell equations
                if 'maxwell' in name.lower() or 'curl' in name.lower():
                    base_weight *= 1.5
                elif 'boundary' in name.lower():
                    base_weight *= 1.2
                
                loss_fn.weight = base_weight
    
    def get_physics_residuals(self) -> Dict[str, float]:
        """Get physics residuals for monitoring."""
        residuals = {}
        for name, history in self.loss_history.items():
            if history:
                residuals[name] = history[-1]
        return residuals