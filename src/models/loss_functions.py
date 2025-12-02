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
    from ..physics.maxwell_equations import MaxwellEquations
    from ..physics.metamaterial import MetamaterialProperties
    from ..physics.boundary_conditions import BoundaryConditions
except ImportError:
    # Fallback for testing when run outside package context (e.g., direct script execution or some IDEs)
    class MaxwellEquations:
        def __init__(self, *args, **kwargs): pass
        def curl_operator(self, *args, **kwargs): return torch.zeros(args[0].shape)
        def curl_E_residual(self, *args, **kwargs): return torch.zeros(args[0].shape[0], 6)
        def curl_H_residual(self, *args, **kwargs): return torch.zeros(args[0].shape[0], 6)
        def divergence_E_residual(self, *args, **kwargs): return torch.zeros(args[0].shape[0], 2)
        def divergence_B_residual(self, *args, **kwargs): return torch.zeros(args[0].shape[0], 2)
    
    class MetamaterialProperties:
        def __init__(self, *args, **kwargs): pass
        def permittivity_tensor(self, *args, **kwargs): return torch.eye(3).unsqueeze(0).expand(args[0].shape[0], -1, -1)
    
    class BoundaryConditions:
        def __init__(self, *args, **kwargs): pass
        def tangential_E_continuity(self, *args, **kwargs): return torch.zeros(args[0].shape[0], 6)


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
                 eps0: float = 8.854e-12,
                 weight: float = 1.0):
        super().__init__(weight)
        self.omega = frequency
        self.mu0 = mu0
        self.eps0 = eps0
        self.maxwell_solver = MaxwellEquations(frequency)
    
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
        
        # Split electromagnetic fields and convert to complex dtype
        E_complex = torch.complex(E[..., 0], E[..., 1])
        H_complex = torch.complex(H[..., 0], H[..., 1])
        
        # Compute curl of E and H using automatic differentiation
        # _compute_curl returns (batch_size, 3, 2) float32, so convert to complex
        curl_E_float = self._compute_curl(E, coords)
        curl_H_float = self._compute_curl(H, coords)
        
        curl_E_complex = torch.complex(curl_E_float[..., 0], curl_E_float[..., 1])
        curl_H_complex = torch.complex(curl_H_float[..., 0], curl_H_float[..., 1])
        
        # Maxwell's equations in frequency domain
        # ∇×E = -iωμH
        # ∇×H = iωεE (assuming no current density)
        
        if material_props is not None:
            mu_r_complex = torch.complex(material_props[:, 0, 0], material_props[:, 0, 1]) # Assuming material_props is (batch_size, 1, 2)
            eps_r_complex = torch.complex(material_props[:, 1, 0], material_props[:, 1, 1]) # Assuming material_props is (batch_size, 3, 2)
            # Need to convert eps_r (3x3 tensor) to complex appropriately. For now, assuming isotropic
            # For simplicity, assuming eps_r is also a scalar in this context
            # This part needs careful handling if eps_r is a tensor.
            # For free space, it's 1.
            # Here, we assume material_props is (batch_size, num_props, 2)
            # and first prop is mu_r, next 3 are eps_r (diagonal components)
            
            # Let's simplify and assume mu_r and eps_r are scalars for now if from material_props
            if material_props.shape[1] == 1: # Only mu_r
                mu_r_complex = torch.complex(material_props[:, 0, 0], material_props[:, 0, 1])
                eps_r_complex = torch.complex(torch.ones_like(mu_r_complex.real), torch.zeros_like(mu_r_complex.real))
            elif material_props.shape[1] == 2: # mu_r and scalar eps_r
                mu_r_complex = torch.complex(material_props[:, 0, 0], material_props[:, 0, 1])
                eps_r_complex = torch.complex(material_props[:, 1, 0], material_props[:, 1, 1])
            else: # Full eps_r tensor
                # This needs proper tensor handling for eps_r
                # For now, let's just use the scalar equivalent if that's what's passed
                mu_r_complex = torch.complex(material_props[:, 0, 0], material_props[:, 0, 1])
                eps_r_complex = torch.complex(torch.ones(E_complex.shape[0], 1, device=E_complex.device), torch.zeros(E_complex.shape[0], 1, device=E_complex.device))

        else:
            # Assume free space, so mu_r = 1.0 + 0j and eps_r = 1.0 + 0j
            mu_r_complex = torch.complex(torch.ones(H_complex.shape[0], 1, device=H_complex.device), torch.zeros(H_complex.shape[0], 1, device=H_complex.device))
            eps_r_complex = torch.complex(torch.ones(E_complex.shape[0], 1, device=E_complex.device), torch.zeros(E_complex.shape[0], 1, device=E_complex.device))
        
        # Maxwell curl residuals
        residual_E = curl_E_complex + 1j * self.omega * self.mu0 * mu_r_complex * H_complex
        residual_H = curl_H_complex - 1j * self.omega * self.eps0 * eps_r_complex * E_complex
        
        # Combine residuals (take real part of squared magnitude)
        loss_E = torch.mean(residual_E.real**2 + residual_E.imag**2)
        loss_H = torch.mean(residual_H.real**2 + residual_H.imag**2)
        
        return loss_E + loss_H
    
    def _compute_curl(self, field: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Compute curl using automatic differentiation."""
        batch_size = field.shape[0]
        # We need to return (batch_size, 3, 2) for real/imag
        curl_output = torch.zeros(batch_size, 3, 2, device=field.device, dtype=field.dtype) # Output will have shape (batch_size, 3, 2)
        
        # Extract field components
        Fx = field[:, 0] # Shape (batch_size, 2)
        Fy = field[:, 1] # Shape (batch_size, 2)
        Fz = field[:, 2] # Shape (batch_size, 2)
        
        # Extract field components (real and imaginary parts separately)
        Fx_real, Fx_imag = Fx[:, 0], Fx[:, 1]
        Fy_real, Fy_imag = Fy[:, 0], Fy[:, 1]
        Fz_real, Fz_imag = Fz[:, 0], Fz[:, 1]
        
        # Compute partial derivatives
        # ∂Fz/∂y - ∂Fy/∂z for curl_x
        grad_Fz_dy_real = torch.autograd.grad(
            outputs=Fz_real.sum(), inputs=coords,
            create_graph=True, retain_graph=True, allow_unused=True
        )[0][:, 1] if coords.shape[1] > 1 else torch.zeros_like(Fz_real)
        grad_Fz_dy_imag = torch.autograd.grad(
            outputs=Fz_imag.sum(), inputs=coords,
            create_graph=True, retain_graph=True, allow_unused=True
        )[0][:, 1] if coords.shape[1] > 1 else torch.zeros_like(Fz_imag)
        dFz_dy = torch.complex(grad_Fz_dy_real, grad_Fz_dy_imag)
        
        grad_Fy_dz_real = torch.autograd.grad(
            outputs=Fy_real.sum(), inputs=coords,
            create_graph=True, retain_graph=True, allow_unused=True
        )[0][:, 2] if coords.shape[1] > 2 else torch.zeros_like(Fy_real)
        grad_Fy_dz_imag = torch.autograd.grad(
            outputs=Fy_imag.sum(), inputs=coords,
            create_graph=True, retain_graph=True, allow_unused=True
        )[0][:, 2] if coords.shape[1] > 2 else torch.zeros_like(Fy_imag)
        dFy_dz = torch.complex(grad_Fy_dz_real, grad_Fy_dz_imag)
        
        curl_x = dFz_dy - dFy_dz
        
        # ∂Fx/∂z - ∂Fz/∂x for curl_y
        grad_Fx_dz_real = torch.autograd.grad(
            outputs=Fx_real.sum(), inputs=coords,
            create_graph=True, retain_graph=True, allow_unused=True
        )[0][:, 2] if coords.shape[1] > 2 else torch.zeros_like(Fx_real)
        grad_Fx_dz_imag = torch.autograd.grad(
            outputs=Fx_imag.sum(), inputs=coords,
            create_graph=True, retain_graph=True, allow_unused=True
        )[0][:, 2] if coords.shape[1] > 2 else torch.zeros_like(Fx_imag)
        dFx_dz = torch.complex(grad_Fx_dz_real, grad_Fx_dz_imag)
        
        grad_Fz_dx_real = torch.autograd.grad(
            outputs=Fz_real.sum(), inputs=coords,
            create_graph=True, retain_graph=True, allow_unused=True
        )[0][:, 0]
        grad_Fz_dx_imag = torch.autograd.grad(
            outputs=Fz_imag.sum(), inputs=coords,
            create_graph=True, retain_graph=True, allow_unused=True
        )[0][:, 0]
        dFz_dx = torch.complex(grad_Fz_dx_real, grad_Fz_dx_imag)
        
        curl_y = dFx_dz - dFz_dx
        
        # ∂Fy/∂x - ∂Fx/∂y for curl_z
        grad_Fy_dx_real = torch.autograd.grad(
            outputs=Fy_real.sum(), inputs=coords,
            create_graph=True, retain_graph=True, allow_unused=True
        )[0][:, 0]
        grad_Fy_dx_imag = torch.autograd.grad(
            outputs=Fy_imag.sum(), inputs=coords,
            create_graph=True, retain_graph=True, allow_unused=True
        )[0][:, 0]
        dFy_dx = torch.complex(grad_Fy_dx_real, grad_Fy_dx_imag)
        
        grad_Fx_dy_real = torch.autograd.grad(
            outputs=Fx_real.sum(), inputs=coords,
            create_graph=True, retain_graph=True, allow_unused=True
        )[0][:, 1] if coords.shape[1] > 1 else torch.zeros_like(Fx_real)
        grad_Fx_dy_imag = torch.autograd.grad(
            outputs=Fx_imag.sum(), inputs=coords,
            create_graph=True, retain_graph=True, allow_unused=True
        )[0][:, 1] if coords.shape[1] > 1 else torch.zeros_like(Fx_imag)
        dFx_dy = torch.complex(grad_Fx_dy_real, grad_Fx_dy_imag)
        
        curl_z = dFy_dx - dFx_dy
        
        curl_output[:, 0, 0] = curl_x.real
        curl_output[:, 0, 1] = curl_x.imag
        curl_output[:, 1, 0] = curl_y.real
        curl_output[:, 1, 1] = curl_y.imag
        curl_output[:, 2, 0] = curl_z.real
        curl_output[:, 2, 1] = curl_z.imag
        
        return curl_output


def _compute_divergence(field: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Compute divergence using automatic differentiation."""
    div = torch.zeros(field.shape[0], 1, device=field.device, dtype=torch.complex64) # Ensure div is complex
    
    for i in range(min(3, field.shape[1])): # Iterate over field components, not coords
        if i < field.shape[1]:  # Make sure field has this component
            field_real = field[:, i, 0] # Extract real part
            field_imag = field[:, i, 1] # Extract imag part
            
            grad_real_output = torch.autograd.grad(
                outputs=field_real.sum(),
                inputs=coords,
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0]
            grad_real = grad_real_output[:, i] if grad_real_output is not None else torch.zeros_like(field_real)
            
            grad_imag_output = torch.autograd.grad(
                outputs=field_imag.sum(),
                inputs=coords,
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0]
            grad_imag = grad_imag_output[:, i] if grad_imag_output is not None else torch.zeros_like(field_imag)
            div[:, 0] += torch.complex(grad_real, grad_imag)
    
    return div


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
        div_E = _compute_divergence(E, coords)
        div_H = _compute_divergence(H, coords)
        
        # Apply material properties
        if material_props is not None:
            eps_r = material_props[:, 1:4]
            D = eps_r * E
            div_D = _compute_divergence(D, coords)
        else:
            div_D = div_E
        
        # Divergence constraints
        if charge_density is not None:
            residual_D = div_D - charge_density
        else:
            residual_D = div_D  # Assume no free charges
            
        residual_B = div_H  # ∇·B = 0 always
        
        return torch.mean(residual_D**2) + torch.mean(residual_B**2)
    
        

class MetamaterialConstitutiveLoss(BaseLoss):
    """
    Enforce metamaterial constitutive relations: D = ε·E, B = μ·H
    
    Args:
        metamaterial_solver: Metamaterial properties calculator
        weight: Loss weight
    """
    
    def __init__(self,
                 metamaterial_solver: Optional[MetamaterialProperties] = None,
                 weight: float = 1.0):
        super().__init__(weight)
        self.metamaterial = metamaterial_solver
    
    def compute(self,
                network: nn.Module,
                coords: torch.Tensor,
                frequency: float,
                **kwargs) -> torch.Tensor:
        """Enforce constitutive relations in metamaterial regions."""
        fields = network(coords)
        E = fields[:, :3]
        H = fields[:, 3:]
        
        # Get material properties
        if self.metamaterial:
            eps_tensor = self.metamaterial.permittivity_tensor(coords)
            # Assume non-magnetic materials (mu_r = 1)
            mu_tensor = torch.eye(3, device=coords.device, dtype=E.dtype).unsqueeze(0).expand(coords.shape[0], -1, -1)
        else:
            # Fallback: assume vacuum/isotropic material
            eps_tensor = torch.eye(3, device=coords.device, dtype=E.dtype).unsqueeze(0).expand(coords.shape[0], -1, -1)
            mu_tensor = torch.eye(3, device=coords.device, dtype=E.dtype).unsqueeze(0).expand(coords.shape[0], -1, -1)        
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
        self.boundary_solver = boundary_solver or BoundaryConditions()
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
        
        z_coords = coords[:, 2] # Extract z-coordinates
        
        # E is (batch_size, 3, 2) where last dim is real/imag
        
        # Calculate the magnitude of each complex component: |E_x|^2 = Ex_real^2 + Ex_imag^2
        Ex_mag_sq = E[:, 0, 0]**2 + E[:, 0, 1]**2
        Ey_mag_sq = E[:, 1, 0]**2 + E[:, 1, 1]**2
        Ez_mag_sq = E[:, 2, 0]**2 + E[:, 2, 1]**2
        
        # Total field magnitude is sqrt(|Ex|^2 + |Ey|^2 + |Ez|^2)
        field_magnitude = torch.sqrt(Ex_mag_sq + Ey_mag_sq + Ez_mag_sq) # Shape (batch_size,)
        
        # Normalize by maximum field to get relative decay
        max_field = torch.max(field_magnitude)
        
        # Ensure expected_decay also has shape (batch_size,)
        expected_decay = torch.exp(-torch.abs(z_coords) / self.decay_length)
        
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
        # E_jump and H_jump are (batch_size, 3, 2)
        # normal_vectors is (batch_size, 3)
        
        # Cross product for real parts
        n_cross_E_real = torch.cross(normal_vectors, E_jump[:, :, 0], dim=1)
        n_cross_H_real = torch.cross(normal_vectors, H_jump[:, :, 0], dim=1)
        
        # Cross product for imaginary parts
        n_cross_E_imag = torch.cross(normal_vectors, E_jump[:, :, 1], dim=1)
        n_cross_H_imag = torch.cross(normal_vectors, H_jump[:, :, 1], dim=1)
        
        # Combine back into (batch_size, 3, 2)
        n_cross_E = torch.stack([n_cross_E_real, n_cross_E_imag], dim=-1)
        n_cross_H = torch.stack([n_cross_H_real, n_cross_H_imag], dim=-1)
        
        # For perfect conductor interface, tangential E should be zero
        # For dielectric interface, tangential E should be continuous
        # Loss is the magnitude squared of the complex vector
        return torch.mean(n_cross_E[..., 0]**2 + n_cross_E[..., 1]**2) + \
               torch.mean(n_cross_H[..., 0]**2 + n_cross_H[..., 1]**2)


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
        div_S = _compute_divergence(S, coords)
        
        # In steady state, ∇·S = 0
        return torch.mean(torch.abs(div_S)**2)
    
    def _compute_divergence(self, field: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Compute divergence using automatic differentiation."""
        div = torch.zeros(field.shape[0], 1, device=field.device, dtype=torch.complex64) # Ensure div is complex
        
        for i in range(min(3, field.shape[1])):
            if i < field.shape[1]:  # Make sure field has this component
                field_real = field[:, i, 0] # Extract real part
                field_imag = field[:, i, 1] # Extract imag part
                
                grad_real_output = torch.autograd.grad(
                    outputs=field_real.sum(),
                    inputs=coords,
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )[0]
                grad_real = grad_real_output[:, i] if grad_real_output is not None else torch.zeros_like(field_real)
                
                grad_imag_output = torch.autograd.grad(
                    outputs=field_imag.sum(),
                    inputs=coords,
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )[0]
                grad_imag = grad_imag_output[:, i] if grad_imag_output is not None else torch.zeros_like(field_imag)
                div[:, 0] += torch.complex(grad_real, grad_imag)
        
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
            print(f"DEBUG: {name} component_loss dtype: {component_loss.dtype}, value: {component_loss.item()}")
            
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