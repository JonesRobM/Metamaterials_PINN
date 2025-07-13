"""
Boundary conditions for electromagnetic fields at metamaterial interfaces.

Implements continuity conditions for tangential E and H fields and
normal displacement/magnetic flux density at dielectric interfaces.
"""

import torch
import numpy as np
from typing import Tuple, Optional


class BoundaryConditions:
    """
    Electromagnetic boundary conditions at metamaterial-dielectric interfaces.
    
    Implements:
    1. Tangential E field continuity: n x (E₂ - E₁) = 0
    2. Tangential H field continuity: n x (H₂ - H₁) = 0  
    3. Normal D field continuity: n · (D₂ - D₁) = 0 (no surface charges)
    4. Normal B field continuity: n · (B₂ - B₁) = 0
    """
    
    def __init__(self, interface_normal: Tuple[float, float, float] = (0, 0, 1),
                 eps0: float = 8.854e-12, mu0: float = 4*np.pi*1e-7):
        """
        Initialise boundary conditions.
        
        Args:
            interface_normal: Unit normal vector pointing from medium 1 to medium 2
            eps0: Permittivity of free space (F/m)
            mu0: Permeability of free space (H/m)
        """
        self.interface_normal = torch.tensor(interface_normal, dtype=torch.float32)
        self.eps0 = eps0
        self.mu0 = mu0
        
        # Normalise the normal vector
        self.interface_normal = self.interface_normal / torch.norm(self.interface_normal)
    
    def cross_product(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute cross product a x b for batch of vectors.
        
        Args:
            a: Vector of shape (N, 3) or (3,)
            b: Vector of shape (N, 3) or (3,)
            
        Returns:
            Cross product of shape (N, 3)
        """
        if a.dim() == 1:
            a = a.unsqueeze(0)
        if b.dim() == 1:
            b = b.unsqueeze(0)
            
        cross = torch.zeros_like(a)
        cross[:, 0] = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
        cross[:, 1] = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
        cross[:, 2] = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        
        return cross
    
    def tangential_E_continuity(self, E1: torch.Tensor, E2: torch.Tensor) -> torch.Tensor:
        """
        Compute residual for tangential E field continuity: n x (E₂ - E₁) = 0
        
        Args:
            E1: Electric field in medium 1, shape (N, 3)
            E2: Electric field in medium 2, shape (N, 3)
            
        Returns:
            Residual vector of shape (N, 3)
        """
        E_diff = E2 - E1
        
        # Expand normal vector to match batch size
        n = self.interface_normal.unsqueeze(0).expand(E_diff.shape[0], -1)
        n = n.to(E_diff.device)
        
        # n × (E₂ - E₁)
        tangential_residual = self.cross_product(n, E_diff)
        
        # Return real and imaginary parts separately
        return torch.cat([tangential_residual.real, tangential_residual.imag], dim=1)
    
    def tangential_H_continuity(self, H1: torch.Tensor, H2: torch.Tensor) -> torch.Tensor:
        """
        Compute residual for tangential H field continuity: n x (H₂ - H₁) = 0
        
        Args:
            H1: Magnetic field in medium 1, shape (N, 3)
            H2: Magnetic field in medium 2, shape (N, 3)
            
        Returns:
            Residual vector of shape (N, 3)
        """
        H_diff = H2 - H1
        
        # Expand normal vector to match batch size
        n = self.interface_normal.unsqueeze(0).expand(H_diff.shape[0], -1)
        n = n.to(H_diff.device)
        
        # n × (H₂ - H₁)
        tangential_residual = self.cross_product(n, H_diff)
        
        # Return real and imaginary parts separately
        return torch.cat([tangential_residual.real, tangential_residual.imag], dim=1)
    
    def normal_D_continuity(self, E1: torch.Tensor, E2: torch.Tensor,
                           eps1_tensor: torch.Tensor, eps2_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute residual for normal D field continuity: n · (D₂ - D₁) = 0
        
        Args:
            E1: Electric field in medium 1, shape (N, 3)
            E2: Electric field in medium 2, shape (N, 3)
            eps1_tensor: Permittivity tensor in medium 1, shape (N, 3, 3)
            eps2_tensor: Permittivity tensor in medium 2, shape (N, 3, 3)
            
        Returns:
            Residual scalar of shape (N, 1)
        """
        # Compute displacement fields D = ε₀εᵣE
        D1 = self.eps0 * torch.einsum('nij,nj->ni', eps1_tensor, E1)
        D2 = self.eps0 * torch.einsum('nij,nj->ni', eps2_tensor, E2)
        
        D_diff = D2 - D1
        
        # Expand normal vector to match batch size
        n = self.interface_normal.unsqueeze(0).expand(D_diff.shape[0], -1)
        n = n.to(D_diff.device)
        
        # n · (D₂ - D₁)
        normal_residual = torch.sum(n * D_diff, dim=1, keepdim=True)
        
        # Return real and imaginary parts separately
        return torch.cat([normal_residual.real, normal_residual.imag], dim=1)
    
    def normal_B_continuity(self, H1: torch.Tensor, H2: torch.Tensor) -> torch.Tensor:
        """
        Compute residual for normal B field continuity: n · (B₂ - B₁) = 0
        
        Args:
            H1: Magnetic field in medium 1, shape (N, 3)
            H2: Magnetic field in medium 2, shape (N, 3)
            
        Returns:
            Residual scalar of shape (N, 1)
        """
        # Assuming non-magnetic materials: B = μ₀H
        B1 = self.mu0 * H1
        B2 = self.mu0 * H2
        
        B_diff = B2 - B1
        
        # Expand normal vector to match batch size
        n = self.interface_normal.unsqueeze(0).expand(B_diff.shape[0], -1)
        n = n.to(B_diff.device)
        
        # n · (B₂ - B₁)
        normal_residual = torch.sum(n * B_diff, dim=1, keepdim=True)
        
        # Return real and imaginary parts separately
        return torch.cat([normal_residual.real, normal_residual.imag], dim=1)
    
    def spp_boundary_conditions(self, E_metamaterial: torch.Tensor, H_metamaterial: torch.Tensor,
                               E_dielectric: torch.Tensor, H_dielectric: torch.Tensor,
                               eps_metamaterial: torch.Tensor, eps_dielectric: float = 1.0) -> torch.Tensor:
        """
        Apply all boundary conditions for SPP at metamaterial-dielectric interface.
        
        Args:
            E_metamaterial: E field in metamaterial, shape (N, 3)
            H_metamaterial: H field in metamaterial, shape (N, 3)
            E_dielectric: E field in dielectric, shape (N, 3)
            H_dielectric: H field in dielectric, shape (N, 3)
            eps_metamaterial: Metamaterial permittivity tensor, shape (N, 3, 3)
            eps_dielectric: Dielectric permittivity (scalar)
            
        Returns:
            Combined boundary condition residuals, shape (N, 16)
        """
        # Create dielectric permittivity tensor (isotropic)
        batch_size = E_metamaterial.shape[0]
        device = E_metamaterial.device
        dtype = torch.complex64 if E_metamaterial.dtype == torch.complex64 else torch.complex128
        
        eps_diel_tensor = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1) * eps_dielectric
        
        # Apply all boundary conditions
        tang_E_res = self.tangential_E_continuity(E_metamaterial, E_dielectric)
        tang_H_res = self.tangential_H_continuity(H_metamaterial, H_dielectric)
        norm_D_res = self.normal_D_continuity(E_metamaterial, E_dielectric, eps_metamaterial, eps_diel_tensor)
        norm_B_res = self.normal_B_continuity(H_metamaterial, H_dielectric)
        
        return torch.cat([tang_E_res, tang_H_res, norm_D_res, norm_B_res], dim=1)
    
    def perfect_conductor_boundary(self, E_tangential: torch.Tensor) -> torch.Tensor:
        """
        Apply perfect electric conductor boundary condition: n × E = 0
        
        Args:
            E_tangential: Tangential components of E field, shape (N, 3)
            
        Returns:
            Residual for PEC boundary condition, shape (N, 3)
        """
        # For PEC, tangential E field must be zero
        tangential_E = E_tangential - torch.sum(E_tangential * self.interface_normal.unsqueeze(0), dim=1, keepdim=True) * self.interface_normal.unsqueeze(0)
        
        return torch.cat([tangential_E.real, tangential_E.imag], dim=1)
    
    def impedance_boundary_condition(self, E_tangential: torch.Tensor, H_tangential: torch.Tensor,
                                   surface_impedance: complex) -> torch.Tensor:
        """
        Apply surface impedance boundary condition: n × E = Z_s (n × H × n)
        
        Args:
            E_tangential: Tangential E field, shape (N, 3)
            H_tangential: Tangential H field, shape (N, 3)
            surface_impedance: Surface impedance Z_s
            
        Returns:
            Residual for impedance boundary condition, shape (N, 3)
        """
        # Expand normal vector
        n = self.interface_normal.unsqueeze(0).expand(E_tangential.shape[0], -1)
        n = n.to(E_tangential.device)
        
        # Compute n × E
        n_cross_E = self.cross_product(n, E_tangential)
        
        # Compute n × H × n = (n · n)H - (n · H)n = H - (n · H)n (since |n| = 1)
        n_dot_H = torch.sum(n * H_tangential, dim=1, keepdim=True)
        n_cross_H_cross_n = H_tangential - n_dot_H * n
        
        # Boundary condition: n × E = Z_s (n × H × n)
        residual = n_cross_E - surface_impedance * n_cross_H_cross_n
        
        return torch.cat([residual.real, residual.imag], dim=1)
    
    def radiation_boundary_condition(self, E_field: torch.Tensor, H_field: torch.Tensor,
                                   k0: float, eps_background: complex = 1.0) -> torch.Tensor:
        """
        Apply first-order absorbing boundary condition for radiation.
        
        Implements: (∂/∂n - ik₀√ε_bg)(E_tan, H_tan) = 0
        
        Args:
            E_field: Electric field, shape (N, 3)
            H_field: Magnetic field, shape (N, 3)
            k0: Free space wavevector
            eps_background: Background permittivity
            
        Returns:
            Residual for ABC, shape (N, 6)
        """
        # This is a simplified ABC - full implementation would require
        # spatial derivatives normal to the boundary
        k_normal = k0 * torch.sqrt(eps_background)
        
        # Simplified residual (proper implementation needs normal derivatives)
        abc_residual_E = -1j * k_normal * E_field
        abc_residual_H = -1j * k_normal * H_field
        
        residual = torch.cat([abc_residual_E, abc_residual_H], dim=1)
        return torch.cat([residual.real, residual.imag], dim=1)
    
    def __repr__(self) -> str:
        return f"BoundaryConditions(normal={self.interface_normal.tolist()})"