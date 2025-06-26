"""
Maxwell's equations implementation for PINN-based SPP modelling.

Implements frequency-domain Maxwell's equations with automatic differentiation
for physics-informed neural network training.
"""

import torch
import numpy as np
from typing import Tuple, Optional


class MaxwellEquations:
    """
    Frequency-domain Maxwell's equations for electromagnetic field computation.
    
    Implements:
    ∇ × E = -iωμ₀H
    ∇ × H = iωε₀εᵣE
    
    Where εᵣ is the relative permittivity tensor of the metamaterial.
    """
    
    def __init__(self, omega: float, mu0: float = 4*np.pi*1e-7, eps0: float = 8.854e-12):
        """
        Initialise Maxwell equations solver.
        
        Args:
            omega: Angular frequency (rad/s)
            mu0: Permeability of free space (H/m)
            eps0: Permittivity of free space (F/m)
        """
        self.omega = omega
        self.mu0 = mu0
        self.eps0 = eps0
        self.c = 1.0 / np.sqrt(mu0 * eps0)
        self.k0 = omega / self.c
        
    def curl_operator(self, field: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute curl of a vector field using automatic differentiation.
        
        Args:
            field: Vector field [Fx, Fy, Fz] of shape (N, 3)
            coords: Coordinates [x, y, z] of shape (N, 3)
            
        Returns:
            Curl of field, shape (N, 3)
        """
        batch_size = field.shape[0]
        curl = torch.zeros_like(field)
        
        # Extract field components
        Fx, Fy, Fz = field[:, 0], field[:, 1], field[:, 2]
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        
        # Compute partial derivatives
        # ∂Fz/∂y
        dFz_dy = torch.autograd.grad(
            Fz, coords, 
            grad_outputs=torch.ones_like(Fz),
            create_graph=True, retain_graph=True
        )[0][:, 1]
        
        # ∂Fy/∂z
        dFy_dz = torch.autograd.grad(
            Fy, coords,
            grad_outputs=torch.ones_like(Fy),
            create_graph=True, retain_graph=True
        )[0][:, 2]
        
        # ∂Fx/∂z
        dFx_dz = torch.autograd.grad(
            Fx, coords,
            grad_outputs=torch.ones_like(Fx),
            create_graph=True, retain_graph=True
        )[0][:, 2]
        
        # ∂Fz/∂x
        dFz_dx = torch.autograd.grad(
            Fz, coords,
            grad_outputs=torch.ones_like(Fz),
            create_graph=True, retain_graph=True
        )[0][:, 0]
        
        # ∂Fy/∂x
        dFy_dx = torch.autograd.grad(
            Fy, coords,
            grad_outputs=torch.ones_like(Fy),
            create_graph=True, retain_graph=True
        )[0][:, 0]
        
        # ∂Fx/∂y
        dFx_dy = torch.autograd.grad(
            Fx, coords,
            grad_outputs=torch.ones_like(Fx),
            create_graph=True, retain_graph=True
        )[0][:, 1]
        
        # Curl components
        curl[:, 0] = dFz_dy - dFy_dz  # (∇ × F)_x
        curl[:, 1] = dFx_dz - dFz_dx  # (∇ × F)_y
        curl[:, 2] = dFy_dx - dFx_dy  # (∇ × F)_z
        
        return curl
    
    def curl_E_residual(self, E_field: torch.Tensor, H_field: torch.Tensor, 
                       coords: torch.Tensor) -> torch.Tensor:
        """
        Compute residual for Faraday's law: ∇ × E + iωμ₀H = 0
        
        Args:
            E_field: Electric field [Ex, Ey, Ez] of shape (N, 3)
            H_field: Magnetic field [Hx, Hy, Hz] of shape (N, 3)
            coords: Coordinates [x, y, z] of shape (N, 3)
            
        Returns:
            Residual vector of shape (N, 3)
        """
        curl_E = self.curl_operator(E_field, coords)
        residual = curl_E + 1j * self.omega * self.mu0 * H_field
        
        # Return real and imaginary parts separately for loss computation
        return torch.cat([residual.real, residual.imag], dim=1)
    
    def curl_H_residual(self, E_field: torch.Tensor, H_field: torch.Tensor,
                       coords: torch.Tensor, epsilon_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute residual for Ampère's law: ∇ × H - iωε₀εᵣE = 0
        
        Args:
            E_field: Electric field [Ex, Ey, Ez] of shape (N, 3)
            H_field: Magnetic field [Hx, Hy, Hz] of shape (N, 3)
            coords: Coordinates [x, y, z] of shape (N, 3)
            epsilon_tensor: Relative permittivity tensor of shape (N, 3, 3)
            
        Returns:
            Residual vector of shape (N, 3)
        """
        curl_H = self.curl_operator(H_field, coords)
        
        # Apply permittivity tensor: εᵣ · E
        # Einstein summation over last dimension
        epsilon_E = torch.einsum('nij,nj->ni', epsilon_tensor, E_field)
        
        residual = curl_H - 1j * self.omega * self.eps0 * epsilon_E
        
        # Return real and imaginary parts separately for loss computation
        return torch.cat([residual.real, residual.imag], dim=1)
    
    def divergence_E_residual(self, E_field: torch.Tensor, coords: torch.Tensor,
                             epsilon_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute residual for Gauss's law: ∇ · (εᵣE) = 0 (no free charges)
        
        Args:
            E_field: Electric field [Ex, Ey, Ez] of shape (N, 3)
            coords: Coordinates [x, y, z] of shape (N, 3)
            epsilon_tensor: Relative permittivity tensor of shape (N, 3, 3)
            
        Returns:
            Residual scalar of shape (N, 1)
        """
        # Apply permittivity tensor
        epsilon_E = torch.einsum('nij,nj->ni', epsilon_tensor, E_field)
        
        # Compute divergence
        div_epsilon_E = torch.zeros(E_field.shape[0], 1, device=E_field.device, dtype=E_field.dtype)
        
        for i in range(3):
            d_component = torch.autograd.grad(
                epsilon_E[:, i], coords,
                grad_outputs=torch.ones_like(epsilon_E[:, i]),
                create_graph=True, retain_graph=True
            )[0][:, i]
            div_epsilon_E[:, 0] += d_component
            
        return torch.cat([div_epsilon_E.real, div_epsilon_E.imag], dim=1)
    
    def divergence_B_residual(self, H_field: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute residual for ∇ · B = 0 (no magnetic monopoles)
        
        Args:
            H_field: Magnetic field [Hx, Hy, Hz] of shape (N, 3)
            coords: Coordinates [x, y, z] of shape (N, 3)
            
        Returns:
            Residual scalar of shape (N, 1)
        """
        div_H = torch.zeros(H_field.shape[0], 1, device=H_field.device, dtype=H_field.dtype)
        
        for i in range(3):
            d_component = torch.autograd.grad(
                H_field[:, i], coords,
                grad_outputs=torch.ones_like(H_field[:, i]),
                create_graph=True, retain_graph=True
            )[0][:, i]
            div_H[:, 0] += d_component
            
        return torch.cat([div_H.real, div_H.imag], dim=1)
    
    def total_residual(self, E_field: torch.Tensor, H_field: torch.Tensor,
                      coords: torch.Tensor, epsilon_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute total Maxwell equation residuals.
        
        Args:
            E_field: Electric field [Ex, Ey, Ez] of shape (N, 3)
            H_field: Magnetic field [Hx, Hy, Hz] of shape (N, 3)
            coords: Coordinates [x, y, z] of shape (N, 3)
            epsilon_tensor: Relative permittivity tensor of shape (N, 3, 3)
            
        Returns:
            Combined residual vector of shape (N, 16):
            [curl_E_real, curl_E_imag, curl_H_real, curl_H_imag, 
             div_E_real, div_E_imag, div_B_real, div_B_imag]
        """
        curl_E_res = self.curl_E_residual(E_field, H_field, coords)
        curl_H_res = self.curl_H_residual(E_field, H_field, coords, epsilon_tensor)
        div_E_res = self.divergence_E_residual(E_field, coords, epsilon_tensor)
        div_B_res = self.divergence_B_residual(H_field, coords)
        
        return torch.cat([curl_E_res, curl_H_res, div_E_res, div_B_res], dim=1)
    
    def poynting_vector(self, E_field: torch.Tensor, H_field: torch.Tensor) -> torch.Tensor:
        """
        Compute Poynting vector S = (1/2) Re(E × H*)
        
        Args:
            E_field: Electric field [Ex, Ey, Ez] of shape (N, 3)
            H_field: Magnetic field [Hx, Hy, Hz] of shape (N, 3)
            
        Returns:
            Poynting vector of shape (N, 3)
        """
        # Cross product E × H*
        Ex, Ey, Ez = E_field[:, 0], E_field[:, 1], E_field[:, 2]
        Hx_conj, Hy_conj, Hz_conj = H_field[:, 0].conj(), H_field[:, 1].conj(), H_field[:, 2].conj()
        
        S = torch.zeros_like(E_field)
        S[:, 0] = Ey * Hz_conj - Ez * Hy_conj
        S[:, 1] = Ez * Hx_conj - Ex * Hz_conj
        S[:, 2] = Ex * Hy_conj - Ey * Hx_conj
        
        return 0.5 * S.real