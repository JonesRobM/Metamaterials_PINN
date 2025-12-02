"""
Maxwell's equations implementation for PINN-based SPP modelling.

Implements frequency-domain Maxwell's equations with automatic differentiation
for physics-informed neural network training.
"""

import torch
import numpy as np
from typing import Tuple, Optional


# Helper function for computing gradient components safely
def _compute_grad_component(output_component: torch.Tensor, full_coords_tensor: torch.Tensor, coord_idx: int,
                           batch_size: int, device: torch.device, dtype: torch.dtype, name: str = "") -> torch.Tensor:
    if output_component.requires_grad:
        # Compute gradient of output_component (N,) with respect to full_coords_tensor (N, 3)
        # This will return a (N, 3) tensor representing [d(out_i)/dx_i, d(out_i)/dy_i, d(out_i)/dz_i]
        grad_all_coords = torch.autograd.grad(
            outputs=output_component,
            inputs=full_coords_tensor, # Pass the full coords tensor
            grad_outputs=torch.ones_like(output_component),
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0]
        if grad_all_coords is not None:
            # We need the gradient with respect to a specific coordinate dimension (coord_idx)
            # grad_all_coords will have shape (N, 3)
            print(f"DEBUG: {name} - output_component: {output_component.shape}, full_coords_tensor: {full_coords_tensor.shape}, grad_all_coords: {grad_all_coords.shape}, grad value: {grad_all_coords[:, coord_idx]}")
            return grad_all_coords[:, coord_idx] # Extract the relevant column
    print(f"DEBUG: {name} - grad is None, returning zeros.")
    return torch.zeros(batch_size, device=device, dtype=dtype)


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
        device = field.device
        
        # Ensure coords requires gradients
        if not coords.requires_grad:
            coords.requires_grad_(True)

        # Extract field components
        Fx = field[:, 0]
        Fy = field[:, 1]
        Fz = field[:, 2]

        # Compute partial derivatives for Fx.real and Fx.imag
        dFx_dx_r = _compute_grad_component(Fx.real, coords, 0, batch_size, device, Fx.real.dtype, "dFx_dx_r")
        dFx_dy_r = _compute_grad_component(Fx.real, coords, 1, batch_size, device, Fx.real.dtype, "dFx_dy_r")
        dFx_dz_r = _compute_grad_component(Fx.real, coords, 2, batch_size, device, Fx.real.dtype, "dFx_dz_r")
        dFx_dx_i = _compute_grad_component(Fx.imag, coords, 0, batch_size, device, Fx.imag.dtype, "dFx_dx_i")
        dFx_dy_i = _compute_grad_component(Fx.imag, coords, 1, batch_size, device, Fx.imag.dtype, "dFx_dy_i")
        dFx_dz_i = _compute_grad_component(Fx.imag, coords, 2, batch_size, device, Fx.imag.dtype, "dFx_dz_i")

        # Compute partial derivatives for Fy.real and Fy.imag
        dFy_dx_r = _compute_grad_component(Fy.real, coords, 0, batch_size, device, Fy.real.dtype, "dFy_dx_r")
        dFy_dy_r = _compute_grad_component(Fy.real, coords, 1, batch_size, device, Fy.real.dtype, "dFy_dy_r")
        dFy_dz_r = _compute_grad_component(Fy.real, coords, 2, batch_size, device, Fy.real.dtype, "dFy_dz_r")
        dFy_dx_i = _compute_grad_component(Fy.imag, coords, 0, batch_size, device, Fy.imag.dtype, "dFy_dx_i")
        dFy_dy_i = _compute_grad_component(Fy.imag, coords, 1, batch_size, device, Fy.imag.dtype, "dFy_dy_i")
        dFy_dz_i = _compute_grad_component(Fy.imag, coords, 2, batch_size, device, Fy.imag.dtype, "dFy_dz_i")

        # Compute partial derivatives for Fz.real and Fz.imag
        dFz_dx_r = _compute_grad_component(Fz.real, coords, 0, batch_size, device, Fz.real.dtype, "dFz_dx_r")
        dFz_dy_r = _compute_grad_component(Fz.real, coords, 1, batch_size, device, Fz.real.dtype, "dFz_dy_r")
        dFz_dz_r = _compute_grad_component(Fz.real, coords, 2, batch_size, device, Fz.real.dtype, "dFz_dz_r")
        dFz_dx_i = _compute_grad_component(Fz.imag, coords, 0, batch_size, device, Fz.imag.dtype, "dFz_dx_i")
        dFz_dy_i = _compute_grad_component(Fz.imag, coords, 1, batch_size, device, Fz.imag.dtype, "dFz_dy_i")
        dFz_dz_i = _compute_grad_component(Fz.imag, coords, 2, batch_size, device, Fz.imag.dtype, "dFz_dz_i")

        # Combine real and imaginary parts to form complex partial derivatives
        dFx_dy = torch.complex(dFx_dy_r, dFx_dy_i)
        dFx_dz = torch.complex(dFx_dz_r, dFx_dz_i)
        dFy_dx = torch.complex(dFy_dx_r, dFy_dx_i)
        dFy_dz = torch.complex(dFy_dz_r, dFy_dz_i)
        dFz_dx = torch.complex(dFz_dx_r, dFz_dx_i)
        dFz_dy = torch.complex(dFz_dy_r, dFz_dy_i)

        # Calculate curl components
        curl_x = dFz_dy - dFy_dz
        curl_y = dFx_dz - dFz_dx
        curl_z = dFy_dx - dFx_dy
        
        # Stack into (N, 3) tensor
        curl = torch.stack([curl_x, curl_y, curl_z], dim=1)
        
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
            Residual scalar of shape (N, 2) (real and imag parts of divergence)
        """
        batch_size = E_field.shape[0]
        device = E_field.device
        
        # Ensure coords requires gradients
        if not coords.requires_grad:
            coords.requires_grad_(True)

        # Apply permittivity tensor: (εᵣ · E)_x, (εᵣ · E)_y, (εᵣ · E)_z
        epsilon_E = torch.einsum('nij,nj->ni', epsilon_tensor, E_field)
        
        # Extract components of (εᵣ · E)
        Dx = epsilon_E[:, 0]
        Dy = epsilon_E[:, 1]
        Dz = epsilon_E[:, 2]

        # Compute d(Dx)/dx
        dDx_dx_r = _compute_grad_component(Dx.real, coords, 0, batch_size, device, Dx.real.dtype, "dDx_dx_r")
        dDx_dx_i = _compute_grad_component(Dx.imag, coords, 0, batch_size, device, Dx.imag.dtype, "dDx_dx_i")
        dDx_dx = torch.complex(dDx_dx_r, dDx_dx_i)

        # Compute d(Dy)/dy
        dDy_dy_r = _compute_grad_component(Dy.real, coords, 1, batch_size, device, Dy.real.dtype, "dDy_dy_r")
        dDy_dy_i = _compute_grad_component(Dy.imag, coords, 1, batch_size, device, Dy.imag.dtype, "dDy_dy_i")
        dDy_dy = torch.complex(dDy_dy_r, dDy_dy_i)

        # Compute d(Dz)/dz
        dDz_dz_r = _compute_grad_component(Dz.real, coords, 2, batch_size, device, Dz.real.dtype, "dDz_dz_r")
        dDz_dz_i = _compute_grad_component(Dz.imag, coords, 2, batch_size, device, Dz.imag.dtype, "dDz_dz_i")
        dDz_dz = torch.complex(dDz_dz_r, dDz_dz_i)
        
        # Divergence = dDx/dx + dDy/dy + dDz/dz
        div_epsilon_E = dDx_dx + dDy_dy + dDz_dz
            
        return torch.cat([div_epsilon_E.real.unsqueeze(1), div_epsilon_E.imag.unsqueeze(1)], dim=1)
    
    def divergence_B_residual(self, H_field: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute residual for ∇ · B = 0 (no magnetic monopoles)
        
        Args:
            H_field: Magnetic field [Hx, Hy, Hz] of shape (N, 3)
            coords: Coordinates [x, y, z] of shape (N, 3)
            
        Returns:
            Residual scalar of shape (N, 2) (real and imag parts of divergence)
        """
        batch_size = H_field.shape[0]
        device = H_field.device

        # Ensure coords requires gradients
        if not coords.requires_grad:
            coords.requires_grad_(True)

        # Extract components of H
        Hx = H_field[:, 0]
        Hy = H_field[:, 1]
        Hz = H_field[:, 2]

        # Compute d(Hx)/dx
        dHx_dx_r = _compute_grad_component(Hx.real, coords, 0, batch_size, device, Hx.real.dtype, "dHx_dx_r")
        dHx_dx_i = _compute_grad_component(Hx.imag, coords, 0, batch_size, device, Hx.imag.dtype, "dHx_dx_i")
        dHx_dx = torch.complex(dHx_dx_r, dHx_dx_i)

        # Compute d(Hy)/dy
        dHy_dy_r = _compute_grad_component(Hy.real, coords, 1, batch_size, device, Hy.real.dtype, "dHy_dy_r")
        dHy_dy_i = _compute_grad_component(Hy.imag, coords, 1, batch_size, device, Hy.imag.dtype, "dHy_dy_i")
        dHy_dy = torch.complex(dHy_dy_r, dHy_dy_i)

        # Compute d(Hz)/dz
        dHz_dz_r = _compute_grad_component(Hz.real, coords, 2, batch_size, device, Hz.real.dtype, "dHz_dz_r")
        dHz_dz_i = _compute_grad_component(Hz.imag, coords, 2, batch_size, device, Hz.imag.dtype, "dHz_dz_i")
        dHz_dz = torch.complex(dHz_dz_r, dHz_dz_i)
        
        # Divergence of B = mu0 * Divergence of H
        # So we compute div H and assume mu0 is handled in constants if needed elsewhere.
        # Here we directly compute div H, as ∇ · B = μ₀ (∇ · H) = 0
        div_H = dHx_dx + dHy_dy + dHz_dz
            
        return torch.cat([div_H.real.unsqueeze(1), div_H.imag.unsqueeze(1)], dim=1)
    
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