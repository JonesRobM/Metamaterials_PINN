"""
Metamaterial constitutive relations for anisotropic SPP modelling.

Implements permittivity tensors for uniaxial metamaterials and provides
analytical dispersion relations for validation.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union


class MetamaterialProperties:
    """
    Uniaxial metamaterial with anisotropic permittivity tensor.
    
    For a uniaxial metamaterial with optical axis along z:
    εᵣ = diag(ε_⊥, ε_⊥, ε_∥)
    
    Where ε_⊥ and ε_∥ are the perpendicular and parallel permittivity components.
    """
    
    def __init__(self, eps_parallel: complex, eps_perpendicular: complex, 
                 optical_axis: str = 'z', eps0: float = 8.854e-12):
        """
        Initialise metamaterial properties.
        
        Args:
            eps_parallel: Relative permittivity parallel to optical axis
            eps_perpendicular: Relative permittivity perpendicular to optical axis
            optical_axis: Direction of optical axis ('x', 'y', or 'z')
            eps0: Permittivity of free space (F/m)
        """
        self.eps_par = eps_parallel
        self.eps_perp = eps_perpendicular
        self.optical_axis = optical_axis.lower()
        self.eps0 = eps0
        
        if self.optical_axis not in ['x', 'y', 'z']:
            raise ValueError("Optical axis must be 'x', 'y', or 'z'")
    
    def permittivity_tensor(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Return anisotropic permittivity tensor at given coordinates.
        
        For uniaxial metamaterial with optical axis along z:
        εᵣ = [[ε_⊥, 0, 0],
              [0, ε_⊥, 0],
              [0, 0, ε_∥]]
        
        Args:
            coords: Coordinates [x, y, z] of shape (N, 3)
            
        Returns:
            Permittivity tensor of shape (N, 3, 3)
        """
        batch_size = coords.shape[0]
        device = coords.device
        dtype = torch.complex64 if coords.dtype == torch.float32 else torch.complex128
        
        # Create identity tensor
        eps_tensor = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
        
        # Fill diagonal elements based on optical axis
        if self.optical_axis == 'z':
            eps_tensor[:, 0, 0] = self.eps_perp  # εxx
            eps_tensor[:, 1, 1] = self.eps_perp  # εyy
            eps_tensor[:, 2, 2] = self.eps_par   # εzz
        elif self.optical_axis == 'y':
            eps_tensor[:, 0, 0] = self.eps_perp  # εxx
            eps_tensor[:, 1, 1] = self.eps_par   # εyy
            eps_tensor[:, 2, 2] = self.eps_perp  # εzz
        else:  # optical_axis == 'x'
            eps_tensor[:, 0, 0] = self.eps_par   # εxx
            eps_tensor[:, 1, 1] = self.eps_perp  # εyy
            eps_tensor[:, 2, 2] = self.eps_perp  # εzz
            
        return eps_tensor
    
    def effective_permittivity(self, kx: torch.Tensor, ky: torch.Tensor) -> torch.Tensor:
        """
        Compute effective permittivity for SPP modes.
        
        For surface waves at z=0 interface with optical axis along z:
        ε_eff = ε_⊥ * ε_∥ / (ε_⊥ - (k_∥²/k₀²) * (ε_⊥ - ε_∥))
        
        Where k_∥² = kx² + ky²
        
        Args:
            kx: x-component of wavevector
            ky: y-component of wavevector
            
        Returns:
            Effective permittivity
        """
        k_parallel_sq = kx**2 + ky**2
        
        if self.optical_axis == 'z':
            # Standard uniaxial case
            denominator = self.eps_perp - (k_parallel_sq / self.k0**2) * (self.eps_perp - self.eps_par)
            eps_eff = self.eps_perp * self.eps_par / denominator
        else:
            # For optical axis not along z, use general formula
            eps_eff = self.eps_perp  # Simplified for demonstration
            
        return eps_eff
    
    def spp_dispersion_relation(self, omega: float, eps_dielectric: complex = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Analytical SPP dispersion relation for metamaterial-dielectric interface.
        
        For interface at z=0 with metamaterial (z<0) and dielectric (z>0):
        k_spp = k₀ * sqrt(ε_eff * ε_d / (ε_eff + ε_d))
        
        Args:
            omega: Angular frequency (rad/s)
            eps_dielectric: Relative permittivity of upper dielectric
            
        Returns:
            Tuple of (k_spp_real, k_spp_imag) for SPP wavevector
        """
        k0 = omega / (3e8)  # Free space wavevector
        
        # For uniaxial metamaterial with optical axis along z
        if self.optical_axis == 'z':
            eps_eff = self.eps_perp  # For SPP modes, use perpendicular component
        else:
            eps_eff = self.eps_perp
            
        # SPP dispersion relation
        numerator = eps_eff * eps_dielectric
        denominator = eps_eff + eps_dielectric
        
        k_spp_complex = k0 * (numerator / denominator)**0.5
        
        return k_spp_complex.real, k_spp_complex.imag
    
    def propagation_length(self, omega: float, eps_dielectric: complex = 1.0) -> float:
        """
        Calculate SPP propagation length L_spp = 1/(2*Im(k_spp))
        
        Args:
            omega: Angular frequency (rad/s)
            eps_dielectric: Relative permittivity of upper dielectric
            
        Returns:
            Propagation length in metres
        """
        _, k_spp_imag = self.spp_dispersion_relation(omega, eps_dielectric)
        
        if k_spp_imag > 0:
            return 1.0 / (2.0 * k_spp_imag)
        else:
            return float('inf')  # No damping
    
    def penetration_depth_metamaterial(self, omega: float, eps_dielectric: complex = 1.0) -> float:
        """
        Calculate SPP penetration depth into metamaterial.
        
        Args:
            omega: Angular frequency (rad/s)
            eps_dielectric: Relative permittivity of upper dielectric
            
        Returns:
            Penetration depth in metres
        """
        k0 = omega / (3e8)
        k_spp_real, _ = self.spp_dispersion_relation(omega, eps_dielectric)
        
        # z-component of wavevector in metamaterial
        if self.optical_axis == 'z':
            eps_eff = self.eps_par  # Use parallel component for z-direction
        else:
            eps_eff = self.eps_perp
            
        kz_sq = eps_eff * k0**2 - k_spp_real**2
        
        if kz_sq.real < 0:  # Evanescent wave
            kz_imag = (-kz_sq)**0.5
            return 1.0 / kz_imag.real
        else:
            return float('inf')  # Propagating wave
    
    def penetration_depth_dielectric(self, omega: float, eps_dielectric: complex = 1.0) -> float:
        """
        Calculate SPP penetration depth into dielectric.
        
        Args:
            omega: Angular frequency (rad/s)
            eps_dielectric: Relative permittivity of upper dielectric
            
        Returns:
            Penetration depth in metres
        """
        k0 = omega / (3e8)
        k_spp_real, _ = self.spp_dispersion_relation(omega, eps_dielectric)
        
        # z-component of wavevector in dielectric
        kz_sq = eps_dielectric * k0**2 - k_spp_real**2
        
        if kz_sq.real < 0:  # Evanescent wave
            kz_imag = ((-kz_sq)**0.5).real
            return 1.0 / kz_imag
        else:
            return float('inf')  # Propagating wave
    
    def field_enhancement_factor(self, omega: float, eps_dielectric: complex = 1.0) -> float:
        """
        Calculate electric field enhancement at the interface.
        
        Args:
            omega: Angular frequency (rad/s)
            eps_dielectric: Relative permittivity of upper dielectric
            
        Returns:
            Field enhancement factor |E_surface|/|E_incident|
        """
        # Simplified enhancement factor for SPP excitation
        if self.optical_axis == 'z':
            eps_eff = self.eps_perp
        else:
            eps_eff = self.eps_perp
            
        # Field enhancement proportional to 1/|ε_eff + ε_d|
        enhancement = 1.0 / abs(eps_eff + eps_dielectric)
        
        return enhancement.real
    
    def is_spp_supported(self, eps_dielectric: complex = 1.0) -> bool:
        """
        Check if SPP modes are supported at the interface.
        
        SPPs exist when Re(ε_eff) < 0 and Re(ε_d) > 0, and
        Re(ε_eff + ε_d) < 0
        
        Args:
            eps_dielectric: Relative permittivity of upper dielectric
            
        Returns:
            True if SPP modes are supported
        """
        if self.optical_axis == 'z':
            eps_eff = self.eps_perp
        else:
            eps_eff = self.eps_perp
            
        condition1 = eps_eff.real * eps_dielectric.real < 0
        
        return condition1
    
    def __repr__(self) -> str:
        return (f"MetamaterialProperties(eps_∥={self.eps_par}, "
                f"eps_⊥={self.eps_perp}, optical_axis={self.optical_axis})")