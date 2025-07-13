"""
Domain sampling strategies for electromagnetic PINN training.

Provides sophisticated sampling methods for metamaterial domains,
with specialised support for Surface Plasmon Polariton modeling.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from abc import ABC, abstractmethod
import warnings
from dataclasses import dataclass
from enum import Enum

try:
    from ..physics.metamaterial import MetamaterialProperties
    from ..physics.boundary_conditions import BoundaryConditions
except ImportError:
    MetamaterialProperties = None
    BoundaryConditions = None


class SamplingStrategy(Enum):
    """Enumeration of available sampling strategies."""
    UNIFORM = "uniform"
    ADAPTIVE = "adaptive"
    STRATIFIED = "stratified"
    INTERFACE_FOCUSED = "interface_focused"
    SPP_OPTIMIZED = "spp_optimized"


@dataclass
class SamplingRegion:
    """
    Definition of a sampling region within the domain.
    
    Attributes:
        name: Region identifier
        bounds: Spatial bounds [(x_min, x_max), ...]
        weight: Relative sampling weight
        material_type: Type of material in region
        density_function: Optional custom density function
    """
    name: str
    bounds: List[Tuple[float, float]]
    weight: float = 1.0
    material_type: str = "dielectric"
    density_function: Optional[Callable] = None


class DomainSampler(ABC):
    """
    Abstract base class for domain sampling strategies.
    
    Args:
        domain_bounds: Overall domain boundaries
        device: Torch device for computations
        dtype: Data type for tensors
        seed: Random seed for reproducibility
    """
    
    def __init__(self, 
                 domain_bounds: List[Tuple[float, float]],
                 device: str = 'cpu',
                 dtype: torch.dtype = torch.float32,
                 seed: Optional[int] = None):
        self.domain_bounds = domain_bounds
        self.spatial_dim = len(domain_bounds)
        self.device = device
        self.dtype = dtype
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Validate domain
        self._validate_domain()
        
        # Derived properties
        self.domain_volume = self._calculate_domain_volume()
        self.domain_center = self._calculate_domain_center()
        
    def _validate_domain(self):
        """Validate domain bounds."""
        for i, (min_val, max_val) in enumerate(self.domain_bounds):
            if min_val >= max_val:
                raise ValueError(f"Invalid domain bounds for dimension {i}: {min_val} >= {max_val}")
    
    def _calculate_domain_volume(self) -> float:
        """Calculate domain volume/area."""
        volume = 1.0
        for min_val, max_val in self.domain_bounds:
            volume *= (max_val - min_val)
        return volume
    
    def _calculate_domain_center(self) -> torch.Tensor:
        """Calculate domain center point."""
        center = torch.zeros(self.spatial_dim, device=self.device, dtype=self.dtype)
        for i, (min_val, max_val) in enumerate(self.domain_bounds):
            center[i] = (min_val + max_val) / 2.0
        return center
    
    @abstractmethod
    def sample_points(self, n_points: int, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Sample points from the domain.
        
        Args:
            n_points: Number of points to sample
            **kwargs: Sampler-specific parameters
            
        Returns:
            Dictionary containing sampled points and metadata
        """
        pass
    
    def _uniform_sample_in_bounds(self, 
                                 n_points: int, 
                                 bounds: Optional[List[Tuple[float, float]]] = None) -> torch.Tensor:
        """Generate uniformly distributed points within bounds."""
        if bounds is None:
            bounds = self.domain_bounds
            
        points = torch.zeros(n_points, len(bounds), device=self.device, dtype=self.dtype)
        
        for i, (min_val, max_val) in enumerate(bounds):
            points[:, i] = torch.rand(n_points, device=self.device, dtype=self.dtype) * \
                          (max_val - min_val) + min_val
        
        return points
    
    def _is_point_in_region(self, point: torch.Tensor, region: SamplingRegion) -> bool:
        """Check if point is within sampling region."""
        for i, (min_val, max_val) in enumerate(region.bounds):
            if i >= len(point) or point[i] < min_val or point[i] > max_val:
                return False
        return True


class UniformSampler(DomainSampler):
    """
    Simple uniform sampling across the entire domain.
    
    Args:
        domain_bounds: Domain boundaries
        exclusion_regions: List of regions to exclude from sampling
        **kwargs: Base class arguments
    """
    
    def __init__(self, 
                 domain_bounds: List[Tuple[float, float]],
                 exclusion_regions: Optional[List[SamplingRegion]] = None,
                 **kwargs):
        super().__init__(domain_bounds, **kwargs)
        self.exclusion_regions = exclusion_regions or []
    
    def sample_points(self, n_points: int, **kwargs) -> Dict[str, torch.Tensor]:
        """Sample points uniformly across domain."""
        if not self.exclusion_regions:
            points = self._uniform_sample_in_bounds(n_points)
            return {
                'points': points,
                'regions': torch.zeros(n_points, dtype=torch.long, device=self.device),
                'weights': torch.ones(n_points, device=self.device, dtype=self.dtype)
            }
        
        # Sample with exclusions
        points = []
        attempts = 0
        max_attempts = n_points * 10
        
        while len(points) < n_points and attempts < max_attempts:
            candidates = self._uniform_sample_in_bounds(n_points * 2)
            
            # Filter exclusions
            for candidate in candidates:
                if len(points) >= n_points:
                    break
                
                excluded = False
                for region in self.exclusion_regions:
                    if self._is_point_in_region(candidate, region):
                        excluded = True
                        break
                
                if not excluded:
                    points.append(candidate)
            
            attempts += 1
        
        if len(points) < n_points:
            warnings.warn(f"Could only generate {len(points)} points instead of {n_points}")
        
        points_tensor = torch.stack(points[:n_points], dim=0)
        return {
            'points': points_tensor,
            'regions': torch.zeros(len(points_tensor), dtype=torch.long, device=self.device),
            'weights': torch.ones(len(points_tensor), device=self.device, dtype=self.dtype)
        }


class StratifiedSampler(DomainSampler):
    """
    Stratified sampling across multiple regions with different densities.
    
    Args:
        domain_bounds: Overall domain boundaries
        regions: List of sampling regions
        normalize_weights: Whether to normalize region weights
        **kwargs: Base class arguments
    """
    
    def __init__(self, 
                 domain_bounds: List[Tuple[float, float]],
                 regions: List[SamplingRegion],
                 normalize_weights: bool = True,
                 **kwargs):
        super().__init__(domain_bounds, **kwargs)
        self.regions = regions
        self.normalize_weights = normalize_weights
        
        if normalize_weights:
            self._normalize_region_weights()
    
    def _normalize_region_weights(self):
        """Normalize region weights to sum to 1."""
        total_weight = sum(region.weight for region in self.regions)
        if total_weight > 0:
            for region in self.regions:
                region.weight /= total_weight
    
    def sample_points(self, n_points: int, **kwargs) -> Dict[str, torch.Tensor]:
        """Sample points using stratified strategy."""
        all_points = []
        all_regions = []
        all_weights = []
        
        for region_id, region in enumerate(self.regions):
            # Calculate points for this region
            region_points = max(1, int(n_points * region.weight))
            
            if region.density_function is not None:
                points = self._sample_with_density(region_points, region)
            else:
                points = self._uniform_sample_in_bounds(region_points, region.bounds)
            
            all_points.append(points)
            all_regions.extend([region_id] * len(points))
            all_weights.extend([region.weight] * len(points))
        
        # Combine all points
        points_tensor = torch.cat(all_points, dim=0)
        regions_tensor = torch.tensor(all_regions, dtype=torch.long, device=self.device)
        weights_tensor = torch.tensor(all_weights, device=self.device, dtype=self.dtype)
        
        # Truncate or pad to exactly n_points
        if len(points_tensor) > n_points:
            indices = torch.randperm(len(points_tensor))[:n_points]
            points_tensor = points_tensor[indices]
            regions_tensor = regions_tensor[indices]
            weights_tensor = weights_tensor[indices]
        elif len(points_tensor) < n_points:
            # Pad with uniform samples
            remaining = n_points - len(points_tensor)
            extra_points = self._uniform_sample_in_bounds(remaining)
            extra_regions = torch.zeros(remaining, dtype=torch.long, device=self.device)
            extra_weights = torch.ones(remaining, device=self.device, dtype=self.dtype)
            
            points_tensor = torch.cat([points_tensor, extra_points], dim=0)
            regions_tensor = torch.cat([regions_tensor, extra_regions], dim=0)
            weights_tensor = torch.cat([weights_tensor, extra_weights], dim=0)
        
        return {
            'points': points_tensor,
            'regions': regions_tensor,
            'weights': weights_tensor,
            'region_info': [region.name for region in self.regions]
        }
    
    def _sample_with_density(self, n_points: int, region: SamplingRegion) -> torch.Tensor:
        """Sample points according to region's density function."""
        points = []
        max_attempts = n_points * 10
        attempts = 0
        
        while len(points) < n_points and attempts < max_attempts:
            candidates = self._uniform_sample_in_bounds(n_points * 2, region.bounds)
            
            for candidate in candidates:
                if len(points) >= n_points:
                    break
                
                # Evaluate density function
                density = region.density_function(candidate.cpu().numpy())
                accept_prob = min(1.0, density)  # Assume density <= 1
                
                if torch.rand(1).item() < accept_prob:
                    points.append(candidate)
            
            attempts += 1
        
        if len(points) < n_points:
            # Fill remaining with uniform samples
            remaining = n_points - len(points)
            extra_points = self._uniform_sample_in_bounds(remaining, region.bounds)
            points.extend(extra_points.tolist())
        
        return torch.stack([torch.tensor(p, device=self.device, dtype=self.dtype) 
                           for p in points[:n_points]], dim=0)


class InterfaceSampler(DomainSampler):
    """
    Sampling focused on interfaces between different materials.
    
    Args:
        domain_bounds: Domain boundaries
        interfaces: List of interface specifications
        interface_thickness: Thickness of interface sampling region
        interface_weight: Relative weight for interface regions
        **kwargs: Base class arguments
    """
    
    def __init__(self, 
                 domain_bounds: List[Tuple[float, float]],
                 interfaces: List[Dict[str, Any]],
                 interface_thickness: float = 1e-7,
                 interface_weight: float = 2.0,
                 **kwargs):
        super().__init__(domain_bounds, **kwargs)
        self.interfaces = interfaces
        self.interface_thickness = interface_thickness
        self.interface_weight = interface_weight
    
    def sample_points(self, n_points: int, interface_fraction: float = 0.5, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Sample points with focus on interfaces.
        
        Args:
            n_points: Total number of points
            interface_fraction: Fraction of points near interfaces
            
        Returns:
            Dictionary with sampled points and interface information
        """
        n_interface = int(n_points * interface_fraction)
        n_bulk = n_points - n_interface
        
        # Sample interface points
        interface_points = self._sample_interface_points(n_interface)
        
        # Sample bulk points
        bulk_points = self._sample_bulk_points(n_bulk)
        
        # Combine and create metadata
        all_points = torch.cat([interface_points, bulk_points], dim=0)
        
        # Create interface labels
        interface_labels = torch.cat([
            torch.ones(len(interface_points), dtype=torch.bool, device=self.device),
            torch.zeros(len(bulk_points), dtype=torch.bool, device=self.device)
        ], dim=0)
        
        # Create weights
        weights = torch.cat([
            torch.full((len(interface_points),), self.interface_weight, 
                      device=self.device, dtype=self.dtype),
            torch.ones(len(bulk_points), device=self.device, dtype=self.dtype)
        ], dim=0)
        
        return {
            'points': all_points,
            'interface_labels': interface_labels,
            'weights': weights,
            'n_interface': len(interface_points),
            'n_bulk': len(bulk_points)
        }
    
    def _sample_interface_points(self, n_points: int) -> torch.Tensor:
        """Sample points near interfaces."""
        if not self.interfaces:
            return torch.empty(0, self.spatial_dim, device=self.device, dtype=self.dtype)
        
        points_per_interface = max(1, n_points // len(self.interfaces))
        all_points = []
        
        for interface in self.interfaces:
            interface_type = interface.get('type', 'plane')
            
            if interface_type == 'plane':
                points = self._sample_plane_interface(points_per_interface, interface)
            elif interface_type == 'cylinder':
                points = self._sample_cylindrical_interface(points_per_interface, interface)
            else:
                warnings.warn(f"Unknown interface type: {interface_type}")
                continue
            
            all_points.append(points)
        
        return torch.cat(all_points, dim=0) if all_points else torch.empty(0, self.spatial_dim, 
                                                                            device=self.device, dtype=self.dtype)
    
    def _sample_plane_interface(self, n_points: int, interface: Dict) -> torch.Tensor:
        """Sample points near planar interface."""
        normal_axis = interface.get('normal_axis', 2)  # Default z-axis
        position = interface.get('position', 0.0)
        
        # Sample in interface-parallel directions
        points = self._uniform_sample_in_bounds(n_points)
        
        # Sample near interface in normal direction
        normal_offset = (torch.rand(n_points, device=self.device) - 0.5) * 2 * self.interface_thickness
        points[:, normal_axis] = position + normal_offset
        
        # Ensure within domain bounds
        min_val, max_val = self.domain_bounds[normal_axis]
        points[:, normal_axis] = torch.clamp(points[:, normal_axis], min_val, max_val)
        
        return points
    
    def _sample_cylindrical_interface(self, n_points: int, interface: Dict) -> torch.Tensor:
        """Sample points near cylindrical interface."""
        radius = interface.get('radius', 1e-6)
        axis = interface.get('axis', 2)
        center = interface.get('center', [0.0, 0.0])
        
        # Sample angular and axial coordinates
        theta = torch.rand(n_points, device=self.device) * 2 * np.pi
        
        # Sample radial positions near interface
        radial_offset = (torch.rand(n_points, device=self.device) - 0.5) * 2 * self.interface_thickness
        r = radius + radial_offset
        
        if axis == 2:  # z-axis cylinder
            x = center[0] + r * torch.cos(theta)
            y = center[1] + r * torch.sin(theta)
            z = torch.rand(n_points, device=self.device) * \
                (self.domain_bounds[2][1] - self.domain_bounds[2][0]) + self.domain_bounds[2][0]
            points = torch.stack([x, y, z], dim=1)
        else:
            raise NotImplementedError(f"Cylindrical interfaces along axis {axis} not implemented")
        
        return points
    
    def _sample_bulk_points(self, n_points: int) -> torch.Tensor:
        """Sample points away from interfaces."""
        return self._uniform_sample_in_bounds(n_points)


class SPPDomainSampler(DomainSampler):
    """
    Specialized sampler for Surface Plasmon Polariton problems.
    
    Args:
        domain_bounds: Domain boundaries
        interface_position: Metal-dielectric interface position
        spp_decay_length: Characteristic SPP field decay length
        metamaterial_bounds: Bounds of metamaterial region
        **kwargs: Base class arguments
    """
    
    def __init__(self, 
                 domain_bounds: List[Tuple[float, float]],
                 interface_position: float = 0.0,
                 spp_decay_length: float = 1e-6,
                 metamaterial_bounds: Optional[List[Tuple[float, float]]] = None,
                 **kwargs):
        super().__init__(domain_bounds, **kwargs)
        self.interface_z = interface_position
        self.decay_length = spp_decay_length
        self.metamaterial_bounds = metamaterial_bounds
        
        # SPP-specific parameters
        self.spp_focus_weight = 3.0  # Weight for SPP decay region
        self.interface_focus_weight = 5.0  # Weight for interface region
        self.propagation_weight = 1.5  # Weight along propagation direction
        
        # Create SPP-optimized regions
        self.spp_regions = self._create_spp_regions()
    
    def _create_spp_regions(self) -> List[SamplingRegion]:
        """Create sampling regions optimized for SPP physics."""
        regions = []
        
        # Interface region (highest priority)
        interface_thickness = 2 * self.decay_length
        z_axis = 2 if self.spatial_dim > 2 else 1
        
        interface_bounds = self.domain_bounds.copy()
        interface_bounds[z_axis] = (
            max(self.interface_z - interface_thickness/2, self.domain_bounds[z_axis][0]),
            min(self.interface_z + interface_thickness/2, self.domain_bounds[z_axis][1])
        )
        
        regions.append(SamplingRegion(
            name="spp_interface",
            bounds=interface_bounds,
            weight=self.interface_focus_weight,
            material_type="interface",
            density_function=self._spp_interface_density
        ))
        
        # Metamaterial region with SPP decay sampling
        if self.metamaterial_bounds:
            regions.append(SamplingRegion(
                name="metamaterial_spp",
                bounds=self.metamaterial_bounds,
                weight=self.spp_focus_weight,
                material_type="metamaterial",
                density_function=self._spp_decay_density
            ))
        
        # Dielectric region
        dielectric_bounds = self.domain_bounds.copy()
        dielectric_bounds[z_axis] = (
            max(self.interface_z, self.domain_bounds[z_axis][0]),
            self.domain_bounds[z_axis][1]
        )
        
        regions.append(SamplingRegion(
            name="dielectric",
            bounds=dielectric_bounds,
            weight=1.0,
            material_type="dielectric",
            density_function=self._dielectric_decay_density
        ))
        
        return regions
    
    def _spp_interface_density(self, point: np.ndarray) -> float:
        """Density function for interface region."""
        z_coord = point[2] if len(point) > 2 else point[1]
        distance = abs(z_coord - self.interface_z)
        return float(np.exp(-distance / (0.1 * self.decay_length)))
    
    def _spp_decay_density(self, point: np.ndarray) -> float:
        """Density function following SPP decay in metamaterial."""
        z_coord = point[2] if len(point) > 2 else point[1]
        distance = abs(z_coord - self.interface_z)
        
        # Higher density closer to interface
        if z_coord < self.interface_z:  # In metamaterial
            return float(np.exp(-distance / self.decay_length))
        else:
            return 0.1  # Low background density
    
    def _dielectric_decay_density(self, point: np.ndarray) -> float:
        """Density function for dielectric region."""
        z_coord = point[2] if len(point) > 2 else point[1]
        distance = abs(z_coord - self.interface_z)
        
        if z_coord > self.interface_z:  # In dielectric
            return float(np.exp(-distance / self.decay_length))
        else:
            return 0.1  # Low background density
    
    def sample_points(self, n_points: int, **kwargs) -> Dict[str, torch.Tensor]:
        """Sample points optimized for SPP physics."""
        # Use stratified sampling with SPP regions
        stratified_sampler = StratifiedSampler(
            self.domain_bounds, self.spp_regions, device=self.device, dtype=self.dtype
        )
        
        result = stratified_sampler.sample_points(n_points, **kwargs)
        
        # Add SPP-specific metadata
        result['interface_position'] = self.interface_z
        result['decay_length'] = self.decay_length
        result['spp_regions'] = [region.name for region in self.spp_regions]
        
        return result


class AdaptiveSampler(DomainSampler):
    """
    Adaptive sampling based on residual feedback from PINN training.
    
    Args:
        domain_bounds: Domain boundaries
        base_sampler: Base sampler for initial points
        adaptation_rate: Rate of adaptation (0-1)
        memory_length: Number of residual evaluations to remember
        **kwargs: Base class arguments
    """
    
    def __init__(self, 
                 domain_bounds: List[Tuple[float, float]],
                 base_sampler: Optional[DomainSampler] = None,
                 adaptation_rate: float = 0.1,
                 memory_length: int = 5,
                 **kwargs):
        super().__init__(domain_bounds, **kwargs)
        self.base_sampler = base_sampler or UniformSampler(domain_bounds, **kwargs)
        self.adaptation_rate = adaptation_rate
        self.memory_length = memory_length
        
        # Adaptive sampling state
        self.residual_history = []
        self.density_estimate = None
        self.adaptation_count = 0
    
    def sample_points(self, n_points: int, **kwargs) -> Dict[str, torch.Tensor]:
        """Sample points using adaptive strategy."""
        if self.density_estimate is None or self.adaptation_count == 0:
            # Use base sampler initially
            return self.base_sampler.sample_points(n_points, **kwargs)
        
        # Use learned density for sampling
        return self._adaptive_sample(n_points, **kwargs)
    
    def update_residuals(self, points: torch.Tensor, residuals: torch.Tensor):
        """Update sampler with residual information."""
        self.residual_history.append({
            'points': points.clone().cpu(),
            'residuals': residuals.clone().cpu(),
            'timestamp': self.adaptation_count
        })
        
        # Keep only recent history
        if len(self.residual_history) > self.memory_length:
            self.residual_history.pop(0)
        
        # Update density estimate
        self._update_density_estimate()
        self.adaptation_count += 1
    
    def _update_density_estimate(self):
        """Update density estimate based on residual history."""
        if not self.residual_history:
            return
        
        # Combine recent residual information
        all_points = []
        all_residuals = []
        
        for entry in self.residual_history:
            all_points.append(entry['points'])
            residuals = entry['residuals']
            
            # Compute residual magnitude
            if residuals.dim() > 1:
                residual_mag = torch.norm(residuals, dim=1)
            else:
                residual_mag = torch.abs(residuals)
            
            all_residuals.append(residual_mag)
        
        points = torch.cat(all_points, dim=0)
        residuals = torch.cat(all_residuals, dim=0)
        
        # Create density estimate (simple kernel density estimation)
        self.density_estimate = self._create_kernel_density(points, residuals)
    
    def _create_kernel_density(self, points: torch.Tensor, residuals: torch.Tensor) -> Callable:
        """Create kernel density estimate from points and residuals."""
        # Normalize residuals to weights
        weights = residuals / (torch.max(residuals) + 1e-8)
        weights = weights + 0.1  # Background density
        
        # Simple Gaussian kernel density
        bandwidth = 0.1 * torch.std(points, dim=0)
        
        def density_function(query_point: np.ndarray) -> float:
            query_tensor = torch.tensor(query_point, dtype=points.dtype)
            
            # Compute Gaussian kernel weights
            distances = torch.norm(points - query_tensor.unsqueeze(0), dim=1) / bandwidth.norm()
            kernel_weights = torch.exp(-0.5 * distances**2)
            
            # Weighted density
            density = torch.sum(weights * kernel_weights) / torch.sum(kernel_weights)
            return float(density.clamp(0, 10))  # Clip extreme values
        
        return density_function
    
    def _adaptive_sample(self, n_points: int, **kwargs) -> Dict[str, torch.Tensor]:
        """Sample using learned density estimate."""
        points = []
        attempts = 0
        max_attempts = n_points * 20
        
        while len(points) < n_points and attempts < max_attempts:
            # Generate candidate points
            candidates = self._uniform_sample_in_bounds(n_points * 2)
            
            # Evaluate density and accept/reject
            for candidate in candidates:
                if len(points) >= n_points:
                    break
                
                density = self.density_estimate(candidate.cpu().numpy())
                accept_prob = min(1.0, density / 10.0)  # Normalize by max density
                
                if torch.rand(1).item() < accept_prob:
                    points.append(candidate)
            
            attempts += 1
        
        # Fill any remaining points uniformly
        if len(points) < n_points:
            remaining = n_points - len(points)
            extra_points = self._uniform_sample_in_bounds(remaining)
            points.extend(extra_points.tolist())
        
        points_tensor = torch.stack([torch.tensor(p, device=self.device, dtype=self.dtype) 
                                   for p in points[:n_points]], dim=0)
        
        return {
            'points': points_tensor,
            'adaptive_iteration': self.adaptation_count,
            'weights': torch.ones(len(points_tensor), device=self.device, dtype=self.dtype),
            'sampling_method': 'adaptive'
        }