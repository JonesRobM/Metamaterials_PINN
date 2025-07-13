"""
Collocation point generation for electromagnetic PINN training.

Implements intelligent sampling strategies for Maxwell's equations in
metamaterial systems, with emphasis on Surface Plasmon Polariton modeling.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from abc import ABC, abstractmethod
import warnings

try:
    from ..physics.metamaterial import MetamaterialProperties
    from ..physics.boundary_conditions import BoundaryConditions
except ImportError:
    MetamaterialProperties = None
    BoundaryConditions = None


class CollocationPointGenerator(ABC):
    """
    Abstract base class for collocation point generation.
    
    Args:
        domain_bounds: Domain boundaries [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
        device: Torch device for tensor operations
        dtype: Data type for tensors
    """
    
    def __init__(self, 
                 domain_bounds: List[Tuple[float, float]],
                 device: str = 'cpu',
                 dtype: torch.dtype = torch.float32):
        self.domain_bounds = domain_bounds
        self.spatial_dim = len(domain_bounds)
        self.device = device
        self.dtype = dtype
        
        # Validate domain bounds
        for i, (min_val, max_val) in enumerate(domain_bounds):
            if min_val >= max_val:
                raise ValueError(f"Invalid domain bounds for dimension {i}: {min_val} >= {max_val}")
    
    @abstractmethod
    def generate_points(self, n_points: int, **kwargs) -> torch.Tensor:
        """
        Generate collocation points.
        
        Args:
            n_points: Number of points to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Points tensor of shape [n_points, spatial_dim]
        """
        pass
    
    def _random_points_in_bounds(self, n_points: int) -> torch.Tensor:
        """Generate uniformly random points within domain bounds."""
        points = torch.zeros(n_points, self.spatial_dim, device=self.device, dtype=self.dtype)
        
        for i, (min_val, max_val) in enumerate(self.domain_bounds):
            points[:, i] = torch.rand(n_points, device=self.device, dtype=self.dtype) * (max_val - min_val) + min_val
        
        return points
    
    def _points_to_device(self, points: torch.Tensor) -> torch.Tensor:
        """Ensure points are on correct device with correct dtype."""
        return points.to(device=self.device, dtype=self.dtype)


class MaxwellPointGenerator(CollocationPointGenerator):
    """
    Generate collocation points for Maxwell's equations in the interior domain.
    
    Args:
        domain_bounds: Spatial domain boundaries
        exclusion_zones: Regions to avoid (e.g., PEC boundaries)
        density_function: Function for non-uniform sampling density
        **kwargs: Base class arguments
    """
    
    def __init__(self, 
                 domain_bounds: List[Tuple[float, float]],
                 exclusion_zones: Optional[List[Callable]] = None,
                 density_function: Optional[Callable] = None,
                 **kwargs):
        super().__init__(domain_bounds, **kwargs)
        self.exclusion_zones = exclusion_zones or []
        self.density_function = density_function
    
    def generate_points(self, n_points: int, **kwargs) -> torch.Tensor:
        """Generate interior domain points for Maxwell's equations."""
        if self.density_function is None and not self.exclusion_zones:
            # Simple uniform sampling
            return self._random_points_in_bounds(n_points)
        
        # Generate more points than needed to account for exclusions
        oversample_factor = 1.5 if self.exclusion_zones else 1.0
        n_candidate = int(n_points * oversample_factor)
        
        points = []
        attempts = 0
        max_attempts = 10 * n_points
        
        while len(points) < n_points and attempts < max_attempts:
            # Generate candidate points
            if self.density_function is not None:
                candidates = self._sample_with_density(n_candidate - len(points))
            else:
                candidates = self._random_points_in_bounds(n_candidate - len(points))
            
            # Filter out excluded points
            if self.exclusion_zones:
                candidates = self._filter_exclusions(candidates)
            
            points.extend(candidates.tolist())
            attempts += 1
        
        if len(points) < n_points:
            warnings.warn(f"Could only generate {len(points)} points instead of {n_points}")
        
        # Convert to tensor and truncate to requested number
        points_tensor = torch.tensor(points[:n_points], device=self.device, dtype=self.dtype)
        return points_tensor
    
    def _sample_with_density(self, n_points: int) -> torch.Tensor:
        """Sample points according to density function using rejection sampling."""
        max_density = 1.0  # Assume normalized density function
        points = []
        
        while len(points) < n_points:
            # Generate candidate points
            candidates = self._random_points_in_bounds(n_points * 2)
            
            # Evaluate density
            densities = torch.tensor([
                self.density_function(point.numpy()) for point in candidates
            ], device=self.device)
            
            # Rejection sampling
            accept_prob = densities / max_density
            accepted = torch.rand(len(candidates), device=self.device) < accept_prob
            
            points.extend(candidates[accepted].tolist())
        
        return torch.tensor(points[:n_points], device=self.device, dtype=self.dtype)
    
    def _filter_exclusions(self, points: torch.Tensor) -> torch.Tensor:
        """Filter out points in exclusion zones."""
        keep_mask = torch.ones(len(points), dtype=torch.bool, device=self.device)
        
        for exclusion_fn in self.exclusion_zones:
            for i, point in enumerate(points):
                if exclusion_fn(point.cpu().numpy()):
                    keep_mask[i] = False
        
        return points[keep_mask]


class BoundaryPointGenerator(CollocationPointGenerator):
    """
    Generate collocation points on boundaries and interfaces.
    
    Args:
        domain_bounds: Spatial domain boundaries
        interface_specs: List of interface specifications
        boundary_specs: List of boundary specifications
        **kwargs: Base class arguments
    """
    
    def __init__(self, 
                 domain_bounds: List[Tuple[float, float]],
                 interface_specs: Optional[List[Dict]] = None,
                 boundary_specs: Optional[List[Dict]] = None,
                 **kwargs):
        super().__init__(domain_bounds, **kwargs)
        self.interface_specs = interface_specs or []
        self.boundary_specs = boundary_specs or []
    
    def generate_points(self, n_points: int, interface_fraction: float = 0.7, **kwargs) -> torch.Tensor:
        """
        Generate boundary and interface points.
        
        Args:
            n_points: Total number of boundary points
            interface_fraction: Fraction of points on interfaces vs domain boundaries
            
        Returns:
            Boundary points tensor
        """
        n_interface = int(n_points * interface_fraction)
        n_boundary = n_points - n_interface
        
        points = []
        
        # Generate interface points
        if self.interface_specs and n_interface > 0:
            interface_points = self._generate_interface_points(n_interface)
            points.append(interface_points)
        
        # Generate domain boundary points
        if n_boundary > 0:
            boundary_points = self._generate_domain_boundary_points(n_boundary)
            points.append(boundary_points)
        
        if not points:
            # Fallback: generate points on domain boundaries
            return self._generate_domain_boundary_points(n_points)
        
        return torch.cat(points, dim=0)
    
    def _generate_interface_points(self, n_points: int) -> torch.Tensor:
        """Generate points on material interfaces."""
        points = []
        points_per_interface = n_points // len(self.interface_specs)
        
        for interface_spec in self.interface_specs:
            interface_type = interface_spec.get('type', 'plane')
            
            if interface_type == 'plane':
                interface_points = self._generate_plane_interface_points(
                    points_per_interface, interface_spec
                )
            elif interface_type == 'cylinder':
                interface_points = self._generate_cylindrical_interface_points(
                    points_per_interface, interface_spec
                )
            else:
                warnings.warn(f"Unknown interface type: {interface_type}")
                continue
            
            points.append(interface_points)
        
        return torch.cat(points, dim=0) if points else torch.empty(0, self.spatial_dim, 
                                                                   device=self.device, dtype=self.dtype)
    
    def _generate_plane_interface_points(self, n_points: int, spec: Dict) -> torch.Tensor:
        """Generate points on planar interface (e.g., z = 0)."""
        normal_axis = spec.get('normal_axis', 2)  # Default to z-axis
        position = spec.get('position', 0.0)  # Default to origin
        
        # Generate points in the plane
        points = self._random_points_in_bounds(n_points)
        points[:, normal_axis] = position
        
        return points
    
    def _generate_cylindrical_interface_points(self, n_points: int, spec: Dict) -> torch.Tensor:
        """Generate points on cylindrical interface."""
        radius = spec.get('radius', 1e-6)
        axis = spec.get('axis', 2)  # Default to z-axis
        center = spec.get('center', [0.0, 0.0])
        
        # Generate cylindrical coordinates
        theta = torch.rand(n_points, device=self.device) * 2 * np.pi
        
        if axis == 2:  # z-axis cylinder
            x = center[0] + radius * torch.cos(theta)
            y = center[1] + radius * torch.sin(theta)
            z = torch.rand(n_points, device=self.device) * \
                (self.domain_bounds[2][1] - self.domain_bounds[2][0]) + self.domain_bounds[2][0]
            points = torch.stack([x, y, z], dim=1)
        else:
            raise NotImplementedError(f"Cylindrical interfaces along axis {axis} not implemented")
        
        return points
    
    def _generate_domain_boundary_points(self, n_points: int) -> torch.Tensor:
        """Generate points on domain boundaries."""
        points_per_face = n_points // (2 * self.spatial_dim)
        points = []
        
        for dim in range(self.spatial_dim):
            for boundary_val in self.domain_bounds[dim]:
                face_points = self._random_points_in_bounds(points_per_face)
                face_points[:, dim] = boundary_val
                points.append(face_points)
        
        return torch.cat(points, dim=0)


class SPPCollocationGenerator(CollocationPointGenerator):
    """
    Specialized collocation point generator for Surface Plasmon Polariton problems.
    
    Args:
        domain_bounds: Spatial domain boundaries
        interface_position: z-coordinate of metal-dielectric interface
        metamaterial_region: Function defining metamaterial region
        spp_decay_length: Characteristic SPP decay length
        **kwargs: Base class arguments
    """
    
    def __init__(self, 
                 domain_bounds: List[Tuple[float, float]],
                 interface_position: float = 0.0,
                 metamaterial_region: Optional[Callable] = None,
                 spp_decay_length: float = 1e-6,
                 **kwargs):
        super().__init__(domain_bounds, **kwargs)
        self.interface_z = interface_position
        self.metamaterial_region = metamaterial_region
        self.decay_length = spp_decay_length
        
        # SPP-specific sampling parameters
        self.interface_weight = 3.0  # Higher density near interface
        self.decay_sampling = True  # Sample according to SPP decay
        
    def generate_points(self, 
                       n_points: int,
                       interior_fraction: float = 0.8,
                       **kwargs) -> Dict[str, torch.Tensor]:
        """
        Generate SPP-optimized collocation points.
        
        Args:
            n_points: Total number of points
            interior_fraction: Fraction for interior vs boundary points
            
        Returns:
            Dictionary with point categories: {'interior', 'interface', 'boundary'}
        """
        n_interior = int(n_points * interior_fraction)
        n_boundary = n_points - n_interior
        
        # Generate interior points with SPP-aware density
        interior_points = self._generate_spp_interior_points(n_interior)
        
        # Generate interface points
        interface_points = self._generate_spp_interface_points(n_boundary // 2)
        
        # Generate boundary points
        boundary_points = self._generate_domain_boundary_points(n_boundary - len(interface_points))
        
        return {
            'interior': interior_points,
            'interface': interface_points,
            'boundary': boundary_points,
            'all': torch.cat([interior_points, interface_points, boundary_points], dim=0)
        }
    
    def _generate_spp_interior_points(self, n_points: int) -> torch.Tensor:
        """Generate interior points with SPP decay-aware density."""
        points = []
        
        # Use rejection sampling with SPP decay profile
        while len(points) < n_points:
            candidates = self._random_points_in_bounds(n_points * 2)
            
            # SPP decay density function
            z_coords = candidates[:, 2] if self.spatial_dim > 2 else candidates[:, 1]
            distance_from_interface = torch.abs(z_coords - self.interface_z)
            
            # Probability proportional to exp(-|z-z0|/δ) + constant background
            decay_prob = torch.exp(-distance_from_interface / self.decay_length)
            background_prob = 0.1
            total_prob = decay_prob + background_prob
            
            # Normalize probabilities
            max_prob = torch.max(total_prob)
            accept_prob = total_prob / max_prob
            
            # Rejection sampling
            accepted = torch.rand(len(candidates), device=self.device) < accept_prob
            points.extend(candidates[accepted].tolist())
        
        points_tensor = torch.tensor(points[:n_points], device=self.device, dtype=self.dtype)
        return points_tensor
    
    def _generate_spp_interface_points(self, n_points: int) -> torch.Tensor:
        """Generate points concentrated on the SPP interface."""
        points = self._random_points_in_bounds(n_points)
        
        # Set z-coordinate to interface position
        if self.spatial_dim > 2:
            points[:, 2] = self.interface_z
        elif self.spatial_dim == 2:
            points[:, 1] = self.interface_z
        
        return points
    
    def _generate_domain_boundary_points(self, n_points: int) -> torch.Tensor:
        """Generate points on domain boundaries."""
        if n_points <= 0:
            return torch.empty(0, self.spatial_dim, device=self.device, dtype=self.dtype)
        
        points_per_face = max(1, n_points // (2 * self.spatial_dim))
        points = []
        
        for dim in range(self.spatial_dim):
            for boundary_val in self.domain_bounds[dim]:
                face_points = self._random_points_in_bounds(points_per_face)
                face_points[:, dim] = boundary_val
                points.append(face_points)
        
        all_points = torch.cat(points, dim=0)
        return all_points[:n_points]  # Truncate to requested number


class AdaptiveCollocationManager:
    """
    Manages adaptive refinement of collocation points based on residuals.
    
    Args:
        base_generator: Base collocation point generator
        max_points: Maximum total number of points
        refinement_frequency: Epochs between refinements
        residual_threshold: Threshold for high-residual regions
    """
    
    def __init__(self, 
                 base_generator: CollocationPointGenerator,
                 max_points: int = 20000,
                 refinement_frequency: int = 1000,
                 residual_threshold: float = 0.9):
        self.base_generator = base_generator
        self.max_points = max_points
        self.refinement_frequency = refinement_frequency
        self.residual_threshold = residual_threshold
        
        # Adaptive sampling state
        self.current_points = None
        self.residual_history = []
        self.refinement_count = 0
        
    def get_current_points(self, n_points: int) -> torch.Tensor:
        """Get current collocation points."""
        if self.current_points is None:
            self.current_points = self.base_generator.generate_points(n_points)
        return self.current_points
    
    def update_residuals(self, points: torch.Tensor, residuals: torch.Tensor):
        """Update residual information for adaptive refinement."""
        self.residual_history.append({
            'points': points.clone(),
            'residuals': residuals.clone(),
            'refinement_step': self.refinement_count
        })
        
        # Keep only recent history
        if len(self.residual_history) > 10:
            self.residual_history.pop(0)
    
    def should_refine(self, epoch: int) -> bool:
        """Check if refinement should occur."""
        return (epoch > 0 and 
                epoch % self.refinement_frequency == 0 and 
                len(self.residual_history) > 0)
    
    def refine_points(self, n_new_points: int) -> torch.Tensor:
        """Refine collocation points based on residual information."""
        if not self.residual_history:
            return self.base_generator.generate_points(n_new_points)
        
        # Get high-residual regions
        recent_data = self.residual_history[-1]
        points = recent_data['points']
        residuals = recent_data['residuals']
        
        # Identify high-residual points
        residual_magnitude = torch.norm(residuals, dim=1) if residuals.dim() > 1 else torch.abs(residuals)
        threshold_val = torch.quantile(residual_magnitude, self.residual_threshold)
        high_residual_mask = residual_magnitude > threshold_val
        
        if not torch.any(high_residual_mask):
            # No high-residual regions, use uniform sampling
            new_points = self.base_generator.generate_points(n_new_points)
        else:
            # Sample around high-residual regions
            high_residual_points = points[high_residual_mask]
            new_points = self._sample_around_points(high_residual_points, n_new_points)
        
        # Update current points
        if self.current_points is not None:
            total_points = torch.cat([self.current_points, new_points], dim=0)
            if len(total_points) > self.max_points:
                # Randomly subsample to maintain max_points
                indices = torch.randperm(len(total_points))[:self.max_points]
                self.current_points = total_points[indices]
            else:
                self.current_points = total_points
        else:
            self.current_points = new_points
        
        self.refinement_count += 1
        return self.current_points
    
    def _sample_around_points(self, 
                             center_points: torch.Tensor, 
                             n_samples: int,
                             radius_factor: float = 0.1) -> torch.Tensor:
        """Sample points around high-residual regions."""
        domain_size = torch.tensor([
            max_val - min_val for min_val, max_val in self.base_generator.domain_bounds
        ], device=center_points.device)
        
        # Sampling radius proportional to domain size
        radius = radius_factor * torch.min(domain_size)
        
        new_points = []
        points_per_center = max(1, n_samples // len(center_points))
        
        for center in center_points:
            # Generate points around this center
            for _ in range(points_per_center):
                # Random offset within radius
                offset = torch.randn_like(center) * radius
                new_point = center + offset
                
                # Ensure within domain bounds
                for i, (min_val, max_val) in enumerate(self.base_generator.domain_bounds):
                    new_point[i] = torch.clamp(new_point[i], min_val, max_val)
                
                new_points.append(new_point)
        
        # Fill remaining points if needed
        remaining = n_samples - len(new_points)
        if remaining > 0:
            extra_points = self.base_generator.generate_points(remaining)
            new_points.extend(extra_points.tolist())
        
        return torch.stack([torch.tensor(p, device=center_points.device, dtype=center_points.dtype) 
                           for p in new_points[:n_samples]], dim=0)