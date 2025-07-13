"""
Evaluation script for trained SPP metamaterial PINN models.

Evaluates model performance on physics metrics, generates visualizations,
and compares against analytical solutions where available.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import pandas as pd
from scipy.interpolate import griddata

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
from config import load_config, get_config
from src.models import ElectromagneticPINN, SPPNetwork, ComplexPINN
from src.physics import MaxwellEquations, MetamaterialProperties, BoundaryConditions
from src.data import SPPDomainSampler
from src.utils.metrics import compute_physics_metrics, compute_spp_metrics


class SPPEvaluator:
    """Comprehensive evaluator for trained SPP metamaterial PINN models."""
    
    def __init__(self, checkpoint_path: str, config_path: Optional[str] = None):
        """Initialize evaluator with trained model."""
        self.checkpoint_path = Path(checkpoint_path)
        self.config = self._load_config(config_path)
        self.device = self._setup_device()
        
        # Load model and training state
        self.model = None
        self.training_history = None
        self.load_checkpoint()
        
        # Initialize physics components
        self.setup_physics()
        
        # Evaluation state
        self.evaluation_results = {}
        self.field_data = {}
        
        # Setup logging
        self.setup_logging()
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration."""
        if config_path:
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            return load_config()
    
    def _setup_device(self) -> torch.device:
        """Setup computation device."""
        device_config = get_config('hardware.device', 'auto')
        if device_config == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_config)
        return device
    
    def setup_logging(self):
        """Setup logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_checkpoint(self):
        """Load trained model from checkpoint."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.config = checkpoint.get('config', self.config)
        
        # Initialize model architecture
        model_type = self.config.get('model', {}).get('type', 'SPPNetwork')
        
        if model_type == 'SPPNetwork':
            self.model = SPPNetwork(
                interface_position=get_config('metamaterial.interface.position', 0.0),
                metal_permittivity=get_config('metamaterial.permittivity.parallel'),
                dielectric_permittivity=get_config('dielectric.superstrate.permittivity'),
                frequency=get_config('physics.angular_frequency'),
                spatial_dim=get_config('domain.spatial_dim', 3),
                hidden_dims=get_config('network.hidden_layers', [64, 64, 64, 64])
            )
        elif model_type == 'ComplexPINN':
            self.model = ComplexPINN(
                spatial_dim=get_config('domain.spatial_dim', 3),
                field_components=get_config('network.output_dim', 6),
                hidden_dims=get_config('network.hidden_layers', [64, 64, 64, 64]),
                frequency=get_config('physics.angular_frequency')
            )
        
        # Load trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load training history
        self.training_history = checkpoint.get('training_history', {})
        
        self.logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def setup_physics(self):
        """Initialize physics solvers for validation."""
        self.physics_solver = MaxwellEquations(
            omega=get_config('physics.angular_frequency'),
            mu0=get_config('physics.constants.vacuum_permeability'),
            eps0=get_config('physics.constants.vacuum_permittivity')
        )
        
        self.metamaterial = MetamaterialProperties(
            eps_parallel=get_config('metamaterial.permittivity.parallel'),
            eps_perpendicular=get_config('metamaterial.permittivity.perpendicular'),
            optical_axis=get_config('metamaterial.optical_axis', 'z')
        )
        
        self.boundary_solver = BoundaryConditions(
            interface_normal=get_config('interface.boundary_type', (0, 0, 1))
        )
    
    def evaluate_maxwell_equations(self, coords: torch.Tensor) -> Dict[str, float]:
        """Evaluate Maxwell equation residuals."""
        self.logger.info("Evaluating Maxwell equation compliance...")
        
        coords.requires_grad_(True)
        
        with torch.no_grad():
            fields = self.model(coords)
            E_field, H_field = self.model.get_fields(coords) if hasattr(self.model, 'get_fields') else (fields[:, :3], fields[:, 3:])
        
        # Get material properties
        eps_tensor = self._get_material_properties(coords)
        
        # Compute Maxwell residuals
        total_residual = self.physics_solver.total_residual(E_field, H_field, coords, eps_tensor)
        
        # Separate residual components
        curl_E_res = self.physics_solver.curl_E_residual(E_field, H_field, coords)
        curl_H_res = self.physics_solver.curl_H_residual(E_field, H_field, coords, eps_tensor)
        div_E_res = self.physics_solver.divergence_E_residual(E_field, coords, eps_tensor)
        div_H_res = self.physics_solver.divergence_B_residual(H_field, coords)
        
        metrics = {
            'total_residual_norm': torch.norm(total_residual).item(),
            'curl_E_residual': torch.norm(curl_E_res).item(),
            'curl_H_residual': torch.norm(curl_H_res).item(),
            'div_E_residual': torch.norm(div_E_res).item(),
            'div_H_residual': torch.norm(div_H_res).item(),
            'max_residual': torch.max(torch.abs(total_residual)).item(),
            'mean_residual': torch.mean(torch.abs(total_residual)).item()
        }
        
        self.logger.info(f"Maxwell residuals - Total: {metrics['total_residual_norm']:.2e}, "
                        f"Max: {metrics['max_residual']:.2e}")
        
        return metrics
    
    def evaluate_spp_properties(self, coords: torch.Tensor) -> Dict[str, float]:
        """Evaluate SPP-specific properties."""
        self.logger.info("Evaluating SPP properties...")
        
        with torch.no_grad():
            fields = self.model(coords)
            E_field, H_field = self.model.get_fields(coords) if hasattr(self.model, 'get_fields') else (fields[:, :3], fields[:, 3:])
        
        # Extract z-coordinate (interface normal direction)
        z_coords = coords[:, 2] if coords.shape[1] > 2 else coords[:, 1]
        interface_z = get_config('metamaterial.interface.position', 0.0)
        
        # Calculate field decay characteristics
        field_magnitude = torch.norm(E_field, dim=1)
        
        # Find interface and decay points
        interface_mask = torch.abs(z_coords - interface_z) < 1e-8
        decay_distances = torch.abs(z_coords - interface_z)
        
        # SPP decay analysis
        decay_metrics = self._analyze_field_decay(z_coords, field_magnitude, interface_z)
        
        # Propagation analysis (if applicable)
        propagation_metrics = self._analyze_propagation_characteristics(coords, E_field)
        
        # Theoretical SPP properties
        theoretical_metrics = self._compute_theoretical_spp_properties()
        
        metrics = {
            **decay_metrics,
            **propagation_metrics,
            **theoretical_metrics
        }
        
        self.logger.info(f"SPP analysis - Decay length: {metrics.get('fitted_decay_length', 0):.2e} m, "
                        f"Propagation: {metrics.get('propagation_constant', 0):.2e} m⁻¹")
        
        return metrics
    
    def _analyze_field_decay(self, z_coords: torch.Tensor, field_magnitude: torch.Tensor, 
                           interface_z: float) -> Dict[str, float]:
        """Analyze SPP field decay characteristics."""
        # Sort by distance from interface
        distances = torch.abs(z_coords - interface_z)
        sort_idx = torch.argsort(distances)
        
        sorted_distances = distances[sort_idx].cpu().numpy()
        sorted_fields = field_magnitude[sort_idx].cpu().numpy()
        
        # Remove points at interface (distance = 0)
        nonzero_mask = sorted_distances > 1e-9
        if not np.any(nonzero_mask):
            return {'fitted_decay_length': 0.0, 'decay_r_squared': 0.0}
        
        distances_fit = sorted_distances[nonzero_mask]
        fields_fit = sorted_fields[nonzero_mask]
        
        # Fit exponential decay: |E| = A * exp(-z/δ)
        try:
            log_fields = np.log(fields_fit + 1e-12)  # Avoid log(0)
            fit_coeffs = np.polyfit(distances_fit, log_fields, 1)
            
            fitted_decay_length = -1.0 / fit_coeffs[0] if fit_coeffs[0] != 0 else float('inf')
            
            # Calculate R²
            log_fields_pred = np.polyval(fit_coeffs, distances_fit)
            ss_res = np.sum((log_fields - log_fields_pred) ** 2)
            ss_tot = np.sum((log_fields - np.mean(log_fields)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            return {
                'fitted_decay_length': abs(fitted_decay_length),
                'decay_r_squared': r_squared,
                'field_at_interface': float(sorted_fields[0]),
                'field_decay_rate': abs(fit_coeffs[0])
            }
        except:
            return {'fitted_decay_length': 0.0, 'decay_r_squared': 0.0}
    
    def _analyze_propagation_characteristics(self, coords: torch.Tensor, E_field: torch.Tensor) -> Dict[str, float]:
        """Analyze SPP propagation characteristics."""
        # Extract x-coordinate (propagation direction)
        x_coords = coords[:, 0]
        
        # Analyze phase progression for complex fields
        if E_field.dtype in [torch.complex64, torch.complex128]:
            # Use dominant field component
            dominant_component = torch.argmax(torch.mean(torch.abs(E_field), dim=0))
            field_component = E_field[:, dominant_component]
            
            phase = torch.angle(field_component)
            
            # Unwrap phase and fit linear progression
            phase_np = phase.cpu().numpy()
            x_np = x_coords.cpu().numpy()
            
            # Sort by x-coordinate
            sort_idx = np.argsort(x_np)
            x_sorted = x_np[sort_idx]
            phase_sorted = phase_np[sort_idx]
            
            # Unwrap phase
            phase_unwrapped = np.unwrap(phase_sorted)
            
            # Fit k*x relationship
            try:
                k_fit = np.polyfit(x_sorted, phase_unwrapped, 1)
                propagation_constant = k_fit[0]
                
                return {
                    'propagation_constant': abs(propagation_constant),
                    'wavelength': 2 * np.pi / abs(propagation_constant) if propagation_constant != 0 else float('inf')
                }
            except:
                return {'propagation_constant': 0.0, 'wavelength': float('inf')}
        
        else:
            # Real fields - analyze standing wave pattern
            field_magnitude = torch.norm(E_field, dim=1)
            return self._analyze_standing_wave_pattern(x_coords, field_magnitude)
    
    def _analyze_standing_wave_pattern(self, x_coords: torch.Tensor, field_magnitude: torch.Tensor) -> Dict[str, float]:
        """Analyze standing wave patterns in real fields."""
        x_np = x_coords.cpu().numpy()
        field_np = field_magnitude.cpu().numpy()
        
        # Sort by x-coordinate
        sort_idx = np.argsort(x_np)
        x_sorted = x_np[sort_idx]
        field_sorted = field_np[sort_idx]
        
        # Find peaks in field magnitude
        try:
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(field_sorted, height=np.max(field_sorted) * 0.1)
            
            if len(peaks) > 1:
                peak_positions = x_sorted[peaks]
                wavelengths = np.diff(peak_positions) * 2  # λ = 2 * (peak spacing)
                avg_wavelength = np.mean(wavelengths)
                propagation_constant = 2 * np.pi / avg_wavelength
                
                return {
                    'propagation_constant': propagation_constant,
                    'wavelength': avg_wavelength,
                    'num_peaks': len(peaks)
                }
        except ImportError:
            pass
        
        return {'propagation_constant': 0.0, 'wavelength': float('inf')}
    
    def _compute_theoretical_spp_properties(self) -> Dict[str, float]:
        """Compute theoretical SPP properties for comparison."""
        omega = get_config('physics.angular_frequency')
        
        try:
            # Theoretical dispersion relation
            k_real, k_imag = self.metamaterial.spp_dispersion_relation(omega)
            L_prop = self.metamaterial.propagation_length(omega)
            depth_meta = self.metamaterial.penetration_depth_metamaterial(omega)
            depth_diel = self.metamaterial.penetration_depth_dielectric(omega)
            
            return {
                'theoretical_k_real': float(k_real),
                'theoretical_k_imag': float(k_imag),
                'theoretical_propagation_length': float(L_prop),
                'theoretical_decay_meta': float(depth_meta),
                'theoretical_decay_diel': float(depth_diel)
            }
        except Exception as e:
            self.logger.warning(f"Could not compute theoretical SPP properties: {e}")
            return {}
    
    def evaluate_boundary_conditions(self, interface_coords: torch.Tensor) -> Dict[str, float]:
        """Evaluate boundary condition compliance at interfaces."""
        self.logger.info("Evaluating boundary conditions...")
        
        interface_coords.requires_grad_(True)
        
        with torch.no_grad():
            fields = self.model(interface_coords)
            E_field, H_field = self.model.get_fields(interface_coords) if hasattr(self.model, 'get_fields') else (fields[:, :3], fields[:, 3:])
        
        # Evaluate continuity conditions
        # For simplicity, evaluate field smoothness near interface
        eps = 1e-8
        normal_dir = 2 if interface_coords.shape[1] > 2 else 1
        
        # Points slightly above and below interface
        coords_above = interface_coords.clone()
        coords_below = interface_coords.clone()
        coords_above[:, normal_dir] += eps
        coords_below[:, normal_dir] -= eps
        
        with torch.no_grad():
            fields_above = self.model(coords_above)
            fields_below = self.model(coords_below)
            
            E_above, H_above = (fields_above[:, :3], fields_above[:, 3:])
            E_below, H_below = (fields_below[:, :3], fields_below[:, 3:])
        
        # Tangential field continuity (simplified)
        tangential_dims = [0, 1] if normal_dir == 2 else [0, 2] if normal_dir == 1 else [1, 2]
        
        E_tang_diff = torch.norm(E_above[:, tangential_dims] - E_below[:, tangential_dims], dim=1)
        H_tang_diff = torch.norm(H_above[:, tangential_dims] - H_below[:, tangential_dims], dim=1)
        
        metrics = {
            'tangential_E_continuity': torch.mean(E_tang_diff).item(),
            'tangential_H_continuity': torch.mean(H_tang_diff).item(),
            'max_E_discontinuity': torch.max(E_tang_diff).item(),
            'max_H_discontinuity': torch.max(H_tang_diff).item()
        }
        
        self.logger.info(f"Boundary conditions - E continuity: {metrics['tangential_E_continuity']:.2e}, "
                        f"H continuity: {metrics['tangential_H_continuity']:.2e}")
        
        return metrics
    
    def evaluate_energy_conservation(self, coords: torch.Tensor) -> Dict[str, float]:
        """Evaluate energy conservation properties."""
        self.logger.info("Evaluating energy conservation...")
        
        coords.requires_grad_(True)
        
        with torch.no_grad():
            fields = self.model(coords)
            E_field, H_field = self.model.get_fields(coords) if hasattr(self.model, 'get_fields') else (fields[:, :3], fields[:, 3:])
        
        # Poynting vector
        S = self.physics_solver.poynting_vector(E_field, H_field)
        
        # Energy density
        eps_tensor = self._get_material_properties(coords)
        electric_energy = 0.5 * torch.real(torch.sum(E_field.conj() * torch.einsum('nij,nj->ni', eps_tensor, E_field), dim=1))
        magnetic_energy = 0.5 * torch.real(torch.sum(H_field.conj() * H_field, dim=1)) / self.physics_solver.mu0
        
        total_energy = electric_energy + magnetic_energy
        
        metrics = {
            'avg_poynting_magnitude': torch.mean(torch.norm(S, dim=1)).item(),
            'avg_electric_energy': torch.mean(electric_energy).item(),
            'avg_magnetic_energy': torch.mean(magnetic_energy).item(),
            'avg_total_energy': torch.mean(total_energy).item(),
            'energy_ratio_E_to_H': (torch.mean(electric_energy) / torch.mean(magnetic_energy)).item(),
            'poynting_flow_x': torch.mean(S[:, 0]).item(),
            'poynting_flow_y': torch.mean(S[:, 1]).item(),
            'poynting_flow_z': torch.mean(S[:, 2]).item()
        }
        
        return metrics
    
    def _get_material_properties(self, coords: torch.Tensor) -> torch.Tensor:
        """Get material properties at coordinates."""
        if hasattr(self.metamaterial, 'permittivity_tensor'):
            return self.metamaterial.permittivity_tensor(coords)
        else:
            batch_size = coords.shape[0]
            eps_tensor = torch.eye(3, device=self.device, dtype=torch.complex64)
            eps_tensor[0, 0] = self.metamaterial.eps_perp
            eps_tensor[1, 1] = self.metamaterial.eps_perp
            eps_tensor[2, 2] = self.metamaterial.eps_par
            return eps_tensor.unsqueeze(0).expand(batch_size, -1, -1)
    
    def generate_field_maps(self, resolution: Tuple[int, int, int] = (100, 50, 100)) -> Dict[str, np.ndarray]:
        """Generate 3D field maps for visualization."""
        self.logger.info("Generating field maps...")
        
        # Create evaluation grid
        domain_bounds = [
            get_config('domain.x_range'),
            get_config('domain.y_range'),
            get_config('domain.z_range')
        ]
        
        x = np.linspace(domain_bounds[0][0], domain_bounds[0][1], resolution[0])
        y = np.linspace(domain_bounds[1][0], domain_bounds[1][1], resolution[1])
        z = np.linspace(domain_bounds[2][0], domain_bounds[2][1], resolution[2])
        
        # For visualization, create 2D slices
        # XZ slice (y = 0)
        y_slice = 0.0
        X, Z = np.meshgrid(x, z)
        Y = np.full_like(X, y_slice)
        
        coords_slice = torch.tensor(
            np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1),
            dtype=torch.float32, device=self.device
        )
        
        # Evaluate fields
        with torch.no_grad():
            fields = self.model(coords_slice)
            E_field, H_field = self.model.get_fields(coords_slice) if hasattr(self.model, 'get_fields') else (fields[:, :3], fields[:, 3:])
        
        # Reshape to grid
        grid_shape = X.shape
        
        # Extract field components
        field_maps = {}
        field_names = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']
        
        for i, name in enumerate(field_names[:3]):
            if E_field.dtype in [torch.complex64, torch.complex128]:
                field_maps[f'{name}_real'] = E_field[:, i].real.cpu().numpy().reshape(grid_shape)
                field_maps[f'{name}_imag'] = E_field[:, i].imag.cpu().numpy().reshape(grid_shape)
                field_maps[f'{name}_magnitude'] = torch.abs(E_field[:, i]).cpu().numpy().reshape(grid_shape)
            else:
                field_maps[name] = E_field[:, i].cpu().numpy().reshape(grid_shape)
        
        for i, name in enumerate(field_names[3:]):
            if H_field.dtype in [torch.complex64, torch.complex128]:
                field_maps[f'{name}_real'] = H_field[:, i].real.cpu().numpy().reshape(grid_shape)
                field_maps[f'{name}_imag'] = H_field[:, i].imag.cpu().numpy().reshape(grid_shape)
                field_maps[f'{name}_magnitude'] = torch.abs(H_field[:, i]).cpu().numpy().reshape(grid_shape)
            else:
                field_maps[name] = H_field[:, i].cpu().numpy().reshape(grid_shape)
        
        # Add coordinate grids
        field_maps['X'] = X
        field_maps['Z'] = Z
        field_maps['Y'] = Y
        
        # Store for later use
        self.field_data = field_maps
        
        return field_maps
    
    def create_visualizations(self, output_dir: str = './evaluation_results'):
        """Create comprehensive visualizations."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Creating visualizations in {output_path}")
        
        # Training history plots
        if self.training_history:
            self._plot_training_history(output_path)
        
        # Field distribution plots
        if self.field_data:
            self._plot_field_distributions(output_path)
        
        # SPP analysis plots
        self._plot_spp_analysis(output_path)
        
        # Physics validation plots
        self._plot_physics_validation(output_path)
    
    def _plot_training_history(self, output_path: Path):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total loss
        if 'total_loss' in self.training_history:
            axes[0, 0].semilogy(self.training_history['total_loss'])
            axes[0, 0].set_title('Total Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True)
        
        # Physics vs boundary loss
        if 'physics_loss' in self.training_history and 'boundary_loss' in self.training_history:
            axes[0, 1].semilogy(self.training_history['physics_loss'], label='Physics')
            axes[0, 1].semilogy(self.training_history['boundary_loss'], label='Boundary')
            axes[0, 1].set_title('Loss Components')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Learning rate
        if 'learning_rate' in self.training_history:
            axes[1, 0].semilogy(self.training_history['learning_rate'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].grid(True)
        
        # Validation loss
        if 'validation_loss' in self.training_history:
            axes[1, 1].semilogy(self.training_history['validation_loss'])
            axes[1, 1].set_title('Validation Loss')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_field_distributions(self, output_path: Path):
        """Plot field distribution maps."""
        if not self.field_data:
            return
        
        interface_z = get_config('metamaterial.interface.position', 0.0)
        
        # Create subplot grid for field components
        field_components = ['Ex', 'Ey', 'Ez']
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        for i, component in enumerate(field_components):
            # Real part
            if f'{component}_real' in self.field_data:
                im1 = axes[0, i].contourf(
                    self.field_data['X'], self.field_data['Z'], 
                    self.field_data[f'{component}_real'], 
                    levels=50, cmap='RdBu_r'
                )
                axes[0, i].axhline(y=interface_z, color='black', linestyle='--', alpha=0.7)
                axes[0, i].set_title(f'{component} (Real Part)')
                axes[0, i].set_xlabel('x (m)')
                axes[0, i].set_ylabel('z (m)')
                plt.colorbar(im1, ax=axes[0, i])
            
            # Magnitude
            if f'{component}_magnitude' in self.field_data:
                im2 = axes[1, i].contourf(
                    self.field_data['X'], self.field_data['Z'], 
                    self.field_data[f'{component}_magnitude'], 
                    levels=50, cmap='viridis'
                )
                axes[1, i].axhline(y=interface_z, color='white', linestyle='--', alpha=0.7)
                axes[1, i].set_title(f'|{component}|')
                axes[1, i].set_xlabel('x (m)')
                axes[1, i].set_ylabel('z (m)')
                plt.colorbar(im2, ax=axes[1, i])
        
        plt.tight_layout()
        plt.savefig(output_path / 'field_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_spp_analysis(self, output_path: Path):
        """Plot SPP-specific analysis."""
        # Sample points along interface normal for decay analysis
        interface_z = get_config('metamaterial.interface.position', 0.0)
        z_range = get_config('domain.z_range')
        
        z_points = np.linspace(z_range[0], z_range[1], 200)
        x_center = 0.0
        y_center = 0.0
        
        coords = torch.tensor(
            [[x_center, y_center, z] for z in z_points],
            dtype=torch.float32, device=self.device
        )
        
        with torch.no_grad():
            fields = self.model(coords)
            E_field, H_field = self.model.get_fields(coords) if hasattr(self.model, 'get_fields') else (fields[:, :3], fields[:, 3:])
        
        field_magnitude = torch.norm(E_field, dim=1).cpu().numpy()
        z_points_shifted = z_points - interface_z
        
        # Create decay plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Linear scale
        ax1.plot(z_points_shifted * 1e6, field_magnitude)
        ax1.axvline(x=0, color='red', linestyle='--', label='Interface')
        ax1.set_xlabel('Distance from Interface (μm)')
        ax1.set_ylabel('|E| Field Magnitude')
        ax1.set_title('SPP Field Distribution')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Log scale (for decay analysis)
        mask = field_magnitude > 1e-10  # Avoid log(0)
        ax2.semilogy(z_points_shifted[mask] * 1e6, field_magnitude[mask])
        ax2.axvline(x=0, color='red', linestyle='--', label='Interface')
        ax2.set_xlabel('Distance from Interface (μm)')
        ax2.set_ylabel('|E| Field Magnitude (log)')
        ax2.set_title('SPP Field Decay (Log Scale)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'spp_decay_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_physics_validation(self, output_path: Path):
        """Plot physics validation metrics."""
        # Sample validation points
        domain_sampler = SPPDomainSampler(
            domain_bounds=[
                get_config('domain.x_range'),
                get_config('domain.y_range'),
                get_config('domain.z_range')
            ],
            interface_position=get_config('metamaterial.interface.position', 0.0),
            device=self.device
        )
        
        val_points = domain_sampler.sample_points(5000)
        coords = val_points['points']
        
        # Evaluate metrics
        maxwell_metrics = self.evaluate_maxwell_equations(coords)
        spp_metrics = self.evaluate_spp_properties(coords)
        energy_metrics = self.evaluate_energy_conservation(coords)
        
        # Create metrics summary plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Maxwell residuals
        residual_types = ['curl_E_residual', 'curl_H_residual', 'div_E_residual', 'div_H_residual']
        residual_values = [maxwell_metrics[key] for key in residual_types]
        
        ax1.bar(residual_types, residual_values)
        ax1.set_yscale('log')
        ax1.set_title('Maxwell Equation Residuals')
        ax1.set_ylabel('Residual Magnitude')
        ax1.tick_params(axis='x', rotation=45)
        
        # Energy conservation
        energy_types = ['avg_electric_energy', 'avg_magnetic_energy']
        energy_values = [energy_metrics[key] for key in energy_types if key in energy_metrics]
        
        if energy_values:
            ax2.bar(energy_types, energy_values)
            ax2.set_title('Energy Components')
            ax2.set_ylabel('Energy Density')
            ax2.tick_params(axis='x', rotation=45)
        
        # SPP characteristics comparison
        if 'fitted_decay_length' in spp_metrics and 'theoretical_decay_meta' in spp_metrics:
            spp_comparison = {
                'Fitted Decay': spp_metrics['fitted_decay_length'],
                'Theoretical Decay': spp_metrics['theoretical_decay_meta']
            }
            ax3.bar(spp_comparison.keys(), spp_comparison.values())
            ax3.set_title('SPP Decay Length Comparison')
            ax3.set_ylabel('Decay Length (m)')
        
        # Model performance summary
        performance_metrics = {
            'Maxwell\nResidual': maxwell_metrics['total_residual_norm'],
            'Boundary\nContinuity': 0,  # Would need interface evaluation
            'Energy\nConservation': energy_metrics.get('avg_total_energy', 0)
        }
        
        ax4.bar(performance_metrics.keys(), performance_metrics.values())
        ax4.set_yscale('log')
        ax4.set_title('Overall Performance Metrics')
        ax4.set_ylabel('Metric Value')
        
        plt.tight_layout()
        plt.savefig(output_path / 'physics_validation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_results(self, output_dir: str = './evaluation_results'):
        """Export evaluation results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Exporting results to {output_path}")
        
        # Export metrics as JSON
        import json
        with open(output_path / 'evaluation_metrics.json', 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        # Export field data as NPZ
        if self.field_data:
            np.savez(output_path / 'field_data.npz', **self.field_data)
        
        # Export summary report
        self._generate_summary_report(output_path)
    
    def _generate_summary_report(self, output_path: Path):
        """Generate text summary report."""
        with open(output_path / 'evaluation_summary.txt', 'w') as f:
            f.write("SPP Metamaterial PINN Evaluation Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Model information
            f.write("Model Information:\n")
            f.write(f"  Checkpoint: {self.checkpoint_path.name}\n")
            f.write(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}\n")
            f.write(f"  Device: {self.device}\n\n")
            
            # Physics validation
            if 'maxwell_metrics' in self.evaluation_results:
                f.write("Maxwell Equation Validation:\n")
                maxwell = self.evaluation_results['maxwell_metrics']
                f.write(f"  Total residual norm: {maxwell.get('total_residual_norm', 'N/A'):.2e}\n")
                f.write(f"  Maximum residual: {maxwell.get('max_residual', 'N/A'):.2e}\n\n")
            
            # SPP properties
            if 'spp_metrics' in self.evaluation_results:
                f.write("SPP Properties:\n")
                spp = self.evaluation_results['spp_metrics']
                f.write(f"  Fitted decay length: {spp.get('fitted_decay_length', 'N/A'):.2e} m\n")
                f.write(f"  Propagation constant: {spp.get('propagation_constant', 'N/A'):.2e} m⁻¹\n\n")
            
            f.write("Evaluation completed successfully.\n")
    
    def run_full_evaluation(self, output_dir: str = './evaluation_results'):
        """Run complete evaluation pipeline."""
        self.logger.info("Starting full SPP PINN evaluation")
        
        # Sample evaluation points
        domain_sampler = SPPDomainSampler(
            domain_bounds=[
                get_config('domain.x_range'),
                get_config('domain.y_range'),
                get_config('domain.z_range')
            ],
            interface_position=get_config('metamaterial.interface.position', 0.0),
            device=self.device
        )
        
        # Physics validation
        val_points = domain_sampler.sample_points(10000)
        coords = val_points['points']
        
        self.evaluation_results['maxwell_metrics'] = self.evaluate_maxwell_equations(coords)
        self.evaluation_results['spp_metrics'] = self.evaluate_spp_properties(coords)
        self.evaluation_results['energy_metrics'] = self.evaluate_energy_conservation(coords)
        
        # Boundary condition evaluation
        interface_coords = val_points['interface']
        if len(interface_coords) > 0:
            self.evaluation_results['boundary_metrics'] = self.evaluate_boundary_conditions(interface_coords)
        
        # Generate field maps and visualizations
        self.generate_field_maps()
        self.create_visualizations(output_dir)
        
        # Export results
        self.export_results(output_dir)
        
        self.logger.info("Evaluation completed successfully")
        return self.evaluation_results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trained SPP metamaterial PINN')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, help='Path to custom config file')
    parser.add_argument('--output', type=str, default='./evaluation_results', help='Output directory')
    parser.add_argument('--device', type=str, help='Device to use (cuda/cpu)')
    parser.add_argument('--resolution', type=int, nargs=3, default=[100, 50, 100], 
                       help='Grid resolution for field maps (nx ny nz)')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = SPPEvaluator(args.checkpoint, args.config)
    
    if args.device:
        evaluator.device = torch.device(args.device)
        evaluator.model = evaluator.model.to(evaluator.device)
    
    # Run evaluation
    try:
        results = evaluator.run_full_evaluation(args.output)
        
        # Print summary
        print("\nEvaluation Summary:")
        print("-" * 30)
        
        if 'maxwell_metrics' in results:
            maxwell = results['maxwell_metrics']
            print(f"Maxwell residual: {maxwell.get('total_residual_norm', 'N/A'):.2e}")
        
        if 'spp_metrics' in results:
            spp = results['spp_metrics']
            print(f"SPP decay length: {spp.get('fitted_decay_length', 'N/A'):.2e} m")
            
        print(f"\nDetailed results saved to: {args.output}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise


if __name__ == '__main__':
    main()