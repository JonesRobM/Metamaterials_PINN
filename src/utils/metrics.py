"""
Metrics for evaluating electromagnetic PINN performance.

Provides comprehensive evaluation metrics for Physics-Informed Neural Networks
applied to electromagnetic problems, particularly Surface Plasmon Polaritons.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
import warnings
from collections import defaultdict

try:
    from ..physics.maxwell_equations import MaxwellEquations
    from ..physics.metamaterial import MetamaterialProperties
    from ..physics.boundary_conditions import BoundaryConditions
except ImportError:
    # Fallback for testing
    MaxwellEquations = None
    MetamaterialProperties = None
    BoundaryConditions = None


@dataclass
class MetricResult:
    """Container for metric evaluation results."""
    name: str
    value: float
    std: Optional[float] = None
    unit: Optional[str] = None
    description: Optional[str] = None
    is_converged: Optional[bool] = None
    threshold: Optional[float] = None


class BaseMetric(ABC):
    """Abstract base class for electromagnetic metrics."""
    
    def __init__(self, name: str, device: str = 'cpu'):
        self.name = name
        self.device = device
        self.history = []
        
    @abstractmethod
    def compute(self, **kwargs) -> MetricResult:
        """Compute the metric value."""
        pass
    
    def update_history(self, result: MetricResult):
        """Update metric history."""
        self.history.append(result.value)
        
    def get_trend(self, window: int = 10) -> str:
        """Get trend direction over recent history."""
        if len(self.history) < window:
            return "insufficient_data"
        
        recent = self.history[-window:]
        slope = np.polyfit(range(len(recent)), recent, 1)[0]
        
        if abs(slope) < 1e-8:
            return "stable"
        elif slope < 0:
            return "decreasing"
        else:
            return "increasing"


class MaxwellResidualMetrics(BaseMetric):
    """Metrics for Maxwell equation residuals."""
    
    def __init__(self, frequency: float, device: str = 'cpu'):
        super().__init__("maxwell_residuals", device)
        self.maxwell_solver = MaxwellEquations(frequency) if MaxwellEquations else None
        self.frequency = frequency
        
    def compute(self, 
                network: torch.nn.Module,
                coords: torch.Tensor,
                epsilon_tensor: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, MetricResult]:
        """
        Compute Maxwell equation residuals.
        
        Args:
            network: PINN network
            coords: Evaluation coordinates
            epsilon_tensor: Material permittivity tensor
            
        Returns:
            Dictionary of residual metrics
        """
        coords = coords.to(self.device)
        coords.requires_grad_(True)
        
        # Get electromagnetic fields
        if hasattr(network, 'get_fields'):
            E_field, H_field = network.get_fields(coords)
        else:
            fields = network(coords)
            if fields.shape[-1] == 2:  # Complex fields
                E_field = fields[:, :3]
                H_field = fields[:, 3:6] if fields.shape[1] >= 6 else torch.zeros_like(E_field)
            else:  # Real fields
                E_field = torch.complex(fields[:, :3], fields[:, 3:6])
                H_field = torch.complex(fields[:, 6:9], fields[:, 9:12])
        
        results = {}
        
        if self.maxwell_solver:
            # Use physics module if available
            if epsilon_tensor is None:
                epsilon_tensor = torch.eye(3, dtype=torch.complex64, device=self.device)\
                    .unsqueeze(0).expand(coords.shape[0], -1, -1)
            
            total_residual = self.maxwell_solver.total_residual(E_field, H_field, coords, epsilon_tensor)
            
            # Split residuals by equation
            curl_E_residual = total_residual[:, :6]  # Real + imag parts
            curl_H_residual = total_residual[:, 6:12]
            div_E_residual = total_residual[:, 12:14]
            div_B_residual = total_residual[:, 14:16]
            
            results.update({
                'curl_E': MetricResult(
                    name='curl_E_residual',
                    value=float(torch.mean(torch.norm(curl_E_residual, dim=1))),
                    std=float(torch.std(torch.norm(curl_E_residual, dim=1))),
                    unit='dimensionless',
                    description='Faraday law residual: ∇×E + iωμ₀H'
                ),
                'curl_H': MetricResult(
                    name='curl_H_residual', 
                    value=float(torch.mean(torch.norm(curl_H_residual, dim=1))),
                    std=float(torch.std(torch.norm(curl_H_residual, dim=1))),
                    unit='dimensionless',
                    description='Ampère law residual: ∇×H - iωε₀εᵣE'
                ),
                'div_E': MetricResult(
                    name='div_E_residual',
                    value=float(torch.mean(torch.abs(div_E_residual))),
                    std=float(torch.std(torch.abs(div_E_residual))),
                    unit='dimensionless',
                    description='Gauss law residual: ∇·(εᵣE)'
                ),
                'div_B': MetricResult(
                    name='div_B_residual',
                    value=float(torch.mean(torch.abs(div_B_residual))),
                    std=float(torch.std(torch.abs(div_B_residual))),
                    unit='dimensionless',
                    description='No magnetic monopoles: ∇·B'
                )
            })
        else:
            # Fallback: simplified residual calculation
            results['total_residual'] = MetricResult(
                name='total_maxwell_residual',
                value=float(torch.mean(torch.norm(E_field, dim=1) + torch.norm(H_field, dim=1))),
                unit='dimensionless',
                description='Simplified Maxwell residual estimate'
            )
        
        return results


class BoundaryConditionMetrics(BaseMetric):
    """Metrics for boundary condition violations."""
    
    def __init__(self, interface_normal: Tuple[float, float, float] = (0, 0, 1), device: str = 'cpu'):
        super().__init__("boundary_conditions", device)
        self.boundary_solver = BoundaryConditions(interface_normal) if BoundaryConditions else None
        self.interface_normal = torch.tensor(interface_normal, device=device)
        
    def compute(self,
                network: torch.nn.Module,
                interface_coords: torch.Tensor,
                epsilon_metamaterial: torch.Tensor,
                epsilon_dielectric: float = 1.0,
                **kwargs) -> Dict[str, MetricResult]:
        """
        Compute boundary condition violations.
        
        Args:
            network: PINN network
            interface_coords: Coordinates on the interface
            epsilon_metamaterial: Metamaterial permittivity tensor
            epsilon_dielectric: Dielectric permittivity
            
        Returns:
            Dictionary of boundary condition metrics
        """
        interface_coords = interface_coords.to(self.device)
        
        # Evaluate fields at interface
        if hasattr(network, 'get_fields'):
            E_field, H_field = network.get_fields(interface_coords)
        else:
            fields = network(interface_coords)
            E_field = fields[:, :3]
            H_field = fields[:, 3:6] if fields.shape[1] >= 6 else torch.zeros_like(E_field)
        
        results = {}
        
        if self.boundary_solver:
            # Use physics module
            eps_diel_tensor = torch.eye(3, dtype=torch.complex64, device=self.device)\
                .unsqueeze(0).expand(interface_coords.shape[0], -1, -1) * epsilon_dielectric
            
            boundary_residuals = self.boundary_solver.spp_boundary_conditions(
                E_field, H_field, E_field, H_field,  # Same fields for simplicity
                epsilon_metamaterial, epsilon_dielectric
            )
            
            results['boundary_total'] = MetricResult(
                name='boundary_total',
                value=float(torch.mean(torch.norm(boundary_residuals, dim=1))),
                std=float(torch.std(torch.norm(boundary_residuals, dim=1))),
                unit='dimensionless',
                description='Total boundary condition violation'
            )
        else:
            # Simplified boundary check
            field_magnitude = torch.norm(E_field, dim=1) + torch.norm(H_field, dim=1)
            results['field_continuity'] = MetricResult(
                name='field_continuity',
                value=float(torch.std(field_magnitude)),
                unit='dimensionless',
                description='Field continuity estimate'
            )
        
        return results


class SPPPhysicsMetrics(BaseMetric):
    """Metrics specific to Surface Plasmon Polariton physics."""
    
    def __init__(self, 
                 frequency: float,
                 interface_position: float = 0.0,
                 expected_k_spp: Optional[complex] = None,
                 device: str = 'cpu'):
        super().__init__("spp_physics", device)
        self.frequency = frequency
        self.interface_z = interface_position
        self.expected_k_spp = expected_k_spp
        
    def compute(self,
                network: torch.nn.Module,
                coords: torch.Tensor,
                metamaterial_props: Optional[Any] = None,
                **kwargs) -> Dict[str, MetricResult]:
        """
        Compute SPP-specific physics metrics.
        
        Args:
            network: PINN network
            coords: Evaluation coordinates
            metamaterial_props: Metamaterial properties object
            
        Returns:
            Dictionary of SPP physics metrics
        """
        coords = coords.to(self.device)
        
        # Get fields
        if hasattr(network, 'get_fields'):
            E_field, H_field = network.get_fields(coords)
        else:
            fields = network(coords)
            E_field = fields[:, :3]
            H_field = fields[:, 3:6] if fields.shape[1] >= 6 else torch.zeros_like(E_field)
        
        results = {}
        
        # Field decay analysis
        z_coords = coords[:, 2] if coords.shape[1] > 2 else coords[:, 1]
        field_magnitude = torch.norm(E_field, dim=1)
        
        # Separate fields above and below interface
        above_interface = z_coords > self.interface_z
        below_interface = z_coords < self.interface_z
        
        if torch.any(above_interface) and torch.any(below_interface):
            decay_above = self._analyze_field_decay(
                z_coords[above_interface] - self.interface_z,
                field_magnitude[above_interface]
            )
            decay_below = self._analyze_field_decay(
                torch.abs(z_coords[below_interface] - self.interface_z),
                field_magnitude[below_interface]
            )
            
            results.update({
                'decay_length_dielectric': MetricResult(
                    name='decay_length_dielectric',
                    value=decay_above,
                    unit='m',
                    description='SPP decay length in dielectric'
                ),
                'decay_length_metamaterial': MetricResult(
                    name='decay_length_metamaterial', 
                    value=decay_below,
                    unit='m',
                    description='SPP decay length in metamaterial'
                )
            })
        
        # Field enhancement at interface
        interface_mask = torch.abs(z_coords - self.interface_z) < 1e-8
        if torch.any(interface_mask):
            interface_field = torch.mean(field_magnitude[interface_mask])
            bulk_field = torch.mean(field_magnitude[~interface_mask])
            enhancement = interface_field / (bulk_field + 1e-10)
            
            results['field_enhancement'] = MetricResult(
                name='field_enhancement',
                value=float(enhancement),
                unit='dimensionless',
                description='Field enhancement at interface'
            )
        
        # Dispersion relation accuracy (if expected k_spp provided)
        if self.expected_k_spp and metamaterial_props:
            try:
                computed_k_spp = self._estimate_wavevector(coords, E_field)
                k_error = abs(computed_k_spp - self.expected_k_spp) / abs(self.expected_k_spp)
                
                results['dispersion_accuracy'] = MetricResult(
                    name='dispersion_accuracy',
                    value=float(k_error),
                    unit='relative error',
                    description='SPP dispersion relation accuracy'
                )
            except Exception as e:
                warnings.warn(f"Could not compute dispersion accuracy: {e}")
        
        return results
    
    def _analyze_field_decay(self, distances: torch.Tensor, field_values: torch.Tensor) -> float:
        """Analyze exponential field decay."""
        if len(distances) < 3:
            return float('nan')
        
        # Sort by distance
        sorted_indices = torch.argsort(distances)
        sorted_distances = distances[sorted_indices]
        sorted_fields = field_values[sorted_indices]
        
        # Fit exponential decay: log(|E|) = log(|E0|) - z/δ
        log_fields = torch.log(sorted_fields + 1e-10)
        
        # Linear regression
        X = torch.stack([torch.ones_like(sorted_distances), sorted_distances], dim=1)
        y = log_fields
        
        try:
            # Solve normal equations
            XtX_inv = torch.inverse(X.T @ X)
            coeffs = XtX_inv @ X.T @ y
            decay_length = -1.0 / coeffs[1]  # δ = -1/slope
            return float(decay_length.clamp(1e-9, 1e-3))  # Physical bounds
        except:
            return float('nan')
    
    def _estimate_wavevector(self, coords: torch.Tensor, E_field: torch.Tensor) -> complex:
        """Estimate SPP wavevector from field phase."""
        x_coords = coords[:, 0]
        
        # Use Ex component for phase analysis
        Ex = E_field[:, 0] if E_field.is_complex() else torch.complex(E_field[:, 0], torch.zeros_like(E_field[:, 0]))
        
        # Sort by x-coordinate
        sorted_indices = torch.argsort(x_coords)
        sorted_x = x_coords[sorted_indices]
        sorted_Ex = Ex[sorted_indices]
        
        # Estimate phase gradient
        phase = torch.angle(sorted_Ex)
        
        # Unwrap phase
        phase_diff = torch.diff(phase)
        phase_diff = ((phase_diff + np.pi) % (2 * np.pi)) - np.pi  # Unwrap
        cumulative_phase = torch.cumsum(torch.cat([phase[:1], phase_diff]), dim=0)
        
        # Linear fit to get k
        x_diff = torch.diff(sorted_x)
        if len(x_diff) > 0:
            k_estimate = torch.mean(phase_diff / x_diff)
            return complex(float(k_estimate), 0)  # Simplified - only real part
        
        return complex(0, 0)


class FieldAccuracyMetrics(BaseMetric):
    """Metrics for field accuracy against analytical solutions."""
    
    def __init__(self, device: str = 'cpu'):
        super().__init__("field_accuracy", device)
        
    def compute(self,
                predicted_fields: torch.Tensor,
                reference_fields: torch.Tensor,
                field_names: Optional[List[str]] = None,
                **kwargs) -> Dict[str, MetricResult]:
        """
        Compute field accuracy metrics.
        
        Args:
            predicted_fields: Network predictions
            reference_fields: Reference solution
            field_names: Names of field components
            
        Returns:
            Dictionary of accuracy metrics
        """
        if field_names is None:
            field_names = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']
        
        predicted_fields = predicted_fields.to(self.device)
        reference_fields = reference_fields.to(self.device)
        
        results = {}
        
        # Overall metrics
        mse = torch.mean((predicted_fields - reference_fields)**2)
        mae = torch.mean(torch.abs(predicted_fields - reference_fields))
        
        # Relative error
        ref_magnitude = torch.norm(reference_fields, dim=1, keepdim=True)
        error_magnitude = torch.norm(predicted_fields - reference_fields, dim=1, keepdim=True)
        relative_error = torch.mean(error_magnitude / (ref_magnitude + 1e-10))
        
        results.update({
            'mse': MetricResult(
                name='mean_squared_error',
                value=float(mse),
                unit='field_units^2',
                description='Mean squared error in field prediction'
            ),
            'mae': MetricResult(
                name='mean_absolute_error',
                value=float(mae), 
                unit='field_units',
                description='Mean absolute error in field prediction'
            ),
            'relative_error': MetricResult(
                name='relative_error',
                value=float(relative_error),
                unit='dimensionless',
                description='Relative error in field magnitude'
            )
        })
        
        # Component-wise analysis
        n_components = min(predicted_fields.shape[1], reference_fields.shape[1], len(field_names))
        for i in range(n_components):
            pred_comp = predicted_fields[:, i]
            ref_comp = reference_fields[:, i]
            
            comp_mse = torch.mean((pred_comp - ref_comp)**2)
            comp_correlation = torch.corrcoef(torch.stack([pred_comp, ref_comp], dim=0))[0, 1]
            
            results[f'{field_names[i]}_mse'] = MetricResult(
                name=f'{field_names[i]}_mse',
                value=float(comp_mse),
                unit='field_units^2',
                description=f'MSE for {field_names[i]} component'
            )
            
            results[f'{field_names[i]}_correlation'] = MetricResult(
                name=f'{field_names[i]}_correlation',
                value=float(comp_correlation) if not torch.isnan(comp_correlation) else 0.0,
                unit='dimensionless',
                description=f'Correlation for {field_names[i]} component'
            )
        
        return results


class EnergyConservationMetrics(BaseMetric):
    """Metrics for energy conservation and Poynting vector analysis."""
    
    def __init__(self, device: str = 'cpu'):
        super().__init__("energy_conservation", device)
        
    def compute(self,
                network: torch.nn.Module,
                coords: torch.Tensor,
                **kwargs) -> Dict[str, MetricResult]:
        """
        Compute energy conservation metrics.
        
        Args:
            network: PINN network
            coords: Evaluation coordinates
            
        Returns:
            Dictionary of energy conservation metrics
        """
        coords = coords.to(self.device)
        coords.requires_grad_(True)
        
        # Get electromagnetic fields
        if hasattr(network, 'get_fields'):
            E_field, H_field = network.get_fields(coords)
        else:
            fields = network(coords)
            E_field = fields[:, :3]
            H_field = fields[:, 3:6] if fields.shape[1] >= 6 else torch.zeros_like(E_field)
        
        results = {}
        
        # Poynting vector calculation
        try:
            S = self._compute_poynting_vector(E_field, H_field)
            
            # Poynting vector divergence
            div_S = self._compute_divergence(S, coords)
            
            results['poynting_divergence'] = MetricResult(
                name='poynting_divergence',
                value=float(torch.mean(torch.abs(div_S))),
                std=float(torch.std(torch.abs(div_S))),
                unit='W/m^3',
                description='Divergence of Poynting vector (energy conservation)'
            )
            
            # Energy density
            energy_density = 0.5 * (torch.norm(E_field, dim=1)**2 + torch.norm(H_field, dim=1)**2)
            
            results['energy_density'] = MetricResult(
                name='energy_density',
                value=float(torch.mean(energy_density)),
                std=float(torch.std(energy_density)),
                unit='J/m^3',
                description='Electromagnetic energy density'
            )
            
        except Exception as e:
            warnings.warn(f"Could not compute energy metrics: {e}")
            results['energy_error'] = MetricResult(
                name='energy_computation_error',
                value=1.0,
                description=f'Energy computation failed: {e}'
            )
        
        return results
    
    def _compute_poynting_vector(self, E_field: torch.Tensor, H_field: torch.Tensor) -> torch.Tensor:
        """Compute Poynting vector S = E × H."""
        if E_field.is_complex():
            # S = (1/2) Re(E × H*)
            H_conj = torch.conj(H_field)
            S = 0.5 * torch.real(torch.cross(E_field, H_conj, dim=1))
        else:
            # Real fields
            S = torch.cross(E_field, H_field, dim=1)
        
        return S
    
    def _compute_divergence(self, vector_field: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Compute divergence of vector field."""
        divergence = torch.zeros(vector_field.shape[0], 1, device=self.device)
        
        for i in range(min(3, coords.shape[1], vector_field.shape[1])):
            grad = torch.autograd.grad(
                outputs=vector_field[:, i].sum(),
                inputs=coords,
                create_graph=True,
                retain_graph=True
            )[0][:, i]
            divergence[:, 0] += grad
        
        return divergence


class TrainingMetrics(BaseMetric):
    """Metrics for monitoring training progress."""
    
    def __init__(self, device: str = 'cpu'):
        super().__init__("training", device)
        self.loss_history = defaultdict(list)
        
    def compute(self,
                total_loss: torch.Tensor,
                loss_components: Dict[str, torch.Tensor],
                epoch: int,
                learning_rate: float,
                **kwargs) -> Dict[str, MetricResult]:
        """
        Compute training progress metrics.
        
        Args:
            total_loss: Total loss value
            loss_components: Individual loss components
            epoch: Current epoch
            learning_rate: Current learning rate
            
        Returns:
            Dictionary of training metrics
        """
        results = {}
        
        # Current loss values
        results['total_loss'] = MetricResult(
            name='total_loss',
            value=float(total_loss),
            description='Total training loss'
        )
        
        for component_name, component_loss in loss_components.items():
            results[f'loss_{component_name}'] = MetricResult(
                name=f'loss_{component_name}',
                value=float(component_loss),
                description=f'Loss component: {component_name}'
            )
        
        # Update history
        self.loss_history['total'].append(float(total_loss))
        for name, loss in loss_components.items():
            self.loss_history[name].append(float(loss))
        
        # Convergence analysis
        if len(self.loss_history['total']) > 10:
            recent_losses = self.loss_history['total'][-10:]
            loss_trend = self.get_trend()
            
            # Loss reduction rate
            if len(self.loss_history['total']) > 1:
                reduction_rate = (self.loss_history['total'][-2] - self.loss_history['total'][-1]) / \
                               (self.loss_history['total'][-2] + 1e-10)
                
                results['loss_reduction_rate'] = MetricResult(
                    name='loss_reduction_rate',
                    value=float(reduction_rate),
                    unit='fractional',
                    description='Rate of loss reduction'
                )
        
        # Learning rate
        results['learning_rate'] = MetricResult(
            name='learning_rate',
            value=learning_rate,
            description='Current learning rate'
        )
        
        return results


class MetricsCollector:
    """Centralized collector for all electromagnetic PINN metrics."""
    
    def __init__(self, 
                 config: Dict[str, Any],
                 device: str = 'cpu'):
        self.config = config
        self.device = device
        
        # Initialize metric calculators
        frequency = config.get('physics', {}).get('frequency', 1e15)
        
        self.metrics = {
            'maxwell': MaxwellResidualMetrics(frequency, device),
            'boundary': BoundaryConditionMetrics(device=device),
            'spp': SPPPhysicsMetrics(frequency, device=device),
            'accuracy': FieldAccuracyMetrics(device),
            'energy': EnergyConservationMetrics(device),
            'training': TrainingMetrics(device)
        }
        
        self.all_results = defaultdict(dict)
        
    def evaluate_all(self,
                    network: torch.nn.Module,
                    coords: torch.Tensor,
                    epoch: int,
                    **kwargs) -> Dict[str, Dict[str, MetricResult]]:
        """
        Evaluate all metrics for current network state.
        
        Args:
            network: PINN network
            coords: Evaluation coordinates
            epoch: Current training epoch
            **kwargs: Additional arguments for specific metrics
            
        Returns:
            Nested dictionary of all metric results
        """
        results = {}
        
        # Maxwell residuals
        try:
            results['maxwell'] = self.metrics['maxwell'].compute(
                network=network, coords=coords, **kwargs
            )
        except Exception as e:
            print(f"Maxwell metrics failed: {e}")
            results['maxwell'] = {}
        
        # SPP physics
        try:
            results['spp'] = self.metrics['spp'].compute(
                network=network, coords=coords, **kwargs
            )
        except Exception as e:
            print(f"SPP metrics failed: {e}")
            results['spp'] = {}
        
        # Energy conservation
        try:
            results['energy'] = self.metrics['energy'].compute(
                network=network, coords=coords, **kwargs
            )
        except Exception as e:
            print(f"Energy metrics failed: {e}")
            results['energy'] = {}
        
        # Store results
        self.all_results[epoch] = results
        
        return results
    
    def get_summary_report(self, recent_epochs: int = 10) -> str:
        """Generate a summary report of recent metrics."""
        if not self.all_results:
            return "No metrics available yet."
        
        recent_epochs_list = sorted(self.all_results.keys())[-recent_epochs:]
        report = []
        
        report.append("=== ELECTROMAGNETIC PINN METRICS SUMMARY ===")
        report.append(f"Epochs analyzed: {recent_epochs_list[0]} to {recent_epochs_list[-1]}")
        report.append("")
        
        # Maxwell residuals trend
        maxwell_losses = []
        for epoch in recent_epochs_list:
            maxwell_results = self.all_results[epoch].get('maxwell', {})
            if 'curl_E' in maxwell_results:
                maxwell_losses.append(maxwell_results['curl_E'].value)
        
        if maxwell_losses:
            report.append(f"Maxwell Residuals:")
            report.append(f"  Current curl E residual: {maxwell_losses[-1]:.2e}")
            report.append(f"  Trend: {self._analyze_trend(maxwell_losses)}")
            report.append("")
        
        # SPP physics
        spp_metrics = self.all_results[recent_epochs_list[-1]].get('spp', {})
        if spp_metrics:
            report.append("SPP Physics:")
            for name, result in spp_metrics.items():
                report.append(f"  {name}: {result.value:.3e} {result.unit or ''}")
            report.append("")
        
        # Energy conservation
        energy_metrics = self.all_results[recent_epochs_list[-1]].get('energy', {})
        if energy_metrics:
            report.append("Energy Conservation:")
            for name, result in energy_metrics.items():
                report.append(f"  {name}: {result.value:.3e} {result.unit or ''}")
        
        return "\n".join(report)
    
    def _analyze_trend(self, values: List[float]) -> str:
        """Analyze trend in metric values."""
        if len(values) < 2:
            return "insufficient data"
        
        if len(values) >= 3:
            slope = np.polyfit(range(len(values)), values, 1)[0]
            if slope < -1e-10:
                return "improving"
            elif slope > 1e-10:
                return "degrading"
            else:
                return "stable"
        else:
            return "improving" if values[-1] < values[0] else "degrading"
    
    def save_metrics(self, filepath: str):
        """Save metrics history to file."""
        import json
        
        # Convert to serializable format
        serializable = {}
        for epoch, epoch_results in self.all_results.items():
            serializable[str(epoch)] = {}
            for category, category_results in epoch_results.items():
                serializable[str(epoch)][category] = {}
                for metric_name, metric_result in category_results.items():
                    serializable[str(epoch)][category][metric_name] = {
                        'value': metric_result.value,
                        'std': metric_result.std,
                        'unit': metric_result.unit,
                        'description': metric_result.description
                    }
        
        with open(filepath, 'w') as f:
            json.dump(serializable, f, indent=2)