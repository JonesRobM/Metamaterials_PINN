"""
Visualization tools for electromagnetic PINN analysis.

Provides comprehensive plotting capabilities for Physics-Informed Neural Networks
applied to electromagnetic problems, particularly Surface Plasmon Polaritons.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import warnings
from dataclasses import dataclass
from pathlib import Path

# Optional advanced plotting libraries
try:
    import plotly.graph_objects as go
    import plotly.subplots as sp
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import seaborn as sns
    sns.set_style("whitegrid")
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


@dataclass
class PlotConfig:
    """Configuration for plot styling and parameters."""
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 150
    colormap: str = 'RdBu_r'
    line_width: float = 1.5
    marker_size: float = 4.0
    font_size: int = 12
    title_size: int = 14
    label_size: int = 10
    save_format: str = 'png'
    transparent: bool = False


class BasePlotter:
    """Base class for electromagnetic plotting utilities."""
    
    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()
        self._setup_matplotlib()
        
    def _setup_matplotlib(self):
        """Configure matplotlib settings."""
        plt.rcParams.update({
            'font.size': self.config.font_size,
            'axes.titlesize': self.config.title_size,
            'axes.labelsize': self.config.label_size,
            'xtick.labelsize': self.config.label_size,
            'ytick.labelsize': self.config.label_size,
            'legend.fontsize': self.config.label_size,
            'figure.dpi': self.config.dpi,
            'savefig.dpi': self.config.dpi,
            'savefig.transparent': self.config.transparent
        })
        
    def save_figure(self, fig: plt.Figure, filename: str, directory: str = "plots"):
        """Save figure with consistent formatting."""
        save_dir = Path(directory)
        save_dir.mkdir(exist_ok=True)
        
        filepath = save_dir / f"{filename}.{self.config.save_format}"
        fig.savefig(filepath, format=self.config.save_format, 
                   bbox_inches='tight', transparent=self.config.transparent)
        print(f"Saved plot: {filepath}")


class EMFieldPlotter(BasePlotter):
    """Plotter for electromagnetic field visualization."""
    
    def __init__(self, config: Optional[PlotConfig] = None):
        super().__init__(config)
        
    def plot_field_2d(self,
                     coords: torch.Tensor,
                     fields: torch.Tensor,
                     field_component: str = 'Ex',
                     plane: str = 'xy',
                     slice_position: float = 0.0,
                     interface_position: Optional[float] = None,
                     title: Optional[str] = None,
                     **kwargs) -> plt.Figure:
        """
        Plot 2D field distribution.
        
        Args:
            coords: Coordinate tensor [N, 3]
            fields: Field tensor [N, 6] or [N, 6, 2] for complex
            field_component: Component to plot ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')
            plane: Plane to plot ('xy', 'xz', 'yz')
            slice_position: Position of slice in third dimension
            interface_position: Position of material interface
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Extract field component
        component_map = {'Ex': 0, 'Ey': 1, 'Ez': 2, 'Hx': 3, 'Hy': 4, 'Hz': 5}
        comp_idx = component_map.get(field_component, 0)
        
        if fields.dim() == 3:  # Complex fields [N, 6, 2]
            field_values = fields[:, comp_idx, 0]  # Real part
        else:  # Real fields [N, 6]
            field_values = fields[:, comp_idx]
        
        # Select slice
        plane_map = {'xy': (0, 1, 2), 'xz': (0, 2, 1), 'yz': (1, 2, 0)}
        x_idx, y_idx, z_idx = plane_map[plane]
        
        # Filter points in slice
        slice_mask = torch.abs(coords[:, z_idx] - slice_position) < 0.1 * torch.std(coords[:, z_idx])
        
        if torch.sum(slice_mask) < 10:
            # If no points in slice, use all points
            slice_mask = torch.ones(len(coords), dtype=torch.bool)
        
        x_coords = coords[slice_mask, x_idx].cpu().detach().numpy()
        y_coords = coords[slice_mask, y_idx].cpu().detach().numpy()
        field_slice = field_values[slice_mask].cpu().detach().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Create scatter plot with field values as colors
        scatter = ax.scatter(x_coords, y_coords, c=field_slice, 
                           cmap=self.config.colormap, s=self.config.marker_size,
                           alpha=0.8)
        
        # Add interface line if specified
        if interface_position is not None and plane in ['xz', 'yz']:
            ax.axhline(y=interface_position, color='black', linestyle='--', 
                      linewidth=2, alpha=0.7, label='Interface')
            ax.legend()
        
        # Formatting
        axis_labels = {'xy': ('x (m)', 'y (m)'), 'xz': ('x (m)', 'z (m)'), 'yz': ('y (m)', 'z (m)')}
        ax.set_xlabel(axis_labels[plane][0])
        ax.set_ylabel(axis_labels[plane][1])
        
        if title is None:
            title = f'{field_component} field in {plane} plane'
        ax.set_title(title)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(f'{field_component} (V/m)' if 'E' in field_component else f'{field_component} (A/m)')
        
        # Equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        return fig
    
    def plot_field_magnitude_2d(self,
                               coords: torch.Tensor,
                               fields: torch.Tensor,
                               field_type: str = 'E',
                               plane: str = 'xz',
                               interface_position: Optional[float] = None,
                               log_scale: bool = True,
                               **kwargs) -> plt.Figure:
        """
        Plot field magnitude in 2D.
        
        Args:
            coords: Coordinate tensor
            fields: Field tensor
            field_type: 'E' or 'H' field
            plane: Plotting plane
            interface_position: Interface position
            log_scale: Use logarithmic color scale
            
        Returns:
            Matplotlib figure
        """
        # Calculate field magnitude
        if field_type == 'E':
            field_components = fields[:, :3]
        else:  # H field
            field_components = fields[:, 3:6]
        
        if fields.dim() == 3:  # Complex fields
            magnitude = torch.norm(field_components[:, :, 0], dim=1)  # Real part magnitude
        else:
            magnitude = torch.norm(field_components, dim=1)
        
        # Select plane coordinates
        plane_map = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
        x_idx, y_idx = plane_map[plane]
        
        x_coords = coords[:, x_idx].cpu().numpy()
        y_coords = coords[:, y_idx].cpu().numpy()
        mag_values = magnitude.cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Apply log scale if requested
        if log_scale:
            mag_values = np.log10(mag_values + 1e-10)
            label = f'log₁₀(|{field_type}|)'
        else:
            label = f'|{field_type}|'
        
        # Scatter plot
        scatter = ax.scatter(x_coords, y_coords, c=mag_values, 
                           cmap='viridis', s=self.config.marker_size, alpha=0.8)
        
        # Interface line
        if interface_position is not None and 'z' in plane:
            if plane == 'xz':
                ax.axhline(y=interface_position, color='white', linestyle='--', 
                          linewidth=2, alpha=0.8, label='Interface')
            elif plane == 'yz':
                ax.axhline(y=interface_position, color='white', linestyle='--', 
                          linewidth=2, alpha=0.8, label='Interface')
            ax.legend()
        
        # Formatting
        axis_labels = {'xy': ('x (m)', 'y (m)'), 'xz': ('x (m)', 'z (m)'), 'yz': ('y (m)', 'z (m)')}
        ax.set_xlabel(axis_labels[plane][0])
        ax.set_ylabel(axis_labels[plane][1])
        ax.set_title(f'{field_type} Field Magnitude')
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(label)
        
        ax.set_aspect('equal', adjustable='box')
        
        return fig
    
    def plot_spp_decay_profile(self,
                              coords: torch.Tensor,
                              fields: torch.Tensor,
                              interface_position: float = 0.0,
                              decay_direction: str = 'z',
                              field_component: str = 'Ex',
                              **kwargs) -> plt.Figure:
        """
        Plot SPP field decay profile perpendicular to interface.
        
        Args:
            coords: Coordinate tensor
            fields: Field tensor
            interface_position: Interface z-position
            decay_direction: Direction of decay ('z' typically)
            field_component: Field component to analyze
            
        Returns:
            Matplotlib figure
        """
        # Extract field component
        component_map = {'Ex': 0, 'Ey': 1, 'Ez': 2, 'Hx': 3, 'Hy': 4, 'Hz': 5}
        comp_idx = component_map.get(field_component, 0)
        
        if fields.dim() == 3:  # Complex fields
            field_values = torch.abs(fields[:, comp_idx, 0] + 1j * fields[:, comp_idx, 1])
        else:
            field_values = torch.abs(fields[:, comp_idx])
        
        # Get decay direction coordinates
        direction_map = {'x': 0, 'y': 1, 'z': 2}
        dir_idx = direction_map[decay_direction]
        decay_coords = coords[:, dir_idx].cpu().numpy()
        field_magnitude = field_values.cpu().numpy()
        
        # Sort by position
        sort_indices = np.argsort(decay_coords)
        sorted_coords = decay_coords[sort_indices]
        sorted_fields = field_magnitude[sort_indices]
        
        # Separate metamaterial and dielectric regions
        metamaterial_mask = sorted_coords < interface_position
        dielectric_mask = sorted_coords > interface_position
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Plot field magnitude
        ax.semilogy(sorted_coords * 1e9, sorted_fields, 'b-', 
                   linewidth=self.config.line_width, label=f'|{field_component}|')
        
        # Highlight interface
        ax.axvline(x=interface_position * 1e9, color='red', linestyle='--', 
                  linewidth=2, alpha=0.7, label='Interface')
        
        # Add exponential decay fits if enough points
        if np.sum(metamaterial_mask) > 3:
            self._fit_exponential_decay(ax, sorted_coords[metamaterial_mask], 
                                       sorted_fields[metamaterial_mask], 
                                       interface_position, 'metamaterial')
        
        if np.sum(dielectric_mask) > 3:
            self._fit_exponential_decay(ax, sorted_coords[dielectric_mask], 
                                       sorted_fields[dielectric_mask], 
                                       interface_position, 'dielectric')
        
        # Formatting
        ax.set_xlabel(f'{decay_direction} position (nm)')
        ax.set_ylabel(f'|{field_component}| (V/m)' if 'E' in field_component else f'|{field_component}| (A/m)')
        ax.set_title(f'SPP Field Decay Profile - {field_component}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def _fit_exponential_decay(self, ax, coords, field_values, interface_pos, region_name):
        """Fit and plot exponential decay."""
        try:
            # Distance from interface
            distances = np.abs(coords - interface_pos)
            
            # Fit exponential: log(|E|) = log(|E0|) - d/δ
            log_fields = np.log(field_values + 1e-10)
            
            # Linear regression
            coeffs = np.polyfit(distances, log_fields, 1)
            decay_length = -1.0 / coeffs[0]
            
            # Plot fit
            fit_distances = np.linspace(distances.min(), distances.max(), 50)
            fit_fields = np.exp(coeffs[1]) * np.exp(-fit_distances / decay_length)
            fit_coords = interface_pos + fit_distances * np.sign(coords[0] - interface_pos)
            
            ax.plot(fit_coords * 1e9, fit_fields, '--', alpha=0.7, 
                   label=f'{region_name} fit (δ={decay_length*1e9:.1f} nm)')
            
        except Exception as e:
            warnings.warn(f"Could not fit decay in {region_name}: {e}")


class ComplexFieldVisualizer(BasePlotter):
    """Visualizer for complex-valued electromagnetic fields."""
    
    def plot_complex_field(self,
                          coords: torch.Tensor,
                          complex_field: torch.Tensor,
                          component: str = 'Ex',
                          **kwargs) -> plt.Figure:
        """
        Plot complex field with magnitude and phase.
        
        Args:
            coords: Coordinate tensor
            complex_field: Complex field tensor [N, components, 2] or [N, components] complex
            component: Field component to plot
            
        Returns:
            Matplotlib figure with magnitude and phase subplots
        """
        component_map = {'Ex': 0, 'Ey': 1, 'Ez': 2, 'Hx': 3, 'Hy': 4, 'Hz': 5}
        comp_idx = component_map.get(component, 0)
        
        # Extract complex values
        if complex_field.dim() == 3:  # [N, components, 2] format
            real_part = complex_field[:, comp_idx, 0]
            imag_part = complex_field[:, comp_idx, 1]
            complex_values = torch.complex(real_part, imag_part)
        else:  # Already complex
            complex_values = complex_field[:, comp_idx]
        
        magnitude = torch.abs(complex_values).cpu().numpy()
        phase = torch.angle(complex_values).cpu().numpy()
        
        # Use x-z plane by default
        x_coords = coords[:, 0].cpu().numpy()
        z_coords = coords[:, 2].cpu().numpy()
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Magnitude plot
        scatter1 = ax1.scatter(x_coords * 1e6, z_coords * 1e6, c=magnitude, 
                              cmap='viridis', s=self.config.marker_size)
        ax1.set_xlabel('x (μm)')
        ax1.set_ylabel('z (μm)')
        ax1.set_title(f'{component} Magnitude')
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('|Field| (V/m)')
        
        # Phase plot
        scatter2 = ax2.scatter(x_coords * 1e6, z_coords * 1e6, c=phase, 
                              cmap='hsv', s=self.config.marker_size, vmin=-np.pi, vmax=np.pi)
        ax2.set_xlabel('x (μm)')
        ax2.set_ylabel('z (μm)')
        ax2.set_title(f'{component} Phase')
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Phase (rad)')
        
        # Equal aspect ratios
        ax1.set_aspect('equal', adjustable='box')
        ax2.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        return fig
    
    def plot_vector_field(self,
                         coords: torch.Tensor,
                         vector_field: torch.Tensor,
                         plane: str = 'xz',
                         subsample: int = 10,
                         **kwargs) -> plt.Figure:
        """
        Plot vector field arrows.
        
        Args:
            coords: Coordinate tensor
            vector_field: Vector field tensor [N, 3] or [N, 3, 2]
            plane: Plane to plot ('xy', 'xz', 'yz')
            subsample: Subsampling factor for arrows
            
        Returns:
            Matplotlib figure
        """
        # Select plane components
        plane_map = {'xy': (0, 1, [0, 1]), 'xz': (0, 2, [0, 2]), 'yz': (1, 2, [1, 2])}
        x_idx, y_idx, vec_indices = plane_map[plane]
        
        # Get coordinates and vectors
        x_coords = coords[::subsample, x_idx].cpu().numpy()
        y_coords = coords[::subsample, y_idx].cpu().numpy()
        
        if vector_field.dim() == 3:  # Complex vectors
            vx = vector_field[::subsample, vec_indices[0], 0].cpu().numpy()
            vy = vector_field[::subsample, vec_indices[1], 0].cpu().numpy()
        else:
            vx = vector_field[::subsample, vec_indices[0]].cpu().numpy()
            vy = vector_field[::subsample, vec_indices[1]].cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Quiver plot
        ax.quiver(x_coords * 1e6, y_coords * 1e6, vx, vy, 
                 angles='xy', scale_units='xy', scale=1, alpha=0.7)
        
        # Formatting
        axis_labels = {'xy': ('x (μm)', 'y (μm)'), 'xz': ('x (μm)', 'z (μm)'), 'yz': ('y (μm)', 'z (μm)')}
        ax.set_xlabel(axis_labels[plane][0])
        ax.set_ylabel(axis_labels[plane][1])
        ax.set_title(f'Vector Field - {plane} plane')
        ax.set_aspect('equal', adjustable='box')
        
        return fig


class TrainingPlotter(BasePlotter):
    """Plotter for training progress visualization."""
    
    def plot_training_history(self,
                             loss_history: Dict[str, List[float]],
                             metrics_history: Optional[Dict[str, List[float]]] = None,
                             **kwargs) -> plt.Figure:
        """
        Plot training loss and metrics history.
        
        Args:
            loss_history: Dictionary of loss component histories
            metrics_history: Dictionary of metric histories
            
        Returns:
            Matplotlib figure
        """
        n_plots = 2 if metrics_history else 1
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 6*n_plots))
        
        if n_plots == 1:
            axes = [axes]
        
        # Plot loss history
        ax1 = axes[0]
        for loss_name, loss_values in loss_history.items():
            epochs = range(len(loss_values))
            ax1.semilogy(epochs, loss_values, label=loss_name, 
                        linewidth=self.config.line_width)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss History')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot metrics history if provided
        if metrics_history and n_plots > 1:
            ax2 = axes[1]
            for metric_name, metric_values in metrics_history.items():
                epochs = range(len(metric_values))
                ax2.plot(epochs, metric_values, label=metric_name,
                        linewidth=self.config.line_width)
            
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Metric Value')
            ax2.set_title('Physics Metrics History')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_loss_components_breakdown(self,
                                     loss_components: Dict[str, float],
                                     **kwargs) -> plt.Figure:
        """
        Plot current loss components as pie chart and bar chart.
        
        Args:
            loss_components: Dictionary of current loss component values
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Prepare data
        names = list(loss_components.keys())
        values = list(loss_components.values())
        
        # Pie chart
        wedges, texts, autotexts = ax1.pie(values, labels=names, autopct='%1.1f%%',
                                          startangle=90)
        ax1.set_title('Loss Components Breakdown')
        
        # Bar chart (log scale)
        bars = ax2.bar(names, values)
        ax2.set_yscale('log')
        ax2.set_ylabel('Loss Value')
        ax2.set_title('Loss Components (Log Scale)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Color bars by magnitude
        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        return fig
    
    def plot_convergence_analysis(self,
                                 total_loss_history: List[float],
                                 window_size: int = 100,
                                 **kwargs) -> plt.Figure:
        """
        Analyze and plot convergence characteristics.
        
        Args:
            total_loss_history: Total loss values over epochs
            window_size: Window for moving average
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        epochs = np.arange(len(total_loss_history))
        loss_values = np.array(total_loss_history)
        
        # Main loss plot with moving average
        ax1.semilogy(epochs, loss_values, alpha=0.7, label='Loss')
        
        if len(loss_values) > window_size:
            moving_avg = np.convolve(loss_values, np.ones(window_size)/window_size, mode='valid')
            avg_epochs = epochs[window_size-1:]
            ax1.semilogy(avg_epochs, moving_avg, 'r-', linewidth=2, 
                        label=f'Moving Average ({window_size} epochs)')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss gradient (rate of change)
        if len(loss_values) > 1:
            loss_gradient = np.gradient(loss_values)
            ax2.plot(epochs[1:], loss_gradient[1:], 'g-', linewidth=self.config.line_width)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss Gradient')
            ax2.set_title('Rate of Loss Change')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return fig


class SPPAnalysisPlotter(BasePlotter):
    """Specialized plotter for SPP physics analysis."""
    
    def plot_dispersion_relation(self,
                                frequencies: np.ndarray,
                                k_values: np.ndarray,
                                analytical_k: Optional[np.ndarray] = None,
                                **kwargs) -> plt.Figure:
        """
        Plot SPP dispersion relation.
        
        Args:
            frequencies: Frequency array
            k_values: Computed wavevector values (complex)
            analytical_k: Analytical reference values
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Real part of k
        ax1.plot(frequencies * 1e-15, np.real(k_values) * 1e-6, 'b-', 
                linewidth=self.config.line_width, label='PINN')
        
        if analytical_k is not None:
            ax1.plot(frequencies * 1e-15, np.real(analytical_k) * 1e-6, 'r--',
                    linewidth=self.config.line_width, label='Analytical')
        
        ax1.set_xlabel('Frequency (PHz)')
        ax1.set_ylabel('Re(k) (μm⁻¹)')
        ax1.set_title('SPP Dispersion Relation - Real Part')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Imaginary part of k
        ax2.plot(frequencies * 1e-15, np.imag(k_values) * 1e-6, 'b-',
                linewidth=self.config.line_width, label='PINN')
        
        if analytical_k is not None:
            ax2.plot(frequencies * 1e-15, np.imag(analytical_k) * 1e-6, 'r--',
                    linewidth=self.config.line_width, label='Analytical')
        
        ax2.set_xlabel('Frequency (PHz)')
        ax2.set_ylabel('Im(k) (μm⁻¹)')
        ax2.set_title('SPP Dispersion Relation - Imaginary Part')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_penetration_depths(self,
                              frequencies: np.ndarray,
                              delta_metal: np.ndarray,
                              delta_dielectric: np.ndarray,
                              **kwargs) -> plt.Figure:
        """
        Plot SPP penetration depths.
        
        Args:
            frequencies: Frequency array
            delta_metal: Penetration depths in metal
            delta_dielectric: Penetration depths in dielectric
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        ax.semilogy(frequencies * 1e-15, delta_metal * 1e9, 'b-',
                   linewidth=self.config.line_width, label='Metamaterial')
        ax.semilogy(frequencies * 1e-15, delta_dielectric * 1e9, 'r-',
                   linewidth=self.config.line_width, label='Dielectric')
        
        ax.set_xlabel('Frequency (PHz)')
        ax.set_ylabel('Penetration Depth (nm)')
        ax.set_title('SPP Penetration Depths')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_field_enhancement_map(self,
                                  coords: torch.Tensor,
                                  field_enhancement: torch.Tensor,
                                  interface_position: float = 0.0,
                                  **kwargs) -> plt.Figure:
        """
        Plot field enhancement factor across the domain.
        
        Args:
            coords: Coordinate tensor
            field_enhancement: Enhancement factor at each point
            interface_position: Interface z-position
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        x_coords = coords[:, 0].cpu().numpy()
        z_coords = coords[:, 2].cpu().numpy()
        enhancement = field_enhancement.cpu().numpy()
        
        # Use log scale for enhancement
        log_enhancement = np.log10(enhancement + 1e-10)
        
        scatter = ax.scatter(x_coords * 1e6, z_coords * 1e6, c=log_enhancement,
                           cmap='hot', s=self.config.marker_size)
        
        # Interface line
        ax.axhline(y=interface_position * 1e6, color='cyan', linestyle='--',
                  linewidth=2, alpha=0.8, label='Interface')
        
        ax.set_xlabel('x (μm)')
        ax.set_ylabel('z (μm)')
        ax.set_title('Field Enhancement Factor')
        ax.legend()
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('log₁₀(Enhancement)')
        
        ax.set_aspect('equal', adjustable='box')
        
        return fig


class DispersionPlotter(BasePlotter):
    """Plotter for dispersion analysis and band structure."""
    
    def plot_band_structure(self,
                           k_points: np.ndarray,
                           frequencies: np.ndarray,
                           mode_labels: Optional[List[str]] = None,
                           **kwargs) -> plt.Figure:
        """
        Plot band structure for metamaterial modes.
        
        Args:
            k_points: Wavevector points
            frequencies: Frequency eigenvalues [k_points, modes]
            mode_labels: Labels for different modes
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        n_modes = frequencies.shape[1] if frequencies.ndim > 1 else 1
        colors = plt.cm.tab10(np.linspace(0, 1, n_modes))
        
        for mode_idx in range(n_modes):
            if frequencies.ndim > 1:
                freq_mode = frequencies[:, mode_idx]
            else:
                freq_mode = frequencies
            
            label = mode_labels[mode_idx] if mode_labels else f'Mode {mode_idx}'
            ax.plot(k_points * 1e-6, freq_mode * 1e-15, 
                   color=colors[mode_idx], linewidth=self.config.line_width,
                   label=label)
        
        ax.set_xlabel('Wavevector k (μm⁻¹)')
        ax.set_ylabel('Frequency (PHz)')
        ax.set_title('Metamaterial Band Structure')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_isofrequency_contours(self,
                                  kx_grid: np.ndarray,
                                  ky_grid: np.ndarray,
                                  frequency_map: np.ndarray,
                                  target_frequency: float,
                                  **kwargs) -> plt.Figure:
        """
        Plot isofrequency contours in k-space.
        
        Args:
            kx_grid: kx meshgrid
            ky_grid: ky meshgrid  
            frequency_map: Frequency values on grid
            target_frequency: Target frequency for contours
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Contour plot
        contours = ax.contour(kx_grid * 1e-6, ky_grid * 1e-6, frequency_map * 1e-15,
                             levels=[target_frequency * 1e-15], colors='red', linewidths=2)
        
        # Background frequency map
        im = ax.contourf(kx_grid * 1e-6, ky_grid * 1e-6, frequency_map * 1e-15,
                        levels=20, cmap='viridis', alpha=0.8)
        
        ax.set_xlabel('kₓ (μm⁻¹)')
        ax.set_ylabel('kᵧ (μm⁻¹)')
        ax.set_title(f'Isofrequency Contours at {target_frequency*1e-15:.2f} PHz')
        ax.set_aspect('equal')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Frequency (PHz)')
        
        return fig


class InteractivePlotter:
    """Interactive plotting using Plotly (if available)."""
    
    def __init__(self):
        if not PLOTLY_AVAILABLE:
            warnings.warn("Plotly not available. Interactive plots disabled.")
        
    def interactive_field_plot(self,
                              coords: torch.Tensor,
                              fields: torch.Tensor,
                              title: str = "Electromagnetic Field",
                              **kwargs):
        """
        Create interactive 3D field plot.
        
        Args:
            coords: Coordinate tensor [N, 3]
            fields: Field tensor [N, components]
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available for interactive plotting")
            return None
        
        # Convert to numpy
        x = coords[:, 0].cpu().numpy()
        y = coords[:, 1].cpu().numpy()
        z = coords[:, 2].cpu().numpy()
        
        # Use first field component for coloring
        field_values = torch.norm(fields[:, :3], dim=1).cpu().numpy()
        
        # Create 3D scatter plot
        fig = go.Figure(data=go.Scatter3d(
            x=x * 1e6,  # Convert to μm
            y=y * 1e6,
            z=z * 1e6,
            mode='markers',
            marker=dict(
                size=3,
                color=field_values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Field Magnitude")
            ),
            text=[f'Field: {val:.3e}' for val in field_values],
            hovertemplate='x: %{x:.2f} μm<br>y: %{y:.2f} μm<br>z: %{z:.2f} μm<br>%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='x (μm)',
                yaxis_title='y (μm)',
                zaxis_title='z (μm)'
            )
        )
        
        return fig
    
    def interactive_training_dashboard(self,
                                     loss_history: Dict[str, List[float]],
                                     metrics_history: Dict[str, List[float]],
                                     **kwargs):
        """
        Create interactive training dashboard.
        
        Args:
            loss_history: Loss component histories
            metrics_history: Metrics histories
            
        Returns:
            Plotly subplot figure
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available for interactive plotting")
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Loss History', 'Loss Components', 'Physics Metrics', 'Convergence'],
            specs=[[{"secondary_y": False}, {"type": "pie"}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Loss history
        for loss_name, loss_values in loss_history.items():
            fig.add_trace(
                go.Scatter(x=list(range(len(loss_values))), y=loss_values,
                          mode='lines', name=loss_name, line=dict(width=2)),
                row=1, col=1
            )
        
        # Loss components pie chart
        if loss_history:
            current_losses = {name: values[-1] for name, values in loss_history.items()}
            fig.add_trace(
                go.Pie(labels=list(current_losses.keys()), values=list(current_losses.values()),
                      name="Loss Components"),
                row=1, col=2
            )
        
        # Physics metrics
        for metric_name, metric_values in metrics_history.items():
            fig.add_trace(
                go.Scatter(x=list(range(len(metric_values))), y=metric_values,
                          mode='lines', name=metric_name, line=dict(width=2)),
                row=2, col=1
            )
        
        # Convergence analysis
        if 'total' in loss_history:
            total_loss = loss_history['total']
            if len(total_loss) > 1:
                gradient = np.gradient(total_loss)
                fig.add_trace(
                    go.Scatter(x=list(range(len(gradient))), y=gradient,
                              mode='lines', name='Loss Gradient', line=dict(width=2)),
                    row=2, col=2
                )
        
        fig.update_layout(height=800, showlegend=True)
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_yaxes(title_text="Loss", type="log", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_yaxes(title_text="Metric Value", row=2, col=1)
        fig.update_xaxes(title_text="Epoch", row=2, col=2)
        fig.update_yaxes(title_text="Loss Gradient", row=2, col=2)
        
        return fig