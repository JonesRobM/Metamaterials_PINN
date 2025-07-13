"""
Training script for SPP metamaterial PINN models.

Implements physics-informed neural network training for Surface Plasmon
Polaritons on metamaterial interfaces using Maxwell's equations.
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
from config import load_config, get_config
from src.models import (
    ElectromagneticPINN, SPPNetwork, ComplexPINN,
    MaxwellCurlLoss, MaxwellDivergenceLoss, SPPBoundaryLoss,
    TangentialContinuityLoss, PowerFlowLoss, EM_CompositeLoss
)
from src.data import SPPDomainSampler, AdaptiveCollocationManager
from src.physics import MaxwellEquations, MetamaterialProperties, BoundaryConditions
from src.utils.metrics import compute_physics_metrics, compute_spp_metrics
from src.utils.plotting import plot_training_progress, plot_field_distribution


class SPPTrainer:
    """Main trainer for SPP metamaterial PINN models."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize trainer with configuration."""
        self.config = load_config() if config_path is None else self._load_custom_config(config_path)
        self.device = self._setup_device()
        self.setup_logging()
        
        # Core components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_function = None
        self.domain_sampler = None
        self.physics_solver = None
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.training_history = {
            'total_loss': [], 'physics_loss': [], 'boundary_loss': [],
            'validation_loss': [], 'learning_rate': []
        }
        
        # Logging
        self.logger = None
        self.tensorboard_writer = None
        
    def _load_custom_config(self, config_path: str) -> Dict:
        """Load custom configuration file."""
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_device(self) -> torch.device:
        """Setup computation device."""
        device_config = get_config('hardware.device', 'auto')
        
        if device_config == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_config)
        
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = get_config('reproducibility.benchmark', True)
            torch.backends.cudnn.deterministic = get_config('reproducibility.deterministic', False)
        
        return device
    
    def setup_logging(self):
        """Setup logging infrastructure."""
        log_dir = Path(get_config('logging.log_file', './logs/training.log')).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, get_config('logging.level', 'INFO')),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(get_config('logging.log_file', './logs/training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # TensorBoard setup
        if get_config('logging.tensorboard.enabled', True):
            tb_dir = get_config('logging.tensorboard.log_dir', './logs/tensorboard')
            self.tensorboard_writer = SummaryWriter(tb_dir)
    
    def setup_model(self):
        """Initialize the neural network model."""
        model_type = get_config('model.type', 'SPPNetwork')
        
        if model_type == 'SPPNetwork':
            self.model = SPPNetwork(
                interface_position=get_config('metamaterial.interface.position', 0.0),
                metal_permittivity=get_config('metamaterial.permittivity.parallel'),
                dielectric_permittivity=get_config('dielectric.superstrate.permittivity'),
                frequency=get_config('physics.angular_frequency'),
                spatial_dim=get_config('domain.spatial_dim', 3),
                hidden_dims=get_config('network.hidden_layers', [64, 64, 64, 64]),
                use_fourier=get_config('network.fourier_features.enabled', True),
                fourier_modes=get_config('network.fourier_features.num_frequencies', 128)
            )
        elif model_type == 'ComplexPINN':
            self.model = ComplexPINN(
                spatial_dim=get_config('domain.spatial_dim', 3),
                field_components=get_config('network.output_dim', 6),
                hidden_dims=get_config('network.hidden_layers', [64, 64, 64, 64]),
                frequency=get_config('physics.angular_frequency'),
                use_fourier=get_config('network.fourier_features.enabled', True)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model = self.model.to(self.device)
        self.logger.info(f"Initialized {model_type} with {self._count_parameters()} parameters")
    
    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def setup_physics(self):
        """Initialize physics solvers."""
        self.physics_solver = MaxwellEquations(
            omega=get_config('physics.angular_frequency'),
            mu0=get_config('physics.constants.vacuum_permeability'),
            eps0=get_config('physics.constants.vacuum_permittivity')
        )
        
        # Metamaterial properties
        self.metamaterial = MetamaterialProperties(
            eps_parallel=get_config('metamaterial.permittivity.parallel'),
            eps_perpendicular=get_config('metamaterial.permittivity.perpendicular'),
            optical_axis=get_config('metamaterial.optical_axis', 'z')
        )
        
        # Boundary conditions
        self.boundary_solver = BoundaryConditions(
            interface_normal=get_config('interface.boundary_type', (0, 0, 1))
        )
    
    def setup_loss_function(self):
        """Initialize composite loss function."""
        frequency = get_config('physics.angular_frequency')
        
        losses = {
            'maxwell_curl': MaxwellCurlLoss(
                frequency=frequency,
                weight=get_config('loss.weights.maxwell_curl_E', 1.0)
            ),
            'maxwell_div': MaxwellDivergenceLoss(
                weight=get_config('loss.weights.maxwell_div_E', 0.1)
            ),
            'spp_boundary': SPPBoundaryLoss(
                spp_wavevector=self._estimate_spp_wavevector(),
                decay_length=get_config('metamaterial.spp_properties.penetration_depth_metal', 1e-6),
                weight=get_config('loss.weights.boundary_tangential_E', 10.0)
            ),
            'tangential_continuity': TangentialContinuityLoss(
                weight=get_config('loss.weights.boundary_tangential_H', 10.0)
            ),
            'power_flow': PowerFlowLoss(
                weight=get_config('loss.weights.regularization', 0.01)
            )
        }
        
        self.loss_function = EM_CompositeLoss(
            losses=losses,
            adaptive_weights=get_config('loss.normalization.enabled', True),
            frequency_dependent=True
        )
    
    def _estimate_spp_wavevector(self) -> float:
        """Estimate SPP wavevector from material properties."""
        omega = get_config('physics.angular_frequency')
        if hasattr(self.metamaterial, 'spp_dispersion_relation'):
            k_real, _ = self.metamaterial.spp_dispersion_relation(omega)
            return float(k_real)
        return omega / 3e8  # Fallback to free-space wavevector
    
    def setup_data_sampling(self):
        """Initialize domain sampling strategy."""
        domain_bounds = [
            get_config('domain.x_range'),
            get_config('domain.y_range'),
            get_config('domain.z_range')
        ]
        
        self.domain_sampler = SPPDomainSampler(
            domain_bounds=domain_bounds,
            interface_position=get_config('metamaterial.interface.position', 0.0),
            spp_decay_length=get_config('metamaterial.spp_properties.penetration_depth_metal', 1e-6),
            device=self.device
        )
        
        # Adaptive collocation management
        if get_config('training.sampling.adaptive_sampling.enabled', True):
            self.collocation_manager = AdaptiveCollocationManager(
                base_generator=self.domain_sampler,
                max_points=get_config('training.sampling.adaptive_sampling.max_points', 20000),
                refinement_frequency=get_config('training.sampling.adaptive_sampling.update_frequency', 1000)
            )
    
    def setup_optimizer(self):
        """Initialize optimizer and learning rate scheduler."""
        optimizer_config = get_config('training.optimizer')
        
        if optimizer_config['type'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=get_config('training.learning_rate'),
                betas=(optimizer_config['beta1'], optimizer_config['beta2']),
                eps=optimizer_config['eps'],
                weight_decay=optimizer_config['weight_decay']
            )
        elif optimizer_config['type'] == 'lbfgs':
            self.optimizer = optim.LBFGS(
                self.model.parameters(),
                lr=get_config('training.learning_rate'),
                max_iter=20,
                tolerance_grad=1e-7,
                tolerance_change=1e-9,
                history_size=100
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_config['type']}")
        
        # Learning rate scheduler
        scheduler_config = get_config('training.scheduler')
        if scheduler_config['type'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config['step_size'],
                gamma=scheduler_config['gamma']
            )
        elif scheduler_config['type'] == 'exponential':
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=scheduler_config['gamma']
            )
        elif scheduler_config['type'] == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=scheduler_config['patience'],
                factor=scheduler_config['gamma']
            )
    
    def sample_collocation_points(self, epoch: int) -> Dict[str, torch.Tensor]:
        """Sample collocation points for current epoch."""
        n_collocation = get_config('training.sampling.n_collocation', 10000)
        n_boundary = get_config('training.sampling.n_boundary', 2000)
        
        # Get domain points
        domain_result = self.domain_sampler.sample_points(
            n_collocation, interior_fraction=0.8
        )
        
        # Combine all points
        points = {
            'collocation': domain_result['interior'],
            'boundary': domain_result['interface'],
            'domain_boundary': domain_result['boundary']
        }
        
        # Add adaptive refinement
        if hasattr(self, 'collocation_manager') and self.collocation_manager.should_refine(epoch):
            refined_points = self.collocation_manager.refine_points(n_collocation // 4)
            points['adaptive'] = refined_points
        
        return points
    
    def compute_loss(self, points: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total physics-informed loss."""
        all_points = torch.cat(list(points.values()), dim=0)
        all_points.requires_grad_(True)
        
        # Forward pass through model
        fields = self.model(all_points)
        
        # Split into E and H fields
        if hasattr(self.model, 'get_fields'):
            E_field, H_field = self.model.get_fields(all_points)
        else:
            E_field = fields[:, :3]
            H_field = fields[:, 3:]
        
        # Compute loss components
        total_loss, loss_dict = self.loss_function.compute(
            network=self.model,
            coords=all_points,
            E_field=E_field,
            H_field=H_field,
            material_props=self._get_material_properties(all_points)
        )
        
        return total_loss, {k: v.item() for k, v in loss_dict.items()}
    
    def _get_material_properties(self, coords: torch.Tensor) -> torch.Tensor:
        """Get material properties at coordinates."""
        if hasattr(self.metamaterial, 'permittivity_tensor'):
            return self.metamaterial.permittivity_tensor(coords)
        else:
            # Fallback: assume uniform metamaterial
            batch_size = coords.shape[0]
            eps_tensor = torch.eye(3, device=self.device, dtype=torch.complex64)
            eps_tensor[0, 0] = self.metamaterial.eps_perp
            eps_tensor[1, 1] = self.metamaterial.eps_perp
            eps_tensor[2, 2] = self.metamaterial.eps_par
            return eps_tensor.unsqueeze(0).expand(batch_size, -1, -1)
    
    def training_step(self, epoch: int) -> Dict[str, float]:
        """Execute single training step."""
        self.model.train()
        
        # Sample collocation points
        points = self.sample_collocation_points(epoch)
        
        def closure():
            self.optimizer.zero_grad()
            total_loss, _ = self.compute_loss(points)
            total_loss.backward()
            return total_loss
        
        if isinstance(self.optimizer, optim.LBFGS):
            loss = self.optimizer.step(closure)
            total_loss = loss.item()
            _, loss_components = self.compute_loss(points)
        else:
            self.optimizer.zero_grad()
            total_loss, loss_components = self.compute_loss(points)
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss = total_loss.item()
        
        # Update learning rate scheduler
        if not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()
        
        return {'total_loss': total_loss, **loss_components}
    
    def validation_step(self, epoch: int) -> Dict[str, float]:
        """Execute validation step."""
        self.model.eval()
        
        with torch.no_grad():
            # Sample validation points
            val_points = self.sample_collocation_points(epoch)
            val_loss, val_components = self.compute_loss(val_points)
            
            # Compute physics metrics
            physics_metrics = self._compute_physics_validation(val_points)
            
        return {
            'val_loss': val_loss.item(),
            **{f'val_{k}': v for k, v in val_components.items()},
            **physics_metrics
        }
    
    def _compute_physics_validation(self, points: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute physics-based validation metrics."""
        all_points = torch.cat(list(points.values()), dim=0)
        
        with torch.no_grad():
            fields = self.model(all_points)
            E_field, H_field = self.model.get_fields(all_points) if hasattr(self.model, 'get_fields') else (fields[:, :3], fields[:, 3:])
            
            # Maxwell equation residuals
            eps_tensor = self._get_material_properties(all_points)
            maxwell_residuals = self.physics_solver.total_residual(E_field, H_field, all_points, eps_tensor)
            
            # SPP-specific metrics
            spp_metrics = {}
            if hasattr(self.model, 'k_spp'):
                spp_metrics = self._compute_spp_validation_metrics(all_points, E_field, H_field)
        
        return {
            'maxwell_residual_norm': torch.norm(maxwell_residuals).item(),
            'field_magnitude_avg': torch.mean(torch.norm(E_field, dim=1)).item(),
            'energy_conservation': self._compute_energy_conservation(E_field, H_field),
            **spp_metrics
        }
    
    def _compute_spp_validation_metrics(self, coords: torch.Tensor, E_field: torch.Tensor, H_field: torch.Tensor) -> Dict[str, float]:
        """Compute SPP-specific validation metrics."""
        # Field decay at interface
        interface_z = get_config('metamaterial.interface.position', 0.0)
        z_coords = coords[:, 2] if coords.shape[1] > 2 else coords[:, 1]
        
        # Find points near interface
        interface_mask = torch.abs(z_coords - interface_z) < 1e-7
        if torch.any(interface_mask):
            interface_field = torch.mean(torch.norm(E_field[interface_mask], dim=1))
        else:
            interface_field = 0.0
        
        # Field decay away from interface
        far_mask = torch.abs(z_coords - interface_z) > 5e-7
        if torch.any(far_mask):
            far_field = torch.mean(torch.norm(E_field[far_mask], dim=1))
            decay_ratio = far_field / (interface_field + 1e-8)
        else:
            decay_ratio = 1.0
        
        return {
            'spp_interface_field': interface_field.item(),
            'spp_decay_ratio': decay_ratio.item()
        }
    
    def _compute_energy_conservation(self, E_field: torch.Tensor, H_field: torch.Tensor) -> float:
        """Compute energy conservation metric."""
        # Poynting vector magnitude
        S = torch.cross(E_field.real, H_field.real, dim=1)
        energy_flow = torch.mean(torch.norm(S, dim=1))
        return energy_flow.item()
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = Path(get_config('checkpointing.checkpoint_dir', './checkpoints'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'loss': self.training_history['total_loss'][-1] if self.training_history['total_loss'] else float('inf'),
            'config': self.config,
            'training_history': self.training_history
        }
        
        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model at epoch {epoch}")
        
        # Clean up old checkpoints (keep last 5)
        checkpoints = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        if len(checkpoints) > 5:
            for old_checkpoint in checkpoints[:-5]:
                old_checkpoint.unlink()
    
    def log_metrics(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Optional[Dict[str, float]] = None):
        """Log training metrics."""
        # Console logging
        lr = self.optimizer.param_groups[0]['lr']
        log_msg = f"Epoch {epoch:5d} | Loss: {train_metrics['total_loss']:.6f} | LR: {lr:.2e}"
        
        if val_metrics:
            log_msg += f" | Val Loss: {val_metrics['val_loss']:.6f}"
        
        self.logger.info(log_msg)
        
        # TensorBoard logging
        if self.tensorboard_writer:
            for key, value in train_metrics.items():
                self.tensorboard_writer.add_scalar(f'Train/{key}', value, epoch)
            
            if val_metrics:
                for key, value in val_metrics.items():
                    self.tensorboard_writer.add_scalar(f'Validation/{key}', value, epoch)
            
            self.tensorboard_writer.add_scalar('Learning_Rate', lr, epoch)
        
        # Update training history
        for key, value in train_metrics.items():
            if key in self.training_history:
                self.training_history[key].append(value)
        
        if val_metrics:
            self.training_history['validation_loss'].append(val_metrics['val_loss'])
        
        self.training_history['learning_rate'].append(lr)
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting SPP metamaterial PINN training")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {self._count_parameters():,}")
        
        epochs = get_config('training.epochs', 10000)
        validation_frequency = get_config('validation.frequency', 100)
        save_frequency = get_config('checkpointing.save_frequency', 1000)
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            
            # Training step
            train_metrics = self.training_step(epoch)
            
            # Validation step
            val_metrics = None
            if epoch % validation_frequency == 0:
                val_metrics = self.validation_step(epoch)
                
                # Update LR scheduler based on validation loss
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
            
            # Logging
            if epoch % get_config('logging.log_frequency', 100) == 0:
                self.log_metrics(epoch, train_metrics, val_metrics)
            
            # Checkpointing
            is_best = False
            if val_metrics and val_metrics['val_loss'] < self.best_loss:
                self.best_loss = val_metrics['val_loss']
                is_best = True
            
            if epoch % save_frequency == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Early stopping check
            if self._should_early_stop():
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Final checkpoint
        self.save_checkpoint(epoch, is_best=False)
        
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
    
    def _should_early_stop(self) -> bool:
        """Check if early stopping criteria are met."""
        patience = get_config('training.early_stopping.patience', 2000)
        min_delta = get_config('training.early_stopping.min_delta', 1e-6)
        
        if len(self.training_history['validation_loss']) < patience:
            return False
        
        recent_losses = self.training_history['validation_loss'][-patience:]
        best_recent = min(recent_losses)
        current_loss = recent_losses[-1]
        
        return (current_loss - best_recent) < min_delta
    
    def setup(self):
        """Setup all training components."""
        self.setup_model()
        self.setup_physics()
        self.setup_loss_function()
        self.setup_data_sampling()
        self.setup_optimizer()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train SPP metamaterial PINN')
    parser.add_argument('--config', type=str, help='Path to custom config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, help='Device to use (cuda/cpu)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Setup trainer
    trainer = SPPTrainer(config_path=args.config)
    
    if args.device:
        trainer.device = torch.device(args.device)
    
    if args.debug:
        trainer.config['logging']['level'] = 'DEBUG'
        trainer.config['training']['epochs'] = 100  # Short run for debugging
    
    # Setup training components
    trainer.setup()
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if trainer.scheduler and checkpoint['scheduler_state_dict']:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        trainer.current_epoch = checkpoint['epoch']
        trainer.training_history = checkpoint.get('training_history', trainer.training_history)
        trainer.logger.info(f"Resumed training from epoch {trainer.current_epoch}")
    
    # Start training
    try:
        trainer.train()
        trainer.logger.info("Training completed successfully")
    except KeyboardInterrupt:
        trainer.logger.info("Training interrupted by user")
        trainer.save_checkpoint(trainer.current_epoch, is_best=False)
    except Exception as e:
        trainer.logger.error(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    main()