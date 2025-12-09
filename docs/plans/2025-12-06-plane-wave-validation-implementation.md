# Plane Wave Validation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create validation script that trains a PINN to discover plane wave solution using only Maxwell's equations (zero data supervision).

**Architecture:** Standalone script in `examples/` directory that samples collocation points, trains ElectromagneticPINN with MaxwellCurlLoss, and validates physics accuracy through residual analysis and wave property verification.

**Tech Stack:** PyTorch, existing PINN architecture (ElectromagneticPINN), MaxwellCurlLoss, matplotlib for visualization, numpy for metrics.

---

## Task 1: Create Examples Directory and Test Infrastructure

**Files:**
- Create: `examples/__init__.py`
- Create: `examples/validate_plane_wave.py`
- Create: `tests/examples/__init__.py`
- Create: `tests/examples/test_validate_plane_wave.py`

**Step 1: Create directory structure**

Run:
```bash
cd ~/.config/superpowers/worktrees/Metamaterials_PINN/plane-wave-validation
mkdir -p examples tests/examples
touch examples/__init__.py tests/examples/__init__.py
```

**Step 2: Create basic test file with imports check**

File: `tests/examples/test_validate_plane_wave.py`
```python
"""Tests for plane wave validation script."""
import pytest
import torch
import numpy as np


def test_imports():
    """Test that all required modules can be imported."""
    try:
        from examples.validate_plane_wave import (
            OMEGA, C, MU0, EPS0, K0, WAVELENGTH
        )
        assert OMEGA > 0
        assert C > 0
        assert MU0 > 0
        assert EPS0 > 0
        assert K0 > 0
        assert WAVELENGTH > 0
    except ImportError:
        pytest.skip("validate_plane_wave not yet implemented")
```

**Step 3: Run test to verify it fails**

Run: `python -m pytest tests/examples/test_validate_plane_wave.py::test_imports -v`
Expected: SKIP (module doesn't exist yet)

**Step 4: Create basic script with physical constants**

File: `examples/validate_plane_wave.py`
```python
"""
Plane Wave Validation for PINN Fundamentals

Trains a PINN to discover a plane wave solution in free space using only
Maxwell's equations (no data supervision). This validates that the physics
loss implementation is correct.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, Optional
import logging

# Physical constants
OMEGA = 2 * np.pi * 1e9  # Angular frequency: 1 GHz
C = 299792458.0  # Speed of light (m/s)
MU0 = 4e-7 * np.pi  # Permeability of free space (H/m)
EPS0 = 8.854187817e-12  # Permittivity of free space (F/m)
K0 = OMEGA / C  # Wave number (rad/m)
WAVELENGTH = 2 * np.pi / K0  # Wavelength (m)

# Training configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 2048
N_EPOCHS = 10000
LEARNING_RATE = 1e-3
```

**Step 5: Run test to verify it passes**

Run: `python -m pytest tests/examples/test_validate_plane_wave.py::test_imports -v`
Expected: PASS

**Step 6: Commit**

```bash
git add examples/ tests/examples/
git commit -m "feat: add plane wave validation script structure with physical constants"
```

---

## Task 2: Implement Collocation Point Sampling

**Files:**
- Modify: `examples/validate_plane_wave.py`
- Modify: `tests/examples/test_validate_plane_wave.py`

**Step 1: Write the failing test**

File: `tests/examples/test_validate_plane_wave.py`
```python
def test_sample_collocation_points():
    """Test collocation point sampling."""
    from examples.validate_plane_wave import sample_collocation_points, WAVELENGTH

    n_points = 100
    coords = sample_collocation_points(n_points)

    # Check shape
    assert coords.shape == (n_points, 3), f"Expected shape ({n_points}, 3), got {coords.shape}"

    # Check dtype
    assert coords.dtype == torch.float32

    # Check requires_grad
    assert coords.requires_grad == True

    # Check bounds (should be in [0, WAVELENGTH])
    assert torch.all(coords >= 0.0)
    assert torch.all(coords <= WAVELENGTH)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/examples/test_validate_plane_wave.py::test_sample_collocation_points -v`
Expected: FAIL with "cannot import name 'sample_collocation_points'"

**Step 3: Write minimal implementation**

File: `examples/validate_plane_wave.py` (add after configuration section)
```python
def sample_collocation_points(n_points: int, device: torch.device = DEVICE) -> torch.Tensor:
    """
    Sample random collocation points uniformly in domain [0, λ]³.

    Args:
        n_points: Number of points to sample
        device: Device to create tensor on

    Returns:
        Tensor of shape (n_points, 3) with requires_grad=True
    """
    coords = torch.rand(n_points, 3, dtype=torch.float32, device=device) * WAVELENGTH
    coords.requires_grad_(True)
    return coords
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/examples/test_validate_plane_wave.py::test_sample_collocation_points -v`
Expected: PASS

**Step 5: Commit**

```bash
git add examples/validate_plane_wave.py tests/examples/test_validate_plane_wave.py
git commit -m "feat: add collocation point sampling for plane wave domain"
```

---

## Task 3: Implement Network Creation

**Files:**
- Modify: `examples/validate_plane_wave.py`
- Modify: `tests/examples/test_validate_plane_wave.py`

**Step 1: Write the failing test**

File: `tests/examples/test_validate_plane_wave.py`
```python
def test_create_network():
    """Test network creation."""
    from examples.validate_plane_wave import create_network, DEVICE

    network = create_network()

    # Check it's a PyTorch module
    assert isinstance(network, nn.Module)

    # Check device
    assert next(network.parameters()).device.type == DEVICE.type

    # Test forward pass
    batch_size = 10
    coords = torch.randn(batch_size, 3, device=DEVICE)
    output = network(coords)

    # Check output shape: [batch, 6, 2] for (Ex,Ey,Ez,Hx,Hy,Hz) × (real,imag)
    assert output.shape == (batch_size, 6, 2), f"Expected shape ({batch_size}, 6, 2), got {output.shape}"

    # Check output dtype is float32
    assert output.dtype == torch.float32
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/examples/test_validate_plane_wave.py::test_create_network -v`
Expected: FAIL with "cannot import name 'create_network'"

**Step 3: Write minimal implementation**

File: `examples/validate_plane_wave.py` (add import at top)
```python
from src.models.pinn_network import ElectromagneticPINN
```

File: `examples/validate_plane_wave.py` (add function)
```python
def create_network(device: torch.device = DEVICE) -> nn.Module:
    """
    Create ElectromagneticPINN for plane wave problem.

    Args:
        device: Device to create network on

    Returns:
        Initialized neural network
    """
    network = ElectromagneticPINN(
        input_dim=3,  # (x, y, z)
        hidden_dims=[256, 256, 256],  # 3 hidden layers
        output_dim=6,  # 6 field components (will be [batch, 6, 2] after internal processing)
        activation='tanh',
        use_fourier_features=False  # Start simple
    ).to(device)

    return network
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/examples/test_validate_plane_wave.py::test_create_network -v`
Expected: PASS

**Step 5: Commit**

```bash
git add examples/validate_plane_wave.py tests/examples/test_validate_plane_wave.py
git commit -m "feat: add network creation for plane wave PINN"
```

---

## Task 4: Implement Training Loop

**Files:**
- Modify: `examples/validate_plane_wave.py`
- Modify: `tests/examples/test_validate_plane_wave.py`

**Step 1: Write the failing test**

File: `tests/examples/test_validate_plane_wave.py`
```python
@pytest.mark.slow
def test_train_pure_physics_convergence():
    """Test that training loop runs and loss decreases (short run)."""
    from examples.validate_plane_wave import create_network, train_pure_physics

    network = create_network()

    # Short training run
    trained_network, history = train_pure_physics(
        network=network,
        n_epochs=100,  # Very short for testing
        batch_size=512,
        learning_rate=1e-3,
        verbose=False
    )

    # Check network is returned
    assert isinstance(trained_network, nn.Module)

    # Check history structure
    assert 'total_loss' in history
    assert 'curl_E_loss' in history
    assert 'curl_H_loss' in history
    assert 'epoch' in history

    # Check loss decreases (at least somewhat)
    initial_loss = history['total_loss'][0]
    final_loss = history['total_loss'][-1]
    assert final_loss < initial_loss, f"Loss should decrease: {initial_loss} -> {final_loss}"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/examples/test_validate_plane_wave.py::test_train_pure_physics_convergence -v`
Expected: FAIL with "cannot import name 'train_pure_physics'"

**Step 3: Write minimal implementation**

File: `examples/validate_plane_wave.py` (add import)
```python
from src.models.loss_functions import MaxwellCurlLoss
```

File: `examples/validate_plane_wave.py` (add function)
```python
def train_pure_physics(
    network: nn.Module,
    n_epochs: int = N_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    device: torch.device = DEVICE,
    verbose: bool = True
) -> Tuple[nn.Module, Dict[str, list]]:
    """
    Train PINN using only Maxwell's equations (no data).

    Args:
        network: Neural network to train
        n_epochs: Number of training epochs
        batch_size: Collocation points per batch
        learning_rate: Initial learning rate
        device: Device for computation
        verbose: Print progress

    Returns:
        Tuple of (trained_network, training_history)
    """
    # Setup loss function
    loss_fn = MaxwellCurlLoss(frequency=OMEGA, mu0=MU0, eps0=EPS0, weight=1.0)

    # Setup optimizer
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=500, verbose=verbose
    )

    # Training history
    history = {
        'epoch': [],
        'total_loss': [],
        'curl_E_loss': [],
        'curl_H_loss': [],
        'learning_rate': []
    }

    # Training loop
    network.train()
    for epoch in range(n_epochs):
        # Sample collocation points
        coords = sample_collocation_points(batch_size, device=device)

        # Forward pass
        optimizer.zero_grad()

        # Compute physics loss
        loss = loss_fn.compute(network=network, coords=coords)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update scheduler
        scheduler.step(loss)

        # Log metrics
        if epoch % 100 == 0 or epoch == n_epochs - 1:
            history['epoch'].append(epoch)
            history['total_loss'].append(loss.item())
            history['curl_E_loss'].append(0.0)  # Placeholder - will compute separately later
            history['curl_H_loss'].append(0.0)  # Placeholder
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])

            if verbose:
                print(f"Epoch {epoch:5d} | Loss: {loss.item():.6e} | LR: {optimizer.param_groups[0]['lr']:.2e}")

    return network, history
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/examples/test_validate_plane_wave.py::test_train_pure_physics_convergence -v`
Expected: PASS (may take ~30 seconds)

**Step 5: Commit**

```bash
git add examples/validate_plane_wave.py tests/examples/test_validate_plane_wave.py
git commit -m "feat: add pure physics training loop for plane wave PINN"
```

---

## Task 5: Implement Validation Metrics

**Files:**
- Modify: `examples/validate_plane_wave.py`
- Modify: `tests/examples/test_validate_plane_wave.py`

**Step 1: Write the failing test**

File: `tests/examples/test_validate_plane_wave.py`
```python
def test_validate_solution():
    """Test solution validation metrics."""
    from examples.validate_plane_wave import create_network, validate_solution, WAVELENGTH

    network = create_network()

    # Run validation
    metrics = validate_solution(network, n_points=1000)

    # Check all expected metrics are present
    expected_keys = [
        'max_curl_E_residual',
        'max_curl_H_residual',
        'mean_curl_E_residual',
        'mean_curl_H_residual',
        'E_H_orthogonality',
        'field_magnitude_E',
        'field_magnitude_H'
    ]

    for key in expected_keys:
        assert key in metrics, f"Missing metric: {key}"
        assert isinstance(metrics[key], (float, np.floating)), f"{key} should be float"
        assert not np.isnan(metrics[key]), f"{key} is NaN"
        assert not np.isinf(metrics[key]), f"{key} is infinite"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/examples/test_validate_plane_wave.py::test_validate_solution -v`
Expected: FAIL with "cannot import name 'validate_solution'"

**Step 3: Write minimal implementation**

File: `examples/validate_plane_wave.py` (add function)
```python
def validate_solution(
    network: nn.Module,
    n_points: int = 10000,
    device: torch.device = DEVICE
) -> Dict[str, float]:
    """
    Validate trained solution by computing physics metrics.

    Args:
        network: Trained neural network
        n_points: Number of validation points
        device: Device for computation

    Returns:
        Dictionary of validation metrics
    """
    network.eval()

    with torch.no_grad():
        # Sample validation points
        coords = sample_collocation_points(n_points, device=device)
        coords.requires_grad_(True)

        # Get fields
        fields = network(coords)  # [n_points, 6, 2]

        # Convert to complex
        E = torch.complex(fields[:, 0:3, 0], fields[:, 0:3, 1])  # [n_points, 3]
        H = torch.complex(fields[:, 3:6, 0], fields[:, 3:6, 1])  # [n_points, 3]

        # Compute Maxwell residuals using loss function
        loss_fn = MaxwellCurlLoss(frequency=OMEGA, mu0=MU0, eps0=EPS0)

        # For detailed residuals, we need to compute curl manually
        # Use the internal _compute_curl method
        curl_E = loss_fn._compute_curl(fields[:, 0:3, :], coords)  # [n_points, 3, 2]
        curl_H = loss_fn._compute_curl(fields[:, 3:6, :], coords)  # [n_points, 3, 2]

        # Convert to complex
        curl_E_complex = torch.complex(curl_E[..., 0], curl_E[..., 1])
        curl_H_complex = torch.complex(curl_H[..., 0], curl_H[..., 1])

        # Maxwell residuals
        residual_E = curl_E_complex + 1j * OMEGA * MU0 * H
        residual_H = curl_H_complex - 1j * OMEGA * EPS0 * E

        # Compute metrics
        metrics = {
            'max_curl_E_residual': torch.max(torch.abs(residual_E)).item(),
            'max_curl_H_residual': torch.max(torch.abs(residual_H)).item(),
            'mean_curl_E_residual': torch.mean(torch.abs(residual_E)).item(),
            'mean_curl_H_residual': torch.mean(torch.abs(residual_H)).item(),
        }

        # E·H orthogonality (should be ~0 for plane waves)
        E_dot_H = torch.sum(E * torch.conj(H), dim=1)  # [n_points]
        E_mag = torch.sqrt(torch.sum(torch.abs(E)**2, dim=1))
        H_mag = torch.sqrt(torch.sum(torch.abs(H)**2, dim=1))
        normalized_dot = E_dot_H / (E_mag * H_mag + 1e-10)
        metrics['E_H_orthogonality'] = torch.mean(torch.abs(normalized_dot)).item()

        # Field magnitudes
        metrics['field_magnitude_E'] = torch.mean(E_mag).item()
        metrics['field_magnitude_H'] = torch.mean(H_mag).item()

    return metrics
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/examples/test_validate_plane_wave.py::test_validate_solution -v`
Expected: PASS

**Step 5: Commit**

```bash
git add examples/validate_plane_wave.py tests/examples/test_validate_plane_wave.py
git commit -m "feat: add solution validation with Maxwell residuals and wave properties"
```

---

## Task 6: Implement Visualization

**Files:**
- Modify: `examples/validate_plane_wave.py`

**Step 1: Write visualization function (no test needed for plotting)**

File: `examples/validate_plane_wave.py` (add function)
```python
def plot_training_history(history: Dict[str, list], save_path: Optional[Path] = None):
    """
    Plot training loss curves.

    Args:
        history: Training history dictionary
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    epochs = history['epoch']
    ax.semilogy(epochs, history['total_loss'], 'b-', linewidth=2, label='Total Loss')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Pure Physics Training: Maxwell Residuals', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training plot to {save_path}")

    plt.show()


def plot_field_slice(
    network: nn.Module,
    z_slice: float = 0.0,
    n_points: int = 50,
    device: torch.device = DEVICE,
    save_path: Optional[Path] = None
):
    """
    Plot 2D slice of electromagnetic fields.

    Args:
        network: Trained network
        z_slice: z-coordinate for slice
        n_points: Resolution (n_points x n_points grid)
        device: Device for computation
        save_path: Optional path to save figure
    """
    network.eval()

    # Create grid
    x = torch.linspace(0, WAVELENGTH, n_points)
    y = torch.linspace(0, WAVELENGTH, n_points)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    Z = torch.full_like(X, z_slice)

    # Stack coordinates
    coords = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1).to(device)

    with torch.no_grad():
        fields = network(coords)  # [n_points^2, 6, 2]

    # Extract E and H (real parts for visualization)
    Ex = fields[:, 0, 0].cpu().reshape(n_points, n_points).numpy()
    Ey = fields[:, 1, 0].cpu().reshape(n_points, n_points).numpy()
    Ez = fields[:, 2, 0].cpu().reshape(n_points, n_points).numpy()
    Hx = fields[:, 3, 0].cpu().reshape(n_points, n_points).numpy()
    Hy = fields[:, 4, 0].cpu().reshape(n_points, n_points).numpy()
    Hz = fields[:, 5, 0].cpu().reshape(n_points, n_points).numpy()

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    components = [
        (Ex, 'Re(Ex)'), (Ey, 'Re(Ey)'), (Ez, 'Re(Ez)'),
        (Hx, 'Re(Hx)'), (Hy, 'Re(Hy)'), (Hz, 'Re(Hz)')
    ]

    for ax, (field, title) in zip(axes.flat, components):
        im = ax.imshow(field.T, origin='lower', extent=[0, WAVELENGTH, 0, WAVELENGTH],
                      cmap='RdBu', aspect='auto')
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('x (m)', fontsize=10)
        ax.set_ylabel('y (m)', fontsize=10)
        plt.colorbar(im, ax=ax)

    plt.suptitle(f'Electromagnetic Fields at z={z_slice:.3f}m', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved field plot to {save_path}")

    plt.show()
```

**Step 2: Commit**

```bash
git add examples/validate_plane_wave.py
git commit -m "feat: add visualization functions for training history and field distributions"
```

---

## Task 7: Add Main Execution and Logging

**Files:**
- Modify: `examples/validate_plane_wave.py`

**Step 1: Add main execution block**

File: `examples/validate_plane_wave.py` (add at end)
```python
def setup_logging(verbose: bool = True):
    """Setup logging configuration."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main execution function."""
    # Setup
    setup_logging(verbose=True)
    logging.info("=" * 70)
    logging.info("Plane Wave Validation - Pure Physics PINN")
    logging.info("=" * 70)

    # Print configuration
    logging.info(f"Device: {DEVICE}")
    logging.info(f"Frequency: {OMEGA/(2*np.pi):.2e} Hz")
    logging.info(f"Wavelength: {WAVELENGTH:.4f} m")
    logging.info(f"Wave number k0: {K0:.4f} rad/m")
    logging.info(f"Batch size: {BATCH_SIZE}")
    logging.info(f"Epochs: {N_EPOCHS}")
    logging.info("=" * 70)

    # Create network
    logging.info("Creating neural network...")
    network = create_network(device=DEVICE)
    n_params = sum(p.numel() for p in network.parameters())
    logging.info(f"Network created with {n_params:,} parameters")

    # Train
    logging.info("\nStarting pure physics training...")
    logging.info("Loss = Maxwell curl residuals only (no data)")
    logging.info("-" * 70)

    trained_network, history = train_pure_physics(
        network=network,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        device=DEVICE,
        verbose=True
    )

    logging.info("-" * 70)
    logging.info("Training complete!")

    # Validate
    logging.info("\nValidating solution...")
    metrics = validate_solution(trained_network, n_points=10000, device=DEVICE)

    logging.info("=" * 70)
    logging.info("VALIDATION METRICS")
    logging.info("=" * 70)
    logging.info(f"Max curl(E) residual:  {metrics['max_curl_E_residual']:.6e}")
    logging.info(f"Max curl(H) residual:  {metrics['max_curl_H_residual']:.6e}")
    logging.info(f"Mean curl(E) residual: {metrics['mean_curl_E_residual']:.6e}")
    logging.info(f"Mean curl(H) residual: {metrics['mean_curl_H_residual']:.6e}")
    logging.info(f"E·H orthogonality:     {metrics['E_H_orthogonality']:.6e}")
    logging.info(f"Mean |E|:              {metrics['field_magnitude_E']:.6e}")
    logging.info(f"Mean |H|:              {metrics['field_magnitude_H']:.6e}")
    logging.info("=" * 70)

    # Check success criteria
    success_threshold = 1e-4  # Relaxed for initial validation
    max_residual = max(metrics['max_curl_E_residual'], metrics['max_curl_H_residual'])

    if max_residual < success_threshold:
        logging.info(f"✓ SUCCESS: Maxwell residuals < {success_threshold:.0e}")
    else:
        logging.warning(f"✗ NEEDS WORK: Maxwell residuals {max_residual:.2e} > {success_threshold:.0e}")

    # Visualize
    logging.info("\nGenerating visualizations...")

    # Create output directory
    output_dir = Path("outputs/plane_wave_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot training history
    plot_training_history(history, save_path=output_dir / "training_history.png")

    # Plot field slice
    plot_field_slice(trained_network, z_slice=0.0, save_path=output_dir / "fields_z0.png")

    # Save model
    model_path = output_dir / "trained_network.pt"
    torch.save(trained_network.state_dict(), model_path)
    logging.info(f"Saved model to {model_path}")

    # Save metrics
    metrics_path = output_dir / "validation_metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write("Plane Wave Validation Metrics\n")
        f.write("=" * 50 + "\n\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.6e}\n")
    logging.info(f"Saved metrics to {metrics_path}")

    logging.info("\nValidation complete!")
    logging.info(f"Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
```

**Step 2: Test running the script**

Run:
```bash
cd ~/.config/superpowers/worktrees/Metamaterials_PINN/plane-wave-validation
python examples/validate_plane_wave.py
```

Expected: Script runs, trains for N_EPOCHS, produces plots and metrics

**Note:** This will take significant time (30 min - 2 hours). For quick verification, you can temporarily set `N_EPOCHS = 1000` in the script.

**Step 3: Commit**

```bash
git add examples/validate_plane_wave.py
git commit -m "feat: add main execution with logging and result saving"
```

---

## Task 8: Add Documentation

**Files:**
- Create: `examples/README.md`

**Step 1: Create examples README**

File: `examples/README.md`
```markdown
# PINN Validation Examples

This directory contains validation experiments for the Metamaterials PINN project.

## Plane Wave Validation

**Script:** `validate_plane_wave.py`

**Purpose:** Validate PINN fundamentals by training a network to discover a plane wave solution using only Maxwell's equations (zero data supervision).

**Usage:**

```bash
python examples/validate_plane_wave.py
```

**What it does:**

1. Creates an `ElectromagneticPINN` network (3 hidden layers, 256 neurons each)
2. Samples random collocation points in domain [0, λ]³
3. Trains using only `MaxwellCurlLoss` (pure physics-informed)
4. Validates by computing Maxwell residuals
5. Generates plots and saves results to `outputs/plane_wave_validation/`

**Success Criteria:**

- **Target:** Maxwell residuals < 1e-6
- **Minimum:** Maxwell residuals < 1e-4
- **Demonstrates:** Physics loss implementation is correct

**Outputs:**

- `outputs/plane_wave_validation/training_history.png` - Loss curves
- `outputs/plane_wave_validation/fields_z0.png` - Field distributions
- `outputs/plane_wave_validation/trained_network.pt` - Saved model
- `outputs/plane_wave_validation/validation_metrics.txt` - Numerical metrics

**Configuration:**

Edit constants in `validate_plane_wave.py`:
- `N_EPOCHS = 10000` - Training epochs
- `BATCH_SIZE = 2048` - Collocation points per batch
- `LEARNING_RATE = 1e-3` - Initial learning rate
- `OMEGA = 2 * np.pi * 1e9` - Frequency (1 GHz)

**Troubleshooting:**

If residuals don't converge below 1e-4:
1. Try longer training (`N_EPOCHS = 20000`)
2. Try different learning rate (`LEARNING_RATE = 5e-4`)
3. Enable Fourier features in `create_network()`
4. Check gradient flow (see debugging guide in design doc)

**Next Steps:**

After successful validation:
- Try different frequencies
- Add boundary conditions
- Progress to SPP problems
```

**Step 2: Commit**

```bash
git add examples/README.md
git commit -m "docs: add README for validation examples"
```

---

## Task 9: Run Full Integration Test

**Files:**
- Modify: `tests/examples/test_validate_plane_wave.py`

**Step 1: Add integration test marker**

File: `tests/examples/test_validate_plane_wave.py` (add at top after imports)
```python
# Mark slow tests
pytest.mark.slow = pytest.mark.skipif(
    not pytest.config.getoption("--run-slow", default=False),
    reason="Slow test, use --run-slow to run"
)
```

**Step 2: Run all tests**

Run:
```bash
python -m pytest tests/examples/test_validate_plane_wave.py -v
```

Expected: All tests PASS (may skip slow tests)

**Step 3: Run with slow tests**

Run:
```bash
python -m pytest tests/examples/test_validate_plane_wave.py -v --run-slow
```

Expected: All tests PASS including training convergence test

**Step 4: Final commit**

```bash
git add tests/examples/test_validate_plane_wave.py
git commit -m "test: add comprehensive test suite for plane wave validation"
```

---

## Task 10: Update Project Documentation

**Files:**
- Modify: `README.md` (project root)

**Step 1: Add validation section to README**

File: `README.md` (find appropriate section and add)
```markdown
## Validation

The project includes validation experiments to verify PINN fundamentals:

### Plane Wave Validation

Pure physics-informed training on free space plane wave:

```bash
python examples/validate_plane_wave.py
```

See `examples/README.md` for details.
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add validation section to project README"
```

---

## Final Verification Checklist

Run these commands to verify everything works:

```bash
# 1. All tests pass
python -m pytest tests/examples/test_validate_plane_wave.py -v

# 2. Script runs without errors (quick test with reduced epochs)
# Edit examples/validate_plane_wave.py temporarily: N_EPOCHS = 1000
python examples/validate_plane_wave.py

# 3. Check outputs generated
ls -la outputs/plane_wave_validation/

# 4. Verify git status clean
git status

# Expected: All committed, working tree clean (or only N_EPOCHS change)
```

---

## Success Criteria

**Minimum Viable:**
- [ ] Script runs without errors
- [ ] Training loss decreases
- [ ] Outputs generated (plots, metrics, model)
- [ ] All tests pass

**Target Success:**
- [ ] Maxwell residuals < 1e-4 after full training
- [ ] Training converges smoothly
- [ ] Plots show wave-like solution
- [ ] Code well-documented

**Stretch:**
- [ ] Maxwell residuals < 1e-6
- [ ] E ⊥ H verified quantitatively
- [ ] Learned wavelength matches analytical

---

## Next Steps After Implementation

1. **Run full experiment** with N_EPOCHS = 10000
2. **Analyze results** - did it converge?
3. **Document findings** in experiment log
4. **If successful:** Add to CI/CD as integration test
5. **If needs work:** Debug using strategies in design doc
6. **Either way:** Merge back to main using @superpowers:finishing-a-development-branch
