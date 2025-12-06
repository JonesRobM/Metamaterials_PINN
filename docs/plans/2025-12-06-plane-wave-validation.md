# Plane Wave Validation Experiment

**Date:** 2025-12-06
**Status:** Design Complete - Ready for Implementation
**Goal:** Validate PINN fundamentals with pure physics-informed training

## Objective

Train a PINN to discover a plane wave solution in free space using **only Maxwell's equations** as constraints, with zero data supervision. This is the ultimate test of whether the physics loss implementation is correct.

## Success Criteria

**Primary Focus: Physics Accuracy**
- Maxwell curl residuals < 1e-6
- Verify ∇×E = -iωμH with high precision
- Verify ∇×H = iωεE with high precision
- Demonstrate that physics constraints alone can discover the solution

**Secondary: Solution Quality**
- E ⊥ H (orthogonality of fields)
- E ⊥ k (transverse wave condition)
- Poynting vector S = E × H* points along propagation direction k
- Phase velocity matches c = 1/√(μ₀ε₀)

## Problem Setup

### Domain
- 3D box: [0, λ] × [0, λ] × [0, λ] where λ is wavelength
- Spatial dimensions in meters

### Physical Parameters
- **Frequency:** Start with 1 GHz (f = 10⁹ Hz) for numerical stability
- **Angular frequency:** ω = 2πf
- **Wave vector:** k = [k₀, 0, 0] propagating in x-direction
- **Wave number:** k₀ = ω/c where c = 1/√(μ₀ε₀)
- **Wavelength:** λ = 2π/k₀ ≈ 0.3 m at 1 GHz
- **Free space:** ε = ε₀ = 8.854×10⁻¹² F/m, μ = μ₀ = 4π×10⁻⁷ H/m

### Boundary Conditions
- Option 1: Periodic boundaries in x-direction
- Option 2: No explicit boundaries (train on interior points only)
- Start with Option 2 for simplicity

### Analytical Solution (for validation only, NOT used in training)
For a plane wave with k = [k₀, 0, 0]:
- **E field:** E = [0, E₀, 0] exp(i(k₀x - ωt))
- **H field:** H = [0, 0, H₀] exp(i(k₀x - ωt))
- **Relationship:** H₀ = (k₀/ωμ₀)E₀ = √(ε₀/μ₀)E₀

## Network Architecture

**Use existing `ElectromagneticPINN`:**
- **Input:** (x, y, z) coordinates [batch, 3]
- **Output:** [batch, 6, 2] electromagnetic fields
  - 6 components: Ex, Ey, Ez, Hx, Hy, Hz
  - 2 values per component: [real, imag]
- **Hidden layers:** 3-4 layers
- **Neurons per layer:** 128-256
- **Activation:** Tanh or `ElectromagneticActivation`
- **Fourier features:** Optional, can try with/without

## Loss Function

**Pure Physics-Informed Training:**
```python
Loss = MaxwellCurlLoss(frequency=ω)
     = Mean( |∇×E + iωμH|² ) + Mean( |∇×H - iωεE|² )
```

**No data loss, no boundary loss - only Maxwell's equations**

Components:
- `curl_E_residual = ∇×E + iωμ₀H`
- `curl_H_residual = ∇×H - iωε₀E`
- Both should approach zero

## Training Procedure

### Hyperparameters
- **Epochs:** 5,000-10,000 (physics-only needs more iterations)
- **Optimizer:** Adam
- **Learning rate:** Start at 1e-3, decay to 1e-4 around epoch 2000
- **LR schedule:** ReduceLROnPlateau or cosine annealing
- **Batch size:** 1024-2048 collocation points
- **Sampling:** Random uniform in domain each epoch

### Training Strategy
1. Initialize network with random weights
2. Each epoch:
   - Sample N collocation points uniformly in domain
   - Compute network output (fields)
   - Compute Maxwell residuals via automatic differentiation
   - Backpropagate and update weights
3. Monitor convergence metrics
4. Save best model based on lowest physics loss

## Monitoring Metrics

### During Training (log every 100 epochs)
1. **Total physics loss**
2. **Curl E residual** (separate)
3. **Curl H residual** (separate)
4. **Field magnitude statistics** (mean, max of |E| and |H|)
5. **Learning rate**

### Post-Training Validation
1. **Maxwell residual distribution:**
   - Evaluate on dense grid (e.g., 50³ points)
   - Plot histogram of residuals
   - Verify max residual < 1e-6

2. **Wave properties:**
   - E·H orthogonality: measure cos(angle) should be ~0
   - E·k orthogonality: should be ~0 (transverse)
   - Poynting vector direction: S should align with k

3. **Wavelength and phase velocity:**
   - Extract spatial frequency from learned solution
   - Verify matches k₀
   - Check phase velocity v = ω/k = c

4. **Field visualization:**
   - Plot Re(E) and Re(H) in 2D slices
   - Animate wave propagation
   - Verify sinusoidal pattern

## Implementation Structure

### File: `examples/validate_plane_wave.py`

```python
# Pseudocode structure

import torch
import torch.nn as nn
from src.models import ElectromagneticPINN, MaxwellCurlLoss
from src.utils.plotting import plot_fields, plot_training_curves
import numpy as np

# Physical constants
omega = 2 * np.pi * 1e9  # 1 GHz
c = 299792458  # m/s
mu0 = 4e-7 * np.pi
eps0 = 8.854e-12
k0 = omega / c
wavelength = 2 * np.pi / k0

# Domain
domain = [0, wavelength] for each dimension

def create_network():
    """Create ElectromagneticPINN instance."""
    return ElectromagneticPINN(
        input_dim=3,
        hidden_dims=[256, 256, 256],
        output_dim=6,  # Will be converted to [batch, 6, 2]
        activation='tanh'
    )

def sample_collocation_points(n_points, domain):
    """Sample random points in domain."""
    return torch.rand(n_points, 3) * wavelength

def train_pure_physics(network, n_epochs=10000):
    """Train with only Maxwell physics loss."""
    loss_fn = MaxwellCurlLoss(frequency=omega)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(...)

    for epoch in range(n_epochs):
        # Sample points
        coords = sample_collocation_points(2048, domain)

        # Compute loss
        loss = loss_fn.compute(network=network, coords=coords)

        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log metrics
        if epoch % 100 == 0:
            log_metrics(epoch, loss, ...)

    return network

def validate_solution(network):
    """Compute physics metrics and visualize."""
    # Evaluate on dense grid
    # Compute residuals
    # Check wave properties
    # Plot fields
    pass

if __name__ == "__main__":
    network = create_network()
    network = train_pure_physics(network)
    validate_solution(network)
```

## Expected Outcomes

### If Training Converges (Success):
- Physics loss decreases steadily to < 1e-6
- Network learns wave-like solution
- Proves that `MaxwellCurlLoss` implementation is correct
- Demonstrates PINN can discover solutions from pure physics
- **Confidence:** Ready to tackle more complex problems

### If Training Struggles (Learning Opportunity):
- Identifies issues with physics loss implementation
- May need gradient scaling, loss balancing, or architecture tuning
- Provides clear debugging target
- Still valuable: know exactly what needs fixing

### Debug Strategies if Needed
1. Check gradient flow (use `torch.autograd.grad` inspection)
2. Try different learning rates (sweep 1e-2 to 1e-5)
3. Add Fourier features for better high-frequency representation
4. Scale loss terms if E and H residuals have different magnitudes
5. Visualize intermediate solutions to see what network is learning

## Next Steps After Validation

### If Successful:
1. **Document results** in experiment log
2. **Add to test suite** as integration test
3. **Try variations:**
   - Different frequencies
   - Different propagation directions
   - Multiple plane waves superposition
4. **Move to next complexity level:**
   - Add boundaries
   - Add materials
   - Surface plasmon polaritons

### If Needs Work:
1. **Systematic debugging** of physics loss
2. **Hyperparameter tuning** experiments
3. **Architecture exploration**
4. **Consider hybrid approach** (Option B or C from design)

## Success Metrics Summary

**Minimum Viable Success:**
- [ ] Maxwell residuals < 1e-4 (relaxed target)
- [ ] Solution shows wave-like behavior
- [ ] Training is stable (doesn't diverge)

**Target Success:**
- [ ] Maxwell residuals < 1e-6
- [ ] E ⊥ H verified
- [ ] Correct wavelength recovered

**Stretch Success:**
- [ ] Maxwell residuals < 1e-8
- [ ] All wave properties perfectly satisfied
- [ ] Works for multiple frequencies

## Resources Required

- **Compute:** CPU sufficient, GPU helpful for larger batches
- **Time:** ~30 minutes to 2 hours per training run
- **Memory:** Minimal (<1GB for network + data)
- **Storage:** ~10MB per saved model

## References

- Maxwell's equations in frequency domain
- PINN methodology (Raissi et al., 2019)
- Plane wave solutions in electromagnetics
