# Surface Plasmon Polaritons on Metamaterials via Physics-Informed Neural Networks

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*Revolutionising electromagnetic metamaterial design through AI-driven physics simulation*

## 🔬 Overview

This project demonstrates a cutting-edge approach to modelling **surface plasmon polaritons (SPPs)** on metamaterial interfaces using **Physics-Informed Neural Networks (PINNs)**. By embedding Maxwell's equations directly into the neural network training process, we achieve unprecedented accuracy in predicting complex electromagnetic phenomena without requiring extensive experimental data.

### Why This Matters

Traditional numerical methods for electromagnetic simulation face significant challenges when dealing with metamaterials:
- **Computational complexity** scales poorly with frequency and structure size
- **Mesh requirements** become prohibitive for subwavelength features
- **Convergence issues** arise near plasmonic resonances
- **Limited generalisability** across parameter spaces

Our PINN approach overcomes these limitations by learning the underlying physics directly, enabling:
- ⚡ **Real-time field prediction** across arbitrary geometries
- 🎯 **Mesh-free computation** with automatic differentiation
- 🔄 **Inverse design capabilities** for optimal metamaterial parameters
- 📈 **Scalable simulation** from nanoscale to macroscale structures

---

## 🧮 The Physics

### Surface Plasmon Polaritons
SPPs are electromagnetic waves that propagate along metal-dielectric interfaces, combining light with collective electron oscillations. In metamaterials, these can be engineered to achieve:

- **Subwavelength confinement** beyond the diffraction limit
- **Enhanced field intensities** for sensing and nonlinear optics  
- **Negative refractive indices** for cloaking applications
- **Broadband operation** through dispersion engineering

### Maxwell's Equations in Frequency Domain
With the `e^{-iωt}` time convention (so lossy media have Im ε > 0 and decaying
waves have Im k > 0), the electromagnetic behaviour is governed by:

```
∇ × E =  iωμ₀H    (Faraday's law)
∇ × H = -iωε₀εᵣE  (Ampère's law)
∇ · (εᵣE) = 0     (Gauss's law)
∇ · H = 0         (No magnetic monopoles)
```

### Metamaterial Constitutive Relations
For uniaxial metamaterials with optical axis along ẑ:

```
εᵣ = [ε⊥  0   0 ]
     [0   ε⊥  0 ]  
     [0   0   ε∥]
```

Where `ε⊥` and `ε∥` are engineered through subwavelength structuring.

---

## 🏗️ Architecture

### 🧠 Physics-Informed Neural Network

Our PINN architecture directly embeds physical laws into the loss function:

```python
L_total = λ₁L_maxwell + λ₂L_boundary + λ₃L_data + λ₄L_initial
```

- **L_maxwell**: Residuals of Maxwell's equations at collocation points
- **L_boundary**: Interface boundary conditions (E/H field continuity)
- **L_data**: Sparse experimental/simulation data (when available)
- **L_initial**: Initial conditions for time-dependent problems

### 📁 Project Structure

```
Metamaterials_PINN/
├── src/
│   ├── constants.py          # Shared physical constants (EPS0, MU0, C0, ETA0)
│   ├── analytical.py         # Closed-form reference solutions (plane wave, point charge)
│   ├── physics/              # Core electromagnetic physics
│   │   ├── maxwell_equations.py    # Frequency-domain Maxwell operators
│   │   ├── metamaterial.py         # Anisotropic constitutive relations
│   │   └── boundary_conditions.py  # Interface continuity conditions
│   ├── models/               # Neural network architectures
│   │   ├── pinn_network.py         # ElectromagneticPINN, ComplexPINN, SPPNetwork, ...
│   │   ├── loss_functions.py       # Physics-informed loss terms
│   │   └── electrostatics_pinn.py  # Small Laplace/point-charge PINN
│   ├── data/
│   │   └── domain_sampler.py       # Uniform / stratified / interface / SPP / adaptive sampling
│   └── utils/
│       ├── plotting.py             # Field and training visualisation
│       └── metrics.py              # Residual, SPP and accuracy metrics
│
├── config/                   # YAML configuration and ConfigManager
├── tests/                    # pytest suite (physics, models, losses, config, examples)
├── scripts/                  # Entry points (all accept --help)
│   ├── validate_physics.py         # Sanity-check the physics module
│   ├── run_tests.py                # Run the whole test suite
│   ├── train_plane_wave_pinn.py    # Train ComplexPINN on a free-space plane wave
│   ├── train_point_charge_pinn.py  # Train ElectrostaticsPINN on a point charge
│   ├── train_spp_pinn.py           # Train SPPNetwork from config/spp_config.yaml
│   ├── visualize_pinn_plane_wave.py
│   ├── visualize_pinn_field.py
│   └── visualize_point_charge.py
├── examples/
│   └── validate_plane_wave.py      # Plane-wave validation example
├── notebooks/                # Standalone PINN tutorials (not part of the SPP pipeline)
├── artifacts/models/         # Trained weights (*.pth)
├── figures/                  # Generated plots
└── docs/
```

---

## ✨ Key Features

### 🎯 **Mesh-Free Simulation**
- No spatial discretisation required
- Automatic handling of complex geometries
- Adaptive resolution based on field gradients

### ⚡ **Real-Time Prediction**  
- Forward pass inference in milliseconds
- Enables interactive design exploration
- Suitable for real-time optimization loops

### 🔄 **Inverse Design Capability**
- Optimize metamaterial parameters for target responses
- Discover novel plasmonic structures
- Multi-objective design optimization

### 📏 **Multi-Scale Modelling**
- Seamless transition from nanoscale to macroscale
- Handles both local field enhancement and far-field radiation
- Automatic resolution adaptation

### 🎨 **Sophisticated Visualisation**
- Real-time field plotting with interactive controls
- 3D electromagnetic field rendering
- Dispersion relation visualisation
- Poynting vector flow analysis

---

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/JonesRobM/Metamaterials_PINN.git
cd Metamaterials_PINN

# Create a virtual environment and install in editable mode with dev/viz extras
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
# (or: uv venv --python 3.12 .venv && uv pip install -r requirements-dev.txt)
```

### Validate the physics implementation
```bash
.venv/bin/python scripts/validate_physics.py
.venv/bin/python -m pytest -q          # or: python scripts/run_tests.py
```

### Basic Usage
```python
import numpy as np
import torch

from src.physics import MaxwellEquations, MetamaterialProperties
from src.models import SPPNetwork, MaxwellCurlLoss
from src.data import UniformSampler

omega = 2 * np.pi * 1e15  # 1 PHz

# Define metamaterial
metamaterial = MetamaterialProperties(
    eps_parallel=4.0 + 0.05j,      # along the optical axis (z, interface normal)
    eps_perpendicular=-2.0 + 0.1j, # in-plane: negative => binds a TM surface wave
    optical_axis='z'
)

# Initialise PINN and physics loss
network = SPPNetwork(spatial_dim=3, hidden_dims=[64, 64, 64], frequency=omega)
loss_fn = MaxwellCurlLoss(frequency=omega)

# Sample collocation points and evaluate the Maxwell residual loss
sampler = UniformSampler(domain_bounds=[(-1e-6, 1e-6)] * 3)
coords = sampler.sample_points(n_points=1000)['points'].requires_grad_(True)
loss = loss_fn(network=network, coords=coords)
```

Full training loops live in `scripts/train_*.py`; see `--help` on each for options.

## ✅ Validation Status

The physics engine is verified against independent ground truth at every layer
(all in the test suite, `pytest -q`):

| Layer | Benchmark | Result |
|---|---|---|
| Differential operators & curl losses | Exact extraordinary wave in a uniaxial crystal (`tests/test_benchmark_anisotropic.py`) | residuals ~1e-15 relative |
| Interface machinery | Exact Fresnel solution, TE/TM, lossy included (`tests/test_benchmark_fresnel.py`) | machine precision; Brewster \|r_p\| ~ 1e-17 |
| SPP analytics | Johnson & Christy silver at 633 nm + independent root-finding (`tests/test_benchmark_spp.py`) | L = 56.4 µm, δ_m = 22.9 nm — all within literature bands |
| Analytical SPP mode | Maxwell + continuity via the validated operators (`tests/test_analytical_spp.py`) | machine precision |

End-to-end experiments (each recovers a known solution from physics + a boundary
anchor; results in `docs/plans/`):

- **Plane wave** — `examples/validate_plane_wave.py`: relative L2 ≈ 8e-4,
  wavelength error 9e-5, E ⊥ H confirmed.
- **Surface plasmon (silver/air, 633 nm)** — `examples/validate_spp.py`:
  the PINN recovers the bound SPP mode (relative L2 ≈ 4e-3, k_spp to 0.008%,
  decay constants < 1%), using a displacement adapter so the physical E_z
  discontinuity at the interface is exact by construction.
- **Anisotropic SPP (uniaxial metamaterial)** — `examples/validate_spp.py
  --case uniaxial`: relative L2 ≈ 9e-3, k_spp to 0.016%, both decay
  constants < 1%.
- **Dispersion recovery (one ω-conditioned network)** — `examples/validate_spp_dispersion.py`:
  a single frequency-conditioned PINN reproduces k_spp(ω) across a 30 %-wide
  band (worst relative L2 0.026, k_spp to 0.15 %) instead of retraining per
  frequency.
- **Design space & inverse design** — `examples/dispersion_analysis.py`
  (bound-mode existence and dispersion maps over the (ε_t, ε_n) plane) and
  `examples/inverse_design.py` (gradient-based design through the
  differentiable dispersion in `src/design.py`: target wavevector, maximum
  propagation under a confinement bound, target field enhancement).

## 🛠️ Technical Details

### Automatic Differentiation
We leverage PyTorch's automatic differentiation to compute spatial derivatives:

```python
def curl_operator(self, field, coords):
    """Compute ∇ × field using automatic differentiation."""
    # Partial derivatives computed via autodiff
    dFz_dy = torch.autograd.grad(Fz, coords, create_graph=True)[0][:, 1]
    # ... (curl computation)
    return curl
```

### Adaptive Sampling
Smart collocation point placement based on:
- **Residual magnitude**: Higher density where physics violations occur
- **Field gradients**: Enhanced resolution near interfaces
- **Geometric features**: Automatic refinement around sharp boundaries

### Loss Function Design
Carefully balanced multi-term loss ensures physical consistency:

```python
L = λ₁‖∇×E - iωμ₀H‖² + λ₂‖∇×H + iωε₀εᵣE‖² + 
    λ₃‖boundary_conditions‖² + λ₄‖training_data‖²
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Funding**: URF\R1\231460
- **Inspiration**: The metamaterials and machine learning communities

---

## 📞 Contact

**Author**: Dr Robert Michael Jones
**Email**: robert.m.jones@kcl.ac.uk  
**Institution**: Department of Physics, King's College London 
**ORCID**: [0000-0002-5422-3088](https://orcid.org/0000-0002-5422-3088)

---

<div align="center">

**⭐ Star this repository if you find it useful! ⭐**

*Advancing the frontiers of computational electromagnetics through AI*

</div>