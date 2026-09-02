# Surface Plasmon Polaritons on Metamaterials via Physics-Informed Neural Networks

[![CI](https://github.com/JonesRobM/Metamaterials_PINN/actions/workflows/ci.yml/badge.svg)](https://github.com/JonesRobM/Metamaterials_PINN/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**[Project site and results →](https://jonesrobm.github.io/Metamaterials_PINN/)**

Physics-informed neural networks (PINNs) that solve frequency-domain Maxwell's
equations for surface plasmon polaritons at metamaterial interfaces, validated
throughout against closed-form and transfer-matrix references.

The headline result: a PINN trained on a **real Ag/silica multilayer** predicts
the bound surface mode substantially more accurately than the effective-medium
approximation normally used for such structures — 430–1700x closer in
Re *k*<sub>spp</sub> and 11–18x closer in Im *k*<sub>spp</sub> across three
training seeds, measured against a transfer-matrix solution.

<p align="center">
  <img src="figures/multilayer/field_profiles.png" width="90%"
       alt="PINN, transfer-matrix and effective-medium field profiles through an Ag/silica multilayer">
</p>

*The E<sub>z</sub> sawtooth (panel b) is the point: the field is discontinuous at all
13 interfaces. The PINN (orange) tracks the transfer-matrix truth (black); the
homogenised model (dashed) draws a smooth curve straight through it.*

---

## What this project does

Solvers for plasmonic metamaterials either mesh the structure (expensive at
subwavelength feature sizes) or homogenise it into an effective medium (cheap,
but approximate in ways that are rarely quantified). This repository explores
the third option — learning the field directly from Maxwell's equations — and
measures how well it works at every step against references that are themselves
independently validated.

Three things distinguish it from a typical PINN demonstration:

1. **Every physics routine is pinned to an external reference.** The curl and
   divergence operators, the anisotropic permittivity path, the interface
   conditions and the surface-plasmon dispersion are each benchmarked against
   exact solutions or literature values, not merely against each other.
2. **Negative results are reported.** Two experiments initially collapsed to the
   trivial zero field; the cause and the fix are documented rather than quietly
   overwritten. Where a target was missed, it says so.
3. **The project's own central approximation is tested.** Homogenising the
   multilayer turns out to carry an O(*a*) surface-term error, an order of
   magnitude stricter than the usual "period much less than lambda" rule — see
   [Known limitations](#known-limitations).

---

## Physics

### Governing equations

With the `e^{-iwt}` time convention (so lossy media have Im eps > 0 and decaying
waves have Im *k* > 0):

```
curl E =  i w mu0 H            (Faraday)
curl H = -i w eps0 eps_bar E   (Ampere)
div (eps_bar E) = 0            (Gauss)
div H = 0                      (no magnetic monopoles)
```

### Uniaxial metamaterials

For a medium with optical axis along z,

```
eps_bar = diag(eps_perp, eps_perp, eps_par)
```

where the two components are engineered by subwavelength structuring. This
repository builds them from a real structure — an Ag/silica multilayer
homogenised by effective-medium theory (`src/effective_medium.py`) — rather than
choosing them by hand, so the design parameters are ones a fabricator could act
on.

### The loss

```
L = w1*||curl E - i w mu0 H||^2 + w2*||curl H + i w eps0 eps_bar E||^2
  + w3*||div D||^2 + w4*||BC||^2 + w5*||anchor||^2
```

The Maxwell residuals alone are minimised exactly by **E = H = 0**. Every
experiment therefore carries a soft boundary anchor taken from the reference
solution; without it, training reliably collapses to the trivial field. This was
learned the hard way — see `docs/plans/`.

---

## Results

Each experiment recovers a known solution from the physics loss plus a boundary
anchor, with no interior data. Errors are relative L2 against the reference
field.

| Experiment                                       | Reference                 |      Field error |    Dispersion error |
| ------------------------------------------------ | ------------------------- | ---------------: | ------------------: |
| Plane wave in free space                         | analytic                  |           7.8e-4 | 8.7e-5 (wavelength) |
| SPP, silver/air at 633 nm                        | analytic                  |           3.9e-3 |              8.3e-5 |
| SPP, uniaxial metamaterial                       | analytic                  |           9.0e-3 |              1.6e-4 |
| Dispersion, one omega-conditioned network        | analytic                  |     1.9e-2 worst |        2.6e-3 worst |
| Design space, one (omega, f)-conditioned network | analytic                  |     3.5e-2 worst |        3.4e-3 worst |
| **Real multilayer, 13 interfaces**         | **transfer matrix** | **5.8e-3** |    **4.9e-5** |

Dispersion error is the relative error in Re *k*<sub>spp</sub> except for the
plane wave, where the recovered wavelength is quoted.

Two results are worth expanding on.

### One network covers a continuum

A single frequency-conditioned network reproduces the whole SPP dispersion curve
of a dispersive metamaterial across a 65%-wide band, rather than needing one
network per frequency. It reproduces the *curvature*, not just the slope: its
residuals about a straight line trace the analytical curvature to 1.7%.

Extending the conditioning to the metal fill fraction *f* gives a surrogate over
the whole fabricable design space (omega, *f*). Gradient descent **through the
trained network** then solves an inverse-design problem — find the fill fraction
achieving a target effective index — matching the closed-form answer to
delta-*f* of about 5e-5.

<p align="center">
  <img src="figures/hmm_surrogate/k_spp_surface.png" width="90%"
       alt="Analytical, PINN and error maps of k_spp over the (omega, f) design space">
</p>

### The multilayer result

At a 30 nm layer period the effective-medium approximation misestimates
Re *k*<sub>spp</sub> by 2.3% and Im *k*<sub>spp</sub> by 34%. A PINN trained on the
actual layers reduces those to 0.005% and 1.9%:

|                          | Re*k*<sub>spp</sub>/k0 |            error | Im*k*<sub>spp</sub> (1/m) |           error |
| ------------------------ | -----------------------: | ---------------: | --------------------------: | --------------: |
| Transfer matrix (truth)  |                 1.058313 |               — |                        8813 |              — |
| Effective medium         |                 1.082930 |            2.33% |                       11778 |           33.6% |
| **PINN (layered)** |       **1.058365** | **0.005%** |              **8982** | **1.92%** |

The result is stable across seeds: three independently trained runs give field
relative L2 of 5.6×10⁻³ ± 0.6×10⁻³, every one reaching target tier with a bound
mode in both regions (`figures/multilayer/seed_variance.json`). The table above
quotes seed 0, the shipped checkpoint.

The mechanism is a *displacement adapter*: the network emits a continuous
D<sub>z</sub>, which is divided by the piecewise-constant eps(z), so the physical
E<sub>z</sub> discontinuity is exact by construction at every interface rather
than something a smooth network must approximate. It generalises from one
interface to many for free, and E<sub>z</sub> ends up the *most* accurately
predicted component.

### Which design choices actually matter

`examples/ablation_study.py` removes one ingredient at a time from the
single-interface case, on an identical (shortened) schedule and seed:

| removed | relative L2 vs control | what breaks |
|---|---:|---|
| boundary anchor | 2.70x | collapses to 0.1% of the correct amplitude |
| displacement adapter | 2.30x | E_z interface jump essentially unlearned (ratio 1.1 where it should be 18.3) |
| per-medium loss weighting | 1.87x | metal-side physics dominates |
| physics-loss ramp | 1.42x | slower, but no collapse |

Only the control recovers a bound mode on *both* sides of the interface.

The anchor result is the one worth dwelling on. Without it, training reaches a
final loss **31,000x lower than the control** — a beautifully converging curve
— while the field it has learned is 0.1% of the correct amplitude. The physics
residual is genuinely minimised; it is just minimised by nothing at all.

### The supervised baseline, and what it means

Plain regression against the transfer-matrix field — no physics — reaches
relative L2 1.4e-3 on the multilayer in ten minutes, beating the 45-minute
PINN's 5.8e-3 by 4x (`figures/ablation/supervised_baseline.json`). That is not
an embarrassment; it is the point. The supervised net needs the full field as
training data, i.e. a problem someone has already solved. The PINN needs only
boundary values and Maxwell's equations. When an exact reference exists,
fitting it (or calling it) is strictly the better tool; physics-informed
training is for when one does not. Few PINN demonstrations run this control.

---

## Validation

Every physics routine is benchmarked against an independent reference:

| Layer                               | Benchmark                                        | Result                                                        |
| ----------------------------------- | ------------------------------------------------ | ------------------------------------------------------------- |
| Differential operators, curl losses | exact extraordinary wave in a uniaxial crystal   | residuals ~1e-15                                              |
| Interface conditions                | exact Fresnel solution, TE/TM, lossy             | machine precision; Brewster\|r_p\| ~ 1e-17                    |
| SPP analytics                       | Johnson & Christy silver at 633 nm               | L = 56.4 um, delta_metal = 22.9 nm — within literature bands |
| Transfer-matrix solver              | Fresnel; closed-form SPP; thin-film branches     | 5e-17; 3.6e-19; correct long/short-range splitting            |
| Analytical SPP mode                 | Maxwell + continuity via the validated operators | machine precision                                             |

Each PINN experiment also pushes its *reference* solution through the identical
validation pipeline as a self-check, so reported errors measure the network
rather than the measurement.

```bash
pytest -q                             # 865 tests
python scripts/validate_physics.py
```

---

## Known limitations

**Homogenisation.** The metamaterial results model the multilayer as a uniform
uniaxial medium. `examples/emt_validity.py` tests that directly against a
transfer-matrix solve of the real stack and finds the leading error is **O(a)**
in the layer period — a termination effect — not the **O((a/lambda)^2)** bulk
term the usual rule guards. Metal- and dielectric-terminated stacks err by equal
magnitudes with opposite signs, which is the signature.

| Layer period | Error in Re*k*<sub>spp</sub> | Error in Im*k*<sub>spp</sub> |
| ------------ | ------------------------------ | ------------------------------ |
| 2 nm         | 0.2%                           | 1.9%                           |
| 10 nm        | 0.9%                           | 9.1%                           |
| 30 nm        | 2.3%                           | 24%                            |

Homogenised *dispersion* is reliable; homogenised **loss figures are optimistic**
unless the period is small. Terminating the stack with a half-thickness metal
layer cancels the O(a) term and restores O(a^2), holding the error under 1% out
to a 60 nm period.

**Cost.** `examples/cost_analysis.py` measures when the design-space surrogate
amortises. Per k_spp query: closed-form dispersion 0.1 µs, transfer matrix on
the real stack 2.3 ms, trained surrogate 1.6 ms. Against the closed form the
surrogate **never** pays for its 82 minutes of training; against the transfer
matrix it needs ~7 million queries. It crosses over at 8–82 queries only
against a reference costing minutes per solve — the full-wave regime this
project has not yet entered. For planar stacks, use the transfer matrix.

**Other caveats.** Silver is a Drude fit with no interband transitions, so it
degrades below ~450 nm. The dispersion experiments hold eps fixed or take it
from effective-medium theory rather than measured data. Training runs are
CPU-bound (~30-80 minutes each); float32 sets the residual floor unless the
L-BFGS refinement runs in float64.

---

## Quick start

```bash
# Environment (uv recommended; a plain venv works too)
uv venv --python 3.12 .venv
uv pip install -r requirements-dev.txt

# Verify the physics implementation and run the test suite
.venv/bin/python scripts/validate_physics.py
.venv/bin/python -m pytest -q
```

Analytics-only studies run in seconds and need no training:

```bash
.venv/bin/python examples/dispersion_analysis.py      # SPP dispersion, design-space maps
.venv/bin/python examples/hyperbolic_metamaterial.py  # multilayer -> uniaxial, band selection
.venv/bin/python examples/inverse_design.py           # gradient-based design, Pareto front
.venv/bin/python examples/emt_validity.py             # where homogenisation breaks down
```

Training experiments take 30-80 minutes on CPU; all checkpoint and support
`--resume`:

```bash
.venv/bin/python examples/validate_spp.py --case uniaxial
.venv/bin/python examples/validate_multilayer.py
```

### Library use

```python
import numpy as np
from src.effective_medium import hmm_permittivities
from src.physics.metamaterial import MetamaterialProperties
from src.constants import C0

omega = 2 * np.pi * C0 / 633e-9

# An Ag/silica multilayer, 30% metal by volume, homogenised
eps_t, eps_n = hmm_permittivities(omega, fill_fraction=0.30, eps_dielectric_layer=2.25)

# Its surface-plasmon mode against air
medium = MetamaterialProperties(eps_n, eps_t, optical_axis="z", omega=omega)
k_spp = medium.spp_wavevector(eps_dielectric=1.0)
print(f"n_eff = {k_spp.real / (omega / C0):.4f}, L = {medium.propagation_length() * 1e6:.1f} um")
```

---

## Repository layout

```
src/
├── physics/            Maxwell operators, boundary conditions, SPP dispersion
├── models/             PINN architectures, loss functions, field formats
├── data/               collocation-point sampling
├── utils/              metrics and plotting
├── analytical.py       closed-form references (plane wave, SPP mode)
├── effective_medium.py multilayer -> uniaxial homogenisation (Drude + layered EMT)
├── transfer_matrix.py  exact layered-stack solver and mode finder
├── design.py           differentiable dispersion for inverse design
└── constants.py        SI constants (c exact, eps0 derived)

examples/               validation experiments and analytics studies
scripts/                training entry points and utilities
tests/                  865 tests, including the benchmark suites
docs/plans/             design documents and per-experiment results
figures/                generated figures and metrics.json per experiment
```

Every experiment writes a `metrics.json` alongside its figures, and each has a
dated results document in `docs/plans/` recording hyperparameters, what was
measured, and what failed.

---

## Docker

```bash
docker build -t metamaterials-pinn .
docker run --rm metamaterials-pinn
```

---

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE).

## Author

**Dr Robert Michael Jones**

*Currently*

Applied AI Scientist, Whitespace

*Previously*
Department of Physics, King's College London
[robert.m.jones@kcl.ac.uk](mailto:robert.m.jones@kcl.ac.uk) ·
[ORCID 0000-0002-5422-3088](https://orcid.org/0000-0002-5422-3088)

Funding: URF\R1\231460
