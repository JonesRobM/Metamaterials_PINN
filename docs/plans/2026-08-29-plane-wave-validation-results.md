# Plane Wave Validation — Results

**Date:** 2026-08-29
**Design:** [2025-12-06-plane-wave-validation.md](2025-12-06-plane-wave-validation.md)
**Implementation plan:** [2025-12-06-plane-wave-validation-implementation.md](2025-12-06-plane-wave-validation-implementation.md)
**Script:** `examples/validate_plane_wave.py` · **Tests:** `tests/examples/test_validate_plane_wave.py`
**Outcome:** physically correct plane wave recovered (rel-L2 error 7.8e-4, wavelength error 9e-5); Maxwell residuals reached 1.5e-3 relative RMS — the design doc's *minimum* tier (1e-4) was **not** reached in the 24-minute CPU budget, although the loss was still decreasing steadily when training stopped.

## What was run

```
.venv/bin/python examples/validate_plane_wave.py --epochs 4000 --n-points 1024 --lbfgs-steps 100 --seed 0
```

Problem: free-space plane wave at f = 1 GHz (k₀ = 20.9585 rad/m, λ = 0.29979 m) on the cube [0, λ]³;
E = ŷ E₀ e^{ik₀x}, H = ẑ (E₀/η₀) e^{ik₀x}, E₀ = 1 V/m, H₀ = 2.6544e-3 A/m. Convention e^{-iωt}.

### As-built deviations from the December plan

| Plan | As built | Why |
|---|---|---|
| Physics loss only, no boundary term ("Option 2") | Curl + divergence residuals in the interior **plus a soft Dirichlet term** (MSE to the analytical field) on the six cube faces, weight 10 | Without a boundary condition the curl equations admit every plane wave, standing wave and the trivial solution E = H = 0; the network collapses to zero amplitude. The plan's `main` task also anticipated a data/boundary term. No interior data are used. |
| Loss in SI units | Trained in a **dimensionless frame** x̂ = x/λ, Ê = E/E₀, Ĥ = η₀H/E₀, i.e. `MaxwellCurlLoss(frequency=2π, mu0=1, eps0=1)` | In SI the two curl residuals differ by η₀² ≈ 1.4e5 in squared magnitude and the H equation is ignored by the optimiser. All reported metrics are in SI, computed through the `PlaneWavePINN` wrapper (coords in metres → SI fields). |
| `MaxwellCurlLoss` only | `MaxwellCurlLoss` + `MaxwellDivergenceLoss` (weight 1) | Cheap and speeds up early convergence. |
| 10 000 Adam epochs, 2048 points | 4000 Adam epochs, 1024 points, then 100 L-BFGS steps on a fixed 4096-point set | 10 000 × 2048 would take ~40 min on this CPU (0.13 s/epoch at 1024 points). Adam plateaus at ~2e-3 dimensionless loss; L-BFGS reduced it a further 40× for the same wall-clock. |
| `ReduceLROnPlateau` | Cosine annealing 1e-3 → 1e-5 | Deterministic schedule; same effect. |
| Duplicated constants / analytical solution | Imported from `src.constants` and `src.analytical` | Code-reuse requirement. |

### Hyperparameters

| | |
|---|---|
| Network | `ElectromagneticPINN` (complex-valued), hidden [128, 128, 128, 128], `complex_tanh`, 128 Fourier features (k ∈ [0.1, 20] per λ; target k̂ = 2π), 117 516 parameters |
| Phase 1 | Adam, lr 1e-3 cosine-annealed to 1e-5, 4000 epochs, 1024 interior + 512 boundary points resampled every epoch |
| Phase 2 | L-BFGS (`max_iter=20`, `history_size=50`, strong-Wolfe), 100 outer steps, fixed 4096 interior + 2048 boundary points |
| Loss | curl + 1·div + 10·boundary, dimensionless |
| Seed / device | 0 / CPU (macOS, 15 cores) |
| Runtime | 611 s Adam + 848 s L-BFGS = **1459 s** |

## Metrics (SI units, 20 000 random interior points)

Residuals are RMS over points, divided by k₀·RMS|field| so that 1.0 means "as large as the curl itself".
The analytical field evaluated through the same pipeline gives 5.8e-8 / 5.3e-8 (float32 noise), which confirms the e^{-iωt} sign convention of `MaxwellCurlLoss` matches `src.analytical.analytical_plane_wave`.

| Metric | Value | Criterion |
|---|---|---|
| Relative L2 error, E | **7.77e-4** | — |
| Relative L2 error, H | **7.61e-4** | — |
| ‖∇×E − iωμ₀H‖ / (k₀‖E‖), RMS | **1.46e-3** | min < 1e-4, target < 1e-6 — not met |
| ‖∇×H + iωε₀E‖ / (k₀‖H‖), RMS | **1.43e-3** | min < 1e-4, target < 1e-6 — not met |
| same, max over points | 3.5e-2 / 3.0e-2 | — |
| ∇·E, ∇·H (relative) | 6.0e-4 / 6.7e-4 | — |
| E·H* / (|E||H|), mean | **5.1e-4** | E ⊥ H — met |
| |Eₓ| / |E|, mean (transversality) | 3.7e-4 | E ⊥ k — met |
| Poynting alignment ⟨Sₓ/|S|⟩ | 1.0000 | S ∥ k — met |
| Impedance ratio (|E|/|H|)/η₀ | 1.0000 | — |
| Wavelength (phase-slope fit) | 0.29977 m, rel. error **8.7e-5** | correct λ — met |
| Wavelength (zero-padded FFT peak) | 0.29979 m (exact bin; resolution λ/64) | — |
| Final / best dimensionless loss | 4.93e-5 | — |

Full numbers: `figures/plane_wave_validation/metrics.json`.

### Success tier reached

- **Minimum** (residuals < 1e-4, wave-like, stable): **partially** — training is stable and the solution is unambiguously the right wave (λ to 1e-4, E⊥H, E⊥k, S∥k, correct impedance), but the residual RMS is 1.5e-3, one order of magnitude above the threshold.
- **Target** (< 1e-6, E⊥H, correct λ): not reached on residuals; E⊥H and λ criteria are met.
- The script's own `success_tier` field therefore reports `not met`.

Interpretation: the *field* error (7.8e-4) is below the *residual* error (1.5e-3) because the residual is a derivative quantity — it amplifies the small high-frequency ripple visible in the error panels of `field_slices.png`, which is concentrated near the cube edges and corners where the soft Dirichlet term and the interior physics compete.

## Figures

All under `figures/plane_wave_validation/`:

- `training_history.png` — total/curl/div/boundary loss vs epoch (L-BFGS phase is the tail after epoch 4000).
- `field_slices.png` — Re(E_y), Re(H_z) at z = λ/2: PINN, analytical, error.
- `line_profile.png` — Re/Im of E_y and H_z along x through the cube centre against the analytical curves.
- `residual_histogram.png` — distribution of the two relative curl residuals on 5000 random points.
- `spectrum.png` — spatial spectrum of E_y along x; single peak at k/k₀ = 1.0000.

Model checkpoint: `artifacts/models/plane_wave_validation.pth` (`{"state_dict", "config", "metrics"}` for `PlaneWavePINN`).

## Observations

1. **Balancing the two curl equations is essential.** In SI units the H-equation residual is 1.4e5× smaller in squared norm than the E-equation residual; the earlier `scripts/train_plane_wave_pinn.py` (1 PHz, SI units, `plane_wave_pinn.pth`) trains essentially on Faraday's law alone. The dimensionless frame fixes this with zero code duplication — `MaxwellCurlLoss(frequency=k₀λ, mu0=1, eps0=1)`.
2. **A boundary condition is not optional.** With physics only, the zero field is a perfect minimiser and Adam finds it within a few hundred epochs (amplitude collapsed to 0.16 in the 200-epoch smoke run). The design doc's "Option 2: no boundaries" cannot work as stated.
3. **Adam stalls, L-BFGS does not.** Adam reached ~2e-3 dimensionless loss and flattened by epoch 3000 regardless of the LR schedule. L-BFGS cut the loss from 2.1e-3 to 4.9e-5 in 100 steps (~14 min) and was still descending ~2× per 25 steps; the residual-vs-step curve suggests another 200–300 steps would reach the 1e-4 relative threshold. Float32 arithmetic will cap achievable residuals around 1e-6–1e-7 (the analytical field already shows 5e-8); reaching the *target* tier realistically needs float64.
4. **Fourier features made the difference between learning and not learning.** The target wavenumber (2π per λ) sits inside the encoder's band [0.1, 20], and the learned spectrum is a single clean line at k₀.
5. The convention check (analytical field → residual 5e-8) is a cheap regression test for the sign conventions in `src/models/loss_functions.py`; it is now in `tests/examples/test_validate_plane_wave.py::test_analytical_network_satisfies_maxwell`.

## Recommended next steps toward the SPP experiment (`scripts/train_spp_pinn.py`)

`scripts/train_spp_pinn.py` has never produced a checkpoint. The plane-wave run points at the likely reasons and a path forward:

1. **Finish the plane-wave tiers first (cheap).** Re-run with `--lbfgs-steps 400` (and optionally `torch.set_default_dtype(torch.float64)`) to confirm the 1e-4 and 1e-6 residual tiers are reachable; this fixes the bar for the SPP problem.
2. **Non-dimensionalise the SPP problem the same way.** Use x̂ = x/λ₀ (λ₀ ≈ 600 nm–1 µm) and Ĥ = η₀H/E₀, so `MaxwellCurlLoss(frequency=2π, mu0=1, eps0=1, ...)` with `epsilon=ε_r(x̂)` gives balanced residuals. Working in metres with 1e15 rad/s frequencies (as the current script does) produces ω²μ₀ε₀ ~ 1e14 m⁻² coefficients and hopeless conditioning.
3. **Impose a boundary/normalisation condition.** For the SPP the analytical mode (`src.physics` SPP wavevector, `SPPNetwork.k_spp`) can be prescribed on the input face x̂ = 0 (soft Dirichlet, as here) plus the radiation/decay condition on the far z faces (`RadiationLoss`, `SPPBoundaryLoss`); otherwise the trivial solution wins exactly as it did here.
4. **Handle the interface explicitly.** Use a two-domain formulation (two `ElectromagneticPINN`s or `SPPNetwork` on each side) with `InterfaceBoundaryLoss` at z = 0 rather than one network across the ε discontinuity — tanh networks cannot represent the kink in E_z, which is where the SPP physics lives.
5. **Keep the two-phase optimiser.** Adam (with resampled points) to ~1e-3, then L-BFGS on a fixed, denser point set; save the best iterate.
6. **Check the Fourier band.** The SPP wavenumber k_spp/k₀ ≈ 1.03–1.1 lies inside the band, but the evanescent decay length in the metal is ~λ₀/30; either add higher-frequency features in ẑ or refine sampling near the interface (`StratifiedSampler`).
7. **Add the analytical-field residual check** for the SPP mode as a unit test before any training run — it costs seconds and would have caught sign/convention problems in `train_spp_pinn.py` immediately.
