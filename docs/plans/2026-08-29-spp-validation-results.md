# SPP Validation — Results

**Date:** 2026-08-31 (runs 1–2), updated 2026-08-31 (run 3 + uniaxial case)
**Script:** `examples/validate_spp.py` (tests: `tests/examples/test_validate_spp.py`)
**Status:** silver **TARGET tier** (run 4, 2026-08-31 evening: the run-3 recipe
with the full 150-step float64 L-BFGS budget resolved the anchor-vs-κ_m
competition entirely — rel L2 E 0.0039 / H 0.0071, κ_m 0.92%);
uniaxial metamaterial case: see its section below.

## Experiment

Recover the bound TM SPP mode at a planar interface from physics + a soft
analytical boundary anchor only (no interior data), mirroring
`examples/validate_plane_wave.py`, per the design work in
`2026-08-29-spp-training-triage.md`. Two material cases (`--case`):

- **silver** (default): λ₀ = 633 nm (ω = 2.9758×10¹⁵ rad/s),
  ε_m = −18.3 + 0.55j (z < 0), ε_d = 1 (z > 0).
- **uniaxial**: type-II metamaterial, optical axis z, in-plane ε_t = −4 + 0.2j,
  normal ε_n = 3 + 0.05j, ε_d = 1 (the anisotropic benchmark of
  `tests/test_analytical_spp.py`).

Ground truth: `src.analytical.analytical_spp_fields` (machine-precision
benchmarked, incl. the uniaxial ε_t/ε_n split). Derived scales (from
`MetamaterialProperties(eps_parallel=ε_n, eps_perpendicular=ε_t, 'z', omega)` —
`eps_parallel` is the component **along the optical axis**, i.e. the normal
component for axis 'z'):

| Scale | silver | uniaxial |
|---|---:|---:|
| k_spp | 1.0209×10⁷ + 8.86×10³j /m (λ_spp = 615.5 nm) | 1.0661×10⁷ + 3.96×10⁴j /m (λ_spp = 589.4 nm) |
| κ_d | 2.386×10⁶ /m (δ_d = 419 nm) | 3.890×10⁶ /m (δ_d = 257 nm) |
| κ_m | 4.368×10⁷ /m (δ_m = 22.9 nm) | 1.558×10⁷ /m (δ_m = 64.2 nm) |
| domain z (−4.4 δ_m … 1.2 δ_d) | −101 … 503 nm | −282 … 308 nm |
| dimensionless k̂_spp, κ̂_d, κ̂_m | 6.46, 1.51, 27.6 | 6.75, 2.46, 9.86 |

Domain (SI): x ∈ [0, 2λ_spp], y ∈ [0, 0.2λ₀], z as above (runs 1–2 used the
narrower z ∈ [−80, 500] nm). Dimensionless frame: coords/λ₀; fields scaled by
[η₀·H₀]×3 + [H₀]×3 with H₀ = 1 A/m, so Ê, Ĥ are O(1) and
`MaxwellCurlLoss(frequency=2π, mu0=1, eps0=1)` is η₀-balanced.

### Where ε_t vs ε_n enters (uniaxial case)

Each use is deliberate; silver is the ε_t = ε_n limit of the same code path.

| Place | Component used | Why |
|---|---|---|
| Mode constants | `MetamaterialProperties(ε_n, ε_t, 'z')` | constructor order is (ε_∥, ε_⊥); ε_∥ lies along the optical axis = surface normal |
| ε tensor in the metamaterial | `diag(ε_t, ε_t, ε_n)` | optical axis z: in-plane xx/yy = ε_t, normal zz = ε_n |
| Curl/divergence losses (metal side) | the full diagonal | ∇×H = −iωε₀(ε·E) and ∇·(ε·E) = 0 mix ε_t (E_x, E_y) and ε_n (E_z) |
| `DisplacementAdapter` divisor below z = 0 | **ε_n** (not ε_t) | the continuous interface quantity is D_z = ε₀ ε_zz E_z |
| Metal loss preconditioning | max(\|ε_t\|, \|ε_n\|) = 4.0 | the curl-H residual penalises Ê errors component-wise by (2π\|ε_comp\|)²; the largest sets the stiffness |
| Fourier band check | κ̂_m = 9.86 < 40 | silver's κ̂_m = 27.6 remains the binding constraint on the (0.1, 40) band |

## Hyperparameters

| Item | Runs 1–2 (2026-08-31 a.m.) | Run 3 / uniaxial (2026-08-31 p.m.) |
|---|---|---|
| Network | `ElectromagneticPINN`, complex MLP 4×128, complex_tanh, 128 Fourier modes, band (0.1, 40) rad/unit | same |
| E_z handling | `DisplacementAdapter`: MLP channel 2 = continuous D̂_z, divided by ε(z) | same, divisor = ε_zz (ε_n below the interface) |
| Interior losses | per-medium curl + divergence, metal terms × 1/\|ε_m\| | same in Adam; **L-BFGS phase raises the metal curl weight to 1/√(max\|ε\|)** (lever d, probe-validated) |
| Interface loss | `TangentialContinuityLoss(offset = 2 nm / λ₀)`, weight 1 | same |
| Anchor | soft Dirichlet vs analytical mode on all six faces, weight 100 | same |
| Physics ramp | 0 → 1 over first 25% of Adam epochs | same |
| Sampling | 2048 pts/epoch, 35% metal, z ∈ [−80, 500] nm | 2048 pts/epoch, **45% metal**, z ∈ [−4.4 δ_m, 1.2 δ_d] |
| Optimiser | Adam lr 1e-3 cosine → 1e-5, 4000 epochs; L-BFGS 40 steps float32 (run 2) | Adam identical; **L-BFGS 50 steps in float64** (~18 s/step, ≈ compute of 100+ float32 steps) |
| Seed / device | 0 / CPU | same |

## Silver metrics (SI units, 20 000 fresh stratified points, ±2 nm guard band excluded)

Run 1 = plane-wave recipe verbatim (collapsed). Run 2 = anchor-100 + ramp +
1/|ε_m| + adapter, L-BFGS 40 float32 steps. Run 3 = run 2 recipe + the
target-tier levers (denser metal sampling, z_min −101 nm, 50 float64 L-BFGS
steps with metal curl weight 1/√|ε_m|). Self-check = the exact analytical mode
through the identical validation pipeline.

| Metric | Run 1 (failed) | Run 2 | Run 3 | **Run 4 (checkpoint)** | Self-check |
|---|---:|---:|---:|---:|---:|
| rel L2 E (overall) | 0.997 | 0.0483 | 0.0677 | **0.0039** | 0.0 |
| rel L2 H (overall) | 0.989 | 0.0509 | 0.0869 | **0.0071** | 0.0 |
| rel L2 E air / metal | 0.997 / 0.999 | 0.0440 / 0.152 | 0.0629 / 0.163 | **0.0034 / 0.0124** | 0 / 0 |
| rel L2 H air / metal | 0.988 / 0.991 | 0.0353 / 0.0802 | 0.0593 / 0.123 | **0.0036 / 0.0111** | 0 / 0 |
| curl-E residual rel, air | 0.748 | 0.0335 | 0.0461 | **0.0088** | 5.1e-08 |
| curl-H residual rel, air | 0.413 | 0.0567 | 0.0999 | **0.0124** | 4.6e-08 |
| curl-E residual rel, metal | 14.0 | 0.791 | 0.464 | **0.0339** | 2.2e-07 |
| curl-H residual rel, metal | 0.886 | 0.175 | 0.132 | **0.0212** | 2.6e-07 |
| k_spp rel. error | 0.966 | 5.6e-4 | 2.3e-3 | **8.3e-5** | 5.2e-11 |
| κ_d fit error | wrong sign | 3.4% | 7.7% | **0.28%** | 2.5e-9 |
| κ_m fit error | wrong order | 15.8% | 14.0% | **0.92%** | 2.1e-9 |
| Decay sign correct (air / metal) | no / yes | yes / yes | yes / yes | yes / yes | yes / yes |
| Continuity at ±2 nm (E / H, rel.) | 0.173 / 0.033 | 0.031 / 0.102 | 0.033 / 0.113 | **0.026 / 0.088** | 0.025 / 0.082 |
| impedance ratio (E_rms/H_rms)/η₀ | 0.867 | 0.914 | 0.876 | **0.868** | 0.915 |
| Boundary MSE at end | 0.04 (stuck) | 5.8e-5 | 2.2e-3 | 4.6e-3 (total) | — |
| Training time | 1214 s | 1431 s | 1925 s | 3791 s | — |
| **Success tier** | **not met** | **minimum** | **minimum** | **TARGET** | (stretch) |

Run 4 (2026-08-31 evening) = the run-3 configuration with the full refinement
budget the run-3 diagnosis called for: 150 float64 L-BFGS steps (vs 50).
The anchor-vs-metal-curl competition disappeared with budget — every metric
improved simultaneously, κ_m by 15×. rel L2 E even clears the stretch bar
(5e-3); the tier remains "target" because rel L2 H (7.1e-3) does not.
Continuity residuals sit at the physical floor (see self-check note below).

Notes on the self-check column: rel L2 is identically 0 (same function); curl
residuals ~1e-7 are the float32 pipeline floor; the continuity residuals of
0.025/0.082 are **physical**, not error — the exact mode's envelope differs
across the 4 nm evaluation gap (exp(−κ_m·2 nm) ≈ 0.92), so ≈8% on H is this
metric's floor for silver (the uniaxial floor is ≈2.3% — gentler decay).

## Target-tier push: what was measured (honest assessment)

**Target tier was not reached.** Failing metrics in run 3: rel L2 H 0.087
(> 0.05), rel L2 E 0.068 (> 0.05), κ_m 14.0% (> 10%). Run 2 remains the best
silver global fit; **the saved checkpoint/figures now hold run 3** (run 2's
were overwritten by the rerun, its numbers are preserved above).

Probes before the full run (8 float64 L-BFGS steps continuing from the *run 2
checkpoint*, new domain/sampling, ~2.5 min each) isolated the levers:

| Refinement variant | rel L2 E / H | κ_m err | κ_d err | metal curl-E res |
|---|---:|---:|---:|---:|
| metal curl weight 1/\|ε_m\| (exponent 1.0) | 0.047 / 0.053 | 19.1% | 4.2% | 0.68 |
| metal curl weight 1/√\|ε_m\| (exponent 0.5) | 0.051 / 0.065 | **5.0%** | 5.6% | **0.46** |

- **Lever d (metal curl re-weighting) is what moves κ_m** — dramatically
  (19% → 5% under identical step budgets) — but it visibly taxes the air-side
  fit and κ_d. It was adopted for the L-BFGS phase only (the Adam phase keeps
  the anti-collapse 1/|ε_m|).
- **Lever b (float64 + more L-BFGS)**: float64 steps cost ~18 s each (≈2× the
  float32 step at 4096 points); 50 steps ≈ 6× run 2's refinement compute.
- **Lever a (45% metal sampling, z_min −101 nm)** made the task itself harder:
  the extra 20 nm of deep metal adds volume where the relative field is
  ~e⁻⁴·⁴ and the air-side point density fell ~15%.

**Diagnosis of the run 3 regression.** The full rerun could not reuse run 2's
converged state (fresh seed-0 training, per protocol), and 50 re-weighted
L-BFGS steps from a fresh Adam solution are not equivalent to run 2's 40
steps *plus* the probe's 8: run 3's L-BFGS ended with the boundary anchor at
MSE 2.2e-3 — 40× less converged than run 2's 5.8e-5 — and the anchor is what
pins the global mode (hence k_spp 5.6e-4 → 2.3e-3 and the rel L2 rise). The
loss was still falling ~1%/step when the budget ran out. The probe shows the
recipe's ceiling is real: refinement continued from the best checkpoint
reaches κ_m ≈ 5% at rel L2 ≈ 0.05/0.065. The two objectives compete at this
network capacity/step budget; reaching all target metrics simultaneously
most likely needs a longer re-weighted refinement (150+ float64 steps, loss
not yet stalled) or a two-stage refinement (anchor-heavy first, then
metal-curl-heavy), not another lever.

## Uniaxial metamaterial case — **target tier**

One full run (`--case uniaxial`, defaults: seed 0, Adam 4000 × 2048 + 50
float64 L-BFGS steps, 1914.9 s ≈ 31.9 min on CPU), no case-specific tuning.
The case is far better conditioned than silver — |ε| contrast 4.0 vs 18.3
(metal curl stiffness ratio vs air 16× vs 335×), E_z jump ×3 vs ×18, κ̂_m 9.9
vs 27.6 — and the same recipe converged an order of magnitude deeper (final
loss 5.8e-3 vs 0.105; anchor MSE 3.2e-6 vs 2.2e-3).

| Metric | **PINN** | Analytical self-check |
|---|---:|---:|
| rel L2 E / H (overall) | **0.0090 / 0.0093** | 0 / 0 |
| rel L2 E air / metal | 0.0067 / 0.0190 | 0 / 0 |
| rel L2 H air / metal | 0.0058 / 0.0133 | 0 / 0 |
| curl-E residual rel, air / metal | 0.0134 / 0.0274 | 4.8e-08 / 9.4e-08 (float32 floor) |
| curl-H residual rel, air / metal | 0.0243 / 0.0243 | ~1e-07 |
| Re k_spp fit (phase slope, z = 50 nm) | 1.0659e7 /m (**0.016% err**) | 7.1e-11 err |
| κ_d fit | 3.884e6 /m (**0.16% err**) | 7.9e-9 err |
| κ_m fit | 1.548e7 /m (**0.64% err**) | 6.2e-10 err |
| Decay sign correct (air / metal) | yes / yes | yes / yes |
| Continuity at ±2 nm (E / H, rel.) | 0.013 / 0.029 | 0.010 / 0.023 (physical floor) |
| impedance ratio (E_rms/H_rms)/η₀ | 0.961 | 0.962 |
| Final / best loss | 5.79e-3 / 5.79e-3 | — |
| **Success tier** | **target** | (stretch) |

- Every target criterion is met with wide margin (rel L2 < 0.05, k_spp < 1%,
  both κ < 10%); the run misses **stretch** only on rel L2 (9.0e-3/9.3e-3 vs
  the 5e-3 bar — k_spp and both κ fits are already inside stretch).
- The anisotropy is genuinely exercised: the recovered κ_m matches the
  uniaxial branch κ_m² = ε_t(k_spp²/ε_n − k₀²) to 0.64% — an isotropic-metal
  model with either ε_t or ε_n alone would put κ_m at a very different value
  (the ε tensor, the adapter's ε_n divisor and the dispersion all have to be
  consistent for the fits to land).
- Continuity residuals sit on the physical floor (self-check 0.010/0.023 —
  smaller than silver's floor because exp(−κ_m·2 nm) ≈ 0.97 here).
- Figures: `uniaxial_field_maps.png`, `uniaxial_decay_profiles.png`,
  `uniaxial_phase_profile.png`, `uniaxial_training_history.png`; checkpoint
  `artifacts/models/spp_validation_uniaxial.pth`.

## Figures (`figures/spp_validation/`)

Silver (run 4): `field_maps.png`, `decay_profiles.png`, `phase_profile.png`,
`training_history.png`. Uniaxial: same set prefixed `uniaxial_`.
`metrics.json` is now per-case: `{"silver": {...}, "uniaxial": {...}}`, each
entry holding metrics + the analytical self-check + figure paths.

Model checkpoints: `artifacts/models/spp_validation.pth` (silver run 4),
`artifacts/models/spp_validation_uniaxial.pth`.

## What failed first, and why (run 1 → run 2)

Run 1 used the plane-wave recipe verbatim (boundary weight 10, single curl/div
loss over all points with a per-point (N,3,3) ε tensor, no ramp). It collapsed
to the trivial E = H = 0 minimiser: the boundary MSE sat at the anchor's mean
square (~0.04) for all 4000 epochs + 50 L-BFGS steps, mean |H| ended at 0.027
(true 0.57). Diagnosis (one iteration, as budgeted):

1. **Pure anchor regression converges easily** (MSE 0.048 → 1.6e-3 in 400
   epochs), so capacity was not the problem — loss balance was.
2. **The metal curl-H residual is stiff**: it contains i·2πε_m·Ê, so an Ê error
   δ in the metal costs (2π·18.3·δ)² ≈ 1.3e4·δ², ~335× the air-side penalty.
   Growing the field toward the mode transits high-loss states; zero fields are
   a strong local optimum against a weight-10 anchor.
3. **The anchor target itself is discontinuous**: E_z jumps by ε_d/ε_m ≈ 1/18
   at z = 0; a continuous MLP fits a smoothed jump whose spurious gradients
   fight the div/curl losses precisely where sampling is densest.

Fixes applied for run 2 (all in `examples/validate_spp.py`; src/ untouched):

- anchor weight 10 → 100, plus a physics ramp (0 → 1 over the first 25% of
  epochs) so the anchor establishes the mode first;
- per-medium interior losses with the metal terms weighted 1/|ε_m|
  (residual preconditioning for the ε contrast);
- `DisplacementAdapter`: the MLP outputs continuous D̂_z and the adapter divides
  by ε(z), making the E_z jump exact by construction.

Intermediate probes (500 epochs each): weight+ramp alone → rel L2 ≈ 0.94;
+rebalancing → 0.89; +adapter → 0.64 and falling. The full run then reached
0.048/0.051.

## Observations

- The η₀-balanced dimensionless frame worked as designed: impedance ratio
  0.876–0.914 across runs vs the mode's true 0.915; no E/H imbalance was ever
  visible in training.
- L-BFGS remains decisive, but *what it is pointed at* matters as much as how
  long it runs: run 2's un-reweighted refinement bought global accuracy; the
  re-weighted refinement buys κ_m at the anchor's expense (see the probe
  table). The refinement objective is the tier bottleneck, not Adam.
- The tangential-continuity metric at ±2 nm has a *physical* floor (~8% on H
  for silver, ~2.3% uniaxial) from the envelope change across the gap; the
  self-check column exposes this and the trained runs sit essentially on it.
- Runtimes on CPU: run 2 ≈ 23.9 min; run 3 ≈ 32.1 min (Adam ≈ 17.5 min +
  50 float64 L-BFGS steps ≈ 15 min); probes ≈ 5 min.

## Next steps

1. Silver target tier: continue the re-weighted float64 L-BFGS well past 50
   steps (it had not stalled), or split the refinement into an anchor-heavy
   stage followed by the metal-curl-heavy stage, starting from the best
   checkpoint rather than a fresh Adam solution.
2. Consider folding the `DisplacementAdapter` idea into `src/models` (a
   general per-medium constitutive adapter) — with the uniaxial case it has
   now been exercised with ε_zz ≠ ε_xx, i.e. a genuinely tensorial medium.
3. `EM_CompositeLoss` adaptive re-weighting still conflicts with the physics
   ramp; driving sub-losses directly remains the recommended pattern.

## Repo defects encountered (not fixed; src/ out of scope)

- `SPPNetwork.forward` (src/models/pinn_network.py) still applies its decay
  envelope to raw coordinates, so it cannot be nondimensionalised as-is (known
  from the triage doc; this experiment used `ElectromagneticPINN` + wrappers
  instead).
- `EM_CompositeLoss` adaptive re-weighting would have fought the physics ramp
  used here; the experiment drives the sub-losses directly (as the plane-wave
  example does), which remains the recommended pattern.
