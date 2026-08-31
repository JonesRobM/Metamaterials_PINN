# SPP Training Path — Triage

**Date:** 2026-08-29
**Status:** Triage complete; quick fixes applied; blocking design work identified
**Context:** `scripts/train_spp_pinn.py` has never produced a converged checkpoint. This
triage lists what was wrong, what was fixed immediately, and what must be designed
properly before the SPP experiment is attempted (the analytics underneath are now
benchmark-validated — see `tests/test_benchmark_spp.py`, `tests/test_benchmark_anisotropic.py`,
`tests/test_benchmark_fresnel.py`).

## Fixed in this pass (cheap, unambiguous)

1. **Factor-2π frequency error.** `config/spp_config.yaml` had `frequency: 4.74e14` — the
   *linear* frequency for λ₀ = 633 nm — but `SPPNetwork` and every loss take *angular*
   ω (rad/s). k₀ and k_spp were wrong by 2π. Now `2.9758e15` with a comment.
2. **Hardcoded decay length.** `SPPBoundaryLoss(decay_length=300e-9)` replaced by the
   dielectric-side penetration depth δ_d ≈ 428 nm derived from
   `MetamaterialProperties` at the configured ε and ω.
3. **Continuity offset larger than the physics.** `TangentialContinuityLoss()` defaulted
   to offset = 1 µm — larger than δ_d and bigger than λ_spp ≈ 616 nm, so the two
   evaluation points straddled entirely different field regions. Now
   `min(δ_d, λ_spp)/50` ≈ 8.6 nm.
4. **Frozen collocation set.** Points were sampled once for 10 000 epochs; now resampled
   every epoch.

## Open design problems (must be solved before the experiment; do NOT just run it)

1. **No amplitude/phase anchor — trivial-minimiser risk (the plane-wave collapse bug,
   again).** `MaxwellCurlLoss` and `TangentialContinuityLoss` are exactly zero for
   E = H = 0, and `SPPBoundaryLoss` normalises by max|E|, so it is amplitude-invariant.
   Nothing pins the solution. The plane-wave fix was a soft Dirichlet boundary term from
   the analytical solution; the SPP analogue is to anchor against the analytical SPP
   mode profile (now trustworthy: dispersion + decay constants are benchmarked to
   machine precision) on a boundary plane or a sparse point set — or to train the
   *correction* to the analytical mode, as the point-charge script does.
2. **Network cannot resolve k₀ from raw SI coordinates.** The Fourier band is
   (0.1, 20) rad per *input unit*; raw coordinates are ~1e-6, so features are constant.
   Use the dimensionless-frame approach that made the plane-wave example converge
   (`examples/validate_plane_wave.py`): coords in λ, fields scaled, η₀-balanced H —
   but note `SPPNetwork.forward` applies its decay envelope to *physical* z, so
   wrapping it in `NondimensionalPINN` as-is would misapply the envelope.
   `SPPNetwork` needs a `coord_scale`-aware envelope (or the envelope folded into the
   wrapper) before nondimensionalising.
3. **η₀ imbalance in the SI-units curl loss.** In SI, the H-equation residual is
   ~1.4×10⁵ smaller than the E-equation residual and Adam ignores it (documented in
   `examples/validate_plane_wave.py`, which trains with `mu0=1, eps0=1` in the
   dimensionless frame instead). The SPP loss must do the same or reweight.
4. **`SPPBoundaryLoss` enforces a single symmetric exp(−|z|/L) decay**, but the real
   mode is asymmetric: δ_d ≈ 428 nm (air) vs δ_m ≈ 22 nm (silver). Either make the loss
   two-sided (δ_d above, δ_m below) or drop it in favour of the analytical-anchor term
   from (1), which encodes the correct profile automatically.
5. **Domain/z-range vs metal-side physics.** z ∈ (−1 µm, 0) in the metal is ~45 decay
   lengths of essentially-zero field; uniform sampling wastes most metal-side points.
   Use `SPPDomainSampler` (built for exactly this) instead of uniform `torch.rand`.

## Recommended experiment structure

Mirror `examples/validate_plane_wave.py`: dimensionless frame, Adam + cosine schedule
then L-BFGS, divergence loss included, soft anchor from the analytical SPP mode,
validation metrics = relative L2 vs the analytical mode, recovered k_spp (FFT along x),
decay-constant fits on both sides vs κ_d, κ_m, and the interface-continuity residuals.
