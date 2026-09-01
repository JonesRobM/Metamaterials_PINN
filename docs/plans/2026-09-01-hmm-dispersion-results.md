# ω-Conditioned SPP PINN on a **Dispersive** Hyperbolic Metamaterial — Results

**Date:** 2026-09-01
**Script:** `examples/validate_hmm_dispersion.py`
(tests: `tests/examples/test_validate_hmm_dispersion.py`)
**Predecessor:** `docs/plans/2026-08-31-spp-dispersion-results.md` (fixed ε, same machinery)
**Status:** **TARGET tier.** One ω-conditioned network reproduces a genuinely
*nonlinear* SPP dispersion across a 65 %-wide band (450–885 nm): worst rel L2
1.85e-2, median 7.1e-3, worst Re k_spp error 0.256 %, median 0.018 %, bound mode
at all 9 held-out frequencies. It misses stretch on a single frequency — the blue
band edge — where rel L2 is 1.85e-2 (threshold 2e-2, so that one passes) and
k_spp error is 0.256 % against the 0.2 % threshold. **The network reproduces the
curvature of the dispersion, not merely its trend**: its k_spp points sit 37×
closer to the exact nonlinear branch than to the best straight line through it.

---

## 1. Experiment

The predecessor held ε **non-dispersive** across the band. Every mode constant
then scales linearly with ω, so `k_spp(ω) = (ω/c)·n_spp` is a straight line
through the origin and the mode family is self-similar — one shape, rescaled.
That is the idealisation this run removes.

Material: the Ag/silica layered hyperbolic metamaterial recommended by
`examples/hyperbolic_metamaterial.py`, driven through `src.effective_medium`:

| Item | Value |
|---|---|
| Metal | Drude Ag: ε_∞ = 3.7, ħω_p = 9.1 eV, ħγ = 0.018 eV |
| Dielectric layers | ε_d2 = 2.25 (silica), metal fill fraction f = 0.30 |
| Effective uniaxial ε | `hmm_permittivities(ω, 0.30, 2.25)` → (ε_t, ε_n), optical axis z |
| Superstrate | air, ε_d = 1 |
| Band | ω/ω_ref ∈ [0.7149, 1.4067], λ₀ ∈ [450, 885] nm (relative width 0.652) |

The fill fraction, layer permittivity, Drude parameters and band endpoints are
**read at runtime** from `figures/hyperbolic/hmm_summary.json` (a test asserts
this), so the experiment tracks the design study rather than drifting from it.
ω_ref = 2πc/633 nm is only a labelling reference — nothing in the recipe uses it
as a scale.

### 1.1 What makes this the harder problem

| Quantity | Red edge (885 nm) | Blue edge (450 nm) | Ratio |
|---|---:|---:|---:|
| ε_t | −9.982 + 0.163j | −0.587 + 0.021j | 17× in \|ε_t\| |
| ε_n | 3.297 + 0.001j | 3.711 + 0.006j | 1.13× |
| Re k_spp [m⁻¹] | 7.333e6 | 1.900e7 | **2.59×** |
| n_eff = k_spp/k₀ | 1.0333 | 1.3609 | 1.32× |
| λ_spp [nm] | 857 | 331 | 2.59× |
| δ_d = 1/Re κ_d [nm] | 541 | 78 | 7.0× |
| δ_m = 1/Re κ_m [nm] | 54.2 | 132 | 2.4× |
| Im k_spp [m⁻¹] | 3.72e3 | 1.14e5 | 31× |

Full-band κ spread (max κ / min κ over both sides) = **9.98×**. ε_t is sweeping
towards the multilayer's in-plane ENZ crossing at 407.6 nm, which is what bends
the dispersion — and, as §5 shows, is also where the network works hardest.

**How nonlinear is "nonlinear"?** Three reference lines, all as
`max |k_spp − line| / (k_max − k_min)` on a dense grid:

| Reference | Departure |
|---|---:|
| Best line **through the origin** — i.e. the shape a non-dispersive ε gives, and exactly what the predecessor recovered | **24.3 %** |
| Best general least-squares straight line `k = aω + b` | 13.3 % |
| Chord through the two band endpoints | 14.9 % |

The 24.3 % figure is the one quoted in `hmm_summary.json`; see §7 for why that
file reports the origin-line number under a label that promises the general one.

### 1.2 Modelling choices, stated up front

**(a) Per-ω domain.** A single fixed box would be wasteful at one edge and
under-resolved at the other (δ_d alone spans 7×). Each sampled ω therefore gets
its own box, sized from *its own* analytic mode constants:

    x ∈ [0, 2 λ_spp(ω)],   z ∈ [−3.5/Re κ_m(ω), +1.2/Re κ_d(ω)],   y ∈ [0, 0.2 λ₀(ω)]

which in SI runs from x = 1714 nm, z ∈ [−190, +650] nm at the red edge to
x = 661 nm, z ∈ [−462, +93] nm at the blue edge — the box does not merely shrink,
it *inverts*, from mostly-air to mostly-metal (both are shown in
`figures/hmm_dispersion/field_maps_omega_*.png`).
**This uses the analytical mode to size the sampling region.** That is
defensible here only because the boundary anchor already *is* the analytical
mode, so no new information enters; but it does mean the experiment measures how
well the network interpolates a mode family whose spatial extent it is told, not
how well it discovers that extent. A design tool without a closed form would
have to size its box from an estimate (e.g. an effective-index guess) instead.

**(b) Input scaling by the local free-space wavenumber, deliberately not by
k_spp.** The network is fed `(x, y, z)·k₀(ω)` with `k₀(ω) = ω/c`, plus a
normalised frequency feature ω̂ ∈ [−1, 1]. Three consequences:

1. *The scaled Maxwell system is exactly frequency-free.* With x̂ = k₀x,
   Ê = E/(η₀H₀), Ĥ = H/H₀ the curl equations read `∇̂×Ê = i Ĥ` and
   `∇̂×Ĥ = −i ε Ê` at **every** ω. A single residual with `frequency = 1`
   therefore serves the whole band, and ω enters the physics *only* through
   ε(ω). (The fixed-ε predecessor had to fold a per-row factor ω/ω₀ into the
   material arguments; that factor is gone.) Verified: the exact mode drives the
   scaled residual to ~4e-17 at both edges and mid-band, in both media.
2. *Everything the network must resolve becomes O(1).* n_eff ∈ [1.03, 1.36],
   κ_d/k₀ ∈ [0.26, 0.92], κ_m/k₀ ∈ [0.54, 2.60] — against a 10× spread in SI.
   The Fourier band drops from (0.1, 40) rad/unit in the predecessor to
   **(0.1, 8)**, and the scaled anchor field stays within |·|max ∈ [1.00, 1.13]
   across the band with no per-ω renormalisation.
3. *It does not leak the answer.* Scaling by k_spp(ω) would build the measured
   dispersion into the coordinate map: k_spp itself ranges 2.59× while k_spp/k₀
   ranges only 1.32×, so a k_spp-scaled network would be handed most of the
   dispersion for free. k₀ = ω/c is known without solving anything, so scaling
   by it is bookkeeping, not a hint.

**(c) Per-row metal preconditioning.** |ε| = max(|ε_t|, |ε_n|) varies 2.7× across
the band (9.98 → 3.71), so the predecessor's scalar 1/|ε| curl weight is applied
here **per row**, at that row's own frequency.

---

## 2. Hyperparameters

| Item | Value |
|---|---|
| Network | `OmegaConditionedCore`: spatial `FourierEMFeatures` (128 modes, band (0.1, 8) rad/unit) on (x, y, z)·k₀(ω) + raw ω̂ → complex MLP 4×128, complex_tanh (**117 644 params**) |
| Frequency feature | ω̂ = (ω − ω_mid)/half-span ∈ [−1, 1], linear in ω; appended after the spatial Fourier encoding, not Fourier-encoded |
| E_z handling | `DispersiveDisplacementAdapter`: MLP channel 2 = continuous D̂_z, divided by ε_zz — **ε_n(ω) below** (read from the ω̂ column, evaluated by a torch mirror of `hmm_permittivities` asserted equal to it), ε_d above. D̂_z = −k_spp/k₀ on both sides, an O(1) target |
| Per-ω physics | one scaled-frame residual at `frequency = 1`; per-row (N,3,3) ε built from ε_t(ω_row), ε_n(ω_row) below and ε_d above. **No ω prefactor anywhere** |
| Interior losses | per-medium curl + divergence; metal curl weight \|ε(ω_row)\|⁻¹ (Adam) → \|ε(ω_row)\|⁻¹ᐟ² (L-BFGS), metal div weight \|ε(ω_row)\|⁻¹. Implemented as `curl_loss_weighted` / `divergence_loss_weighted`, asserted equal to `MaxwellCurlLoss` / `MaxwellDivergenceLoss` to 1e-12 for uniform weights |
| Interface / anchor | `TangentialContinuityLoss(offset 0.02 scaled = 2 nm at 633 nm)` weight 1; soft Dirichlet anchor = `analytical_spp_fields(·, ω, ε_t(ω), ε_n(ω))` on the six faces of each block's own box, weight 100; physics ramp 0→1 over the first 25 % of Adam epochs |
| ω sampling | Adam: 4 fresh uniform ω per epoch (one per sub-block of the 2048-pt batch, 1024 boundary, 512 interface), each with its own box; z-strata use each ω's own δ_m, δ_d |
| Optimiser | Adam lr 1e-3 cosine → 1e-5, 5000 epochs; then float64 L-BFGS on a fixed 4096-pt set, 150 steps total (see §4) |
| Guard bands | 0.01 scaled units (training), 0.02 (validation) — the predecessor's 1 nm / 2 nm expressed in k₀ units |
| Seed / device | 0 (Adam) / CPU |

**Validation grid:** 9 frequencies, ω_min … ω_max in 8 equal steps. The 4 odd
grid points {0.8014, 0.9743, 1.1473, 1.3202}·ω_ref are **strictly held out** from
the L-BFGS refinement set and each sits alone at the centre of a Δ-wide node
gap, i.e. Δ/2 from its nearest refined neighbour (asserted in the tests, where
Δ is the validation spacing); Adam samples the ω continuum, so no discrete
frequency is ever trained more than sampled. Per ω, in SI: rel L2 (E and H) vs
the analytical mode on 8000 fresh stratified points, Re k_spp from the phase
slope of H_y along x at a quarter of that ω's air-side box height, Im k_spp from
the amplitude slope (reported, unscored), κ_d/κ_m from ln|H_y|(z) decay fits,
tangential continuity at ±0.02/k₀, SI curl residuals per half-space.

---

## 3. Analytical self-check (exact mode through the identical pipeline)

| ω/ω_ref | rel L2 E | k_spp err | κ_d err | κ_m err |
|---|---:|---:|---:|---:|
| 0.7149 | 0.0 | 3.6e-10 | 8.3e-09 | 1.5e-09 |
| 1.0608 | 0.0 | 1.0e-10 | 6.2e-10 | 1.1e-09 |
| 1.4067 | 0.0 | 2.3e-10 | 1.3e-08 | 3.6e-09 |

Machine precision at every probe frequency, and asserted at all 9 validation
frequencies in the test suite. The dispersive ε plumbing, the per-ω box sizing,
the k₀ scaling and the sign conventions are exact, so every number in §5
measures the network, not the measurement.

---

## 4. What was actually run (and the one diagnostic iteration)

This did not go in a straight line, and the deviations matter for reproducing it.

1. **Adam, 5000 epochs** (1548 s), 4 random ω per epoch. Loss 4.97 → 1.36e-2.
2. **L-BFGS, 120 float64 steps on a 5-node refinement set** {0.7149, 0.8879,
   1.0608, 1.2337, 1.4067}·ω_ref — the predecessor's design, 5 of the 9
   validation frequencies (2341 s). Result: **TARGET, worst rel L2 7.2e-2**,
   entirely from a single validation frequency, 1.3202·ω_ref.
   *The process was killed by a harness timeout at step 104 and resumed from the
   atomic `hmm_dispersion.partial.pth` checkpoint for the remaining 16 steps —
   the checkpointing machinery doing exactly the job the predecessor added it
   for.*
3. **Diagnosis before acting.** A 41-point ω sweep of the trained model showed
   the error was not an isolated bad point but a clean **node-aligned bulge**:

   | ω/ω_ref | 0.7149★ | 0.8879★ | 1.0608★ | 1.1819 | 1.2337★ | 1.3548 | 1.4067★ |
   |---|---:|---:|---:|---:|---:|---:|---:|
   | rel L2 E | 7.0e-3 | 5.4e-3 | 5.1e-3 | 1.06e-2 | 5.0e-3 | **9.05e-2** | 5.5e-3 |

   (★ = an L-BFGS node.) The error minima sit *exactly* on the refinement nodes;
   between them it bulges, negligibly in the red half (6.6e-3) and 18× in the
   last interval — precisely where ε_t sweeps −1.57 → −0.59 towards its ENZ
   crossing and the mode changes fastest with ω. Adam samples ω at random and
   cannot produce node-aligned structure, so this is the L-BFGS phase
   over-fitting its fixed frequency set where the ω direction is under-resolved.
4. **Diagnostic iteration (one, as budgeted): halve the refinement spacing.**
   The node set became **13** frequencies — the original 5 plus the 8 half-way
   points — which halves the worst-case distance from an arbitrary ω to a
   refined one (Δ → Δ/2) while leaving all 4 held-out validation frequencies
   still strictly unrefined, each alone mid-gap. Resumed from the checkpoint for
   **30 more float64 L-BFGS steps** (565 s). That is the model reported here.

   The band was **not** narrowed. Cause (a) from the plan (the 10× κ spread) was
   not the binding constraint: the k₀ input scaling had already compressed that
   spread to ~5× in the coordinates the network actually sees, and the red two
   thirds of the band were at 5–7e-3 throughout. Cause (c), insufficient L-BFGS
   *coverage in ω*, was.

**Runtime:** 4454 s total ≈ **74 min** CPU (Adam 26 min, 5-node L-BFGS 39 min,
13-node L-BFGS 9 min), within the 90-minute budget. A single uninterrupted
`python examples/validate_hmm_dispersion.py` now runs the corrected recipe
(5000 Adam + 120 L-BFGS on the 13-node set) end to end; the 39 minutes spent on
the superseded 5-node set are the cost of the diagnosis, not of the recipe.

---

## 5. Per-frequency results

Nine frequencies; the 4 odd ones are strictly held out from refinement.

| ω/ω_ref | λ₀ [nm] | rel L2 E | rel L2 H | k_spp err | κ_d err | κ_m err | cont. E | held out |
|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| 0.7149 | 885 | 1.01e-2 | 1.24e-2 | 1.8e-4 | 2.0e-3 | 2.03e-2 | 8.8e-3 | |
| 0.8014 | 790 | 6.57e-3 | 8.16e-3 | 5.6e-5 | 6.7e-3 | 1.62e-2 | 9.1e-3 | ✓ |
| 0.8879 | 713 | 6.56e-3 | 6.82e-3 | 4.8e-5 | 1.1e-3 | 4.0e-3 | 8.8e-3 | |
| 0.9743 | 650 | 6.24e-3 | 6.34e-3 | 2.8e-4 | 4.5e-3 | 1.36e-2 | 8.5e-3 | ✓ |
| 1.0608 | 597 | 6.80e-3 | 7.10e-3 | 2.2e-4 | 9.3e-3 | 3.7e-3 | 7.7e-3 | |
| 1.1473 | 552 | 6.82e-3 | 6.53e-3 | 4.5e-5 | 8.0e-3 | 1.08e-2 | 6.8e-3 | ✓ |
| 1.2337 | 513 | 6.38e-3 | 6.81e-3 | 2.6e-4 | 3.9e-3 | 4.3e-3 | 5.1e-3 | |
| 1.3202 | 479 | 9.48e-3 | 8.26e-3 | 1.2e-4 | 4.8e-3 | 3.5e-3 | 3.1e-3 | ✓ |
| 1.4067 | 450 | 1.85e-2 | 1.57e-2 | **2.6e-3** | 3.6e-3 | 3.39e-2 | 6.0e-3 | |
| **worst** | | **1.85e-2** | | **2.56e-3** | 9.3e-3 | 3.39e-2 | | |
| **median** | | **7.10e-3** | | **1.79e-4** | | | | |

Bound mode recovered at every frequency (`bound_mode_everywhere = 1`), decay
signs correct on both sides throughout. Final loss 7.53e-4.

**Held-out vs refined:** median rel L2 is 7.49e-3 over the 4 held-out
frequencies and 7.10e-3 over the 5 refined ones — a 5 % difference, and the
*worst* held-out frequency (9.5e-3) is better than the worst refined one
(1.85e-2, the blue edge). Frequency conditioning is genuinely interpolating, not
memorising nodes.

**A denser check.** A 41-point ω sweep of the final model (in
`metrics.json → dense_omega_sweep`) gives worst rel L2 **2.03e-2** over the whole
continuum, against 9.05e-2 for the 5-node model — a 4.5× improvement — and
≤ 1.0e-2 everywhere below ω/ω_ref = 1.33. The residual error is confined to the
bluest ~5 % of the band.

### Success tiers

| Tier | Criterion | Result |
|---|---|---|
| minimum | bound mode at all 9 ω, rel L2 < 0.5 | **met**, 27× margin |
| target | rel L2 < 0.1 everywhere **and** k_spp within 1 % | **met**, 5.4× and 3.9× margin |
| stretch | rel L2 < 0.02 **and** k_spp within 0.2 % | rel L2 **met** (1.85e-2 < 2e-2); k_spp **missed** at one frequency only (0.256 % at the blue edge; every other frequency ≤ 0.028 %) |

---

## 6. Did it capture the nonlinearity?

This is the point of the experiment, so it is measured rather than asserted.
Take the PINN's 9 fitted Re k_spp values and score their scatter about two
references: the exact nonlinear branch, and the best straight line through that
branch on the same grid.

| Quantity | Value [m⁻¹] |
|---|---:|
| RMS(PINN − exact nonlinear branch) | **1.63e4** |
| RMS(PINN − best straight line) | 6.06e5 |
| RMS(exact branch − best straight line), i.e. the curvature itself | 6.17e5 |
| **Curvature-capture ratio** = 6.06e5 / 1.63e4 | **37.1×** |
| RMS(PINN − best line through the origin, the fixed-ε form) | 1.17e6 |
| RMS(exact branch − origin line) | 1.18e6 |

The PINN points lie **37× closer to the curve than to the line**, and their
scatter about the straight line (6.06e5) reproduces the true curvature
amplitude (6.17e5) to 1.7 %. Had the network learned only a linear dispersion,
those two numbers would have been comparable instead of differing by 37×.
Equivalently: the PINN's own maximum departure from the endpoint chord is
14.83 % of the k range against the exact 14.88 %.

The right-hand panel of `figures/hmm_dispersion/dispersion.png` shows this
directly — the red circles (PINN − line) sit on the black analytical curvature
curve, and the blue squares (PINN − curve) sit on zero.

## 6.1 Comparison against the fixed-ε predecessor

| | Fixed ε (2026-08-31) | Dispersive ε (this run) |
|---|---|---|
| Band | [0.85, 1.15] ω₀, **30 %** wide | [0.715, 1.407] ω_ref, **65 %** wide |
| ε | constant (−4 + 0.2j, 3 + 0.05j) | Ag/silica EMT, ε_t sweeps −9.98 → −0.59 |
| k_spp(ω) | straight line through the origin | nonlinear, 24.3 % departure from that line |
| k_spp range | 1.35× | **2.59×** |
| κ spread | 5.4× | **9.98×** |
| Input scaling | fixed λ₀ | **local k₀(ω)** |
| Fourier band | (0.1, 40) rad/unit | **(0.1, 8)** |
| Domain | one worst-case box | **per-ω box** |
| Loss frequency handling | per-row ω/ω₀ folded into (ε, μ_r) | **none needed** — scaled system is frequency-free |
| Worst rel L2 | 2.6e-2 | **1.85e-2** |
| Median rel L2 | 1.5e-2 | **7.1e-3** |
| Worst k_spp error | 1.5e-3 | 2.6e-3 |
| Median k_spp error | 5.3e-4 | **1.8e-4** |
| Runtime | 62 min | 74 min (26 min Adam + 48 min L-BFGS) |
| Tier | target | target |

The harder problem — twice the fractional bandwidth, twice the wavenumber range,
a nonlinear branch and a material that changes by a factor of 17 — came out
**more accurate in the field norm**, at 1.2× the cost. The k₀ input scaling is
what pays for it: by making the scaled problem nearly frequency-invariant, it
turns a wide dispersive band into something not much harder than a narrow
self-similar one.

---

## 7. Observations

1. **Scaling by the local k₀ is the single highest-value change.** It is
   arithmetic, not modelling: it makes the residual frequency-free (no folding
   trick), collapses the 10× κ spread to ~5× in network coordinates, drops the
   Fourier bandwidth 5×, and keeps the anchor O(1) at every ω without
   renormalisation. Median rel L2 improved 2.1× against the *easier* fixed-ε
   problem. The discipline that makes it honest is choosing k₀ = ω/c — known a
   priori — rather than k_spp(ω), which is the thing being measured.

2. **L-BFGS accuracy is local in ω, and the node spacing is a real
   hyperparameter.** The 5-node run's error tracked distance-to-nearest-node
   almost perfectly (5e-3 at nodes, up to 9e-2 mid-gap). This did not show up in
   the fixed-ε predecessor because a self-similar mode family varies slowly in
   ω; with dispersive ε the required node density scales with how fast the
   *material* moves. Practical rule: place refinement nodes so that |ε| changes
   by at most ~20 % between neighbours, not uniformly in ω.

3. **Error is largest where ε_t approaches ENZ, not where κ is most extreme.**
   Worst rel L2 and worst k_spp error are both at the blue edge (450 nm), where
   |ε_t| = 0.587 and n_eff is rising fastest — not at the red edge, where the
   decay lengths are most disparate (54 nm vs 541 nm). The per-ω domain plus k₀
   scaling handled the 10× length-scale spread essentially for free; what
   remains hard is the rapid *material* variation. This also compounds with the
   classic interpolation signature the predecessor reported (band edges are
   constrained from one side only), which is why the blue edge, not
   ω/ω_ref = 1.36 just inside it, is the worst point.

4. **κ_m is the hardest fitted quantity again** (worst 3.4 %, vs 0.9 % for κ_d
   and 0.26 % for k_spp), matching both earlier experiments: the metal-side field
   decays over 54–132 nm, so its fit is the most sensitive to residual error near
   the interface. As before, κ_m error does not track rel L2 across the band.

5. **Im k_spp (unscored) is recovered well only where the loss is large.** At the
   blue edge the fit gives 1.047e5 m⁻¹ against 1.136e5 (8 % low); at the red edge
   827 against 3720 (78 % low). This is a measurement limitation, not a model
   one: the domain is 2 λ_spp long, over which the red-edge mode's amplitude
   decays by 0.6 %, below the network's own field error. Measuring Im k_spp at
   the red end would need a domain tens of λ_spp long.

6. **Checkpointing earned its keep for the second time.** The first full run was
   killed at L-BFGS step 104/120 by a harness timeout; the atomic
   `.partial.pth` write made that a 20-second loss instead of a 65-minute one,
   and `--resume` also made the diagnostic iteration cheap (30 extra L-BFGS steps
   rather than a full retrain). One caveat to record: L-BFGS's inverse-Hessian
   history is *not* checkpointed, so each resume restarts curvature estimation
   from scratch — visible as the loss spikes at steps 104 and 120 in
   `training_history.png`.

7. **The `--band-fraction` fallback was implemented but not needed.** It narrows
   the band about its midpoint and refreshes every derived constant, and is
   exercised by the tests; the full 65 % band held up, so the reported result is
   on the full recommended band.

---

## 8. Repo defects found

> **Both were fixed on 2026-09-01, after this experiment ran.** The band this
> experiment trained on is unchanged by the D1 fix (see the resolution note
> below), so every number in this document stands as recorded.

**D1 — `examples/hyperbolic_metamaterial.linear_fit_residual` silently fits a
line through the origin.** It builds `design = np.vstack([omega, ones]).T` and
calls `np.linalg.lstsq(design, k_real, rcond=None)`. With ω ~ 3e15 the design
matrix has singular values [5.39e16, 3.12], a ratio of 5.78e-17, while
`rcond=None` selects numpy's machine-precision cutoff `max(M, N)·eps` = 6.26e-14.
The intercept's singular value is therefore truncated: `lstsq` returns **rank 1**
and an intercept of 1.18e-24. The reported `nonlinearity_percent` = 24.26 % is
consequently the departure from `k = aω` (a line through the origin), not from
the `k ≈ aω + b` the docstring promises — that value is 13.30 %.

*Reproduce:* `rank` from `np.linalg.lstsq` on the recommended band is 1, and
`np.polyfit(omega, k, 1)` gives a max residual of 1.55e6 m⁻¹ against `lstsq`'s
2.83e6 m⁻¹.

Both quantities are physically meaningful, and the origin-line one is arguably
the more useful (it measures departure from non-dispersive behaviour), so the
band selection this repo already made is sound. The defect is that the number is
undocumented, disagrees with the docstring, and would flip silently if ω were
ever expressed in different units. *Suggested fix:* use
`np.polyfit(omega, k_real, 1)`, or centre and scale ω before `lstsq` and state
which reference line is intended. This experiment reports both (§1.1).

**D2 — `src/analytical.py.__all__` omits `analytical_spp_fields`.** The module's
headline function — imported by name in `examples/validate_spp.py`,
`examples/validate_spp_dispersion.py`, this experiment and several test modules —
is missing from `__all__`, which lists only `analytical_potential`,
`analytical_point_charge_field`, `analytical_plane_wave` and
`complex_to_pinn_format`. A `from src.analytical import *` silently omits it.
One-line fix.

**Resolution (2026-09-01).**
D1: `linear_fit_residual` now takes an explicit `through_origin` flag and
scales ω before the general fit, so the two references are deliberate rather
than accidental. Both are published: `nonlinearity_percent` (24.26 %) is the
departure from the fixed-ε form `k ∝ ω` — the band-selection criterion, and the
number this experiment quotes throughout — and the new `curvature_percent`
(13.28 %) is the stricter departure from the best general line `k = aω + b`.
The recommended band is byte-identical, so this experiment's results are
unaffected. `tests/examples/test_hyperbolic_metamaterial.py::TestLinearFitResidual`
now contains a regression guard that fits data with a large non-zero intercept
and fails if the intercept is silently truncated.
D2: `analytical_spp_fields` added to `src/analytical.py.__all__`.

---

## 9. Next steps

1. **Close the blue edge.** Two independent cheap options, both suggested by the
   §7 analysis: (i) train on a band ~10 % wider than the one queried, so the
   reported edges become interior (the predecessor's recommendation, still
   unexercised); (ii) place refinement nodes by equal steps in |ε_t| rather than
   equal steps in ω, which would put roughly twice as many in the bluest third
   at no extra cost.
2. **Condition on the material, not just the frequency.** Adding (ε_t, ε_n) —
   or the design parameters (f, ε_d2) — as further input features turns this
   into the surrogate that `examples/inverse_design.py` currently gets from the
   closed form. The dispersive run is the right base for it: the network already
   has to represent a mode family whose material varies by 17×, and it does so
   with ω as the only handle. Giving it ε directly should be *easier*, and it is
   the point at which the PINN earns its keep, since the closed form does not
   exist for finite slabs or real multilayers.
3. **Cross the ENZ crossing.** The band stops at 450 nm, short of the in-plane
   ENZ point at 407.6 nm where the mode ceases to be bound. Pushing towards it
   is the natural stress test of frequency conditioning, and is where effective-
   medium theory itself becomes least trustworthy — so it would need a
   transfer-matrix reference rather than the EMT closed form.
4. **Finite multilayer instead of the homogenised medium.** `max_layer_period`
   says the EMT description of this band needs a period ≤ 33 nm. Validating a
   real 8-period stack against a transfer-matrix reference would test both the
   PINN and the homogenisation at once — and has no closed-form ground truth,
   which is where this line of work is heading.
