# A PINN on the **Real Ag/Silica Multilayer** — Does Keeping the Layers Beat Homogenising Them? — Results

**Date:** 2026-09-01
**Script:** `examples/validate_multilayer.py`
(tests: `tests/examples/test_validate_multilayer.py`, plus the additive
`TestPermittivityProfile` / `TestModeFieldProfile` in `tests/test_transfer_matrix.py`)
**Predecessors:** `docs/plans/2026-08-29-spp-validation-results.md` (single
interface, the DisplacementAdapter recipe), `docs/plans/2026-09-01-hmm-dispersion-results.md`
(the dimensionless frame and the k₀ input scaling), and the study that motivates
this one — `examples/emt_validity.py`, which measured how wrong homogenisation is.

**Status:** **TARGET tier.** Trained on the real 13-interface stack, the PINN
reaches rel L2 = **5.75e-3** against the transfer-matrix field (43× closer than
the homogenised model) and recovers **Re k_spp to 0.005 %** where EMT is 2.33 %
off — a **472×** improvement — and **Im k_spp to 1.9 %** where EMT is 33.6 % off,
a **17.5×** improvement. The E_z sawtooth is exact by construction and comes out
as the *most* accurate field component. It misses stretch on one number: Im k_spp
is 1.9 % against a 1 % threshold, and its 1σ across probe lines is 15 % — see
§9.2, which is the honest caveat on the headline.

---

## 1. The question

Every metamaterial result in this project so far has replaced the Ag/silica
multilayer by a single homogeneous uniaxial medium (ε_t, ε_n). `emt_validity.py`
showed that substitution carries an **O(a) surface-term error** — not the
O((a/λ)²) bulk correction the `a ≪ λ` rule guards against, but a term left by
truncating the periodic stack at the air interface. At a 30 nm period it is 2.3 %
in Re k_spp and a third in Im k_spp.

So: train the PINN on the **actual layered ε(z)** and ask whether it lands closer
to the transfer-matrix truth than the homogenised model does. Ground truth is
`src.transfer_matrix` (validated: Fresnel to 5e-17, single-interface SPP to
3.6e-19, both branches of a thin metal film).

---

## 2. The methodological point: the displacement adapter generalises for free

The recipe that won the single-interface and dispersive-HMM runs contains one
structural idea, the **DisplacementAdapter**: the network emits a *continuous*
normal displacement D̂_z and the adapter divides it by the local ε_zz, making the
E_z jump exact by construction rather than smoothing it.

That idea does not care how many interfaces there are. For a TM mode
`F(x, z) = F(z) e^{i k_x x}`, Ampère's law `∇ × H = −i ω ε₀ ε E` gives, with
`∂_x → i k_x` and `∂_y → 0`,

```
(∇×H)_z = i k_x H_y  =  −i ω ε₀ ε E_z
   ⟹   E_z = −k_x H_y / (ω ε₀ ε(z))
   ⟹   D_z = ε₀ ε E_z = −k_x H_y / ω        ← no ε left in it
```

The transfer matrix's interface matrix `D(j→j+1)` *is* the statement that H_y and
E_x are continuous at every boundary; D_z inherits H_y's continuity for free. So
across all thirteen interfaces of this stack the only discontinuous component is
E_z, and it is exactly `D_z/ε(z)`. **The adapter is unchanged; only ε(z) is
swapped from a two-valued step for the real stack's piecewise-constant profile.**

This was verified, not assumed. `tests/test_transfer_matrix.py::TestModeFieldProfile`
checks on this very stack that

* H_y, E_x and D_z are continuous at all 13 interfaces (< 1e-7 relative),
* E_z jumps by exactly ε_below/ε_above at each of them (< 1e-6 relative), and
  that every jump is real (min |ratio − 1| > 0.5),
* the profile obeys `H_y'' = (k_x² − ε k₀²) H_y` inside each layer,

and `tests/examples/test_validate_multilayer.py` checks that the torch adapter
divides by *the ε of the layer the point is in*.

---

## 3. Structure, and why this period and period count

Ag (Drude: ε_∞ = 3.7, ħω_p = 9.1 eV, ħγ = 0.018 eV → ε_Ag = −17.883 + 0.198j at
633 nm) / silica (ε = 2.25), metal fill f = 0.30, air above, **λ₀ = 633 nm
fixed**. Termination: **metal** — a full `f·a` = 9 nm silver layer against the
air, the `'metal'` case of `examples/emt_validity.multilayer_stack` and the
natural stack whose EMT error is large. A semi-infinite silver substrate closes
the stack from below, so the mode is strictly bound on both sides.

**Chosen: a = 30 nm, N = 6** (9 nm Ag + 21 nm SiO₂ per period, 180 nm of stack,
13 interfaces).

### Why a = 30 nm

| a (nm) | 2π/a in k₀ | N at ~180 nm | n_eff (TMM) | Im k (TMM) | EMT err Re | EMT err Im |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 63.3 | 18 | 1.07293 | 1.06e4 | 0.93 % | 11.4 % |
| 20 | 31.6 | 9 | 1.06491 | 9.64e3 | 1.69 % | 22.2 % |
| **30** | **21.1** | **6** | **1.05831** | **8.81e3** | **2.33 %** | **33.6 %** |
| 60 | 10.5 | 4 (240 nm) | 1.04498 | 6.91e3 | 3.63 % | 70.4 % |

(errors normalised by the TMM value; `emt_validity.py` normalises by the EMT
value, which is why it quotes 24 % where this table says 34 %.)

30 nm is the sweet spot between *EMT being clearly wrong* and *the layers being
resolvable*. At 10 nm the Re k_spp discrepancy is under 1 % — close to what the
PINN's own field error would be, so a win would not be readable — and the Fourier
band would have to reach 190 in k₀ units instead of 63. At 60 nm resolution is
easy but the structure is so far outside any homogenisation regime that beating
EMT proves little. 30 nm is also exactly the period the project's own
`max_layer_period` = 33 nm rule would wave through, which is the point
`emt_validity.py` was making.

### Why N = 6

At a = 30 nm the finite stack has converged to the semi-infinite answer by N = 6:

| N | Na (nm) | n_eff | Im k | \|ΔRe\|/Re vs N = 32 | \|ΔIm\|/Im vs N = 32 |
|---:|---:|---:|---:|---:|---:|
| 2 | 60 | 1.05303 | 7057 | 5.1e-3 | 2.1e-1 |
| 4 | 120 | 1.05759 | 8475 | 8.0e-4 | 4.7e-2 |
| **6** | **180** | **1.05831** | **8813** | **1.2e-4** | **9.5e-3** |
| 8 | 240 | 1.05842 | 8881 | 1.9e-5 | 1.8e-3 |
| 16 | 480 | 1.05844 | 8897 | 1.0e-8 | 1.7e-6 |

So the 2.3 % / 34 % gap to EMT is homogenisation error, not truncation: at N = 6
the stack is within 0.012 % (Re) and 0.95 % (Im) of the semi-infinite limit,
i.e. 190× and 35× smaller than the EMT gap it is being used to measure. N = 6 was
also the largest stack the tractability probe cleared comfortably inside its
time box.

Note that the truth the PINN is scored against is the exact mode of *this finite
stack*, so the residual truncation is not an error in the comparison at all — it
only matters for reading the EMT gap as a homogenisation error.

### The mode

| Quantity | Value |
|---|---|
| k_spp (TMM, truth) | 1.050486e7 + 8.8126e3 j m⁻¹ (n_eff = 1.05831) |
| k_spp (EMT prediction) | 1.074920e7 + 1.1778e4 j m⁻¹ (n_eff = 1.08293) |
| **EMT error to beat** | **Re 2.326 %, Im 33.65 %** |
| κ_d (air side) | 3.439e6 m⁻¹ (δ_d = 291 nm) |
| Bloch envelope in the stack | κ_env = 1.469e7 m⁻¹ (68 nm), from \|H_y\| at the interfaces |
| Domain | x ∈ [0, 1196 nm] = 2 λ_spp, y ∈ [0, 127 nm], z ∈ [−210, 349] nm |

The domain reaches 30 nm into the silver substrate below the stack, so the bottom
anchor face sits inside a homogeneous medium rather than on a layer boundary.

---

## 4. Resolution — the two choices the layering forces

### 4.1 Fourier band: anisotropic, and sized by a

The stack's own spatial frequency is 2π/a = **21.1 k₀**. The 1-D dispersion run
used a band of (0.1, 8)·k₀ — it would not even reach the fundamental. Two
decisions follow:

* **Top of the z band = 3 × 2π/a = 63.3.** H_y and E_x are continuous but
  *kinked* at every interface (their z-derivatives jump with ε, since
  `∂_z H_y = i ω ε₀ ε E_x`), and a kink needs harmonics well past the
  fundamental. Three harmonics is the smallest multiple for which the probe
  reached the accuracy the anchor needs.
* **The band must be anisotropic.** `src.models.FourierEMFeatures` samples
  wave-vector *directions* isotropically. Here the required band along z is 63
  while the field contains exactly one wavenumber along x, k_spp/k₀ = 1.06;
  isotropic sampling would spend most of its modes manufacturing high-frequency
  ripple along x that the solution does not have. `LayeredFourierFeatures`
  therefore splits its 128 modes into 64 axis-aligned in z (log-spaced 0.1→63.3),
  24 axis-aligned in x (0.1→6), and 40 oblique modes with k_x, k_y uniform in
  ±6 and |k_z| log-spaced to 63.3 with a random sign.

### 4.2 Collocation sampling: uniform inside the stack

The thinnest layer is 9 nm out of a 559 nm domain. The single-interface recipe's
purely exponential z-stratification would starve the deep layers — the field
falls by e⁻³ across the stack — so 70 % of the below-interface points are drawn
**uniformly in z** and only 30 % biased toward the top interface with the
measured Bloch envelope (68 nm). Measured at the run's 2048 interior points
(`points_per_layer`, logged every run):

* **53 points per 9 nm silver layer**, 114 per 21 nm silica layer,
* **40 in the worst (deepest) layer** — many points per layer everywhere.

A ±0.5 nm guard band is excluded around **every** interface (11 % of each silver
layer): ε jumps there, so the autograd `∂_z(ε E_x)` inside the divergence
residual is a delta function rather than a number.

---

## 5. Hyperparameters

| Item | Value | Note |
|---|---|---|
| Frame | x̂ = k₀·x, Ê = E/(η₀H₀), Ĥ = H/H₀ | makes the curl equations frequency-free (ω̂ = 1) |
| Network | 128×4 complex MLP, complex_tanh | 133 900 parameters |
| Fourier | 64 z + 24 x + 40 oblique modes, k_min 0.1 | 259 input features |
| Adapter | `LayeredDisplacementAdapter` over the 14-medium ε table | |
| Interior loss | curl + divergence, **per material** (Ag / silica / air) | ε now takes three values, not two |
| Metal preconditioner | \|ε_Ag\|⁻¹ (Adam), \|ε_Ag\|⁻¹ᐟ² (L-BFGS) | anti-collapse; the curl-H residual penalises an Ê error in silver ~320× harder |
| Continuity | all 13 interfaces, offset 0.5 nm | |
| Anchor | soft Dirichlet on the six faces = the **TMM** profile, weight 100 | not the analytical SPP |
| Physics ramp | 0 → 1 over the first 25 % of epochs | |
| Phase 1 | 4000 Adam epochs, 2048 points, lr 1e-3 cosine | |
| Phase 2 | 60 float64 L-BFGS steps, 4096 points | |
| Seed | 0 | |

**Continuity offset.** 0.5 nm rather than the 1.5–2 nm the layer thickness would
allow. The loss compares the field at ±offset, and the exact mode is *not* equal
at those two points: H_y varies by ≈ 2κδ across the gap, which is a residual the
true solution cannot avoid. With κ_Ag ≈ 4.3e7 m⁻¹ that floor is ~4 % at 0.5 nm
and ~13 % at 1.5 nm — large enough to bias the solution. Every continuity number
below has to be read against that floor; the exact TMM field scores ~0.03 on it,
not 0.

---

## 6. Tractability probe (run first, before any physics)

Before the full run: can a network of this capacity even *represent* the TMM
field of this stack, by direct supervised regression with no physics loss? If
not, no PINN objective could rescue it and the honest move would be to reduce N,
widen a, or grow the network.

**2500 Adam epochs at 4096 points, 48.8 s, seed 0 — rel L2 = 7.53e-3.**

| Region | probe rel L2 |
|---|---:|
| overall | **7.53e-3** |
| air (z > 0) | 6.01e-3 |
| inside the stack | 1.66e-2 |
| silver substrate (z < −180 nm) | 2.93e-1 |

Comfortably inside the stretch threshold (0.03), so the architecture was never
the binding constraint and `(a, N) = (30 nm, 6)` was accepted without reduction.
Two things the probe told us that mattered:

* The **substrate is the weak region even for pure regression** (29 %). The field
  there is ~1.4 % of its peak, so it contributes almost nothing to a global L2 and
  the fit simply does not care about it. That is a property of the metric, not a
  failure — but every substrate number below has to be read that way.
* It also fixed the Fourier band: 3 harmonics of 2π/a was the smallest multiple
  that reached this accuracy.

**The probe rel L2 is also a useful floor — and the full PINN beat it** (5.75e-3
vs 7.53e-3). Different budgets (4000 Adam + 60 float64 L-BFGS vs 2500 float32
Adam), so this is not a controlled comparison, but it does say the physics
objective plus the TMM anchor is at least as good a training signal here as
direct supervision on the answer.

---

## 7. What was run

One main run, no restarts for physics reasons, no fallback rungs needed:

```
python examples/validate_multilayer.py --epochs 4000 --n-points 2048 \
       --lbfgs-steps 60 --probe-epochs 2500 --seed 0
```

**45.2 min of training** (Adam 24.1 min, float64 L-BFGS 21.1 min at ~21 s/step)
plus 48.8 s of probe and ~1 min of validation and figures — **≈ 48 min CPU total**,
inside the 100 min budget with the diagnostic iteration unused for retraining.

Final training loss 7.10e-4 (from 4.35 at epoch 0); the L-BFGS phase alone took
it from 1.06e-2 to 7.10e-4, a factor of 15, and **was still descending at step
60** (see §11).

The one diagnostic iteration was spent not on retraining but on the `Im k_spp`
estimator (§9.2), which needed no retraining and could therefore be done from
the saved checkpoint with `--eval-only`.

---

## 8. Self-checks

**TMM field through the identical pipeline** (the convention check):

| Metric | Value |
|---|---:|
| rel L2 vs itself | **0.0** (exact) |
| Re k_spp fit error | 2.6e-11 |
| Im k_spp fit error | 4.3e-8 |
| κ_d fit error | 8.8e-10 |
| E_z layer contrast error | < 1e-6 |
| curl-E residual (air) | ~1e-7 |
| success tier | stretch |

This required making the TMM reference a **differentiable torch module** rather
than a numpy round-trip: the pipeline's curl residual differentiates the network,
and a detached reference reports `∇×E = 0` — a curl residual of order 1 — for a
field that is exact. `test_tmm_reference_module_is_differentiable` pins that.

**EMT field through the identical pipeline** (the baseline, not a check):
rel L2 0.244 overall, 0.600 in the stack, 0.181 in the air; k_spp errors 2.33 %
and 33.6 % by construction. Tier: *not met*.

---

## 9. Results

### 9.1 The headline — the three-way k_spp comparison

| | Re k_spp / k₀ | error | Im k_spp (m⁻¹) | error |
|---|---:|---:|---:|---:|
| **TMM (truth)** | 1.058313 | — | 8812.6 | — |
| **EMT (homogenised)** | 1.082930 | **2.326 %** | 11778.0 | **33.65 %** |
| **PINN (layered)** | 1.058365 | **0.0049 %** | 8981.9 | **1.92 %** |

**Error ratios (EMT error ÷ PINN error): Re 472×, Im 17.5×.**

**The PINN beats the homogenised model on both components, decisively.** On the
real part it is not close: the PINN's 0.005 % is at the level of the fit's own
resolution, and the fit agrees to 0.001–0.02 % at *every one of 20 independent
probe heights* (worst line 0.022 %).

### 9.2 The honest caveat on Im k_spp

`Im k_spp` is measured from `−d ln|H_y|/dx`, and over the 2 λ_spp box the true
mode's amplitude falls by only ~1 %. The field error in the air is 0.26 %, so a
single-line fit is a small difference of nearly equal numbers:

| Statistic over the 20 probe lines | Value |
|---|---:|
| signal-weighted mean (the reported number) | 8982 (**1.92 %** high) |
| 1σ across lines | 1341 = **15.2 % of the TMM value** |
| full spread (max − min) | 52 % of the TMM value |
| **worst single line** | **36.2 %** |

So: the *aggregate* is good and robustly so — unweighted mean 3.0 %, median
2.1 %, signal-weighted 1.9 %, i.e. every sensible aggregate lands between 1.7 %
and 3.2 % — but any *individual* line is worth ±15 %, and the worst one (36 %) is
marginally worse than EMT. The 17.5× win on Im k is real but should be quoted as
"an aggregate estimate, 1σ ≈ 15 %", not as a 1.9 % measurement. The money plot
carries the ±1σ bar for exactly this reason.

The scatter is a property of the *estimator's dynamic range*, not of the mode:
run on the exact TMM field the same estimator has zero spread (asserted in the
tests). And the estimator only works in the air — inside the stack a log-slope
fit along x returns −158 % to +12 %, because the amplitudes there are e-foldings
smaller and carry the layer structure.

### 9.3 Field accuracy against the TMM profile

| Region | PINN rel L2 | EMT rel L2 | PINN advantage |
|---|---:|---:|---:|
| **overall** | **5.75e-3** | 2.47e-1 | **43×** |
| air (z > 0) | 2.60e-3 | 1.82e-1 | 70× |
| inside the stack | 1.83e-2 | 6.17e-1 | 34× |
| silver substrate | 1.04e-1 | 2.06e0 | 20× |

Per component: E 5.75e-3, H 4.94e-3, **E_z 3.17e-3**, E_x 1.32e-2.

That E_z — the discontinuous component, the one a plain MLP cannot represent — is
the *most* accurate component is the displacement adapter doing its job: the
network only ever has to fit the smooth D̂_z, and the sawtooth is arithmetic.

### 9.4 Does it have the layer structure?

| Metric | PINN | TMM | error |
|---|---:|---:|---:|
| E_z contrast (mean \|E_z\| in silica / in silver) | 6.142 | 6.107 | **0.57 %** |
| κ_d (air-side decay) | 3.422e6 m⁻¹ | 3.439e6 | 0.49 % |
| Bloch envelope κ in the stack | 1.445e7 m⁻¹ | 1.469e7 | 1.61 % |
| impedance ratio E_rms/(η₀H_rms) | 0.9968 | 1 | 0.32 % |

Bound on both sides (positive decay in the air and into the stack). The contrast
number is worth reading twice: the naive expectation from D_z continuity alone is
|ε_Ag|/ε_silica = 7.95, and the TMM's actual 6.11 is that reduced by the
envelope's decay across each period — the PINN reproduces the *actual* value to
0.6 %, not the naive one.

### 9.5 Residuals

| Residual | air | silica | silver |
|---|---:|---:|---:|
| curl-E, RMS / (k₀·E_rms) | 0.019 | 0.072 | 0.192 |
| curl-H, RMS / (k₀·H_rms) | 0.035 | 0.188 | 0.419 |

Tangential continuity across all 13 interfaces: E 0.0086, H 0.029 — both at or
below the ~4 % floor the ±0.5 nm offset formulation imposes on the exact mode
(§5), so continuity is saturated.

The silver-side curl residual (0.19 / 0.42) is the weak spot, exactly as in the
single-interface and dispersive-HMM predecessors — the |ε_Ag|⁻¹ᐟ² L-BFGS
preconditioner still under-weights it. It does not show up in the field error
because the anchor and the other residuals pin the solution anyway.

### 9.6 Success tier

**TARGET.**

| Tier | Requirement | Achieved? |
|---|---|---|
| minimum | bound mode, rel L2 < 0.5, correct layer structure | ✅ (5.75e-3, contrast 0.57 %) |
| **target** | rel L2 < 0.1 **and** both k components closer than EMT | ✅ **(43× / 472× / 17.5×)** |
| stretch | rel L2 < 0.03 **and** both k within 1 % of TMM | ❌ — rel L2 5.75e-3 passes easily, **Re k 0.005 % passes, Im k 1.92 % does not** |

The stretch tier fails on one number only. Given §9.2, pushing Im k under 1 %
would mean improving the estimator's baseline (a wider box) rather than the
network — see §11.

---

## 10. Observations

1. **The methodological claim held exactly.** The DisplacementAdapter needed no
   structural change to go from 1 interface to 13 — only the ε(z) table. E_z came
   out as the most accurate component of the whole solution. This is the reusable
   result: for TM modes in *any* planar stack, D_z = −k_x H_y/ω carries no ε, so
   a network that emits D̂_z and divides by ε(z) gets every interface exactly right
   by construction, however many there are.
2. **Re k_spp is essentially exact and Im k_spp is not, for a structural reason.**
   The phase advances 4π across the box while the amplitude changes by 1 %. Any
   fixed-box mode solver has this asymmetry; it is not specific to PINNs, and it
   is why the loss figure needs an error bar and the index does not.
3. **The physics objective matched direct supervision** on the same architecture
   (5.75e-3 vs 7.53e-3), which is a mild but genuine argument that the anchor is
   not doing all the work.
4. **Isotropic Fourier features are the wrong tool for a layered problem** and had
   to be replaced. The required band along z is 60× the band along x; sampling
   directions isotropically spends the modes in the wrong place. The anisotropic
   split (§4.1) is probably the single most transferable engineering decision
   here.
5. **Uniform z-sampling inside the stack matters.** The predecessor's purely
   exponential stratification would have put a handful of points in the deepest
   period; the measured 40-per-worst-layer came from deliberately overriding it.
6. **The substrate is the worst region for both the PINN (10 %) and the probe
   (29 %).** Both are dominated by the same thing: a global L2 has no reason to
   care about a region carrying 1.4 % of the field. It is a metric artefact, but
   it means "rel L2 5.8e-3" should not be read as "0.6 % everywhere".

---

## 11. Limitations, honestly

* **Im k_spp is quoted to 1.9 % with a 15 % 1σ.** The number is right; the
  measurement is not sharp. Do not carry the 1.9 % forward as a precision claim.
* **The anchor is the TMM profile on all six faces.** As in every predecessor
  experiment, the network is not *discovering* the mode from physics alone — it
  is being told the mode on the boundary and asked to fill in the interior
  consistently with Maxwell. The result therefore demonstrates that the layered
  representation is learnable and accurate, not that the PINN could find k_spp
  unaided. (The k_spp comparison is still meaningful: the fit is done in the
  interior, at heights the anchor does not pin, and a network that merely
  interpolated its boundary data sloppily would not reach 0.005 %.)
* **One frequency, one geometry, one seed.** No ω conditioning, no sweep over a,
  no seed ensemble. The 472× and 17.5× ratios are single-run numbers.
* **L-BFGS had not converged at 60 steps** — the loss was still falling
  monotonically. The stopping point was the time budget, not a plateau.
* **The silver-side curl residual (0.42) is large in absolute terms.** The
  solution is accurate despite it, which means the metal-side physics is being
  carried by the anchor more than by its own residual.
* **The stack is only 6 periods deep.** Deeper stacks put more of the domain in
  the low-amplitude region where the global L2 stops constraining the solution;
  the substrate number is a preview of what that would look like.
* **`--eval-only` was used to re-score the saved model** after the `Im k`
  estimator was improved from a 5-line median to a 20-line signal-weighted mean.
  The weighting rule is principled a priori (better-conditioned log-fits where
  the signal is larger) and both the spread and the *worst* single line are
  reported, so nothing is being selected post hoc — but the sequence is recorded
  here rather than hidden. The trained weights were never touched.

---

## 12. Repo defects found (reported, not fixed)

1. **`src/models/pinn_network.py::FourierEMFeatures` samples 3-D directions
   non-uniformly.** It draws `theta = rand()*π`, `phi = rand()*2π` and builds
   `(k sinθ cosφ, k sinθ sinφ, k cosθ)`. Uniform θ is *not* uniform on the sphere
   — the density goes as 1/sin θ relative to uniform, so the wave-vectors cluster
   toward the ±z poles. The fix is `theta = arccos(1 − 2u)`. Nothing in the repo
   depends on isotropy for correctness, and for the layered problem the bias
   happens to point the right way, but the class documents itself as "spherical
   sampling" and is not.
2. **`FourierEMFeatures` docstring units are wrong.** `frequency_range` is
   documented as "(k_min, k_max) wavenumber range in absolute units (rad/m)", but
   every caller in the repo (`validate_spp`, `validate_spp_dispersion`,
   `validate_hmm_dispersion`) works in a dimensionless frame and passes rad per
   *input unit*. Relatedly, `dc_scale`'s comment describes rescaling "physical
   coordinates of order 1e-6 m" to O(1) while the default is 1.0, which does no
   such thing.
3. **A live trap, hit and guarded here: `nn.Module.to(dtype)` silently destroys
   complex buffers.** `Module._apply` converts *complex* buffers with whatever
   dtype it is handed, so the float64 L-BFGS promotion turns a registered
   `complex128` permittivity table into a real `float64` one — `ε_Ag = −17.88 +
   0.20j` becomes `−17.88`, a lossless metal, and `Im k_spp` becomes
   unmeasurable. The existing adapters escape only by accident, because they
   store python `complex` scalars rather than tensors. Anything storing a
   material table as a buffer will hit this. Guarded in
   `LayeredDisplacementAdapter` (plain tensors, lazy device move) with a
   regression test; worth a note wherever the L-BFGS promotion is documented.

---

## 13. Next steps

1. **Widen the box in x** to 6–8 λ_spp and re-measure Im k_spp. The amplitude
   change becomes 3–4 % instead of 1 %, which should cut the 15 % 1σ by the same
   factor and is the only change likely to reach the stretch tier.
2. **More L-BFGS.** It was still descending at step 60; 150–200 steps is ~35 min
   more and costs nothing in complexity.
3. **Sweep the period.** With the machinery in place (`--period-nm`,
   `--n-periods`), a = 10 / 20 / 30 / 60 nm at fixed N·a would turn this single
   point into the PINN's own version of the `emt_validity.py` curve — and would
   show where the layered PINN's advantage over EMT disappears.
4. **Drop the anchor to fewer faces** (or to the far faces only) and see whether
   the interior physics still locates the mode. That is the experiment that would
   turn "the layered representation is learnable" into "the PINN found it".
5. **Region-weighted loss for the substrate.** A relative (per-point normalised)
   field loss would stop the global L2 ignoring the deep low-amplitude region;
   the probe shows this is a metric problem, not a capacity one.
6. **ω conditioning on the layered stack.** The natural composition of this run
   with `validate_hmm_dispersion.py` — but note that a and the Fourier band are
   both frequency-dependent, so the band would have to be sized for the whole
   sweep at once.
