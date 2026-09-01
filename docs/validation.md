---
title: Validation
---

# Validation

[← Overview](index.html)

A physics-informed network can only be as trustworthy as the operators it is
built from. Every physics routine here is pinned to an **independent** reference
— an exact solution, a literature value, or a separately validated solver — not
to another part of this codebase.

| Layer | Benchmark | Result |
|---|---|---|
| Differential operators, curl losses | exact extraordinary wave in a uniaxial crystal | residuals ~10⁻¹⁵ |
| Interface conditions | exact Fresnel solution, TE and TM, lossy included | machine precision; Brewster \|r_p\| ~ 10⁻¹⁷ |
| SPP analytics | Johnson & Christy silver at 633 nm | L = 56.4 µm, δ_metal = 22.9 nm — inside literature bands |
| Transfer-matrix solver | Fresnel; closed-form SPP; thin-film branches | 5×10⁻¹⁷; 3.6×10⁻¹⁹; correct long/short-range splitting |
| Analytical SPP mode | Maxwell and continuity via the validated operators | machine precision |

## Why these particular tests

**The anisotropic path needed a case where E ∦ D.** A scalar permittivity test
cannot distinguish a correct tensor implementation from several wrong ones. The
extraordinary wave in a uniaxial crystal has non-parallel **E** and **D**, so it
exercises the tensor contraction properly. As a control, feeding the same fields
an isotropic permittivity raises the residual by 26 orders of magnitude.

**The interface tests discriminate rather than merely pass.** Tangential **E**
and normal **D** are continuous while normal **E** is *discontinuous* by the
permittivity ratio, and the tests assert both facts. Detuning the reflection
coefficient by 5% is detected.

**The transfer-matrix solver reproduces textbook film physics.** A thin metal
film supports coupled symmetric and antisymmetric branches; both converge to the
single-interface value as the film thickens, and the long-range branch's
propagation length grows more than fiftyfold as it thins.

## Self-checking the measurement

Each PINN experiment pushes its own *reference* solution through the identical
validation pipeline. If the analytical field scores a relative L2 of exactly
zero and recovers its wavevector to 10⁻¹⁰, then a reported network error of
10⁻³ is a property of the network, not of the metric.

This caught a real issue. The interface-continuity metric evaluated at ±2 nm has
a *physical* floor of roughly 8% for a strongly confined silver mode, because
the exact field genuinely differs across a 4 nm gap. Without the self-check that
floor would have looked like network error.

## Do the design choices earn their place?

Benchmarks establish that the operators are right. They say nothing about
whether the *recipe* built on them is necessary. `examples/ablation_study.py`
removes one ingredient at a time from the single-interface case, every arm on
an identical schedule and seed:

| removed | relative L2 vs control | amplitude recovered | bound mode |
|---|---:|---:|:--:|
| *(control)* | 1.00x | 73% | both sides |
| boundary anchor | 2.70x | **0.1%** | air only |
| displacement adapter | 2.30x | 30% | neither |
| per-medium weighting | 1.87x | 42% | neither |
| physics-loss ramp | 1.42x | 58% | metal only |

All four earn their place, and only the control recovers a bound mode on both
sides of the interface — a sharper discriminator than the error norm.

Two findings are worth stating separately.

**A loss curve is not evidence.** The no-anchor arm reaches a final loss
**31,000 times lower than the control**, descending smoothly throughout, while
the field it learned is one part in a thousand of the correct amplitude. The
Maxwell residual really is minimised — by E = H = 0. Any PINN result reported
without a reference solution should be read with this in mind.

![Training curves for the five ablation arms](assets/ablation_training_curves.png)

The green curve is the run that learned nothing. Each arm minimises its own
objective, so the levels are not comparable — the shape is the point.

**The adapter's effect is not local.** It was introduced to make the E_z jump
exact, and removing it does leave that jump essentially unlearned (a ratio of
1.1 where the exact value is 18.3). But it also degrades the metal-side
curl-H residual by 5.9x. Getting the discontinuity right by construction makes
the interior physics tractable, not just the interface.

The caveat, stated plainly: this runs a deliberately shortened schedule, so the
control here is far from the production result (relative L2 0.37 against
0.0039). The arms are comparable to each other, not to the headline numbers.
One case, one frequency, one seed.

## Reproducing

```bash
pytest -q                             # 865 tests
python scripts/validate_physics.py    # standalone physics checks
```

Continuous integration runs the non-slow subset on Python 3.10 and 3.12 with a
coverage floor, plus a smoke run of the analytics-only studies.
