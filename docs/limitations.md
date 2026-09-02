---
title: Limitations
---

# Limitations

[← Overview](index.html)

## Homogenising the multilayer is worse than the usual rule suggests

Most results in this project model the Ag/silica multilayer as a uniform
uniaxial medium. The standard guard for that approximation is that the layer
period should be much smaller than a wavelength — here about 33 nm.

Testing it directly against a transfer-matrix solve of the real stack shows that
guard is roughly an order of magnitude too loose for a *surface* mode. The
leading error is **O(a)** in the layer period, not the **O((a/λ)²)** bulk term
the rule addresses, because the dominant contribution is a **termination**
effect. The signature is unambiguous: metal-terminated and dielectric-terminated
stacks err by equal magnitudes with opposite signs.

![Effective-medium error against layer period](assets/emt_period_sweep.png)

| Layer period | Error in Re *k*<sub>spp</sub> | Error in Im *k*<sub>spp</sub> |
|---|---|---|
| 2 nm | 0.2% | 1.9% |
| 10 nm | 0.9% | 9.1% |
| 30 nm | 2.3% | 24% |

So homogenised **dispersion** is reliable, but homogenised **loss figures** —
propagation lengths, quality factors, anything derived from Im *k* — are
optimistic unless the period is genuinely small.

There is a clean fix. Terminating the stack with a half-thickness metal layer
places the effective boundary at that layer's centre, cancels the O(a) term and
restores O(a²) scaling, holding the error below 1% out to a 60 nm period.

## Other caveats

**Material model.** Silver is represented by a Drude fit with no interband
transitions. It is within 3% of Johnson & Christy at 633 nm but degrades below
about 450 nm, which sets the blue edge of every band used here. Its imaginary
part is roughly three times too small, so Drude-based propagation lengths are
optimistic even before homogenisation enters.

**Idealisations.** The dispersion experiments either hold ε fixed across the
band or take it from effective-medium theory; neither uses measured data. The
geometry is a planar, laterally infinite interface throughout — no roughness,
no finite beams, no edges.

**Numerics.** Training is CPU-bound at roughly 30–80 minutes per experiment.
Single precision sets a residual floor near 10⁻³ relative, so the refinement
phase runs in double precision. Frequency-conditioned networks are least
accurate at their band edges, an interpolation effect: train on a band about 10%
wider than the one to be queried.

**When does the surrogate pay for itself? Mostly, it doesn't — yet.**
Measured per-query costs for k_spp(ω, f) (`examples/cost_analysis.py`):
closed-form dispersion 0.1 µs, transfer matrix on the real six-period stack
2.3 ms, trained surrogate 1.6 ms. Against the closed form, 82 minutes of
training never amortises. Against the transfer matrix, it would take about
seven million queries. The crossover only becomes reasonable — 8 to 82 queries
— against a reference costing minutes per solve, i.e. the 2-D/3-D full-wave
regime this project has not yet entered. For planar stacks the transfer matrix
is the right tool, and this page says so with measurements.

**Scope of the headline result.** The multilayer experiment is one structure at
one frequency. It shows the method can capture physics that homogenisation
misses; it does not establish that it does so across all geometries, and the
transfer matrix remains both faster and exact for planar stacks. The case for
the network is generalisation to geometries where no such reference exists —
which this project has not yet demonstrated.
