r"""
Material-Conditioned SPP Surrogate over a 2-D (ω, f) Design Space

Trains **one** network that covers a two-dimensional *fabricable* design space of
Ag/silica hyperbolic metamaterials against air: angular frequency ω **and** the
multilayer's metal fill fraction f. The permittivity pair the mode sees,
``(ε_t, ε_n) = hmm_permittivities(ω, f, ε_silica)``, is not a free parameter — it
is whatever the Rytov effective medium of a real stack with that fill fraction
gives at that frequency. Every point of the rectangle therefore names a
structure someone could grow.

Where this sits
---------------
* ``examples/validate_spp_dispersion.py`` — fixed ε, ω-conditioned (1 input axis,
  a self-similar mode family).
* ``examples/validate_hmm_dispersion.py`` — dispersive ε(ω), ω-conditioned
  (1 input axis, a genuinely nonlinear branch). Reached TARGET tier, worst
  rel L2 1.85e-2. **This experiment inherits its architecture and recipe
  wholesale** and adds one conditioning axis.
* ``examples/inverse_design.py`` + :mod:`src.design` — differentiable inverse
  design over ``(ε_t, ε_n)`` treated as *free* complex parameters, i.e. over a
  space most of whose points no multilayer realises.

The surrogate closes that gap: it is conditioned on the *design variable*
(f), not on the derived material, so a gradient step in f is a step to a
neighbouring manufacturable stack.

The k₀(ω) input-scaling trick (unchanged, and still the key)
------------------------------------------------------------
The network is fed ``(x, y, z)·k₀(ω)`` with ``k₀(ω) = ω/c``. With x̂ = k₀x,
Ê = E/(η₀H₀), Ĥ = H/H₀ the curl equations are exactly ``∇̂×Ê = i Ĥ`` and
``∇̂×Ĥ = −i ε Ê`` **at every ω**, so a single ``frequency = 1`` residual serves
the whole rectangle and ω enters the physics only through ε(ω, f). It also
compresses the dynamic range: over this design space the SI decay constants
span 9.7× while their scaled counterparts κ̂_d ∈ [0.27, 0.90] and
κ̂_m ∈ [0.37, 2.66] are all O(1). And it leaks nothing — k₀ = ω/c is known
without solving anything, whereas scaling by k_spp would hand the network most
of the answer.

The design space (verified, not assumed)
----------------------------------------
* **f ∈ [0.15, 0.40]** — the range over which ``fill_fraction_scan`` in
  ``figures/hyperbolic/hmm_summary.json`` reports a qualifying band (its f = 0.45
  and 0.50 entries have none: too much metal drives ε_t strongly negative and
  pushes the mode onto the light line).
* **ω = the intersection over that f range of the per-f qualifying bands**, inset
  by :data:`DESIGN_INSET` of its span at each end so the rectangle does not sit
  on a band edge. That is ω/ω_ref ∈ [0.826, 1.067], λ₀ ∈ [593, 767] nm.
* :func:`verify_design_space` re-derives this at runtime on a dense 2-D grid
  using ``MetamaterialProperties.is_spp_supported`` **plus** the non-radiative
  gate ``Re k_spp > √ε_d·k₀``, and reports the bound fraction (1.000) and the
  worst margin (n_eff = 1.037, i.e. 3.7 % above the light line, at the
  f = 0.40 / red corner).

Note the rectangle is *conservative*: the qualifying bands carry extra quality
criteria (κ spread ≤ 10, ≥ 15 % nonlinearity, n_eff ratio ≥ 1.15, L/λ_spp ≥ 10)
beyond boundedness, so the merely-bound region is wider — see the results doc.

Does f actually move the answer?
--------------------------------
Yes, and by a comparable amount to ω. Across the rectangle
n_eff = Re k_spp/k₀ spans 1.037 → 1.344. At fixed mid-band ω, sweeping f from
0.15 to 0.40 changes Re k_spp by −12 % (n_eff 1.19 → 1.05); at the blue edge the
same sweep changes n_eff by −20 %. At fixed mid f, sweeping ω across the band
changes Re k_spp by +30 %. A surrogate that ignored f would be wrong by far more
than the target tolerance, which is the point.

Conditioning
------------
* **Input**: 5 features — (x, y, z)·k₀(ω), plus ω̂ ∈ [−1, 1] and f̂ ∈ [−1, 1],
  each linear over its own range. Fourier features encode the *spatial* columns
  only; ω̂ and f̂ are appended raw, because both dependences are smooth and
  putting random 8 rad/unit directions on them would invent oscillations along
  the design axes.
* **Per-row material**: the interior ε tensor is built per row from
  ε_t(ω_row, f_row), ε_n(ω_row, f_row) below the interface and ε_d = 1 above.
  The displacement adapter's divisor is ε_n(ω_row, f_row); the boundary anchor is
  the analytical mode at each block's own (ω, f); each block's box is sized from
  its own (ω, f) analytic scales. Two rows differing only in f carry genuinely
  different material — asserted in the tests, and the whole point of the
  experiment.
* **Batching**: each epoch draws :data:`N_BLOCKS` = 6 (ω, f) pairs by *jittered
  stratification* — a 3 × 2 grid of equal cells with one uniform sample per cell
  — so every epoch covers the rectangle evenly rather than clumping the way
  6 independent uniform draws would. L-BFGS refines on a fixed 5 × 5 tensor grid
  of nodes (corners included).

Validation: a 4 × 4 grid of held-out (ω, f) points placed at the *centres* of the
refinement grid's cells, so no validation point is a refinement node and each sits
at the worst-case distance from the nearest one. Per point, in SI: rel L2 vs
:func:`src.analytical.analytical_spp_fields`, Re k_spp by phase-slope fit, κ_d and
κ_m by decay fits, tangential-continuity residuals. The analytical mode is pushed
through the identical pipeline at 3 (ω, f) points as a convention self-check.
Success tiers: minimum = bound everywhere and rel L2 < 0.5; target = rel L2 < 0.1
everywhere and k_spp within 1 %; stretch = rel L2 < 0.03 and k_spp within 0.5 %.

Inverse design through the surrogate
------------------------------------
:func:`inverse_design_demo` solves "which fill fraction gives effective index
n_eff at this ω?" by gradient descent **through the trained network**: k_spp is
recovered from the network's own H_y along a probe line by an unwrapped-phase
least-squares solve (:func:`k_spp_from_network`), which is differentiable end to
end, so ∂n_eff/∂f̂ comes straight from autograd. It is cross-checked against a
1-D root find on the closed form. Here the closed form exists, so the demo
**validates** the method rather than being necessary; its value is that the same
loop runs unchanged where no closed form exists (finite slabs, the next
experiment).

Usage::

    python examples/validate_hmm_surrogate.py [--epochs 4500] [--n-points 2048]
        [--lr 1e-3] [--seed 0] [--device cpu] [--lbfgs-steps 90]
        [--lbfgs-dtype {float64,float32}] [--f-min 0.15] [--f-max 0.40]
        [--resume] [--quick]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analytical import analytical_spp_fields, complex_to_pinn_format  # noqa: E402
from src.constants import C0, EPS0, ETA0, MU0  # noqa: E402
from src.effective_medium import drude_parameters_ev, hmm_permittivities  # noqa: E402
from src.models import (  # noqa: E402
    ElectromagneticPINN,
    FourierEMFeatures,
    TangentialContinuityLoss,
    to_complex,
)
from src.physics import MaxwellEquations  # noqa: E402
from src.physics.metamaterial import MetamaterialProperties  # noqa: E402

logger = logging.getLogger("validate_hmm_surrogate")

# --------------------------------------------------------------------------- material / design
HMM_SUMMARY_PATH = REPO_ROOT / "figures" / "hyperbolic" / "hmm_summary.json"

#: Fraction of the raw ω-intersection removed at *each* end, so the rectangle
#: does not sit exactly on a qualifying-band edge (the scan resolves those
#: edges on a 401-point grid, and a band edge is where the mode is weakest).
DESIGN_INSET = 0.02


def load_hmm_design(path: Path = HMM_SUMMARY_PATH) -> Dict:
    """
    Read the material, the Drude fit and the per-f qualifying bands from
    ``hmm_summary.json``.

    Read at import time rather than hardcoded, so this experiment tracks
    ``examples/hyperbolic_metamaterial.py`` if the recommendation moves. The
    ``fill_fraction_scan`` entries whose ``band`` is ``None`` (f ≥ 0.45) are
    dropped: those fill fractions have no qualifying band at all.
    """
    with open(path) as fh:
        summary = json.load(fh)
    design = summary["design"]
    drude_ev = design["drude_model"]
    scan = [
        (float(e["fill_fraction"]), tuple(float(v) for v in e["band"]["omega_over_omega_ref"]))
        for e in summary["fill_fraction_scan"]
        if e.get("band") is not None
    ]
    return {
        "eps_dielectric_layer": float(design["eps_dielectric_layer"]),
        "eps_superstrate": float(design["eps_superstrate"]),
        "omega_ref": float(design["omega_ref"]),
        "drude": drude_parameters_ev(
            eps_inf=drude_ev["eps_inf"],
            hbar_omega_p_ev=drude_ev["hbar_omega_p_eV"],
            hbar_gamma_ev=drude_ev["hbar_gamma_eV"],
        ),
        "fill_fraction_bands": sorted(scan),
        "reference_fill_fraction": float(design["fill_fraction"]),
    }


DESIGN = load_hmm_design()
EPS_D2 = DESIGN["eps_dielectric_layer"]  # silica layers
EPS_D = DESIGN["eps_superstrate"]  # air half-space above z = 0
DRUDE = DESIGN["drude"]  # {'eps_inf', 'omega_p', 'gamma'} in SI
OMEGA_REF = DESIGN["omega_ref"]  # 2πc/633 nm, the reference for ω/ω_ref labels
FILL_BANDS = DESIGN["fill_fraction_bands"]  # [(f, (ω_lo/ω_ref, ω_hi/ω_ref)), ...]
REFERENCE_FILL = DESIGN["reference_fill_fraction"]  # the 1-parameter run's single f
H0 = 1.0  # A/m

FULL_F_MIN, FULL_F_MAX = 0.15, 0.40


def omega_intersection(f_min: float, f_max: float, inset: float = DESIGN_INSET) -> Tuple[float, float]:
    """
    The ω window bound for **every** f in ``[f_min, f_max]``, from the scan.

    Takes the intersection of the qualifying bands of every scanned fill
    fraction inside the range (the band edges move monotonically with f, and
    both ends of the range are scanned points, so the intersection is attained
    at the endpoints), then insets it by ``inset`` of its span at each end.
    """
    bands = [b for f, b in FILL_BANDS if f_min - 1e-9 <= f <= f_max + 1e-9]
    if not bands:
        raise ValueError(f"no scanned fill fraction lies in [{f_min}, {f_max}]")
    lo = max(b[0] for b in bands) * OMEGA_REF
    hi = min(b[1] for b in bands) * OMEGA_REF
    if not lo < hi:
        raise ValueError("the per-f qualifying bands do not overlap on this f range")
    span = hi - lo
    return lo + inset * span, hi - inset * span


# Module-level design-space state; :func:`set_design_space` refreshes everything
# derived from it (the documented fallback is to shrink the f range).
F_MIN, F_MAX = FULL_F_MIN, FULL_F_MAX
OMEGA_MIN, OMEGA_MAX = omega_intersection(F_MIN, F_MAX)
OMEGA_MID = 0.5 * (OMEGA_MIN + OMEGA_MAX)
OMEGA_HALF_SPAN = 0.5 * (OMEGA_MAX - OMEGA_MIN)
F_MID = 0.5 * (F_MIN + F_MAX)
F_HALF_SPAN = 0.5 * (F_MAX - F_MIN)

#: L-BFGS refinement nodes: a 5 × 5 tensor grid on (ω̂, f̂) including the corners.
N_NODE_OMEGA, N_NODE_FILL = 5, 5
#: Held-out validation grid: the 4 × 4 *centres* of that grid's cells, so no
#: validation point is a node and each sits at the worst-case distance from one.
VALIDATION_POINTS: Tuple[Tuple[float, float], ...] = ()
LBFGS_POINTS: Tuple[Tuple[float, float], ...] = ()
SELF_CHECK_POINTS: Tuple[Tuple[float, float], ...] = ()


def _grid_from_hats(w_hats: Sequence[float], f_hats: Sequence[float]) -> Tuple[Tuple[float, float], ...]:
    return tuple(
        (omega_from_hat(w), fill_from_hat(f)) for w in w_hats for f in f_hats
    )


def set_design_space(f_min: float = FULL_F_MIN, f_max: float = FULL_F_MAX) -> None:
    """
    Set the (ω, f) rectangle and refresh every derived module constant.

    ``f_min``/``f_max`` choose the fill-fraction range; the ω range then follows
    as the intersection of those fill fractions' qualifying bands. Provided for
    the documented failure path: if the full rectangle will not converge inside
    the compute budget, narrowing f (e.g. to [0.20, 0.35]) and reporting the
    reduced space honestly beats reporting failure on the full one.
    """
    global F_MIN, F_MAX, F_MID, F_HALF_SPAN
    global OMEGA_MIN, OMEGA_MAX, OMEGA_MID, OMEGA_HALF_SPAN
    global VALIDATION_POINTS, LBFGS_POINTS, SELF_CHECK_POINTS
    if not 0.0 < f_min < f_max < 1.0:
        raise ValueError("need 0 < f_min < f_max < 1")
    F_MIN, F_MAX = float(f_min), float(f_max)
    F_MID, F_HALF_SPAN = 0.5 * (F_MIN + F_MAX), 0.5 * (F_MAX - F_MIN)
    OMEGA_MIN, OMEGA_MAX = omega_intersection(F_MIN, F_MAX)
    OMEGA_MID = 0.5 * (OMEGA_MIN + OMEGA_MAX)
    OMEGA_HALF_SPAN = 0.5 * (OMEGA_MAX - OMEGA_MIN)

    node_w = np.linspace(-1.0, 1.0, N_NODE_OMEGA)
    node_f = np.linspace(-1.0, 1.0, N_NODE_FILL)
    LBFGS_POINTS = _grid_from_hats(node_w.tolist(), node_f.tolist())
    # Cell centres of the node grid: strictly held out, and maximally far from
    # every node in both directions.
    val_w = 0.5 * (node_w[:-1] + node_w[1:])
    val_f = 0.5 * (node_f[:-1] + node_f[1:])
    VALIDATION_POINTS = _grid_from_hats(val_w.tolist(), val_f.tolist())
    SELF_CHECK_POINTS = (
        (OMEGA_MIN, F_MIN),
        (OMEGA_MID, F_MID),
        (OMEGA_MAX, F_MAX),
    )


# --------------------------------------------------------------------------- material helpers
def hmm_eps(omega: float, fill: float) -> Tuple[complex, complex]:
    """``(ε_t, ε_n)`` of the Ag/silica stack at ``(omega, fill)`` — the design point."""
    eps_t, eps_n = hmm_permittivities(omega, fill, EPS_D2, **DRUDE)
    return complex(eps_t), complex(eps_n)


def hmm_eps_torch(
    omega: torch.Tensor, fill: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorised, **differentiable** torch mirror of :func:`hmm_eps` (complex128).

    Same Drude + Rytov algebra as :mod:`src.effective_medium`, in torch so the
    displacement adapter can build its ε_n(ω, f) divisor from the (ω̂, f̂)
    columns inside a forward pass — and so the inverse-design demo can
    differentiate that divisor with respect to f. ``tests`` assert it agrees
    with ``hmm_permittivities`` to machine precision.
    """
    w = omega.to(torch.float64).to(torch.complex128)
    f = fill.to(torch.float64).to(torch.complex128)
    eps_m = DRUDE["eps_inf"] - DRUDE["omega_p"] ** 2 / (w**2 + 1j * DRUDE["gamma"] * w)
    eps_t = f * eps_m + (1.0 - f) * EPS_D2
    eps_n = eps_m * EPS_D2 / (f * EPS_D2 + (1.0 - f) * eps_m)
    return eps_t, eps_n


@lru_cache(maxsize=65536)
def mode_constants(omega: float, fill: float) -> Tuple[complex, complex, complex]:
    """Analytical ``(k_spp, κ_d, κ_m)`` at the design point ``(omega, fill)``."""
    eps_t, eps_n = hmm_eps(float(omega), float(fill))
    material = MetamaterialProperties(eps_n, eps_t, "z")
    return material.decay_constants(float(omega), EPS_D, "x")


def k0_of(omega: float) -> float:
    """Local free-space wavenumber ω/c — the input-scaling factor."""
    return float(omega) / C0


def omega_hat(omega: float) -> float:
    """Normalised frequency feature, linear on the band: ω̂ ∈ [−1, 1]."""
    return (float(omega) - OMEGA_MID) / OMEGA_HALF_SPAN


def omega_from_hat(w_hat: float) -> float:
    """Inverse of :func:`omega_hat`."""
    return OMEGA_MID + OMEGA_HALF_SPAN * float(w_hat)


def fill_hat(fill: float) -> float:
    """Normalised fill-fraction feature, linear on ``[F_MIN, F_MAX]``: f̂ ∈ [−1, 1]."""
    return (float(fill) - F_MID) / F_HALF_SPAN


def fill_from_hat(f_hat: float) -> float:
    """Inverse of :func:`fill_hat`."""
    return F_MID + F_HALF_SPAN * float(f_hat)


def omega_from_hat_torch(w_hat: torch.Tensor) -> torch.Tensor:
    """Tensor form of :func:`omega_from_hat`."""
    return OMEGA_MID + OMEGA_HALF_SPAN * w_hat.to(torch.float64)


def fill_from_hat_torch(f_hat: torch.Tensor) -> torch.Tensor:
    """Tensor form of :func:`fill_from_hat` (differentiable, used by the adapter)."""
    return F_MID + F_HALF_SPAN * f_hat.to(torch.float64)


def eps_scale(omega: float, fill: float) -> float:
    """Stiffness scale of the lower medium, ``max(|ε_t|, |ε_n|)`` at ``(ω, f)``."""
    eps_t, eps_n = hmm_eps(omega, fill)
    return max(abs(eps_t), abs(eps_n))


# --------------------------------------------------------------------------- design-space check
def is_bound(omega: float, fill: float, margin: float = 0.0) -> Tuple[bool, float]:
    """
    Independent bound-mode test at ``(omega, fill)``, returning ``(bound, n_eff)``.

    Both gates from ``examples/hyperbolic_metamaterial.py`` are applied:
    ``MetamaterialProperties.is_spp_supported`` (the unsquared matching
    condition on the ``Re κ > 0`` branch) **and** the non-radiative gate
    ``Re k_spp > (1 + margin)·√ε_d·k₀``, because with loss the matching
    condition also admits radiative quasi-roots below the light line.
    """
    eps_t, eps_n = hmm_eps(omega, fill)
    material = MetamaterialProperties(eps_n, eps_t, "z")
    supported = material.is_spp_supported(EPS_D, "x")
    k = material.spp_wavevector(omega, EPS_D, "x")
    n_eff = k.real / k0_of(omega)
    return bool(supported and n_eff > (1.0 + margin) * math.sqrt(EPS_D)), float(n_eff)


def verify_design_space(
    n_omega: int = 61, n_fill: int = 41, margin: float = 0.0
) -> Dict[str, float]:
    """
    Sweep a dense 2-D grid of the rectangle and report how bound it is.

    Returns the bound fraction, the worst (smallest) n_eff and where it occurs,
    the n_eff range, and how much Re k_spp moves along each axis — the last
    being the "is the second conditioning axis worth having?" number.
    """
    omegas = np.linspace(OMEGA_MIN, OMEGA_MAX, n_omega)
    fills = np.linspace(F_MIN, F_MAX, n_fill)
    n_eff = np.zeros((n_omega, n_fill))
    bound = np.zeros((n_omega, n_fill), dtype=bool)
    for i, w in enumerate(omegas):
        for j, f in enumerate(fills):
            bound[i, j], n_eff[i, j] = is_bound(float(w), float(f), margin)
    k = n_eff * (omegas[:, None] / C0)
    worst = np.unravel_index(int(np.argmin(n_eff)), n_eff.shape)
    mid_w, mid_f = n_omega // 2, n_fill // 2
    return {
        "n_grid": float(n_omega * n_fill),
        "bound_fraction": float(bound.mean()),
        "n_eff_min": float(n_eff.min()),
        "n_eff_max": float(n_eff.max()),
        "worst_margin_over_light_line": float(n_eff.min() - math.sqrt(EPS_D)),
        "worst_omega_over_omega_ref": float(omegas[worst[0]] / OMEGA_REF),
        "worst_fill_fraction": float(fills[worst[1]]),
        "k_spp_min_per_m": float(k.min()),
        "k_spp_max_per_m": float(k.max()),
        "k_spp_ratio_over_rectangle": float(k.max() / k.min()),
        "k_spp_ratio_along_f_at_mid_omega": float(k[mid_w, -1] / k[mid_w, 0]),
        "k_spp_ratio_along_omega_at_mid_f": float(k[-1, mid_f] / k[0, mid_f]),
        "k_spp_ratio_along_f_at_blue_edge": float(k[-1, -1] / k[-1, 0]),
        "k_spp_ratio_along_f_at_red_edge": float(k[0, -1] / k[0, 0]),
    }


# --------------------------------------------------------------------------- scaled frame
E_SCALE = ETA0 * abs(H0)
H_SCALE = abs(H0)
FIELD_SCALE = torch.tensor([E_SCALE] * 3 + [H_SCALE] * 3, dtype=torch.float32).view(1, 6, 1)

# In the k₀-scaled frame the guard band and continuity offset of the fixed-ε run
# (1 nm and 2 nm at λ₀ = 633 nm) become 0.01 and 0.02 scaled units exactly.
GUARD_HAT = 0.01
VAL_GUARD_HAT = 0.02
CONTINUITY_OFFSET_HAT = 0.02

# Per-(ω, f) box, in units of that design point's own analytic scales.
X_PERIODS = 2.0  # x ∈ [0, 2 λ_spp]
Z_METAL_DEPTHS = 3.5  # z ≥ −3.5 / Re κ_m
Z_AIR_DEPTHS = 1.2  # z ≤ +1.2 / Re κ_d
Y_WAVELENGTHS = 0.2  # y ∈ [0, 0.2 λ₀(ω)] — thin, the mode is y-invariant


def domain_hat(omega: float, fill: float) -> Tuple[float, float, float, float]:
    """``(x̂_max, ŷ_max, ẑ_min, ẑ_max)`` of the design point's box, k₀(ω)-scaled."""
    k, kappa_d, kappa_m = mode_constants(omega, fill)
    k0 = k0_of(omega)
    x_max = X_PERIODS * 2.0 * np.pi / (k.real / k0)
    y_max = Y_WAVELENGTHS * 2.0 * np.pi
    z_min = -Z_METAL_DEPTHS / (kappa_m.real / k0)
    z_max = Z_AIR_DEPTHS / (kappa_d.real / k0)
    return float(x_max), float(y_max), float(z_min), float(z_max)


def domain_si(omega: float, fill: float) -> Tuple[float, float, float, float]:
    """The same box in metres (``domain_hat`` divided by ``k₀(ω)``)."""
    k0 = k0_of(omega)
    return tuple(v / k0 for v in domain_hat(omega, fill))  # type: ignore[return-value]


# --------------------------------------------------------------------------- defaults
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2048
N_EPOCHS = 4500
LEARNING_RATE = 1e-3
#: (ω, f) blocks per epoch, drawn by jittered stratification on this cell grid.
STRATA_OMEGA, STRATA_FILL = 3, 2
N_BLOCKS = STRATA_OMEGA * STRATA_FILL
BOUNDARY_WEIGHT = 100.0
DIVERGENCE_WEIGHT = 1.0
CONTINUITY_WEIGHT = 1.0
PHYSICS_RAMP_FRAC = 0.25
# Metal-side loss preconditioning, applied PER ROW: max(|ε_t|, |ε_n|) varies 3.6×
# over the rectangle (2.69 at the f = 0.15 blue corner, 9.70 at the f = 0.40 red one).
CURL_POWER_ADAM = 1.0  # weight = |ε(row)|^-1
CURL_POWER_LBFGS = 0.5  # weight = |ε(row)|^-1/2
DIV_POWER = 1.0
METAL_FRACTION = 0.45
AIR_UNIFORM_FLOOR = 0.3
QUICK_EPOCHS = 200
LBFGS_STEPS = 90
LBFGS_POINTS_FACTOR = 3  # 25 refinement nodes need more points than the 1-D run's 13
LBFGS_DTYPE = torch.float64
FOURIER_MODES = 128
FOURIER_K_RANGE = (0.1, 8.0)  # scaled wavenumbers reach only κ̂_m = 2.66
HIDDEN_DIMS = (128, 128, 128, 128)

set_design_space()

FIGURES_DIR = REPO_ROOT / "figures" / "hmm_surrogate"
MODEL_PATH = REPO_ROOT / "artifacts" / "models" / "hmm_surrogate.pth"


# --------------------------------------------------------------------------- analytical
def analytical_fields_si(coords: torch.Tensor, omega: float, fill: float) -> torch.Tensor:
    """Analytical SPP ``(E, H)`` at SI ``coords`` and design point, ``[N, 6, 2]``."""
    eps_t, eps_n = hmm_eps(omega, fill)
    E, H = analytical_spp_fields(coords, omega, eps_t, eps_n, eps_dielectric=EPS_D, H0=H0)
    return torch.cat([complex_to_pinn_format(E), complex_to_pinn_format(H)], dim=1).to(
        coords.dtype if coords.is_floating_point() else torch.float32
    )


def analytical_fields_hat(coords_hat: torch.Tensor, omega: float, fill: float) -> torch.Tensor:
    """Analytical mode at k₀(ω)-scaled coords, in the core's scaled field units."""
    fields = analytical_fields_si(coords_hat / k0_of(omega), omega, fill)
    return fields / FIELD_SCALE.to(device=fields.device, dtype=fields.dtype)


class AnalyticalHMMSPP(nn.Module):
    """Exact SI mode with the surrogate's ``(coords, ω, f)`` interface (self-check)."""

    def forward(self, coords: torch.Tensor, omega: float, fill: float) -> torch.Tensor:
        return analytical_fields_si(coords, omega, fill)

    def at_point(self, omega: float, fill: float) -> nn.Module:
        return _FixedPoint(self, omega, fill)


# --------------------------------------------------------------------------- network
class DesignConditionedCore(nn.Module):
    """
    Scaled field network conditioned on the design point.

    Consumes ``[x̂, ŷ, ẑ, ω̂, f̂]`` of shape ``(N, 5)`` with x̂ = k₀(ω)·x. Fourier
    features encode the spatial columns only; the raw (ω̂, f̂) features are
    appended afterwards (see the module docstring). Returns ``(N, 6, 2)``, with
    channel 2 carrying the *continuous* D̂_z that the adapter converts to Ê_z.
    """

    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = HIDDEN_DIMS,
        fourier_modes: int = FOURIER_MODES,
        fourier_k_range: Tuple[float, float] = FOURIER_K_RANGE,
    ):
        super().__init__()
        self.fourier = FourierEMFeatures(3, fourier_modes, fourier_k_range, include_dc=True)
        self.mlp = ElectromagneticPINN(
            spatial_dim=self.fourier.output_dim + 2,  # spatial features + ω̂ + f̂
            field_components=6,
            hidden_dims=list(hidden_dims),
            complex_valued=True,
            frequency=OMEGA_REF,  # selects the time-harmonic input layout (no +1 column)
            use_fourier=False,  # spatial Fourier encoding is done here instead
            activation_type="complex_tanh",
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        features = self.fourier(coords[:, :3])
        return self.mlp(torch.cat([features, coords[:, 3:5]], dim=1))


class DesignDisplacementAdapter(nn.Module):
    """
    Displacement adapter whose ε_zz divisor depends on **both** ω and f.

    Same idea as ``examples/validate_spp.DisplacementAdapter``: the wrapped MLP
    represents the continuous normal displacement D̂_z on channel 2, and this
    module divides it by the local ε_zz so the returned Ê_z carries the exact
    interface jump a continuous MLP cannot represent. The predecessor's divisor
    below the interface was ε_n(ω); here it is ε_n(ω, f), read from the last two
    columns of the 5-column input. Above the interface it is the constant ε_d.

    Over this design space ε_n spans 2.69 → 4.15, and at fixed ω the f
    dependence alone moves it by ~50 % — so the divisor genuinely has to be a
    function of f, not of ω only.
    """

    def __init__(self, mlp: nn.Module, eps_above: complex = EPS_D):
        super().__init__()
        self.mlp = mlp
        self.eps_above = complex(eps_above)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        out = self.mlp(coords)  # [N, 6, 2]; channel 2 carries D̂_z
        fields = torch.complex(out[..., 0], out[..., 1])  # [N, 6]
        omega = omega_from_hat_torch(coords[:, 3])
        fill = fill_from_hat_torch(coords[:, 4])
        _, eps_n = hmm_eps_torch(omega, fill)
        eps_below = eps_n.to(dtype=fields.dtype, device=fields.device)
        eps_above = torch.as_tensor(self.eps_above, dtype=fields.dtype, device=fields.device)
        eps = torch.where(coords[:, 2] < 0, eps_below, eps_above.expand_as(eps_below))
        e_z = fields[:, 2] / eps
        fields = torch.cat([fields[:, :2], e_z.unsqueeze(1), fields[:, 3:]], dim=1)
        return torch.stack([fields.real, fields.imag], dim=-1)


class ConditionColumnNet(nn.Module):
    """
    3-column spatial view of the 5-input core with fixed per-row (ω̂, f̂) columns.

    The differential operators require ``coords`` with at most 3 columns, so the
    stored condition block — aligned row-for-row with the batch — is appended
    inside the forward. Gradients flow through the spatial columns only, which
    is exactly the curl/divergence semantics (neither ∂/∂ω nor ∂/∂f enters
    Maxwell's equations).
    """

    def __init__(self, core: nn.Module, cond: torch.Tensor):
        super().__init__()
        self.core = core
        self.cond = cond  # (N, 2) = [ω̂, f̂], no grad

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.core(torch.cat([coords, self.cond.to(coords.dtype)], dim=1))


class _FixedPoint(nn.Module):
    """SI 3-column module at a fixed design point: ``forward(coords_m) -> [N, 6, 2]``."""

    def __init__(self, parent: nn.Module, omega: float, fill: float):
        super().__init__()
        self.parent = parent
        self.omega = float(omega)
        self.fill = float(fill)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.parent(coords, self.omega, self.fill)


class HMMSurrogatePINN(nn.Module):
    """
    SI-unit wrapper: ``forward(coords_m, omega, fill)`` scales coordinates by
    k₀(ω), appends (ω̂, f̂) and rescales the network's dimensionless output to SI
    fields. ``at_point(omega, fill)`` returns a 3-column SI module for the
    validation pipeline.
    """

    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core
        self.register_buffer("field_scale", FIELD_SCALE.clone())

    def forward(self, coords: torch.Tensor, omega: float, fill: float) -> torch.Tensor:
        coords_hat = coords * k0_of(omega)
        cond = coords_hat.new_tensor([omega_hat(omega), fill_hat(fill)]).expand(
            coords_hat.shape[0], 2
        )
        scale = self.field_scale.to(coords_hat.dtype)
        return self.core(torch.cat([coords_hat, cond], dim=1)) * scale

    def at_point(self, omega: float, fill: float) -> nn.Module:
        return _FixedPoint(self, omega, fill)


def create_network(
    hidden_dims: Tuple[int, ...] = HIDDEN_DIMS,
    fourier_modes: int = FOURIER_MODES,
    device: torch.device = DEVICE,
) -> HMMSurrogatePINN:
    """Build the (ω, f)-conditioned HMM SPP surrogate (Fourier features + adapter)."""
    mlp = DesignConditionedCore(hidden_dims=hidden_dims, fourier_modes=fourier_modes)
    adapter = DesignDisplacementAdapter(mlp, eps_above=EPS_D)
    return HMMSurrogatePINN(adapter).to(device)


# --------------------------------------------------------------------------- permittivity rows
def eps_tensor_rows(omegas: torch.Tensor, fills: torch.Tensor, metal: bool) -> torch.Tensor:
    """
    Per-row ``(N, 3, 3)`` complex permittivity for the interior loss.

    Below the interface each row carries ``diag(ε_t, ε_t, ε_n)`` evaluated at
    **that row's own (ω, f)** — the entire point of the material-conditioned
    experiment; above it, ``diag(ε_d, ε_d, ε_d)``. No ω prefactor is needed
    anywhere: in the k₀-scaled frame Maxwell's curl equations are
    frequency-free, so the design point enters only here (and in the adapter).
    """
    w = omegas.reshape(-1).to(torch.float64)
    if metal:
        eps_t, eps_n = hmm_eps_torch(w, fills.reshape(-1).to(torch.float64))
        diag = torch.stack([eps_t, eps_t, eps_n], dim=1)  # (N, 3)
    else:
        diag = torch.full(
            (w.shape[0], 3), complex(EPS_D), dtype=torch.complex128, device=w.device
        )
    return torch.diag_embed(diag)


# --------------------------------------------------------------------------- sampling
def stratified_design_points(
    n_omega: int = STRATA_OMEGA, n_fill: int = STRATA_FILL, generator=None
) -> List[Tuple[float, float]]:
    """
    One jittered-stratified draw of ``n_omega × n_fill`` design points.

    The rectangle is cut into equal cells and exactly one uniform sample is
    taken from each, so a single epoch already covers the whole space. Pure
    uniform draws of the same size clump: with 6 points the expected largest
    empty axis-aligned gap is a substantial fraction of the rectangle, and the
    network would then see some regions only every few epochs.
    """
    u = torch.rand(n_omega, n_fill, 2, generator=generator)
    points = []
    for i in range(n_omega):
        for j in range(n_fill):
            w_hat = -1.0 + 2.0 * (i + float(u[i, j, 0])) / n_omega
            f_hat = -1.0 + 2.0 * (j + float(u[i, j, 1])) / n_fill
            points.append((omega_from_hat(w_hat), fill_from_hat(f_hat)))
    return points


def _sample_z_metal(
    n: int, depth_hat: float, z_min_hat: float, guard: float, device: torch.device
) -> torch.Tensor:
    """|ẑ| ~ truncated Exp(scale = 1/κ̂_m) on [guard, |ẑ_min|], returned negative."""
    span = -z_min_hat - guard
    u = torch.rand(n, device=device)
    z_abs = guard - depth_hat * torch.log1p(-u * (1.0 - math.exp(-span / depth_hat)))
    return -z_abs


def _sample_z_air(
    n: int, depth_hat: float, z_max_hat: float, guard: float, device: torch.device
) -> torch.Tensor:
    """ẑ on [guard, ẑ_max]: truncated Exp(scale = 1/κ̂_d) mixed with a uniform floor."""
    n_uniform = int(round(AIR_UNIFORM_FLOOR * n))
    n_exp = n - n_uniform
    span = z_max_hat - guard
    u = torch.rand(n_exp, device=device)
    z_exp = guard - depth_hat * torch.log1p(-u * (1.0 - math.exp(-span / depth_hat)))
    z_uni = guard + span * torch.rand(n_uniform, device=device)
    return torch.cat([z_exp, z_uni])


def sample_collocation_hat(
    n_points: int,
    omega: float,
    fill: float,
    guard: float = GUARD_HAT,
    device: torch.device = DEVICE,
) -> torch.Tensor:
    """
    Stratified interior points of the design point's box, in k₀(ω)-scaled units.

    Both the extents and the exponential z-strata use that ``(ω, f)``'s analytic
    scales, so every design point's near-field is sampled at the same relative
    resolution.
    """
    _, kappa_d, kappa_m = mode_constants(omega, fill)
    k0 = k0_of(omega)
    x_max, y_max, z_min, z_max = domain_hat(omega, fill)
    n_metal = int(round(METAL_FRACTION * n_points))
    n_air = n_points - n_metal
    z = torch.cat(
        [
            _sample_z_metal(n_metal, k0 / kappa_m.real, z_min, guard, device),
            _sample_z_air(n_air, k0 / kappa_d.real, z_max, guard, device),
        ]
    )
    x = torch.rand(n_points, device=device) * x_max
    y = torch.rand(n_points, device=device) * y_max
    return torch.stack([x, y, z], dim=1)


def sample_collocation_si(
    n_points: int,
    omega: float,
    fill: float,
    guard: float = GUARD_HAT,
    device: torch.device = DEVICE,
) -> torch.Tensor:
    """:func:`sample_collocation_hat` in metres, with ``requires_grad`` set."""
    coords = sample_collocation_hat(n_points, omega, fill, guard, device) / k0_of(omega)
    coords.requires_grad_(True)
    return coords


def _avoid_guard(z: torch.Tensor, guard: float) -> torch.Tensor:
    return torch.where(z.abs() < guard, torch.where(z < 0, -guard, guard), z)


def sample_boundary_hat(
    n_points: int,
    omega: float,
    fill: float,
    guard: float = GUARD_HAT,
    device: torch.device = DEVICE,
) -> torch.Tensor:
    """Points on the six faces of the design point's scaled box, ``n//6`` per face."""
    x_max, y_max, z_min, z_max = domain_hat(omega, fill)
    per_face = max(1, n_points // 6)
    low = torch.tensor([0.0, 0.0, z_min], device=device)
    high = torch.tensor([x_max, y_max, z_max], device=device)
    faces = []
    for axis in range(3):
        for value in (low[axis], high[axis]):
            pts = low + torch.rand(per_face, 3, device=device) * (high - low)
            pts[:, axis] = value
            if axis != 2:
                pts[:, 2] = _avoid_guard(pts[:, 2], guard)
            faces.append(pts)
    return torch.cat(faces, dim=0)


def sample_interface_hat(
    n_points: int, omega: float, fill: float, device: torch.device = DEVICE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Points on ẑ = 0 within the design point's box, and their unit normals ẑ."""
    x_max, y_max, _, _ = domain_hat(omega, fill)
    x = torch.rand(n_points, device=device) * x_max
    y = torch.rand(n_points, device=device) * y_max
    coords = torch.stack([x, y, torch.zeros_like(x)], dim=1)
    normals = torch.zeros_like(coords)
    normals[:, 2] = 1.0
    return coords, normals


def sample_training_batch(
    n_int: int,
    n_bc: int,
    n_if: int,
    points: Sequence[Tuple[float, float]],
    device: torch.device = DEVICE,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """
    One multi-design-point training batch of per-(ω, f) blocks, in scaled units.

    Each block lives in its own box (sized by its own analytic scales) and
    carries its own (ω̂, f̂) condition columns, its own per-row ε(ω, f) and its
    own stiffness scale max(|ε_t|, |ε_n|) for the metal loss weighting. Boundary
    anchors are the analytical mode at each block's own design point.
    """
    n_pt = len(points)
    int_blocks, bc_blocks, tgt_blocks, if_blocks, nrm_blocks = [], [], [], [], []
    c_int, c_bc, c_if, om_int, fl_int = [], [], [], [], []
    for i, (omega, fill) in enumerate(points):
        ni = n_int // n_pt + (n_int % n_pt if i == n_pt - 1 else 0)
        nb = max(6, n_bc // n_pt)
        nf = max(1, n_if // n_pt)
        cond = torch.tensor([omega_hat(omega), fill_hat(fill)], dtype=dtype, device=device)

        c = sample_collocation_hat(ni, omega, fill, device=device).detach().to(dtype)
        int_blocks.append(c)
        c_int.append(cond.expand(c.shape[0], 2))
        om_int.append(torch.full((c.shape[0],), float(omega), dtype=torch.float64, device=device))
        fl_int.append(torch.full((c.shape[0],), float(fill), dtype=torch.float64, device=device))

        b = sample_boundary_hat(nb, omega, fill, device=device).to(dtype)
        bc_blocks.append(b)
        c_bc.append(cond.expand(b.shape[0], 2))
        with torch.no_grad():
            tgt_blocks.append(analytical_fields_hat(b, omega, fill).to(dtype))

        f_pts, nrm = sample_interface_hat(nf, omega, fill, device=device)
        if_blocks.append(f_pts.to(dtype))
        nrm_blocks.append(nrm.to(dtype))
        c_if.append(cond.expand(f_pts.shape[0], 2))

    coords = torch.cat(int_blocks)
    cond_col = torch.cat(c_int)
    om_col = torch.cat(om_int)
    fl_col = torch.cat(fl_int)
    metal = coords[:, 2] < 0
    eps_t_m, eps_n_m = hmm_eps_torch(om_col[metal], fl_col[metal])
    return {
        "coords_air": coords[~metal].clone().requires_grad_(True),
        "coords_metal": coords[metal].clone().requires_grad_(True),
        "cond_air": cond_col[~metal],
        "cond_metal": cond_col[metal],
        "eps_air": eps_tensor_rows(om_col[~metal], fl_col[~metal], metal=False),
        "eps_metal": eps_tensor_rows(om_col[metal], fl_col[metal], metal=True),
        "eps_scale_metal": torch.maximum(eps_t_m.abs(), eps_n_m.abs()),
        "boundary": torch.cat(bc_blocks),
        "cond_bc": torch.cat(c_bc),
        "target": torch.cat(tgt_blocks),
        "iface": torch.cat(if_blocks),
        "normals": torch.cat(nrm_blocks),
        "cond_if": torch.cat(c_if),
    }


# --------------------------------------------------------------------------- interior losses
# One scaled-frame Maxwell operator serves the whole design space: with x̂ = k₀(ω)x
# the curl equations are ∇̂×Ê = i Ĥ and ∇̂×Ĥ = −i ε Ê at every (ω, f).
_SCALED_MAXWELL = MaxwellEquations(1.0, mu0=1.0, eps0=1.0)


def curl_loss_weighted(
    core: nn.Module,
    coords: torch.Tensor,
    cond: torch.Tensor,
    eps_rows: torch.Tensor,
    row_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Scaled-frame curl residual with **per-row** weights.

    Identical to ``MaxwellCurlLoss(frequency=1, mu0=1, eps0=1).compute(...)``
    (asserted in the tests) except that each row's squared residual is
    multiplied by ``row_weight``. The per-row form is needed here because the
    metal-side preconditioner |ε|^-p depends on the row's *design point*.
    """
    net = ConditionColumnNet(core, cond)
    E, H = to_complex(net(coords))
    curl_E = _SCALED_MAXWELL.curl_operator(E, coords)
    curl_H = _SCALED_MAXWELL.curl_operator(H, coords)
    eps_E = torch.einsum("nij,nj->ni", eps_rows.to(E.dtype), E)
    res_E = curl_E - 1j * H
    res_H = curl_H + 1j * eps_E
    if row_weight is None:
        return torch.mean(res_E.abs() ** 2) + torch.mean(res_H.abs() ** 2)
    w = row_weight.reshape(-1, 1).to(res_E.real.dtype)
    return torch.mean(w * res_E.abs() ** 2) + torch.mean(w * res_H.abs() ** 2)


def divergence_loss_weighted(
    core: nn.Module,
    coords: torch.Tensor,
    cond: torch.Tensor,
    eps_rows: torch.Tensor,
    row_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``∇̂·(εÊ) = 0`` and ``∇̂·Ĥ = 0`` with per-row weights (see :func:`curl_loss_weighted`)."""
    net = ConditionColumnNet(core, cond)
    E, H = to_complex(net(coords))
    div_D = _SCALED_MAXWELL.divergence_operator(
        torch.einsum("nij,nj->ni", eps_rows.to(E.dtype), E), coords
    )
    div_H = _SCALED_MAXWELL.divergence_operator(H, coords)
    if row_weight is None:
        return torch.mean(div_D.abs() ** 2) + torch.mean(div_H.abs() ** 2)
    w = row_weight.reshape(-1).to(div_D.real.dtype)
    return torch.mean(w * div_D.abs() ** 2) + torch.mean(w * div_H.abs() ** 2)


# --------------------------------------------------------------------------- training
def _write_checkpoint(
    path: Path, state: dict, loss: float, phase: str, history: Optional[Dict[str, list]] = None
) -> None:
    """
    Atomically save the best weights so far (rename is atomic on POSIX).

    The loss history rides along so a ``--resume`` run can continue the curve
    instead of restarting it: this experiment is trained in wall-clock-limited
    chunks, and a training-history figure covering only the last chunk would be
    a figure of the last chunk, not of the training.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "state_dict": {k: v.cpu() for k, v in state.items()},
            "best_loss": loss,
            "phase": phase,
            "history": history,
        },
        tmp,
    )
    tmp.replace(path)


def load_checkpoint_into(network: HMMSurrogatePINN, path: Path) -> float:
    """Load core weights from a training checkpoint; returns its recorded loss."""
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    network.core.load_state_dict(blob["state_dict"])
    return float(blob.get("best_loss", float("nan")))


def load_history(path: Path) -> Optional[Dict[str, list]]:
    """The loss history stored in a checkpoint, if it has one."""
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    history = blob.get("history")
    return {k: list(v) for k, v in history.items()} if history else None


def train(
    network: HMMSurrogatePINN,
    n_epochs: int = N_EPOCHS,
    n_points: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    boundary_weight: float = BOUNDARY_WEIGHT,
    divergence_weight: float = DIVERGENCE_WEIGHT,
    continuity_weight: float = CONTINUITY_WEIGHT,
    physics_ramp_frac: float = PHYSICS_RAMP_FRAC,
    device: torch.device = DEVICE,
    log_every: int = 100,
    lbfgs_steps: int = 0,
    lbfgs_dtype: torch.dtype = LBFGS_DTYPE,
    checkpoint_path: Optional[Path] = None,
    initial_history: Optional[Dict[str, list]] = None,
    initial_best_loss: float = float("inf"),
) -> Tuple[HMMSurrogatePINN, Dict[str, list]]:
    """
    Train the (ω, f)-conditioned core on the design-space rectangle.

    Phase 1 (Adam, cosine LR): each epoch draws :data:`N_BLOCKS` design points by
    jittered stratification, one per sub-block, each with its own box and its own
    ε(ω, f); interior physics ramps 0 → 1 over the first ``physics_ramp_frac`` of
    epochs, with the metal curl/divergence residual weighted per row by
    |ε(row)|^-1. Phase 2 (L-BFGS, ``lbfgs_dtype``): a fixed batch spanning the
    25 nodes of :data:`LBFGS_POINTS` with the metal curl weight raised to
    |ε|^-1/2. Returns the network (best iterate restored) and the loss history.

    ``initial_history`` / ``initial_best_loss`` continue a chunked run: the new
    epochs are numbered after the stored ones and the best-so-far bar is
    inherited, so a resumed chunk cannot checkpoint a *worse* iterate than the
    one it started from (its cosine schedule restarts at the full learning
    rate, so its early epochs are genuinely worse).
    """
    core = network.core
    cont_loss = TangentialContinuityLoss(offset=CONTINUITY_OFFSET_HAT)

    optimizer = torch.optim.Adam(core.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, n_epochs), eta_min=learning_rate * 1e-2
    )
    keys = ("epoch", "total", "curl", "div", "continuity", "boundary", "lr", "wall_s")
    history: Dict[str, list] = {k: list((initial_history or {}).get(k, [])) for k in keys}
    epoch0 = int(max(history["epoch"])) + 1 if history["epoch"] else 0
    # Cumulative wall clock across chunks, so the reported training time is the
    # whole run's and not just this process's.
    wall0 = float(history["wall_s"][-1]) if history["wall_s"] else 0.0
    best_loss = float(initial_best_loss)
    best_state = {k: v.detach().clone() for k, v in core.state_dict().items()}

    n_boundary = max(6 * N_BLOCKS, n_points // 2)
    n_interface = max(N_BLOCKS, n_points // 4)
    ramp_epochs = max(1, int(physics_ramp_frac * n_epochs))

    def compute_losses(batch: Dict[str, torch.Tensor], ramp: float = 1.0,
                       curl_power: float = CURL_POWER_ADAM):
        w_curl_m = batch["eps_scale_metal"] ** (-curl_power)
        w_div_m = batch["eps_scale_metal"] ** (-DIV_POWER)
        l_curl = curl_loss_weighted(
            core, batch["coords_air"], batch["cond_air"], batch["eps_air"]
        ) + curl_loss_weighted(
            core, batch["coords_metal"], batch["cond_metal"], batch["eps_metal"], w_curl_m
        )
        l_div = divergence_loss_weighted(
            core, batch["coords_air"], batch["cond_air"], batch["eps_air"]
        ) + divergence_loss_weighted(
            core, batch["coords_metal"], batch["cond_metal"], batch["eps_metal"], w_div_m
        )
        l_cont = cont_loss.compute(
            network=ConditionColumnNet(core, batch["cond_if"]),
            interface_coords=batch["iface"],
            normal_vectors=batch["normals"],
        )
        l_bc = torch.mean(
            (ConditionColumnNet(core, batch["cond_bc"])(batch["boundary"]) - batch["target"]) ** 2
        )
        total = (
            ramp * (l_curl + divergence_weight * l_div + continuity_weight * l_cont)
            + boundary_weight * l_bc
        )
        return total, l_curl, l_div, l_cont, l_bc

    core.train()
    t0 = time.perf_counter()
    for epoch in range(n_epochs):
        points = stratified_design_points()
        batch = sample_training_batch(n_points, n_boundary, n_interface, points, device=device)
        ramp = min(1.0, (epoch + 1) / ramp_epochs)
        optimizer.zero_grad(set_to_none=True)
        loss, l_curl, l_div, l_cont, l_bc = compute_losses(batch, ramp=ramp)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        history["epoch"].append(epoch0 + epoch)
        history["total"].append(loss_val)
        history["curl"].append(l_curl.item())
        history["div"].append(l_div.item())
        history["continuity"].append(l_cont.item())
        history["boundary"].append(l_bc.item())
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["wall_s"].append(wall0 + time.perf_counter() - t0)

        if ramp >= 1.0 and loss_val < best_loss and math.isfinite(loss_val):
            best_loss = loss_val
            best_state = {k: v.detach().clone() for k, v in core.state_dict().items()}
        if checkpoint_path is not None and epoch % log_every == 0 and math.isfinite(best_loss):
            _write_checkpoint(
                checkpoint_path, best_state, best_loss, f"adam:{epoch0 + epoch}", history
            )

        if epoch % log_every == 0 or epoch == n_epochs - 1:
            logger.info(
                "epoch %5d | total %.3e | curl %.3e | div %.3e | cont %.3e | bc %.3e | "
                "lr %.2e | %.0fs",
                epoch, loss_val, l_curl.item(), l_div.item(), l_cont.item(), l_bc.item(),
                optimizer.param_groups[0]["lr"], time.perf_counter() - t0,
            )

    if lbfgs_steps > 0:
        core.load_state_dict(best_state)
        if lbfgs_dtype == torch.float64:
            core.to(torch.float64)
        logger.info("L-BFGS phase (%s) on %d fixed design points",
                    lbfgs_dtype, len(LBFGS_POINTS))
        batch = sample_training_batch(
            LBFGS_POINTS_FACTOR * n_points,
            LBFGS_POINTS_FACTOR * n_boundary,
            LBFGS_POINTS_FACTOR * n_interface,
            LBFGS_POINTS,
            device=device,
            dtype=lbfgs_dtype,
        )
        lbfgs = torch.optim.LBFGS(
            core.parameters(), lr=1.0, max_iter=20, history_size=50,
            tolerance_grad=1e-12, tolerance_change=1e-14, line_search_fn="strong_wolfe",
        )
        parts: Dict[str, float] = {}

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            loss, l_curl, l_div, l_cont, l_bc = compute_losses(
                batch, curl_power=CURL_POWER_LBFGS
            )
            loss.backward()
            parts.update(
                curl=l_curl.item(), div=l_div.item(), cont=l_cont.item(), bc=l_bc.item()
            )
            return loss

        for step in range(lbfgs_steps):
            loss_val = float(lbfgs.step(closure).detach())
            history["epoch"].append(epoch0 + n_epochs + step)
            history["total"].append(loss_val)
            history["curl"].append(parts["curl"])
            history["div"].append(parts["div"])
            history["continuity"].append(parts["cont"])
            history["boundary"].append(parts["bc"])
            history["lr"].append(float("nan"))
            history["wall_s"].append(wall0 + time.perf_counter() - t0)
            if loss_val < best_loss and math.isfinite(loss_val):
                best_loss = loss_val
                best_state = {k: v.detach().clone() for k, v in core.state_dict().items()}
                if checkpoint_path is not None:
                    _write_checkpoint(
                        checkpoint_path, best_state, best_loss, f"lbfgs:{step}", history
                    )
            logger.info(
                "lbfgs %3d | total %.3e | curl %.3e | div %.3e | cont %.3e | bc %.3e | %.0fs",
                step, loss_val, parts["curl"], parts["div"], parts["cont"], parts["bc"],
                time.perf_counter() - t0,
            )
            if not math.isfinite(loss_val):
                logger.warning("L-BFGS produced a non-finite loss; stopping refinement")
                break

    core.load_state_dict(best_state)
    network.to(torch.float32)
    # Always write the final checkpoint, even if this chunk never beat the
    # inherited best: otherwise its slice of the history would be lost.
    if checkpoint_path is not None and math.isfinite(best_loss):
        _write_checkpoint(checkpoint_path, best_state, best_loss, "final", history)
    logger.info("restored best weights (loss %.3e)", best_loss)
    return network, history


# --------------------------------------------------------------------------- validation
def _relative_l2(pred: torch.Tensor, ref: torch.Tensor) -> float:
    return (
        torch.linalg.vector_norm(pred - ref) / torch.linalg.vector_norm(ref).clamp_min(1e-30)
    ).item()


def estimate_k_spp(
    net3: nn.Module, omega: float, fill: float, n_line: int = 512, device: torch.device = DEVICE
) -> Dict[str, float]:
    """
    Re k_spp from the phase slope of H_y along x (and Im k_spp from ln|H_y|).

    The reporting counterpart of the differentiable :func:`k_spp_from_network`;
    this one uses ``numpy.unwrap`` and is a *measurement*, not part of any
    gradient path. The probe line sits at a quarter of the design point's own
    air-side box height, so it is in the near field everywhere in the rectangle.
    """
    k_ref = mode_constants(omega, fill)[0]
    x_max, y_max, _, z_max = domain_si(omega, fill)
    x = torch.linspace(0.0, x_max, n_line, device=device)
    coords = torch.stack(
        [x, torch.full_like(x, y_max / 2), torch.full_like(x, 0.25 * z_max)], dim=1
    )
    with torch.no_grad():
        _, H = to_complex(net3(coords))
    hy = H[:, 1].cpu().numpy().astype(np.complex128)
    xs = coords[:, 0].cpu().numpy().astype(np.float64)
    k_fit = float(np.polyfit(xs, np.unwrap(np.angle(hy)), 1)[0])
    k_imag_fit = -float(np.polyfit(xs, np.log(np.abs(hy) + 1e-30), 1)[0])
    out = {
        "k_spp_fit": k_fit,
        "k_spp_rel_error": float(abs(k_fit - k_ref.real) / k_ref.real),
        "k_spp_analytical": float(k_ref.real),
        "n_eff_fit": k_fit / k0_of(omega),
        "n_eff_analytical": float(k_ref.real) / k0_of(omega),
        "k_spp_imag_fit": k_imag_fit,
        "k_spp_imag_analytical": float(k_ref.imag),
    }
    if k_ref.imag != 0:
        out["k_spp_imag_rel_error"] = float(abs(k_imag_fit - k_ref.imag) / abs(k_ref.imag))
    return out


def fit_decay_constants(
    net3: nn.Module, omega: float, fill: float, n_line: int = 200,
    guard: float = VAL_GUARD_HAT, device: torch.device = DEVICE,
) -> Dict[str, float]:
    """κ fits from ln|H_y| vs z on each side of the interface, over the point's own box."""
    _, kappa_d, kappa_m = mode_constants(omega, fill)
    x_max, y_max, z_min, z_max = domain_si(omega, fill)
    guard_si = guard / k0_of(omega)
    x = 0.25 * x_max
    out: Dict[str, float] = {}
    for side, z_lo, z_hi, kappa_ref, sign, name in [
        ("air", guard_si, 0.9 * z_max, kappa_d.real, -1.0, "kappa_d"),
        ("metal", 0.95 * z_min, -guard_si, kappa_m.real, 1.0, "kappa_m"),
    ]:
        z = torch.linspace(z_lo, z_hi, n_line, device=device)
        coords = torch.stack([torch.full_like(z, x), torch.full_like(z, y_max / 2), z], dim=1)
        with torch.no_grad():
            _, H = to_complex(net3(coords))
        log_hy = np.log(np.abs(H[:, 1].cpu().numpy().astype(np.complex128)) + 1e-30)
        slope = float(np.polyfit(z.cpu().numpy().astype(np.float64), log_hy, 1)[0])
        kappa_fit = sign * slope
        out[f"{name}_fit"] = kappa_fit
        out[f"{name}_fit_rel_error"] = float(abs(kappa_fit - kappa_ref) / kappa_ref)
        out[f"{name}_analytical"] = float(kappa_ref)
        out[f"decay_sign_correct_{side}"] = float(kappa_fit > 0)
    return out


def continuity_residuals(
    net3: nn.Module, omega: float, fill: float, n_points: int = 2000,
    offset: float = VAL_GUARD_HAT, device: torch.device = DEVICE,
) -> Dict[str, float]:
    """Tangential continuity residual at z = ±offset/k₀(ω), relative to the field RMS."""
    coords_hat, normals = sample_interface_hat(n_points, omega, fill, device=device)
    coords = coords_hat / k0_of(omega)
    off = offset / k0_of(omega)
    with torch.no_grad():
        E_p, H_p = to_complex(net3(coords + off * normals))
        E_m, H_m = to_complex(net3(coords - off * normals))
        n = normals.to(E_p.dtype)
        res_E = torch.linalg.vector_norm(torch.linalg.cross(n, E_p - E_m, dim=1), dim=1)
        res_H = torch.linalg.vector_norm(torch.linalg.cross(n, H_p - H_m, dim=1), dim=1)
        E_rms = torch.sqrt(
            torch.mean(torch.sum(E_p.abs() ** 2 + E_m.abs() ** 2, dim=1) / 2)
        ).clamp_min(1e-30)
        H_rms = torch.sqrt(
            torch.mean(torch.sum(H_p.abs() ** 2 + H_m.abs() ** 2, dim=1) / 2)
        ).clamp_min(1e-30)
    return {
        "continuity_E_rel": (torch.sqrt(torch.mean(res_E**2)) / E_rms).item(),
        "continuity_H_rel": (torch.sqrt(torch.mean(res_H**2)) / H_rms).item(),
    }


def validate_at_point(
    model: nn.Module, omega: float, fill: float, n_points: int = 6000,
    device: torch.device = DEVICE,
) -> Dict[str, float]:
    """
    SI-unit validation of ``model.at_point(omega, fill)`` on fresh stratified points.

    Keys: rel L2 vs the analytical mode (overall and per half-space), SI curl
    residual RMS / (k₀·RMS field) per half-space, k_spp phase/amplitude fits,
    κ_d/κ_m decay fits, tangential-continuity residuals, the material at this
    design point.
    """
    net3 = model.at_point(omega, fill)
    net3.eval()
    k0 = k0_of(omega)
    eps_t, eps_n = hmm_eps(omega, fill)
    coords = sample_collocation_si(n_points, omega, fill, guard=VAL_GUARD_HAT, device=device)
    fields = net3(coords)
    E, H = to_complex(fields)

    maxwell = MaxwellEquations(omega, mu0=MU0, eps0=EPS0)
    curl_E = maxwell.curl_operator(E, coords)
    curl_H = maxwell.curl_operator(H, coords)
    z = coords[:, 2].detach()
    metal = z < 0
    eps_m_row = torch.tensor([eps_t, eps_t, eps_n], dtype=E.dtype, device=device).view(1, 3)
    eps_d_row = torch.full((1, 3), complex(EPS_D), dtype=E.dtype, device=device)
    eps_diag = torch.where(metal.view(-1, 1), eps_m_row, eps_d_row)
    res_E = curl_E - 1j * omega * MU0 * H
    res_H = curl_H + 1j * omega * EPS0 * eps_diag * E

    with torch.no_grad():
        ref = analytical_fields_si(coords.detach(), omega, fill)
        E_ref, H_ref = to_complex(ref.to(fields.dtype))

        metrics: Dict[str, float] = {
            "omega": float(omega),
            "fill_fraction": float(fill),
            "omega_over_omega_ref": float(omega / OMEGA_REF),
            "omega_hat": omega_hat(omega),
            "fill_hat": fill_hat(fill),
            "wavelength_nm": float(2 * np.pi * C0 / omega * 1e9),
            "eps_t_re": eps_t.real, "eps_t_im": eps_t.imag,
            "eps_n_re": eps_n.real, "eps_n_im": eps_n.imag,
            "rel_l2_E": _relative_l2(E, E_ref),
            "rel_l2_H": _relative_l2(H, H_ref),
            "rel_l2_total": _relative_l2(fields, ref.to(fields.dtype)),
        }
        for side, mask in (("air", ~metal), ("metal", metal)):
            metrics[f"rel_l2_E_{side}"] = _relative_l2(E[mask], E_ref[mask])
            metrics[f"rel_l2_H_{side}"] = _relative_l2(H[mask], H_ref[mask])
            E_rms = torch.sqrt(torch.mean(torch.sum(E[mask].abs() ** 2, 1))).clamp_min(1e-30)
            H_rms = torch.sqrt(torch.mean(torch.sum(H[mask].abs() ** 2, 1))).clamp_min(1e-30)
            rE = torch.linalg.vector_norm(res_E[mask], dim=1)
            rH = torch.linalg.vector_norm(res_H[mask], dim=1)
            metrics[f"curl_E_residual_rel_{side}"] = (
                torch.sqrt(torch.mean(rE**2)) / (k0 * E_rms)
            ).item()
            metrics[f"curl_H_residual_rel_{side}"] = (
                torch.sqrt(torch.mean(rH**2)) / (k0 * H_rms)
            ).item()

        E_rms_all = torch.sqrt(torch.mean(torch.sum(E.abs() ** 2, 1))).clamp_min(1e-30)
        H_rms_all = torch.sqrt(torch.mean(torch.sum(H.abs() ** 2, 1))).clamp_min(1e-30)
        metrics["impedance_ratio"] = ((E_rms_all / H_rms_all) / ETA0).item()

    metrics.update(estimate_k_spp(net3, omega, fill, device=device))
    metrics.update(fit_decay_constants(net3, omega, fill, device=device))
    metrics.update(continuity_residuals(net3, omega, fill, device=device))
    return metrics


def point_key(omega: float, fill: float) -> str:
    """Stable dict key for a design point: ``"w0.8500_f0.1875"``."""
    return f"w{omega / OMEGA_REF:.4f}_f{fill:.4f}"


def validate_grid(
    model: nn.Module,
    points: Sequence[Tuple[float, float]] = (),
    n_points: int = 6000,
    device: torch.device = DEVICE,
) -> Dict[str, Dict[str, float]]:
    """Run :func:`validate_at_point` at each held-out design point."""
    points = points or VALIDATION_POINTS
    return {
        point_key(w, f): validate_at_point(model, w, f, n_points, device) for w, f in points
    }


def summarise(per_point: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Worst/median statistics over the held-out grid, plus the success tier."""
    rel = [max(m["rel_l2_E"], m["rel_l2_H"]) for m in per_point.values()]
    k_err = [m["k_spp_rel_error"] for m in per_point.values()]
    kappa_err = [
        max(m["kappa_d_fit_rel_error"], m["kappa_m_fit_rel_error"]) for m in per_point.values()
    ]
    bound = all(
        m["decay_sign_correct_air"] > 0 and m["decay_sign_correct_metal"] > 0
        for m in per_point.values()
    )
    summary = {
        "n_points_validated": float(len(per_point)),
        "f_min": float(F_MIN), "f_max": float(F_MAX),
        "omega_min": float(OMEGA_MIN), "omega_max": float(OMEGA_MAX),
        "omega_min_over_omega_ref": float(OMEGA_MIN / OMEGA_REF),
        "omega_max_over_omega_ref": float(OMEGA_MAX / OMEGA_REF),
        "worst_rel_l2": float(max(rel)),
        "median_rel_l2": float(np.median(rel)),
        "worst_k_spp_rel_error": float(max(k_err)),
        "median_k_spp_rel_error": float(np.median(k_err)),
        "worst_kappa_rel_error": float(max(kappa_err)),
        "bound_mode_everywhere": float(bound),
    }
    summary["success_tier"] = success_tier(summary)
    return summary


def success_tier(summary: Dict[str, float]) -> str:
    """Tiers: minimum (bound everywhere, rel L2 < 0.5), target (rel L2 < 0.1 and
    k_spp < 1 % everywhere), stretch (rel L2 < 0.03 and k_spp < 0.5 %)."""
    rel = summary["worst_rel_l2"]
    k_err = summary["worst_k_spp_rel_error"]
    bound = summary["bound_mode_everywhere"] > 0
    if bound and rel < 0.03 and k_err < 0.005:
        return "stretch"
    if bound and rel < 0.1 and k_err < 0.01:
        return "target"
    if bound and rel < 0.5:
        return "minimum"
    return "not met"


# --------------------------------------------------------------------------- surface scan
def design_space_maps(
    model: nn.Module,
    n_omega: int = 13,
    n_fill: int = 9,
    n_points: int = 1500,
    device: torch.device = DEVICE,
) -> Dict[str, np.ndarray]:
    """
    Sweep the whole rectangle and return the arrays the headline figure needs.

    Cheap relative to training (no autograd, no Maxwell residuals): per grid
    point a 512-sample probe line for the k_spp phase fit and ``n_points``
    stratified samples for the field rel L2.
    """
    omegas = np.linspace(OMEGA_MIN, OMEGA_MAX, n_omega)
    fills = np.linspace(F_MIN, F_MAX, n_fill)
    k_pinn = np.zeros((n_omega, n_fill))
    k_exact = np.zeros((n_omega, n_fill))
    rel_l2 = np.zeros((n_omega, n_fill))
    for i, w in enumerate(omegas):
        for j, f in enumerate(fills):
            net3 = model.at_point(float(w), float(f))
            net3.eval()
            fit = estimate_k_spp(net3, float(w), float(f), device=device)
            k_pinn[i, j] = fit["k_spp_fit"]
            k_exact[i, j] = fit["k_spp_analytical"]
            coords = sample_collocation_hat(
                n_points, float(w), float(f), guard=VAL_GUARD_HAT, device=device
            ) / k0_of(float(w))
            with torch.no_grad():
                pred = net3(coords)
                ref = analytical_fields_si(coords, float(w), float(f)).to(pred.dtype)
            rel_l2[i, j] = _relative_l2(pred, ref)
    return {
        "omegas": omegas,
        "fills": fills,
        "k_pinn": k_pinn,
        "k_exact": k_exact,
        "k_rel_error": np.abs(k_pinn - k_exact) / np.abs(k_exact),
        "rel_l2": rel_l2,
    }


# --------------------------------------------------------------------------- differentiable k_spp
def _lstsq_slope(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Slope of the least-squares line ``y ≈ a + b x``, via ``torch.linalg.lstsq``.

    Differentiable in both arguments (the solution of an over-determined
    full-rank system is a smooth function of the data), which is the whole
    reason the phase fit is written this way rather than with ``numpy.polyfit``.
    """
    design = torch.stack([torch.ones_like(x), x], dim=1)
    solution = torch.linalg.lstsq(design, y.unsqueeze(1)).solution
    return solution[1, 0]


def k_spp_from_network(
    core: nn.Module,
    omega: float,
    f_hat: torch.Tensor,
    n_line: int = 512,
    z_fraction: float = 0.25,
    device: torch.device = DEVICE,
) -> torch.Tensor:
    """
    **Differentiable** Re k_spp (1/m) recovered from the network's own H_y.

    The estimator, and why each step keeps the gradient:

    1. Sample H_y on a probe line along x at fixed y and ẑ = ``z_fraction``·ẑ_max.
       The line's *geometry* is computed from ``f_hat.detach()`` — where we
       choose to look is not part of the definition of k_spp, so no gradient is
       taken through it; the gradient that matters flows through the f̂ **input
       column**, which reaches both the MLP and the adapter's ε_n(ω, f) divisor.
    2. Unwrap the phase by cumulatively summing the principal arguments of the
       successive ratios ``H_y(x_{j+1})/H_y(x_j)``. ``numpy.unwrap`` is a
       branch-counting operation on detached numbers; this is the same thing
       written out of differentiable pieces (``torch.angle`` and ``cumsum``).
       It is exact whenever the per-sample phase advance stays below π, which
       here is ~0.03 rad.
    3. Fit the slope by a least-squares solve (:func:`_lstsq_slope`).

    Because the abscissa is x̂ = k₀x, the slope *is* n_eff = Re k_spp/k₀; the
    return value multiplies it back to SI.
    """
    fill = fill_from_hat(float(f_hat.detach().reshape(())))
    x_max, y_max, _, z_max = domain_hat(omega, fill)
    dtype = next(core.parameters()).dtype
    x_hat = torch.linspace(0.0, x_max, n_line, device=device, dtype=dtype)
    coords = torch.stack(
        [x_hat, torch.full_like(x_hat, y_max / 2), torch.full_like(x_hat, z_fraction * z_max)],
        dim=1,
    )
    w_col = torch.full_like(x_hat, omega_hat(omega)).unsqueeze(1)
    f_col = f_hat.to(dtype).reshape(1, 1).expand(n_line, 1)
    out = core(torch.cat([coords, w_col, f_col], dim=1))
    h_y = torch.complex(out[:, 4, 0], out[:, 4, 1])

    ratio = h_y[1:] / h_y[:-1]
    d_phi = torch.angle(ratio)
    phi = torch.cat([torch.zeros(1, device=device, dtype=d_phi.dtype), torch.cumsum(d_phi, 0)])
    n_eff = _lstsq_slope(x_hat, phi)
    return n_eff * k0_of(omega)


def closed_form_fill_for_index(
    omega: float, n_target: float, f_lo: float = None, f_hi: float = None, tol: float = 1e-10
) -> float:
    """
    The same problem solved analytically: the f with ``Re k_spp/k₀ = n_target``.

    A bisection on ``n_eff(f) − n_target`` using ``hmm_permittivities`` and
    ``MetamaterialProperties`` directly. n_eff decreases monotonically with f
    over this rectangle (more metal ⇒ more negative ε_t ⇒ the mode is pushed
    towards the light line), so the root is unique and bracketing is trivial.
    """
    lo = F_MIN if f_lo is None else float(f_lo)
    hi = F_MAX if f_hi is None else float(f_hi)

    def residual(f: float) -> float:
        _, n_eff = is_bound(omega, f)
        return n_eff - n_target

    r_lo, r_hi = residual(lo), residual(hi)
    if r_lo * r_hi > 0:
        raise ValueError(
            f"n_target={n_target:.4f} is outside [{residual(hi) + n_target:.4f}, "
            f"{residual(lo) + n_target:.4f}] at this ω"
        )
    for _ in range(200):
        if hi - lo <= tol:
            break
        mid = 0.5 * (lo + hi)
        r_mid = residual(mid)
        if (r_lo < 0.0) != (r_mid < 0.0):
            hi = mid
        else:
            lo, r_lo = mid, r_mid
    return 0.5 * (lo + hi)


def inverse_design_demo(
    model: nn.Module,
    targets: Sequence[Tuple[float, float]] = (),
    n_steps: int = 200,
    lr: float = 0.05,
    device: torch.device = DEVICE,
) -> List[Dict[str, float]]:
    """
    Inverse design **through the surrogate**: find f achieving a target n_eff at ω.

    For each ``(omega, n_target)`` the optimiser descends
    ``(n_eff_PINN(f) − n_target)²`` in the unconstrained variable u with
    ``f̂ = tanh(u)`` (so f can never leave the rectangle the network was trained
    on), where n_eff_PINN comes from :func:`k_spp_from_network` — i.e. every
    gradient is autograd through the trained network, not through any formula
    for k_spp.

    Each result is cross-checked against :func:`closed_form_fill_for_index`.
    **The closed form exists here, so this validates the method rather than
    being necessary.** Its value is that the identical loop runs where no
    closed form does — a finite slab, a real layer stack, a patterned surface —
    which is the next experiment.

    Returns one record per target with the target, the f the surrogate found,
    the closed-form f, their difference, and the n_eff each actually delivers.
    """
    core = model.core
    core.eval()
    records: List[Dict[str, float]] = []
    for omega, n_target in targets:
        u = torch.zeros((), device=device, requires_grad=True)  # start at the centre, f̂ = 0
        opt = torch.optim.Adam([u], lr=lr)
        history = []
        for _ in range(n_steps):
            opt.zero_grad(set_to_none=True)
            f_hat = torch.tanh(u)
            k = k_spp_from_network(core, omega, f_hat, device=device)
            n_eff = k / k0_of(omega)
            loss = (n_eff - n_target) ** 2
            loss.backward()
            opt.step()
            history.append(float(n_eff.detach()))
        f_found = fill_from_hat(float(torch.tanh(u).detach()))
        f_closed = closed_form_fill_for_index(omega, n_target)
        _, n_at_found = is_bound(omega, f_found)
        _, n_at_closed = is_bound(omega, f_closed)
        with torch.no_grad():
            n_pinn_at_found = float(
                k_spp_from_network(core, omega, torch.tanh(u).detach(), device=device)
            ) / k0_of(omega)
        records.append(
            {
                "omega": float(omega),
                "omega_over_omega_ref": float(omega / OMEGA_REF),
                "wavelength_nm": float(2 * np.pi * C0 / omega * 1e9),
                "n_eff_target": float(n_target),
                "fill_fraction_pinn": float(f_found),
                "fill_fraction_closed_form": float(f_closed),
                "fill_fraction_abs_error": float(abs(f_found - f_closed)),
                "fill_fraction_rel_error": float(abs(f_found - f_closed) / f_closed),
                "n_eff_true_at_pinn_f": float(n_at_found),
                "n_eff_true_at_closed_form_f": float(n_at_closed),
                "n_eff_error_at_pinn_f": float(abs(n_at_found - n_target)),
                "n_eff_pinn_reports_at_its_f": float(n_pinn_at_found),
                "n_eff_history": history,
            }
        )
        logger.info(
            "inverse design ω/ω_ref %.4f: target n_eff %.4f -> f_PINN %.5f, "
            "f_closed %.5f (Δf %.2e), true n_eff at f_PINN %.4f",
            omega / OMEGA_REF, n_target, f_found, f_closed,
            abs(f_found - f_closed), n_at_found,
        )
    return records


def default_inverse_targets() -> List[Tuple[float, float]]:
    """
    Three (ω, n_eff) problems whose answers are *interior* to the f range.

    Each target is the exact analytical n_eff of a chosen ground-truth fill
    fraction (0.20, 0.275, 0.35) at a chosen ω, so the correct answer is known
    and comfortably away from the bracket ends.
    """
    out = []
    for w_hat, f_true in ((-0.6, 0.20), (0.0, 0.275), (0.6, 0.35)):
        omega = omega_from_hat(w_hat)
        _, n_eff = is_bound(omega, f_true)
        out.append((omega, n_eff))
    return out


# --------------------------------------------------------------------------- plots
def plot_k_surface(maps: Dict[str, np.ndarray], out_dir: Path = FIGURES_DIR) -> str:
    """
    Headline: the k_spp(ω, f) surface — analytical, PINN, and the error map.

    This is the deliverable that shows a *surrogate* rather than a single
    solution: one network, evaluated over a whole 2-D design space.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    x = maps["omegas"] / OMEGA_REF
    y = maps["fills"]
    X, Y = np.meshgrid(x, y, indexing="ij")
    k_exact = maps["k_exact"] / 1e7
    k_pinn = maps["k_pinn"] / 1e7
    err = 100.0 * maps["k_rel_error"]
    vmin, vmax = min(k_exact.min(), k_pinn.min()), max(k_exact.max(), k_pinn.max())

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.9))
    for ax, data, title, kw in [
        (axes[0], k_exact, r"Analytical  Re $k_{\rm spp}(\omega, f)$", dict(vmin=vmin, vmax=vmax, cmap="viridis")),
        (axes[1], k_pinn, r"PINN surrogate  Re $k_{\rm spp}(\omega, f)$", dict(vmin=vmin, vmax=vmax, cmap="viridis")),
        (axes[2], err, r"$|k_{\rm PINN} - k_{\rm exact}| / k_{\rm exact}$", dict(cmap="magma")),
    ]:
        im = ax.pcolormesh(X, Y, data, shading="auto", **kw)
        if data is not err:
            # Iso-k_spp contours of the exact surface, on both k panels only:
            # on the error panel they would compete with the node markers.
            cs = ax.contour(X, Y, k_exact, levels=8, colors="w", linewidths=0.7, alpha=0.75)
            ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")
        ax.set_xlabel(r"$\omega / \omega_{\rm ref}$   ($\omega_{\rm ref} = 2\pi c/633$ nm)")
        ax.set_title(title, fontsize=11)
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("%" if data is err else r"$10^7$ m$^{-1}$", fontsize=9)
    for w, f in VALIDATION_POINTS:
        axes[2].plot(w / OMEGA_REF, f, "o", ms=5, mfc="none", mec="deepskyblue", mew=1.4)
    for w, f in LBFGS_POINTS:
        axes[2].plot(w / OMEGA_REF, f, "+", ms=6, color="0.75", mew=1.2)
    axes[0].set_ylabel("metal fill fraction $f$")
    fig.suptitle(
        "One material-conditioned PINN over the whole (ω, f) design space of an "
        "Ag/silica multilayer\n"
        "grey +: L-BFGS refinement nodes   ·   blue ○: held-out validation points",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    p = out_dir / "k_spp_surface.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


def plot_error_maps(
    maps: Dict[str, np.ndarray],
    per_point: Dict[str, Dict[str, float]],
    out_dir: Path = FIGURES_DIR,
) -> str:
    """Field rel-L2 over the design space, plus the held-out points' own errors."""
    out_dir.mkdir(parents=True, exist_ok=True)
    x = maps["omegas"] / OMEGA_REF
    y = maps["fills"]
    X, Y = np.meshgrid(x, y, indexing="ij")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9))
    im = axes[0].pcolormesh(
        X, Y, np.log10(np.maximum(maps["rel_l2"], 1e-8)), shading="auto", cmap="magma"
    )
    cb = fig.colorbar(im, ax=axes[0])
    cb.set_label(r"$\log_{10}$ rel $L_2$ (E, H together)")
    for w, f in LBFGS_POINTS:
        axes[0].plot(w / OMEGA_REF, f, "+", ms=6, color="0.8", mew=1.2)
    for w, f in VALIDATION_POINTS:
        axes[0].plot(w / OMEGA_REF, f, "o", ms=5, mfc="none", mec="deepskyblue", mew=1.4)
    axes[0].set_xlabel(r"$\omega / \omega_{\rm ref}$")
    axes[0].set_ylabel("metal fill fraction $f$")
    axes[0].set_title("Field error over the design space\n"
                      "(grey +: refinement nodes, blue ○: held-out)", fontsize=10)

    vals = list(per_point.values())
    f_of = np.array([m["fill_fraction"] for m in vals])
    rel = np.array([max(m["rel_l2_E"], m["rel_l2_H"]) for m in vals])
    w_of = np.array([m["omega_over_omega_ref"] for m in vals])
    for f_val in np.unique(f_of):
        sel = np.argsort(w_of[f_of == f_val])
        axes[1].semilogy(
            w_of[f_of == f_val][sel], rel[f_of == f_val][sel], "o-", ms=5,
            label=f"f = {f_val:.4f}",
        )
    axes[1].set_xlabel(r"$\omega / \omega_{\rm ref}$")
    axes[1].set_ylabel(r"held-out rel $L_2$  (max of E, H)")
    axes[1].set_title("Held-out grid, by fill fraction", fontsize=10)
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "error_maps.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


def plot_fill_slices(
    model: nn.Module,
    maps: Dict[str, np.ndarray],
    omegas: Sequence[float] = (),
    out_dir: Path = FIGURES_DIR,
    device: torch.device = DEVICE,
) -> str:
    """
    ``k_spp`` vs ``f`` at fixed ω — the cut that shows the second axis working.

    If the surrogate had learned only the ω dependence (a defensible failure
    mode, since ω moves k_spp further than f does) these curves would be flat.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    omegas = omegas or (OMEGA_MIN, OMEGA_MID, OMEGA_MAX)
    f_dense = np.linspace(F_MIN, F_MAX, 121)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    colours = plt.cm.viridis(np.linspace(0.1, 0.85, len(omegas)))
    for omega, colour in zip(omegas, colours, strict=True):
        k_curve = np.array([mode_constants(float(omega), float(f))[0].real for f in f_dense])
        axes[0].plot(f_dense, k_curve / 1e7, "-", color=colour, lw=1.8,
                     label=rf"analytical, $\omega/\omega_{{\rm ref}}$ = {omega / OMEGA_REF:.3f}")
        f_pts = np.linspace(F_MIN, F_MAX, 11)
        k_pts = []
        for f in f_pts:
            net3 = model.at_point(float(omega), float(f))
            net3.eval()
            k_pts.append(estimate_k_spp(net3, float(omega), float(f), device=device)["k_spp_fit"])
        axes[0].plot(f_pts, np.array(k_pts) / 1e7, "o", color=colour, ms=6, mfc="none", mew=1.6)
        axes[1].semilogy(
            f_pts, np.abs(np.array(k_pts) - np.interp(f_pts, f_dense, k_curve)) /
            np.interp(f_pts, f_dense, k_curve), "o-", color=colour, ms=5,
            label=rf"$\omega/\omega_{{\rm ref}}$ = {omega / OMEGA_REF:.3f}",
        )
    axes[0].set_xlabel("metal fill fraction $f$")
    axes[0].set_ylabel(r"Re $k_{\rm spp}$  [$10^7$ m$^{-1}$]")
    axes[0].set_title("Fixed-ω slices: PINN points on the analytical curve", fontsize=10)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("metal fill fraction $f$")
    axes[1].set_ylabel(r"relative error in Re $k_{\rm spp}$")
    axes[1].set_title("Same slices, error", fontsize=10)
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "fill_slices.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


def plot_inverse_design(
    records: Sequence[Dict[str, float]], out_dir: Path = FIGURES_DIR
) -> str:
    """Convergence of the through-the-surrogate inverse design, vs the closed form."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    colours = plt.cm.plasma(np.linspace(0.15, 0.8, len(records)))
    for rec, colour in zip(records, colours, strict=True):
        hist = np.asarray(rec["n_eff_history"], dtype=float)
        axes[0].plot(hist, color=colour, lw=1.5,
                     label=rf"$\omega/\omega_{{\rm ref}}$ = {rec['omega_over_omega_ref']:.3f}")
        axes[0].axhline(rec["n_eff_target"], color=colour, ls="--", lw=1.0)
    axes[0].set_xlabel("Adam step (on the fill fraction)")
    axes[0].set_ylabel(r"$n_{\rm eff}$ reported by the surrogate")
    axes[0].set_title("Inverse design through the trained PINN\n"
                      "(dashed: targets)", fontsize=10)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    idx = np.arange(len(records))
    width = 0.36
    axes[1].bar(idx - width / 2, [r["fill_fraction_pinn"] for r in records], width,
                label="f found through the PINN", color="#4c72b0")
    axes[1].bar(idx + width / 2, [r["fill_fraction_closed_form"] for r in records], width,
                label="f from the closed-form root find", color="#dd8452")
    for i, r in enumerate(records):
        axes[1].text(i, max(r["fill_fraction_pinn"], r["fill_fraction_closed_form"]) + 0.006,
                     rf"$\Delta f$ = {r['fill_fraction_abs_error']:.1e}",
                     ha="center", fontsize=8)
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels(
        [rf"$\omega/\omega_{{\rm ref}}$ = {r['omega_over_omega_ref']:.3f}" + "\n"
         + rf"$n_{{\rm eff}}^*$ = {r['n_eff_target']:.4f}" for r in records], fontsize=8
    )
    axes[1].set_ylabel("metal fill fraction $f$")
    axes[1].set_ylim(F_MIN - 0.02, F_MAX + 0.04)
    axes[1].set_title("Surrogate answer vs closed form\n"
                      "(the closed form exists here; that is the point of the check)",
                      fontsize=10)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "inverse_design.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


def plot_field_maps(
    model: nn.Module, omega: float, fill: float, out_dir: Path = FIGURES_DIR,
    device: torch.device = DEVICE,
) -> str:
    """Re H_y in the x-z plane at one design point: PINN vs analytical vs difference."""
    out_dir.mkdir(parents=True, exist_ok=True)
    net3 = model.at_point(omega, fill)
    net3.eval()
    nm = 1e9
    nx, nz = 160, 120
    x_max, y_max, z_min, z_max = domain_si(omega, fill)
    x = torch.linspace(0.0, x_max, nx, device=device)
    zv = torch.linspace(z_min, z_max, nz, device=device)
    X, Z = torch.meshgrid(x, zv, indexing="ij")
    coords = torch.stack(
        [X.flatten(), torch.full_like(X.flatten(), y_max / 2), Z.flatten()], dim=1
    )
    with torch.no_grad():
        pred = net3(coords)[:, 4, 0].reshape(nx, nz).cpu().numpy()
        ref = analytical_fields_si(coords, omega, fill)[:, 4, 0].reshape(nx, nz).cpu().numpy()
    Xn, Zn = X.cpu().numpy() * nm, Z.cpu().numpy() * nm
    vmax = np.abs(ref).max()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for ax, (data, title, lim) in zip(
        axes,
        [
            (pred, "PINN Re H_y [A/m]", vmax),
            (ref, "Analytical Re H_y [A/m]", vmax),
            (pred - ref, "Difference", np.abs(pred - ref).max() + 1e-30),
        ],
        strict=True,
    ):
        im = ax.pcolormesh(Xn, Zn, data, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
        ax.axhline(0.0, color="k", lw=0.8, ls="--")
        ax.set_title(title)
        ax.set_xlabel("x [nm]")
        fig.colorbar(im, ax=ax)
    axes[0].set_ylabel("z [nm]")
    eps_t, eps_n = hmm_eps(omega, fill)
    fig.suptitle(
        f"HMM SPP at ω = {omega / OMEGA_REF:.4f} ω_ref "
        f"(λ₀ = {2 * np.pi * C0 / omega * 1e9:.0f} nm), f = {fill:.3f}: "
        f"ε_t = {eps_t:.3f}, ε_n = {eps_n:.3f}; x-z plane, interface at z = 0"
    )
    fig.tight_layout()
    p = out_dir / f"field_maps_w{omega / OMEGA_REF:.4f}_f{fill:.4f}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


def plot_history(history: Dict[str, list], out_dir: Path = FIGURES_DIR) -> str:
    """
    Training-curve figure, split into its two phases.

    Thousands of Adam epochs and ~10² L-BFGS steps share no useful x-axis: on
    one axis the refinement — where most of the final accuracy is won —
    collapses into a sliver at the right edge. The phases are therefore drawn
    side by side, identified by the learning rate (L-BFGS rows record NaN).

    The run is trained in wall-clock-limited chunks, each resumed from the
    previous checkpoint with a fresh cosine cycle; the dotted verticals mark
    those **warm restarts**, which is why the Adam curve has a sawtooth.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = ("total", "curl", "div", "continuity", "boundary")
    epoch = np.asarray(history["epoch"], dtype=float)
    lr = np.asarray(history["lr"], dtype=float)
    is_lbfgs = np.isnan(lr)
    n_adam = int((~is_lbfgs).sum())
    adam_lr = lr[~is_lbfgs]
    adam_epoch = epoch[~is_lbfgs]
    restarts = adam_epoch[1:][adam_lr[1:] > adam_lr[:-1] * 1.5] if n_adam > 1 else np.array([])

    if n_adam == 0 or int(is_lbfgs.sum()) == 0:
        fig, axis = plt.subplots(figsize=(8, 5))
        panels = [(axis, np.ones_like(epoch, dtype=bool), "epoch", "training", False)]
    else:
        fig, ax_pair = plt.subplots(
            1, 2, figsize=(12, 5), sharey=True,
            gridspec_kw={"width_ratios": [2.2, 1]},
        )
        panels = [
            (ax_pair[0], ~is_lbfgs, "Adam epoch", "Phase 1: Adam (cosine LR)", False),
            (ax_pair[1], is_lbfgs, "L-BFGS step", "Phase 2: float64 L-BFGS refinement", True),
        ]

    for ax, mask, xlabel, title, renumber in panels:
        # The L-BFGS rows are appended after the Adam ones, so their stored
        # "epoch" continues the Adam count; on their own axis they read as steps.
        x = np.arange(int(mask.sum()), dtype=float) if renumber else epoch[mask]
        for key in keys:
            ax.semilogy(x, np.asarray(history[key], dtype=float)[mask], label=key, linewidth=1)
        if not renumber:
            for r in restarts:
                ax.axvline(r, color="0.4", ls=":", lw=1.0)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, which="both")
    panels[0][0].set_ylabel("loss (dimensionless, k₀-scaled frame)")
    panels[0][0].legend(fontsize=8)
    fig.suptitle("(ω, f)-conditioned HMM SPP surrogate training")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = out_dir / "training_history.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


# --------------------------------------------------------------------------- main
def write_metrics_json(
    path: Path, per_point: Dict, summary: Dict, self_check: Dict, design_space: Dict,
    inverse: Sequence[Dict], figures: Dict[str, str], run_info: Dict,
) -> None:
    """Write the held-out metrics, summary, self-check and inverse design to JSON."""
    with open(path, "w") as fh:
        json.dump(
            {
                "per_point": per_point,
                "summary": summary,
                "analytical_self_check": self_check,
                "design_space": design_space,
                "inverse_design": [
                    {k: v for k, v in rec.items() if k != "n_eff_history"} for rec in inverse
                ],
                "figures": figures,
                "run_info": run_info,
            },
            fh,
            indent=2,
        )


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--epochs", type=int, default=N_EPOCHS)
    p.add_argument("--n-points", type=int, default=BATCH_SIZE,
                   help=f"interior collocation points per epoch (split over {N_BLOCKS} blocks)")
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=str(DEVICE))
    p.add_argument("--lbfgs-steps", type=int, default=LBFGS_STEPS,
                   help="L-BFGS outer steps after Adam (0 disables)")
    p.add_argument("--lbfgs-dtype", choices=("float64", "float32"), default="float64")
    p.add_argument("--f-min", type=float, default=FULL_F_MIN,
                   help="lower metal fill fraction of the design space")
    p.add_argument("--f-max", type=float, default=FULL_F_MAX,
                   help="upper metal fill fraction (the ω range follows from the pair)")
    p.add_argument("--resume", action="store_true",
                   help="warm-start from <model-out>.partial.pth if it exists")
    p.add_argument("--quick", action="store_true",
                   help=f"smoke run: {QUICK_EPOCHS} epochs, 512 points, no L-BFGS")
    p.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    p.add_argument("--model-out", type=Path, default=MODEL_PATH)
    return p.parse_args(argv)


def main(argv=None) -> Dict:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    set_design_space(args.f_min, args.f_max)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    n_epochs, n_points, lbfgs_steps = (
        (QUICK_EPOCHS, 512, 0) if args.quick else (args.epochs, args.n_points, args.lbfgs_steps)
    )
    n_val_points = 2000 if args.quick else 6000
    lbfgs_dtype = torch.float64 if args.lbfgs_dtype == "float64" else torch.float32

    design_space = verify_design_space(*((21, 15) if args.quick else (61, 41)))
    logger.info(
        "design space: f ∈ [%.3f, %.3f], ω/ω_ref ∈ [%.4f, %.4f] (λ₀ [%.0f, %.0f] nm)",
        F_MIN, F_MAX, OMEGA_MIN / OMEGA_REF, OMEGA_MAX / OMEGA_REF,
        2 * np.pi * C0 / OMEGA_MAX * 1e9, 2 * np.pi * C0 / OMEGA_MIN * 1e9,
    )
    logger.info(
        "verified on %d points: bound fraction %.4f, n_eff ∈ [%.4f, %.4f] "
        "(worst margin %.4f above the light line, at ω/ω_ref %.3f, f %.3f)",
        int(design_space["n_grid"]), design_space["bound_fraction"],
        design_space["n_eff_min"], design_space["n_eff_max"],
        design_space["worst_margin_over_light_line"],
        design_space["worst_omega_over_omega_ref"], design_space["worst_fill_fraction"],
    )
    logger.info(
        "k_spp moves ×%.3f over the rectangle; ×%.3f along f at mid ω, ×%.3f along f at the "
        "blue edge, ×%.3f along ω at mid f",
        design_space["k_spp_ratio_over_rectangle"],
        design_space["k_spp_ratio_along_f_at_mid_omega"],
        design_space["k_spp_ratio_along_f_at_blue_edge"],
        design_space["k_spp_ratio_along_omega_at_mid_f"],
    )
    for label, (w, f) in (("red/low-f", (OMEGA_MIN, F_MIN)), ("blue/high-f", (OMEGA_MAX, F_MAX))):
        eps_t, eps_n = hmm_eps(w, f)
        k, kd, km = mode_constants(w, f)
        logger.info(
            "%-11s corner: ε_t=%s ε_n=%s | n_eff %.4f, κ̂_d %.3f, κ̂_m %.3f, box x̂≤%.2f "
            "ẑ∈[%.2f, %.2f]",
            label, f"{eps_t:.4f}", f"{eps_n:.4f}",
            k.real / k0_of(w), kd.real / k0_of(w), km.real / k0_of(w),
            domain_hat(w, f)[0], domain_hat(w, f)[2], domain_hat(w, f)[3],
        )
    logger.info(
        "device=%s epochs=%d n_points=%d (%d blocks/epoch) lr=%.1e lbfgs_steps=%d (%s, %d nodes) "
        "seed=%d", device, n_epochs, n_points, N_BLOCKS, args.lr, lbfgs_steps,
        args.lbfgs_dtype, len(LBFGS_POINTS), args.seed,
    )

    self_points = SELF_CHECK_POINTS[:2] if args.quick else SELF_CHECK_POINTS
    self_check = validate_grid(AnalyticalHMMSPP(), self_points, n_points=4000, device=device)
    for key, m in self_check.items():
        logger.info(
            "self-check %s: rel_l2_E %.2e, k_spp err %.2e, κ_d err %.2e, κ_m err %.2e",
            key, m["rel_l2_E"], m["k_spp_rel_error"],
            m["kappa_d_fit_rel_error"], m["kappa_m_fit_rel_error"],
        )

    network = create_network(device=device)
    logger.info("network parameters: %d", sum(p.numel() for p in network.parameters()))

    t0 = time.perf_counter()
    checkpoint_path = args.model_out.with_suffix(".partial.pth")
    prior_history: Optional[Dict[str, list]] = None
    prior_best = float("inf")
    if args.resume:
        if checkpoint_path.exists():
            prior_best = load_checkpoint_into(network, checkpoint_path)
            prior_history = load_history(checkpoint_path)
            logger.info(
                "resuming from %s (loss %.3e, %d recorded steps)", checkpoint_path, prior_best,
                len(prior_history["epoch"]) if prior_history else 0,
            )
        else:
            logger.warning("--resume given but %s does not exist; training fresh", checkpoint_path)
    network, history = train(
        network, n_epochs=n_epochs, n_points=n_points, learning_rate=args.lr, device=device,
        lbfgs_steps=lbfgs_steps, lbfgs_dtype=lbfgs_dtype, checkpoint_path=checkpoint_path,
        initial_history=prior_history, initial_best_loss=prior_best,
    )
    train_time = time.perf_counter() - t0
    logger.info("training time %.1f s (this chunk)", train_time)

    per_point = validate_grid(network, VALIDATION_POINTS, n_points=n_val_points, device=device)
    summary = summarise(per_point)
    total_train_time = float(history["wall_s"][-1]) if history.get("wall_s") else train_time
    summary.update(
        train_time_s=total_train_time, train_time_this_chunk_s=train_time,
        total_adam_epochs=float(sum(1 for v in history["lr"] if not math.isnan(v))),
        total_lbfgs_steps=float(sum(1 for v in history["lr"] if math.isnan(v))),
        epochs=float(n_epochs), n_points=float(n_points),
        lbfgs_steps=float(lbfgs_steps), lr=args.lr, seed=float(args.seed),
        final_loss=history["total"][-1], best_loss=min(history["total"]),
    )
    logger.info("%8s | %7s | %8s | %10s | %10s | %10s | %10s | %10s",
                "ω/ω_ref", "f", "λ₀ nm", "rel_l2_E", "rel_l2_H", "k_spp err", "κ_d err", "κ_m err")
    for m in per_point.values():
        logger.info(
            "%8.4f | %7.4f | %8.0f | %10.3e | %10.3e | %10.3e | %10.3e | %10.3e",
            m["omega_over_omega_ref"], m["fill_fraction"], m["wavelength_nm"],
            m["rel_l2_E"], m["rel_l2_H"], m["k_spp_rel_error"],
            m["kappa_d_fit_rel_error"], m["kappa_m_fit_rel_error"],
        )
    for k, v in summary.items():
        logger.info("%-44s %s", k, f"{v:.4e}" if isinstance(v, float) else v)

    maps = design_space_maps(
        network, *((7, 5) if args.quick else (13, 9)),
        n_points=800 if args.quick else 1500, device=device,
    )
    inverse = inverse_design_demo(
        network, default_inverse_targets(), n_steps=60 if args.quick else 200, device=device
    )

    figures = {
        "k_spp_surface": plot_k_surface(maps, out_dir=args.figures_dir),
        "error_maps": plot_error_maps(maps, per_point, out_dir=args.figures_dir),
        "fill_slices": plot_fill_slices(network, maps, out_dir=args.figures_dir, device=device),
        "inverse_design": plot_inverse_design(inverse, out_dir=args.figures_dir),
        # Opposite corners: the boxes and the material differ several-fold.
        "field_maps_low_f_red": plot_field_maps(network, OMEGA_MIN, F_MIN,
                                                out_dir=args.figures_dir, device=device),
        "field_maps_high_f_blue": plot_field_maps(network, OMEGA_MAX, F_MAX,
                                                  out_dir=args.figures_dir, device=device),
        "training_history": plot_history(history, out_dir=args.figures_dir),
    }

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": network.state_dict(),
            "config": {
                "hidden_dims": list(HIDDEN_DIMS), "fourier_modes": FOURIER_MODES,
                "fourier_k_range": FOURIER_K_RANGE,
                "eps_dielectric_layer": EPS_D2, "eps_superstrate": EPS_D, "drude": DRUDE,
                "omega_ref": OMEGA_REF, "omega_min": OMEGA_MIN, "omega_max": OMEGA_MAX,
                "f_min": F_MIN, "f_max": F_MAX, "H0": H0,
                "E_scale": E_SCALE, "H_scale": H_SCALE,
                "input_scaling": "coords * k0(omega), k0 = omega / c; then omega_hat, fill_hat",
                "domain_rule": {
                    "x_periods": X_PERIODS, "z_metal_depths": Z_METAL_DEPTHS,
                    "z_air_depths": Z_AIR_DEPTHS, "y_wavelengths": Y_WAVELENGTHS,
                },
            },
            "summary": summary,
        },
        args.model_out,
    )
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    run_info = {
        "hmm_summary_source": str(HMM_SUMMARY_PATH.relative_to(REPO_ROOT)),
        "design_rectangle": {
            "omega": [OMEGA_MIN, OMEGA_MAX],
            "omega_over_omega_ref": [OMEGA_MIN / OMEGA_REF, OMEGA_MAX / OMEGA_REF],
            "wavelength_nm": [2 * np.pi * C0 / OMEGA_MAX * 1e9, 2 * np.pi * C0 / OMEGA_MIN * 1e9],
            "fill_fraction": [F_MIN, F_MAX],
            "inset_fraction": DESIGN_INSET,
        },
        "lbfgs_points": [[w / OMEGA_REF, f] for w, f in LBFGS_POINTS],
        "validation_points": [[w / OMEGA_REF, f] for w, f in VALIDATION_POINTS],
        "n_blocks_per_epoch": N_BLOCKS, "strata": [STRATA_OMEGA, STRATA_FILL],
        "lbfgs_dtype": args.lbfgs_dtype, "quick": bool(args.quick),
    }
    write_metrics_json(
        args.figures_dir / "metrics.json", per_point, summary, self_check, design_space,
        inverse, figures, run_info,
    )
    np.savez(
        args.figures_dir / "design_space_maps.npz",
        **{k: np.asarray(v) for k, v in maps.items()},
    )
    logger.info("saved model to %s, figures + metrics.json to %s", args.model_out,
                args.figures_dir)
    logger.info("success tier: %s", summary["success_tier"])
    return {
        "per_point": per_point,
        "summary": summary,
        "self_check": self_check,
        "design_space": design_space,
        "inverse_design": inverse,
    }


if __name__ == "__main__":
    main()
