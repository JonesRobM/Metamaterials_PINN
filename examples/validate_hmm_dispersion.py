r"""
Frequency-Conditioned SPP PINN on a **Dispersive** Hyperbolic Metamaterial

Trains ONE network conditioned on frequency to recover the bound TM SPP mode of
an Ag/silica layered hyperbolic metamaterial (HMM) against air, across the band
recommended by ``examples/hyperbolic_metamaterial.py``. The difference from
``examples/validate_spp_dispersion.py`` — the fixed-ε predecessor — is that the
permittivities are now **physically dispersive**: ε_t(ω), ε_n(ω) come from the
Drude/effective-medium stack in :mod:`src.effective_medium`, so k_spp(ω) is a
genuinely *nonlinear* branch rather than a straight line through the origin.

Why this is the harder experiment
---------------------------------
With ε fixed, every mode constant scales linearly with ω and the mode family is
self-similar: one shape, rescaled. Here ε_t sweeps ≈ −10 → −0.6 across the band
(approaching the multilayer's in-plane ENZ crossing), so

* Re k_spp spans 2.59× and departs from the best straight line by up to 24 % of
  its own range (``nonlinearity_percent`` in ``figures/hyperbolic/hmm_summary.json``);
* the transverse decay constants span ~10× (decay lengths 54 nm → 541 nm);
* λ_spp spans 331 nm → 857 nm.

Two modelling choices follow from that spread, and both are stated up front
because they are choices, not derivations.

**1. Per-ω domain (a modelling choice).** One fixed box would be wasteful at one
band edge and under-resolved at the other, so each sampled ω gets its own box
sized from *its own* analytic scales::

    x ∈ [0, 2 λ_spp(ω)],   z ∈ [−3.5 / Re κ_m(ω), +1.2 / Re κ_d(ω)],   y thin.

This uses the analytical mode to size the sampling region — legitimate here
because the anchor boundary condition already *is* the analytical mode, so no
new information is injected; but it does mean the experiment measures how well
the network interpolates a mode family whose extent it is told, not how well it
discovers the extent.

**2. Input scaling by the LOCAL FREE-SPACE wavenumber (deliberately not k_spp).**
The network is fed ``(x, y, z)·k₀(ω)`` with ``k₀(ω) = ω/c``, plus a normalised
frequency feature. Two reasons:

* *It removes most of the cross-band variation.* In these units the mode's
  wavenumbers are k_spp/k₀ = n_eff ∈ [1.03, 1.36], κ_d/k₀ ∈ [0.26, 0.92] and
  κ_m/k₀ ∈ [0.54, 2.60] — everything O(1), against a 10× spread in SI. It also
  makes the *scaled Maxwell system frequency-free*: with x̂ = k₀x, Ê = E/(η₀H₀),
  Ĥ = H/H₀ the curl equations are exactly ``∇̂×Ê = i Ĥ`` and ``∇̂×Ĥ = −i ε Ê``
  at every ω, so a single ``frequency = 1`` residual serves the whole band and
  ω enters the physics **only** through ε(ω). (The fixed-ε predecessor had to
  fold a per-row factor ω/ω₀ into the material arguments; that factor is gone.)
* *It does not leak the answer.* Scaling by k_spp(ω) instead would build the
  measured dispersion into the coordinate map: k_spp itself ranges 2.6× while
  k_spp/k₀ ranges only 1.03–1.36, so a k_spp-scaled network would be handed
  most of the dispersion for free. k₀(ω) = ω/c is known without solving
  anything, so scaling by it is bookkeeping, not a hint.

Recipe (inherited from ``examples/validate_spp_dispersion.py``)
---------------------------------------------------------------
Dimensionless frame; a displacement adapter making the E_z jump exact — now
with an **ω-dependent divisor** ε_n(ω) below the interface (the predecessor's
divisor was a constant); boundary anchor weight 100 with a physics ramp;
per-medium curl weighting, here applied **per row** as |ε(ω_row)|^-1 (Adam) and
|ε(ω_row)|^-1/2 (L-BFGS) because |ε| itself varies 2.7× across the band; Adam
then a float64 L-BFGS refinement; atomic checkpointing with ``--resume``.

Frequency conditioning
----------------------
* **Input**: 4 features — (x, y, z)·k₀(ω) plus ω̂ ∈ [−1, 1] linear in ω over the
  band. Fourier features encode the *spatial* columns only (band (0.1, 8)
  rad/unit comfortably covers the largest scaled wavenumber, κ_m/k₀ = 2.60);
  ω̂ is appended raw, because the ω dependence is smooth and putting random
  8 rad/unit directions on it would invent oscillations along the frequency
  axis.
* **Per-ω material**: the interior ε is built per row from ε_t(ω_row), ε_n(ω_row)
  below the interface and ε_d = 1 above, as an (N, 3, 3) complex tensor. This is
  the whole experiment: two rows at different ω carry genuinely different ε.
* **Batching**: each epoch samples ``N_FREQ_SUB = 4`` fresh uniform ω's, one per
  sub-block (each with its own box); L-BFGS refines on a fixed float64 set of 13
  frequencies including both ends. The node *spacing* matters: with only 5 nodes
  the converged error was ≈ 5e-3 at each node but bulged to 9e-2 between the two
  bluest ones, where ε_t sweeps fastest towards its ENZ crossing (see the results
  doc). Halving the spacing removes the bulge.

Validation: 9 frequencies spanning the band (both ends included; the 4 odd grid
points are strictly held out from the L-BFGS set). Per ω, in SI: rel L2 vs
:func:`src.analytical.analytical_spp_fields` (E and H), Re k_spp by a phase-slope
fit, κ_d and κ_m by decay fits, tangential-continuity residuals. The analytical
mode is pushed through the identical pipeline at 3 frequencies as a convention
self-check. Success tiers: minimum = bound mode at all 9 ω and rel L2 < 0.5;
target = rel L2 < 0.1 everywhere and k_spp within 1 %; stretch = rel L2 < 0.02
and k_spp within 0.2 %.

Usage::

    python examples/validate_hmm_dispersion.py [--epochs 5000] [--n-points 2048]
        [--lr 1e-3] [--seed 0] [--device cpu] [--lbfgs-steps 120]
        [--lbfgs-dtype {float64,float32}] [--band-fraction 1.0] [--resume] [--quick]
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
from typing import Dict, Optional, Sequence, Tuple

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
from src.experiments import (  # noqa: E402
    SCALED_MAXWELL,
    ColumnConditionedNet,
    DisplacementAdapter,
    LinearFeature,
    TrainingConfig,
    add_core_args,
    add_output_args,
    banded_success_tier,
    k0_of,
    load_core_checkpoint,
    measurement,
    plot_two_phase_history,
    relative_l2,
    run_training,
    weighted_curl_loss,
    weighted_divergence_loss,
    write_checkpoint,
    write_json_report,
)
from src.models import (  # noqa: E402
    ElectromagneticPINN,
    FourierEMFeatures,
    TangentialContinuityLoss,
    to_complex,
)
from src.physics import MaxwellEquations  # noqa: E402
from src.physics.metamaterial import MetamaterialProperties  # noqa: E402

logger = logging.getLogger("validate_hmm_dispersion")

# --------------------------------------------------------------------------- material / band
HMM_SUMMARY_PATH = REPO_ROOT / "figures" / "hyperbolic" / "hmm_summary.json"


def load_hmm_design(path: Path = HMM_SUMMARY_PATH) -> Dict:
    """
    Read the HMM design and its recommended band from ``hmm_summary.json``.

    The numbers (fill fraction, layer permittivity, Drude parameters, band
    endpoints) are read at import time rather than hardcoded, so this experiment
    tracks ``examples/hyperbolic_metamaterial.py`` if the recommendation moves.
    """
    with open(path) as fh:
        summary = json.load(fh)
    design = summary["design"]
    band = summary["recommended_band"]
    drude_ev = design["drude_model"]
    return {
        "fill_fraction": float(design["fill_fraction"]),
        "eps_dielectric_layer": float(design["eps_dielectric_layer"]),
        "eps_superstrate": float(design["eps_superstrate"]),
        "omega_ref": float(design["omega_ref"]),
        "lambda_ref": float(design["lambda_ref_nm"]) * 1e-9,
        "drude": drude_parameters_ev(
            eps_inf=drude_ev["eps_inf"],
            hbar_omega_p_ev=drude_ev["hbar_omega_p_eV"],
            hbar_gamma_ev=drude_ev["hbar_gamma_eV"],
        ),
        "omega_band": (float(band["omega"][0]), float(band["omega"][1])),
        "band_summary": band,
    }


DESIGN = load_hmm_design()
FILL_FRACTION = DESIGN["fill_fraction"]
EPS_D2 = DESIGN["eps_dielectric_layer"]  # silica layers
EPS_D = DESIGN["eps_superstrate"]  # air half-space above z = 0
DRUDE = DESIGN["drude"]  # {'eps_inf', 'omega_p', 'gamma'} in SI
OMEGA_REF = DESIGN["omega_ref"]  # 2πc/633 nm, the reference used for ω/ω_ref labels
LAMBDA_REF = DESIGN["lambda_ref"]
H0 = 1.0  # A/m

# Full recommended band; ``--band-fraction`` shrinks it about the midpoint (the
# documented fallback if the 10× κ spread proves too much for one network).
FULL_OMEGA_MIN, FULL_OMEGA_MAX = DESIGN["omega_band"]
BAND_FRACTION = 1.0
OMEGA_MIN, OMEGA_MAX = FULL_OMEGA_MIN, FULL_OMEGA_MAX
OMEGA_MID = 0.5 * (OMEGA_MIN + OMEGA_MAX)
OMEGA_HALF_SPAN = 0.5 * (OMEGA_MAX - OMEGA_MIN)
#: ω -> the network's ω̂ input column, and back; refreshed by :func:`set_band_fraction`.
OMEGA_FEATURE = LinearFeature.centred(OMEGA_MID, OMEGA_HALF_SPAN)


def set_band_fraction(fraction: float) -> None:
    """
    Shrink the training/validation band to ``fraction`` of the recommended one,
    centred on its midpoint, and refresh every band-derived module constant.

    Provided for the documented failure path: if the full 10× spread of decay
    constants defeats a single network, narrowing to the middle 60–70 % and
    reporting the reduced band honestly beats reporting failure on the full one.
    """
    global BAND_FRACTION, OMEGA_MIN, OMEGA_MAX, OMEGA_MID, OMEGA_HALF_SPAN
    global OMEGA_FEATURE, VALIDATION_OMEGAS, LBFGS_OMEGAS, SELF_CHECK_OMEGAS
    if not 0.0 < fraction <= 1.0:
        raise ValueError("band fraction must lie in (0, 1]")
    BAND_FRACTION = float(fraction)
    mid = 0.5 * (FULL_OMEGA_MIN + FULL_OMEGA_MAX)
    half = 0.5 * (FULL_OMEGA_MAX - FULL_OMEGA_MIN) * BAND_FRACTION
    OMEGA_MIN, OMEGA_MAX = mid - half, mid + half
    OMEGA_MID, OMEGA_HALF_SPAN = mid, half
    OMEGA_FEATURE = LinearFeature.centred(mid, half)
    VALIDATION_OMEGAS = tuple(np.linspace(OMEGA_MIN, OMEGA_MAX, 9))
    # L-BFGS refinement nodes: the 5 even validation points PLUS the 8 half-way
    # points between consecutive validation points — 13 in all. The 4 *odd*
    # validation frequencies stay strictly held out (they are integer multiples
    # of Δ; every added node is a half-integer multiple).
    #
    # The first run of this experiment used only the 5 even points and produced a
    # clean node-aligned error signature: rel L2 ≈ 5e-3 *at* each node, bulging
    # between them, harmlessly (6e-3) in the red half but reaching 9e-2 in the
    # last interval — exactly where ε_t sweeps −1.57 → −0.59 towards the in-plane
    # ENZ crossing and the mode changes fastest with ω. Adding the half-way nodes
    # halves the worst-case distance from an arbitrary ω to the nearest refined
    # one (Δ → Δ/2) and removes the bulge; note the node gaps are deliberately
    # NOT uniform — each held-out validation frequency sits alone mid-gap, so it
    # is never refined. See the results doc's "Diagnostic iteration".
    delta = (OMEGA_MAX - OMEGA_MIN) / 8.0
    LBFGS_OMEGAS = tuple(
        sorted(list(VALIDATION_OMEGAS[::2]) + [OMEGA_MIN + (i + 0.5) * delta for i in range(8)])
    )
    SELF_CHECK_OMEGAS = (OMEGA_MIN, OMEGA_MID, OMEGA_MAX)


def hmm_eps(omega: float) -> Tuple[complex, complex]:
    """``(ε_t, ε_n)`` of the Ag/silica stack at ``omega`` — the dispersive material."""
    eps_t, eps_n = hmm_permittivities(omega, FILL_FRACTION, EPS_D2, **DRUDE)
    return complex(eps_t), complex(eps_n)


def hmm_eps_torch(omega: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorised torch mirror of :func:`hmm_eps` (complex128), for per-row use.

    Same Drude + Rytov algebra as :mod:`src.effective_medium`, in torch so the
    displacement adapter can build its ε_n(ω) divisor from the ω̂ column inside
    a forward pass. ``tests`` assert it agrees with ``hmm_permittivities`` to
    machine precision.
    """
    w = omega.to(torch.float64)
    eps_m = DRUDE["eps_inf"] - DRUDE["omega_p"] ** 2 / (
        w.to(torch.complex128) ** 2 + 1j * DRUDE["gamma"] * w.to(torch.complex128)
    )
    eps_t = FILL_FRACTION * eps_m + (1.0 - FILL_FRACTION) * EPS_D2
    eps_n = eps_m * EPS_D2 / (FILL_FRACTION * EPS_D2 + (1.0 - FILL_FRACTION) * eps_m)
    return eps_t, eps_n


@lru_cache(maxsize=8192)
def mode_constants(omega: float) -> Tuple[complex, complex, complex]:
    """Analytical ``(k_spp, κ_d, κ_m)`` at ``omega`` for the dispersive HMM."""
    eps_t, eps_n = hmm_eps(float(omega))
    material = MetamaterialProperties(eps_n, eps_t, "z")
    return material.decay_constants(float(omega), EPS_D, "x")


def omega_hat(omega: float) -> float:
    """Normalised frequency feature, linear on the band: ω̂ ∈ [−1, 1]."""
    return OMEGA_FEATURE.to_hat(omega)


def omega_from_hat(w_hat: float) -> float:
    """Inverse of :func:`omega_hat`."""
    return OMEGA_FEATURE.from_hat(w_hat)


def omega_from_hat_torch(w_hat: torch.Tensor) -> torch.Tensor:
    """Tensor form of :func:`omega_from_hat`."""
    return OMEGA_FEATURE.from_hat_torch(w_hat)


def eps_scale(omega: float) -> float:
    """Stiffness scale of the lower medium, ``max(|ε_t|, |ε_n|)`` at ``omega``."""
    eps_t, eps_n = hmm_eps(omega)
    return max(abs(eps_t), abs(eps_n))


# --------------------------------------------------------------------------- scaled frame
E_SCALE = ETA0 * abs(H0)
H_SCALE = abs(H0)
FIELD_SCALE = torch.tensor([E_SCALE] * 3 + [H_SCALE] * 3, dtype=torch.float32).view(1, 6, 1)

# In the k₀-scaled frame the guard band and continuity offset of the fixed-ε run
# (1 nm and 2 nm at λ₀ = 633 nm) become 0.01 and 0.02 scaled units exactly.
GUARD_HAT = 0.01
VAL_GUARD_HAT = 0.02
CONTINUITY_OFFSET_HAT = 0.02

# Per-ω box, in units of that ω's own analytic scales.
X_PERIODS = 2.0  # x ∈ [0, 2 λ_spp(ω)]
Z_METAL_DEPTHS = 3.5  # z ≥ −3.5 / Re κ_m(ω)
Z_AIR_DEPTHS = 1.2  # z ≤ +1.2 / Re κ_d(ω)
Y_WAVELENGTHS = 0.2  # y ∈ [0, 0.2 λ₀(ω)] — thin, the mode is y-invariant


def domain_hat(omega: float) -> Tuple[float, float, float, float]:
    """``(x̂_max, ŷ_max, ẑ_min, ẑ_max)`` of the ω's own box in k₀(ω)-scaled units."""
    k, kappa_d, kappa_m = mode_constants(omega)
    k0 = k0_of(omega)
    x_max = X_PERIODS * 2.0 * np.pi / (k.real / k0)
    y_max = Y_WAVELENGTHS * 2.0 * np.pi
    z_min = -Z_METAL_DEPTHS / (kappa_m.real / k0)
    z_max = Z_AIR_DEPTHS / (kappa_d.real / k0)
    return float(x_max), float(y_max), float(z_min), float(z_max)


def domain_si(omega: float) -> Tuple[float, float, float, float]:
    """The same box in metres (``domain_hat`` divided by ``k₀(ω)``)."""
    k0 = k0_of(omega)
    return tuple(v / k0 for v in domain_hat(omega))  # type: ignore[return-value]


# --------------------------------------------------------------------------- defaults
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2048
N_EPOCHS = 5000
LEARNING_RATE = 1e-3
N_FREQ_SUB = 4
BOUNDARY_WEIGHT = 100.0
DIVERGENCE_WEIGHT = 1.0
CONTINUITY_WEIGHT = 1.0
PHYSICS_RAMP_FRAC = 0.25
# Metal-side loss preconditioning, applied PER ROW because |ε(ω)| varies 2.7×
# across this band (9.98 at the red edge, 3.71 at the blue edge).
CURL_POWER_ADAM = 1.0  # weight = |ε(ω_row)|^-1
CURL_POWER_LBFGS = 0.5  # weight = |ε(ω_row)|^-1/2
DIV_POWER = 1.0
METAL_FRACTION = 0.45
AIR_UNIFORM_FLOOR = 0.3
QUICK_EPOCHS = 200
LBFGS_STEPS = 120
LBFGS_POINTS_FACTOR = 2
LBFGS_DTYPE = torch.float64
FOURIER_MODES = 128
FOURIER_K_RANGE = (0.1, 8.0)  # scaled wavenumbers reach only κ_m/k₀ = 2.60
HIDDEN_DIMS = (128, 128, 128, 128)

VALIDATION_OMEGAS: Tuple[float, ...] = ()
LBFGS_OMEGAS: Tuple[float, ...] = ()
SELF_CHECK_OMEGAS: Tuple[float, ...] = ()
set_band_fraction(1.0)

FIGURES_DIR = REPO_ROOT / "figures" / "hmm_dispersion"
MODEL_PATH = REPO_ROOT / "artifacts" / "models" / "hmm_dispersion.pth"


# --------------------------------------------------------------------------- analytical
def analytical_fields_si(coords: torch.Tensor, omega: float) -> torch.Tensor:
    """Analytical SPP ``(E, H)`` at SI ``coords`` and ``omega``, ``[N, 6, 2]`` layout."""
    eps_t, eps_n = hmm_eps(omega)
    E, H = analytical_spp_fields(coords, omega, eps_t, eps_n, eps_dielectric=EPS_D, H0=H0)
    return torch.cat([complex_to_pinn_format(E), complex_to_pinn_format(H)], dim=1).to(
        coords.dtype if coords.is_floating_point() else torch.float32
    )


def analytical_fields_hat(coords_hat: torch.Tensor, omega: float) -> torch.Tensor:
    """Analytical mode at k₀(ω)-scaled coords, in the core's scaled field units."""
    fields = analytical_fields_si(coords_hat / k0_of(omega), omega)
    return fields / FIELD_SCALE.to(device=fields.device, dtype=fields.dtype)


class AnalyticalHMMSPP(nn.Module):
    """Exact SI mode with the PINN's ``(coords, omega)`` interface (pipeline self-check)."""

    def forward(self, coords: torch.Tensor, omega: float) -> torch.Tensor:
        return analytical_fields_si(coords, omega)

    def at_omega(self, omega: float) -> nn.Module:
        return _FixedOmega(self, omega)


# --------------------------------------------------------------------------- network
class OmegaConditionedCore(nn.Module):
    """
    Scaled field network conditioned on frequency.

    Consumes ``[x̂, ŷ, ẑ, ω̂]`` of shape ``(N, 4)`` with x̂ = k₀(ω)·x. Fourier
    features encode the spatial columns only; the raw ω̂ feature is appended
    afterwards (see the module docstring). Returns ``(N, 6, 2)``, with channel 2
    carrying the *continuous* D̂_z that the adapter converts to Ê_z.
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
            spatial_dim=self.fourier.output_dim + 1,  # spatial features + ω̂
            field_components=6,
            hidden_dims=list(hidden_dims),
            complex_valued=True,
            frequency=OMEGA_REF,  # selects the time-harmonic input layout (no +1 column)
            use_fourier=False,  # spatial Fourier encoding is done here instead
            activation_type="complex_tanh",
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        features = self.fourier(coords[:, :3])
        return self.mlp(torch.cat([features, coords[:, 3:4]], dim=1))


class DispersiveDisplacementAdapter(DisplacementAdapter):
    """
    Displacement adapter with an **ω-dependent** ε_zz divisor.

    :class:`src.experiments.DisplacementAdapter` supplies the construction —
    channel 2 of the wrapped MLP is the continuous ``D̂_z``, divided here by the
    local ε_zz so the returned Ê_z carries the exact interface jump. What is
    specific to this experiment is the divisor: below the interface it is
    ε_n(ω), read from the ω̂ column of the 4-column input and evaluated with
    :func:`hmm_eps_torch`, because the material is dispersive. Above it is the
    constant ε_d.

    D̂_z is also the better-conditioned target across this band: the analytical
    mode has D̂_z = −k_spp/k₀ on *both* sides, an O(1) quantity, while Ê_z jumps
    by ε_n ≈ 3.3–3.7.
    """

    def __init__(self, mlp: nn.Module, eps_above: complex = EPS_D):
        super().__init__(mlp)
        self.eps_above = complex(eps_above)

    def eps_zz_at(self, coords: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        omega = omega_from_hat_torch(coords[:, 3])
        _, eps_n = hmm_eps_torch(omega)
        eps_below = eps_n.to(dtype=dtype, device=coords.device)
        eps_above = torch.as_tensor(self.eps_above, dtype=dtype, device=coords.device)
        return torch.where(coords[:, 2] < 0, eps_below, eps_above.expand_as(eps_below))


#: 3-column spatial view of the 4-input core with a fixed per-row ω̂ column;
#: see :class:`src.experiments.ColumnConditionedNet` for why the ω̂ column is
#: appended inside the forward rather than passed to the differential operators.
OmegaColumnNet = ColumnConditionedNet


class _FixedOmega(nn.Module):
    """SI 3-column module at a fixed ω: ``forward(coords_m) -> [N, 6, 2]``."""

    def __init__(self, parent: nn.Module, omega: float):
        super().__init__()
        self.parent = parent
        self.omega = float(omega)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.parent(coords, self.omega)


class HMMDispersionPINN(nn.Module):
    """
    SI-unit wrapper: ``forward(coords_m, omega)`` scales coordinates by k₀(ω),
    appends ω̂ and rescales the network's dimensionless output to SI fields.
    ``at_omega(omega)`` returns a 3-column SI module for the validation pipeline.
    """

    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core
        self.register_buffer("field_scale", FIELD_SCALE.clone())

    def forward(self, coords: torch.Tensor, omega: float) -> torch.Tensor:
        coords_hat = coords * k0_of(omega)
        w = coords_hat.new_full((coords_hat.shape[0], 1), omega_hat(omega))
        scale = self.field_scale.to(coords_hat.dtype)
        return self.core(torch.cat([coords_hat, w], dim=1)) * scale

    def at_omega(self, omega: float) -> nn.Module:
        return _FixedOmega(self, omega)


def create_network(
    hidden_dims: Tuple[int, ...] = HIDDEN_DIMS,
    fourier_modes: int = FOURIER_MODES,
    device: torch.device = DEVICE,
) -> HMMDispersionPINN:
    """Build the ω-conditioned HMM SPP PINN (Fourier spatial features + ω̂ + adapter)."""
    mlp = OmegaConditionedCore(hidden_dims=hidden_dims, fourier_modes=fourier_modes)
    adapter = DispersiveDisplacementAdapter(mlp, eps_above=EPS_D)
    return HMMDispersionPINN(adapter).to(device)


# --------------------------------------------------------------------------- permittivity rows
def eps_tensor_rows(omegas: torch.Tensor, metal: bool) -> torch.Tensor:
    """
    Per-row ``(N, 3, 3)`` complex permittivity for the interior loss.

    Below the interface each row carries ``diag(ε_t(ω), ε_t(ω), ε_n(ω))`` at
    **that row's own frequency** — the entire point of the dispersive
    experiment; above it, ``diag(ε_d, ε_d, ε_d)``. No ω prefactor is needed
    anywhere: in the k₀-scaled frame Maxwell's curl equations are
    frequency-free, so ω enters only here.
    """
    w = omegas.reshape(-1).to(torch.float64)
    if metal:
        eps_t, eps_n = hmm_eps_torch(w)
        diag = torch.stack([eps_t, eps_t, eps_n], dim=1)  # (N, 3)
    else:
        diag = torch.full((w.shape[0], 3), complex(EPS_D), dtype=torch.complex128,
                          device=w.device)
    return torch.diag_embed(diag)


# --------------------------------------------------------------------------- sampling
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
    n_points: int, omega: float, guard: float = GUARD_HAT, device: torch.device = DEVICE
) -> torch.Tensor:
    """
    Stratified interior points of ``omega``'s own box, in k₀(ω)-scaled units.

    Both the extents and the exponential z-strata use that ω's analytic scales,
    so every frequency's near-field is sampled at the same relative resolution.
    """
    _, kappa_d, kappa_m = mode_constants(omega)
    k0 = k0_of(omega)
    x_max, y_max, z_min, z_max = domain_hat(omega)
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
    n_points: int, omega: float, guard: float = GUARD_HAT, device: torch.device = DEVICE
) -> torch.Tensor:
    """:func:`sample_collocation_hat` in metres, with ``requires_grad`` set."""
    coords = sample_collocation_hat(n_points, omega, guard, device) / k0_of(omega)
    coords.requires_grad_(True)
    return coords


def _avoid_guard(z: torch.Tensor, guard: float) -> torch.Tensor:
    return torch.where(z.abs() < guard, torch.where(z < 0, -guard, guard), z)


def sample_boundary_hat(
    n_points: int, omega: float, guard: float = GUARD_HAT, device: torch.device = DEVICE
) -> torch.Tensor:
    """Points on the six faces of ``omega``'s scaled box, ``n_points // 6`` per face."""
    x_max, y_max, z_min, z_max = domain_hat(omega)
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
    n_points: int, omega: float, device: torch.device = DEVICE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Points on ẑ = 0 within ``omega``'s box, and their unit normals ẑ."""
    x_max, y_max, _, _ = domain_hat(omega)
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
    omegas: Sequence[float],
    device: torch.device = DEVICE,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """
    One multi-frequency training batch of per-ω blocks, all in scaled units.

    Each block lives in its own box (sized by that ω's analytic scales) and
    carries its own ω̂ column, ω column, per-row ε(ω) and per-row stiffness
    scale |ε(ω)| used for the metal loss weighting. Boundary anchors are the
    analytical mode at each block's own ω.
    """
    n_om = len(omegas)
    int_blocks, bc_blocks, tgt_blocks, if_blocks, nrm_blocks = [], [], [], [], []
    w_int, w_bc, w_if, om_int = [], [], [], []
    for i, omega in enumerate(omegas):
        ni = n_int // n_om + (n_int % n_om if i == n_om - 1 else 0)
        nb = max(6, n_bc // n_om)
        nf = max(1, n_if // n_om)
        w = omega_hat(omega)

        c = sample_collocation_hat(ni, omega, device=device).detach().to(dtype)
        int_blocks.append(c)
        w_int.append(torch.full((c.shape[0], 1), w, dtype=dtype, device=device))
        om_int.append(torch.full((c.shape[0],), float(omega), dtype=torch.float64, device=device))

        b = sample_boundary_hat(nb, omega, device=device).to(dtype)
        bc_blocks.append(b)
        w_bc.append(torch.full((b.shape[0], 1), w, dtype=dtype, device=device))
        with torch.no_grad():
            tgt_blocks.append(analytical_fields_hat(b, omega).to(dtype))

        f, nrm = sample_interface_hat(nf, omega, device=device)
        if_blocks.append(f.to(dtype))
        nrm_blocks.append(nrm.to(dtype))
        w_if.append(torch.full((f.shape[0], 1), w, dtype=dtype, device=device))

    coords = torch.cat(int_blocks)
    w_col = torch.cat(w_int)
    om_col = torch.cat(om_int)
    metal = coords[:, 2] < 0
    eps_t_m, eps_n_m = hmm_eps_torch(om_col[metal])
    return {
        "coords_air": coords[~metal].clone().requires_grad_(True),
        "coords_metal": coords[metal].clone().requires_grad_(True),
        "w_air": w_col[~metal],
        "w_metal": w_col[metal],
        "eps_air": eps_tensor_rows(om_col[~metal], metal=False),
        "eps_metal": eps_tensor_rows(om_col[metal], metal=True),
        "eps_scale_metal": torch.maximum(eps_t_m.abs(), eps_n_m.abs()),
        "omega_air": om_col[~metal],
        "omega_metal": om_col[metal],
        "boundary": torch.cat(bc_blocks),
        "w_bc": torch.cat(w_bc),
        "target": torch.cat(tgt_blocks),
        "iface": torch.cat(if_blocks),
        "normals": torch.cat(nrm_blocks),
        "w_if": torch.cat(w_if),
    }


# --------------------------------------------------------------------------- interior losses
# One scaled-frame Maxwell operator serves the whole band: with x̂ = k₀(ω)x the
#: Maxwell's operators in the scaled frame — no explicit ω at any frequency.
_SCALED_MAXWELL = SCALED_MAXWELL


def curl_loss_weighted(
    core: nn.Module,
    coords: torch.Tensor,
    w_col: torch.Tensor,
    eps_rows: torch.Tensor,
    row_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Scaled-frame curl residual at the batch's own frequencies, weighted per row.

    The per-row form is needed because the metal-side preconditioner |ε|^-p
    depends on the row's frequency. See :func:`src.experiments.weighted_curl_loss`.
    """
    return weighted_curl_loss(OmegaColumnNet(core, w_col), coords, eps_rows, row_weight)


def divergence_loss_weighted(
    core: nn.Module,
    coords: torch.Tensor,
    w_col: torch.Tensor,
    eps_rows: torch.Tensor,
    row_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``∇̂·(εÊ) = 0`` and ``∇̂·Ĥ = 0`` with per-row weights (see :func:`curl_loss_weighted`)."""
    return weighted_divergence_loss(
        OmegaColumnNet(core, w_col), coords, eps_rows, row_weight
    )


# --------------------------------------------------------------------------- training
#: Atomic best-weights checkpointing (see :func:`src.experiments.write_checkpoint`).
_write_checkpoint = write_checkpoint
load_checkpoint_into = load_core_checkpoint


def train(
    network: HMMDispersionPINN,
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
) -> Tuple[HMMDispersionPINN, Dict[str, list]]:
    """
    Train the ω-conditioned core on the dispersive band.

    Phase 1 draws ``N_FREQ_SUB`` fresh uniform ω's each epoch, one per
    sub-block, each with its own box and its own ε(ω), and weights the metal
    curl/divergence residual per row by |ε(ω_row)|^-1 (anti-collapse). Phase 2
    refines on a fixed batch spanning :data:`LBFGS_OMEGAS` with the metal curl
    weight raised to |ε|^-1/2. The schedule around the physics — ramp,
    best-iterate tracking, checkpointing — is :func:`src.experiments.run_training`.
    """
    core = network.core
    cont_loss = TangentialContinuityLoss(offset=CONTINUITY_OFFSET_HAT)

    def compute_losses(batch: Dict[str, torch.Tensor], ramp: float = 1.0,
                       curl_power: float = CURL_POWER_ADAM):
        w_curl_m = batch["eps_scale_metal"] ** (-curl_power)
        w_div_m = batch["eps_scale_metal"] ** (-DIV_POWER)
        l_curl = curl_loss_weighted(
            core, batch["coords_air"], batch["w_air"], batch["eps_air"]
        ) + curl_loss_weighted(
            core, batch["coords_metal"], batch["w_metal"], batch["eps_metal"], w_curl_m
        )
        l_div = divergence_loss_weighted(
            core, batch["coords_air"], batch["w_air"], batch["eps_air"]
        ) + divergence_loss_weighted(
            core, batch["coords_metal"], batch["w_metal"], batch["eps_metal"], w_div_m
        )
        l_cont = cont_loss.compute(
            network=OmegaColumnNet(core, batch["w_if"]),
            interface_coords=batch["iface"],
            normal_vectors=batch["normals"],
        )
        l_bc = torch.mean(
            (OmegaColumnNet(core, batch["w_bc"])(batch["boundary"]) - batch["target"]) ** 2
        )
        total = (
            ramp * (l_curl + divergence_weight * l_div + continuity_weight * l_cont)
            + boundary_weight * l_bc
        )
        return total, l_curl, l_div, l_cont, l_bc

    def sample_epoch(n_int, n_bc, n_if, dtype):
        """Adam: a fresh uniform draw from the band, one ω per sub-block."""
        omegas = (OMEGA_MIN + (OMEGA_MAX - OMEGA_MIN) * torch.rand(N_FREQ_SUB)).tolist()
        return sample_training_batch(n_int, n_bc, n_if, omegas, device=device, dtype=dtype)

    def sample_fixed(n_int, n_bc, n_if, dtype):
        """L-BFGS: the fixed refinement nodes :data:`LBFGS_OMEGAS`."""
        return sample_training_batch(
            n_int, n_bc, n_if, LBFGS_OMEGAS, device=device, dtype=dtype
        )

    return run_training(
        network,
        TrainingConfig(
            n_epochs=n_epochs,
            n_points=n_points,
            n_boundary=max(6 * N_FREQ_SUB, n_points // 2),
            n_interface=max(N_FREQ_SUB, n_points // 4),
            learning_rate=learning_rate,
            physics_ramp_frac=physics_ramp_frac,
            lbfgs_steps=lbfgs_steps,
            lbfgs_dtype=lbfgs_dtype,
            lbfgs_points_factor=LBFGS_POINTS_FACTOR,
            log_every=log_every,
        ),
        sample_epoch,
        compute_losses,
        logger,
        lbfgs_sample_batch=sample_fixed,
        lbfgs_loss_kwargs={"curl_power": CURL_POWER_LBFGS},
        checkpoint_path=checkpoint_path,
        lbfgs_note=f" on {len(LBFGS_OMEGAS)} fixed frequencies",
    )


# --------------------------------------------------------------------------- validation
#: ``‖pred − ref‖ / ‖ref‖`` — see :func:`src.experiments.relative_l2`.
_relative_l2 = relative_l2


def estimate_k_spp(
    net3: nn.Module, omega: float, n_line: int = 512, device: torch.device = DEVICE
) -> Dict[str, float]:
    """
    Re k_spp from the phase slope of H_y along x (and Im k_spp from ln|H_y|).

    The probe line sits at a quarter of the ω's own air-side box height, so it
    is in the near field at every frequency (a fixed 50 nm would be 0.9 δ_d at
    the blue edge and 0.09 δ_d at the red edge).
    """
    k_ref = mode_constants(omega)[0]
    x_max, y_max, _, z_max = domain_si(omega)
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
        "k_spp_imag_fit": k_imag_fit,
        "k_spp_imag_analytical": float(k_ref.imag),
    }
    if k_ref.imag != 0:
        out["k_spp_imag_rel_error"] = float(abs(k_imag_fit - k_ref.imag) / abs(k_ref.imag))
    return out


def fit_decay_constants(
    net3: nn.Module, omega: float, n_line: int = 200, guard: float = VAL_GUARD_HAT,
    device: torch.device = DEVICE,
) -> Dict[str, float]:
    """κ fits from ln|H_y| vs z on each side of the interface, over the ω's own box."""
    _, kappa_d, kappa_m = mode_constants(omega)
    x_max, y_max, z_min, z_max = domain_si(omega)
    guard_si = guard / k0_of(omega)
    x = 0.25 * x_max
    return measurement.fit_decay_constants(
        net3, kappa_d.real, kappa_m.real,
        x=x, y=y_max / 2, z_min=z_min, z_max=z_max, guard=guard_si,
        n_line=n_line, device=device,
    )


def continuity_residuals(
    net3: nn.Module, omega: float, n_points: int = 2000, offset: float = VAL_GUARD_HAT,
    device: torch.device = DEVICE,
) -> Dict[str, float]:
    """Tangential continuity residual at z = ±offset/k₀(ω), relative to the field RMS."""
    coords_hat, normals = sample_interface_hat(n_points, omega, device=device)
    coords = coords_hat / k0_of(omega)
    off = offset / k0_of(omega)
    return measurement.continuity_residuals(net3, coords, normals, off)


def validate_at_omega(
    model: nn.Module, omega: float, n_points: int = 8000, device: torch.device = DEVICE
) -> Dict[str, float]:
    """
    SI-unit validation of ``model.at_omega(omega)`` on fresh stratified points.

    Keys: rel L2 vs the analytical mode (overall and per half-space), SI curl
    residual RMS / (k₀·RMS field) per half-space, k_spp phase/amplitude fits,
    κ_d/κ_m decay fits, tangential-continuity residuals, the material at this ω.
    """
    net3 = model.at_omega(omega)
    net3.eval()
    k0 = k0_of(omega)
    eps_t, eps_n = hmm_eps(omega)
    coords = sample_collocation_si(n_points, omega, guard=VAL_GUARD_HAT, device=device)
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
        ref = analytical_fields_si(coords.detach(), omega)
        E_ref, H_ref = to_complex(ref.to(fields.dtype))

        metrics: Dict[str, float] = {
            "omega": float(omega),
            "omega_over_omega_ref": float(omega / OMEGA_REF),
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

    metrics.update(estimate_k_spp(net3, omega, device=device))
    metrics.update(fit_decay_constants(net3, omega, device=device))
    metrics.update(continuity_residuals(net3, omega, device=device))
    return metrics


def validate_band(
    model: nn.Module,
    omegas: Sequence[float] = (),
    n_points: int = 8000,
    device: torch.device = DEVICE,
) -> Dict[str, Dict[str, float]]:
    """Run :func:`validate_at_omega` at each ω; keys are ``f"{ω/ω_ref:.4f}"``."""
    omegas = omegas or VALIDATION_OMEGAS
    return {
        f"{omega / OMEGA_REF:.4f}": validate_at_omega(model, omega, n_points, device)
        for omega in omegas
    }


def straight_line_reference(omegas: Sequence[float]) -> np.ndarray:
    """Least-squares straight line through the *analytical* Re k_spp(ω) on the band."""
    om = np.asarray(list(omegas), dtype=float)
    k = np.array([mode_constants(float(w))[0].real for w in om])
    return np.polyval(np.polyfit(om, k, 1), om)


def origin_line_reference(omegas: Sequence[float]) -> np.ndarray:
    """
    Best line **through the origin**, ``k = (ω/c)·n̄`` — i.e. the shape a
    *non-dispersive* ε would produce, and therefore exactly what the fixed-ε
    predecessor experiment recovered.

    Departure from this line is the most meaningful "how much dispersion is
    there" number for comparing the two experiments. It also happens to be the
    quantity ``examples/hyperbolic_metamaterial.linear_fit_residual`` reports
    as ``nonlinearity_percent`` (24.3 % here), because its unscaled
    ``lstsq(design, ...)`` with ``rcond=None`` silently truncates the intercept
    as rank-deficient — see the results doc's "repo defects".
    """
    om = np.asarray(list(omegas), dtype=float)
    k = np.array([mode_constants(float(w))[0].real for w in om])
    slope = float(np.dot(om, k) / np.dot(om, om))
    return slope * om


def chord_reference(omegas: Sequence[float]) -> np.ndarray:
    """
    Straight line through the analytical Re k_spp at the two band *endpoints*.

    This is the reference ``examples/hyperbolic_metamaterial.py`` uses for its
    ``nonlinearity_percent``, so quoting departures from it keeps this
    experiment's numbers comparable with ``figures/hyperbolic/hmm_summary.json``.
    """
    om = np.asarray(list(omegas), dtype=float)
    ends = np.array([OMEGA_MIN, OMEGA_MAX])
    k_ends = np.array([mode_constants(float(w))[0].real for w in ends])
    slope = (k_ends[1] - k_ends[0]) / (ends[1] - ends[0])
    return k_ends[0] + slope * (om - ends[0])


def nonlinearity_metrics(per_freq: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Does the PINN reproduce the *curvature* of k_spp(ω), or only its trend?

    Compares the scatter of the PINN's fitted k_spp about (a) the exact
    nonlinear branch and (b) the best straight line through that branch on the
    same grid. If the network had merely learned a linear dispersion, the two
    would be comparable; the ratio ``curvature_capture`` is how many times
    tighter the PINN sits on the curve than on the line.
    """
    om = np.array([m["omega"] for m in per_freq.values()])
    order = np.argsort(om)
    om = om[order]
    vals = list(per_freq.values())
    k_pinn = np.array([vals[i]["k_spp_fit"] for i in order])
    k_exact = np.array([vals[i]["k_spp_analytical"] for i in order])
    k_line = straight_line_reference(om)
    k_chord = chord_reference(om)

    # The band's own nonlinearity is a property of the curve, not of the 9-point
    # grid, so it is measured on a dense sweep (the coarse grid under-reports the
    # maximum departure).
    om_dense = np.linspace(OMEGA_MIN, OMEGA_MAX, 401)
    k_dense = np.array([mode_constants(float(w))[0].real for w in om_dense])
    k_dense_chord = chord_reference(om_dense)
    k_dense_line = straight_line_reference(om_dense)
    k_dense_origin = origin_line_reference(om_dense)
    span_dense = float(k_dense.max() - k_dense.min())
    k_origin = origin_line_reference(om)

    rms_curve = float(np.sqrt(np.mean((k_pinn - k_exact) ** 2)))
    rms_line = float(np.sqrt(np.mean((k_pinn - k_line) ** 2)))
    rms_line_exact = float(np.sqrt(np.mean((k_exact - k_line) ** 2)))
    k_span = float(k_exact.max() - k_exact.min())
    return {
        "nonlinearity_percent_chord": float(
            100.0 * np.abs(k_dense - k_dense_chord).max() / max(span_dense, 1e-30)
        ),
        "nonlinearity_percent_lsq_line": float(
            100.0 * np.abs(k_dense - k_dense_line).max() / max(span_dense, 1e-30)
        ),
        "nonlinearity_percent_origin_line": float(
            100.0 * np.abs(k_dense - k_dense_origin).max() / max(span_dense, 1e-30)
        ),
        "rms_residual_about_origin_line_per_m": float(
            np.sqrt(np.mean((k_pinn - k_origin) ** 2))
        ),
        "rms_analytical_curvature_about_origin_line_per_m": float(
            np.sqrt(np.mean((k_exact - k_origin) ** 2))
        ),
        "pinn_departure_from_chord_percent": float(
            100.0 * np.abs(k_pinn - k_chord).max() / max(k_span, 1e-30)
        ),
        "rms_residual_about_analytical_curve_per_m": rms_curve,
        "rms_residual_about_straight_line_per_m": rms_line,
        "rms_analytical_curvature_about_line_per_m": rms_line_exact,
        "max_analytical_curvature_about_line_per_m": float(np.abs(k_exact - k_line).max()),
        "curvature_capture_ratio": float(rms_line / max(rms_curve, 1e-30)),
        "curvature_as_percent_of_k_span": float(100.0 * rms_line_exact / max(k_span, 1e-30)),
        "k_spp_span_per_m": k_span,
        "k_spp_ratio_max_over_min": float(k_exact.max() / k_exact.min()),
    }


def summarise(per_freq: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Band-level worst/median statistics, the nonlinearity check and the tier."""
    rel = [max(m["rel_l2_E"], m["rel_l2_H"]) for m in per_freq.values()]
    k_err = [m["k_spp_rel_error"] for m in per_freq.values()]
    kappa_err = [
        max(m["kappa_d_fit_rel_error"], m["kappa_m_fit_rel_error"]) for m in per_freq.values()
    ]
    bound = all(
        m["decay_sign_correct_air"] > 0 and m["decay_sign_correct_metal"] > 0
        for m in per_freq.values()
    )
    summary = {
        "n_frequencies": float(len(per_freq)),
        "band_fraction": float(BAND_FRACTION),
        "omega_min": float(OMEGA_MIN),
        "omega_max": float(OMEGA_MAX),
        "worst_rel_l2": float(max(rel)),
        "median_rel_l2": float(np.median(rel)),
        "worst_k_spp_rel_error": float(max(k_err)),
        "median_k_spp_rel_error": float(np.median(k_err)),
        "worst_kappa_rel_error": float(max(kappa_err)),
        "bound_mode_everywhere": float(bound),
    }
    summary.update(nonlinearity_metrics(per_freq))
    summary["success_tier"] = success_tier(summary)
    return summary


def success_tier(summary: Dict[str, float]) -> str:
    """Tiers: minimum (bound everywhere, rel L2 < 0.5), target (rel L2 < 0.1 and
    k_spp < 1% at every ω), stretch (rel L2 < 0.02 and k_spp < 0.2%)."""
    return banded_success_tier(summary, stretch=(0.02, 0.002), target=(0.1, 0.01))


# --------------------------------------------------------------------------- plots
def plot_dispersion(per_freq: Dict[str, Dict[str, float]], out_dir: Path = FIGURES_DIR) -> str:
    """Headline: PINN k_spp(ω) on the nonlinear branch, plus the curvature residual."""
    out_dir.mkdir(parents=True, exist_ok=True)
    om = np.array([m["omega"] for m in per_freq.values()])
    order = np.argsort(om)
    om = om[order]
    vals = list(per_freq.values())

    def get(key: str) -> np.ndarray:
        return np.array([vals[i][key] for i in order])

    om_dense = np.linspace(OMEGA_MIN, OMEGA_MAX, 300)
    modes = [mode_constants(float(w)) for w in om_dense]
    k_dense = np.array([m[0].real for m in modes])
    kd_dense = np.array([m[1].real for m in modes])
    km_dense = np.array([m[2].real for m in modes])
    line_coeffs = np.polyfit(om_dense, k_dense, 1)
    k_line_dense = np.polyval(line_coeffs, om_dense)
    k_origin_dense = origin_line_reference(om_dense)
    x_dense = om_dense / OMEGA_REF
    x_pts = om / OMEGA_REF
    k_line_pts = np.polyval(line_coeffs, om)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))

    ax = axes[0, 0]
    ax.plot(x_dense, k_dense / 1e7, "k-", lw=1.8, label=r"analytical $k_{spp}(\omega)$, $\epsilon(\omega)$ dispersive")
    ax.plot(x_dense, k_line_dense / 1e7, "--", color="0.55", lw=1.4,
            label="best straight line through it")
    ax.plot(x_dense, k_origin_dense / 1e7, ":", color="0.35", lw=1.4,
            label=r"$k = \bar{n}\,\omega/c$ (non-dispersive $\epsilon$, i.e. the fixed-$\epsilon$ run)")
    ax.plot(x_pts, get("k_spp_fit") / 1e7, "ro", ms=7, mfc="none", mew=1.8,
            label="PINN phase-slope fit")
    ax.set_xlabel(r"$\omega / \omega_{\mathrm{ref}}$   ($\omega_{\mathrm{ref}} = 2\pi c/633$ nm)")
    ax.set_ylabel(r"Re $k_{\mathrm{spp}}$  [$10^7$ m$^{-1}$]")
    ax.set_title("Dispersion of the Ag/silica HMM SPP\n(one ω-conditioned PINN)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[0, 1]
    ax.plot(x_dense, (k_dense - k_line_dense) / 1e6, "k-", lw=1.8,
            label="analytical curvature (curve − line)")
    ax.plot(x_pts, (get("k_spp_fit") - k_line_pts) / 1e6, "ro", ms=7, mfc="none", mew=1.8,
            label="PINN − line")
    ax.plot(x_pts, (get("k_spp_fit") - get("k_spp_analytical")) / 1e6, "bs", ms=5,
            label="PINN − analytical curve")
    ax.axhline(0.0, color="0.55", ls="--", lw=1.4)
    ax.set_xlabel(r"$\omega / \omega_{\mathrm{ref}}$")
    ax.set_ylabel(r"$k_{\mathrm{spp}} - $ straight line  [$10^6$ m$^{-1}$]")
    ax.set_title("Curvature residual about the best straight line\n(does the PINN track the bend?)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.semilogy(x_dense, kd_dense / 1e6, "k-", lw=1.2, label=r"analytical $\kappa_d$")
    ax.semilogy(x_dense, km_dense / 1e6, "k--", lw=1.2, label=r"analytical $\kappa_m$")
    ax.semilogy(x_pts, get("kappa_d_fit") / 1e6, "bo", ms=6, mfc="none", label=r"PINN $\kappa_d$")
    ax.semilogy(x_pts, get("kappa_m_fit") / 1e6, "rs", ms=6, mfc="none", label=r"PINN $\kappa_m$")
    ax.set_xlabel(r"$\omega / \omega_{\mathrm{ref}}$")
    ax.set_ylabel(r"decay constant [$10^6$ m$^{-1}$]")
    ax.set_title(r"Transverse decay constants (~10$\times$ spread)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.semilogy(x_pts, get("rel_l2_E"), "o-", label="rel L2 E")
    ax.semilogy(x_pts, get("rel_l2_H"), "s-", label="rel L2 H")
    for f in LBFGS_OMEGAS:
        ax.axvline(f / OMEGA_REF, color="k", alpha=0.13, lw=6)
    ax.set_xlabel(r"$\omega / \omega_{\mathrm{ref}}$")
    ax.set_ylabel("relative L2 error vs analytical")
    ax.set_title("Field error across the band\n(grey bands: L-BFGS refinement frequencies)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    lam_lo = 2 * np.pi * C0 / OMEGA_MAX * 1e9
    lam_hi = 2 * np.pi * C0 / OMEGA_MIN * 1e9
    fig.suptitle(
        f"Dispersive Ag/silica HMM (f = {FILL_FRACTION:g}, "
        rf"$\epsilon_{{d2}}$ = {EPS_D2:g}) against air, "
        rf"$\lambda_0 \in$ [{lam_lo:.0f}, {lam_hi:.0f}] nm",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = out_dir / "dispersion.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


def plot_field_maps(
    model: nn.Module, omega: float, out_dir: Path = FIGURES_DIR, device: torch.device = DEVICE
) -> str:
    """Re H_y in the x-z plane at one ω: PINN vs analytical vs difference."""
    out_dir.mkdir(parents=True, exist_ok=True)
    net3 = model.at_omega(omega)
    net3.eval()
    nm = 1e9
    nx, nz = 160, 120
    x_max, y_max, z_min, z_max = domain_si(omega)
    x = torch.linspace(0.0, x_max, nx, device=device)
    zv = torch.linspace(z_min, z_max, nz, device=device)
    X, Z = torch.meshgrid(x, zv, indexing="ij")
    coords = torch.stack(
        [X.flatten(), torch.full_like(X.flatten(), y_max / 2), Z.flatten()], dim=1
    )
    with torch.no_grad():
        pred = net3(coords)[:, 4, 0].reshape(nx, nz).cpu().numpy()
        ref = analytical_fields_si(coords, omega)[:, 4, 0].reshape(nx, nz).cpu().numpy()
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
    eps_t, eps_n = hmm_eps(omega)
    fig.suptitle(
        f"HMM SPP at ω = {omega / OMEGA_REF:.4f} ω_ref (λ₀ = {2 * np.pi * C0 / omega * 1e9:.0f} nm), "
        f"ε_t = {eps_t:.3f}, ε_n = {eps_n:.3f}; x-z plane, interface at z = 0"
    )
    fig.tight_layout()
    p = out_dir / f"field_maps_omega_{omega / OMEGA_REF:.4f}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


def plot_history(history: Dict[str, list], out_dir: Path = FIGURES_DIR) -> str:
    """Training-curve figure; see :func:`src.experiments.plot_two_phase_history`."""
    return plot_two_phase_history(
        history, out_dir, "ω-conditioned dispersive-HMM SPP PINN training"
    )


# --------------------------------------------------------------------------- main
def write_metrics_json(
    path: Path, per_freq: Dict, summary: Dict, self_check: Dict, figures: Dict[str, str],
    run_info: Dict,
) -> None:
    """Write the per-frequency metrics, band summary and self-check to JSON."""
    write_json_report(path, {
        "per_frequency": per_freq,
        "summary": summary,
        "analytical_self_check": self_check,
        "figures": figures,
        "run_info": run_info,
    })


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_core_args(
        p, epochs=N_EPOCHS, n_points=BATCH_SIZE, lr=LEARNING_RATE,
        device=str(DEVICE), lbfgs_steps=LBFGS_STEPS,
        n_points_help="interior collocation points per epoch "
                      f"(split over {N_FREQ_SUB} sub-frequencies)",
    )
    p.add_argument("--band-fraction", type=float, default=1.0,
                   help="fraction of the recommended band to use, centred on its midpoint")
    add_output_args(
        p, quick_epochs=QUICK_EPOCHS, figures_dir=FIGURES_DIR, model_out=MODEL_PATH,
    )
    return p.parse_args(argv)


def main(argv=None) -> Dict:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    set_band_fraction(args.band_fraction)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    n_epochs, n_points, lbfgs_steps = (
        (QUICK_EPOCHS, 512, 0) if args.quick else (args.epochs, args.n_points, args.lbfgs_steps)
    )
    n_val_points = 2000 if args.quick else 8000
    lbfgs_dtype = torch.float64 if args.lbfgs_dtype == "float64" else torch.float32

    eps_t_lo, eps_n_lo = hmm_eps(OMEGA_MIN)
    eps_t_hi, eps_n_hi = hmm_eps(OMEGA_MAX)
    k_lo, kd_lo, km_lo = mode_constants(OMEGA_MIN)
    k_hi, kd_hi, km_hi = mode_constants(OMEGA_MAX)
    logger.info(
        "Ag/silica HMM f=%.2f eps_d2=%.2f vs air; band ω/ω_ref [%.4f, %.4f] "
        "(λ₀ [%.0f, %.0f] nm, band fraction %.2f)",
        FILL_FRACTION, EPS_D2, OMEGA_MIN / OMEGA_REF, OMEGA_MAX / OMEGA_REF,
        2 * np.pi * C0 / OMEGA_MAX * 1e9, 2 * np.pi * C0 / OMEGA_MIN * 1e9, BAND_FRACTION,
    )
    logger.info("dispersive ε: red edge ε_t=%.4f ε_n=%.4f | blue edge ε_t=%.4f ε_n=%.4f",
                eps_t_lo, eps_n_lo, eps_t_hi, eps_n_hi)
    logger.info(
        "mode: k_spp [%.3e, %.3e] /m (n_eff %.3f→%.3f), κ_d [%.3e, %.3e], κ_m [%.3e, %.3e]; "
        "scaled box x̂ [0, %.2f]→[0, %.2f], ẑ [%.2f, %.2f]→[%.2f, %.2f]",
        k_lo.real, k_hi.real, k_lo.real / k0_of(OMEGA_MIN), k_hi.real / k0_of(OMEGA_MAX),
        kd_lo.real, kd_hi.real, km_lo.real, km_hi.real,
        domain_hat(OMEGA_MIN)[0], domain_hat(OMEGA_MAX)[0],
        domain_hat(OMEGA_MIN)[2], domain_hat(OMEGA_MIN)[3],
        domain_hat(OMEGA_MAX)[2], domain_hat(OMEGA_MAX)[3],
    )
    logger.info(
        "device=%s epochs=%d n_points=%d (%d ω/epoch) lr=%.1e lbfgs_steps=%d (%s, %d fixed ω) "
        "seed=%d", device, n_epochs, n_points, N_FREQ_SUB, args.lr, lbfgs_steps,
        args.lbfgs_dtype, len(LBFGS_OMEGAS), args.seed,
    )

    self_check_omegas = SELF_CHECK_OMEGAS[:2] if args.quick else SELF_CHECK_OMEGAS
    self_check = validate_band(AnalyticalHMMSPP(), self_check_omegas, n_points=4000, device=device)
    for key, m in self_check.items():
        logger.info(
            "self-check ω/ω_ref = %s: rel_l2_E %.2e, k_spp err %.2e, κ_d err %.2e, κ_m err %.2e",
            key, m["rel_l2_E"], m["k_spp_rel_error"],
            m["kappa_d_fit_rel_error"], m["kappa_m_fit_rel_error"],
        )

    network = create_network(device=device)
    logger.info("network parameters: %d", sum(p.numel() for p in network.parameters()))

    t0 = time.perf_counter()
    checkpoint_path = args.model_out.with_suffix(".partial.pth")
    if args.resume:
        if checkpoint_path.exists():
            logger.info("resuming from %s (loss %.3e)", checkpoint_path,
                        load_checkpoint_into(network, checkpoint_path))
        else:
            logger.warning("--resume given but %s does not exist; training fresh", checkpoint_path)
    network, history = train(
        network, n_epochs=n_epochs, n_points=n_points, learning_rate=args.lr, device=device,
        lbfgs_steps=lbfgs_steps, lbfgs_dtype=lbfgs_dtype, checkpoint_path=checkpoint_path,
    )
    train_time = time.perf_counter() - t0
    logger.info("training time %.1f s", train_time)

    per_freq = validate_band(network, VALIDATION_OMEGAS, n_points=n_val_points, device=device)
    summary = summarise(per_freq)
    summary.update(
        train_time_s=train_time, epochs=float(n_epochs), n_points=float(n_points),
        lbfgs_steps=float(lbfgs_steps), lr=args.lr, seed=float(args.seed),
        final_loss=history["total"][-1], best_loss=min(history["total"]),
    )
    logger.info("%8s | %8s | %10s | %10s | %10s | %10s | %10s", "ω/ω_ref", "λ₀ nm", "rel_l2_E",
                "rel_l2_H", "k_spp err", "κ_d err", "κ_m err")
    for key, m in per_freq.items():
        logger.info(
            "%8s | %8.0f | %10.3e | %10.3e | %10.3e | %10.3e | %10.3e",
            key, m["wavelength_nm"], m["rel_l2_E"], m["rel_l2_H"], m["k_spp_rel_error"],
            m["kappa_d_fit_rel_error"], m["kappa_m_fit_rel_error"],
        )
    for k, v in summary.items():
        logger.info("%-44s %s", k, f"{v:.4e}" if isinstance(v, float) else v)

    figures = {
        "dispersion": plot_dispersion(per_freq, out_dir=args.figures_dir),
        # Both edges: the boxes differ several-fold, which is the per-ω domain in action.
        "field_maps_blue_edge": plot_field_maps(network, OMEGA_MAX, out_dir=args.figures_dir,
                                                device=device),
        "field_maps_red_edge": plot_field_maps(network, OMEGA_MIN, out_dir=args.figures_dir,
                                               device=device),
        "training_history": plot_history(history, out_dir=args.figures_dir),
    }

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": network.state_dict(),
            "config": {
                "hidden_dims": list(HIDDEN_DIMS), "fourier_modes": FOURIER_MODES,
                "fourier_k_range": FOURIER_K_RANGE,
                "fill_fraction": FILL_FRACTION, "eps_dielectric_layer": EPS_D2,
                "eps_superstrate": EPS_D, "drude": DRUDE,
                "omega_ref": OMEGA_REF, "omega_min": OMEGA_MIN, "omega_max": OMEGA_MAX,
                "band_fraction": BAND_FRACTION, "H0": H0,
                "E_scale": E_SCALE, "H_scale": H_SCALE,
                "input_scaling": "coords * k0(omega), k0 = omega / c",
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
        "band_omega": [OMEGA_MIN, OMEGA_MAX],
        "band_wavelength_nm": [2 * np.pi * C0 / OMEGA_MAX * 1e9, 2 * np.pi * C0 / OMEGA_MIN * 1e9],
        "lbfgs_omegas_over_omega_ref": [f / OMEGA_REF for f in LBFGS_OMEGAS],
        "validation_omegas_over_omega_ref": [f / OMEGA_REF for f in VALIDATION_OMEGAS],
        "n_freq_sub": N_FREQ_SUB, "lbfgs_dtype": args.lbfgs_dtype, "quick": bool(args.quick),
        "band_fraction": BAND_FRACTION,
    }
    write_metrics_json(
        args.figures_dir / "metrics.json", per_freq, summary, self_check, figures, run_info
    )
    logger.info("saved model to %s, figures + metrics.json to %s", args.model_out,
                args.figures_dir)
    logger.info("success tier: %s", summary["success_tier"])
    return {"per_frequency": per_freq, "summary": summary, "self_check": self_check}


if __name__ == "__main__":
    main()
