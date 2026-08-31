"""
Surface-Plasmon-Polariton Validation for PINN Fundamentals

Trains a PINN to recover the bound TM SPP mode at a planar interface and checks
it against the machine-precision analytical mode
:func:`src.analytical.analytical_spp_fields`. Two material cases are supported
(``--case``):

- ``silver`` (default): isotropic silver/air at λ₀ = 633 nm,
  ε_m = −18.3 + 0.55j (z < 0), ε_d = 1 (z > 0).
- ``uniaxial``: type-II uniaxial metamaterial (optical axis z) with in-plane
  ε_t = −4 + 0.2j, normal ε_n = 3 + 0.05j, against ε_d = 1 — the anisotropic
  case benchmarked in ``tests/test_analytical_spp.py``.

The interior is constrained by Maxwell's curl and divergence equations with
the piecewise permittivity (:class:`src.models.MaxwellCurlLoss` with a
diagonal ε spec per medium), tangential continuity is enforced across the
interface (:class:`src.models.TangentialContinuityLoss`) and the solution is
made unique by a *soft Dirichlet* term pinning the fields on the six faces of
the domain to the analytical mode. No interior data are used.

Where ε_t vs ε_n enters (uniaxial case; silver is the ε_t = ε_n limit)
----------------------------------------------------------------------
The lower medium is uniaxial with optical axis z (the interface normal), so
its permittivity tensor is ``diag(ε_t, ε_t, ε_n)``:

- **Mode constants**: ``MetamaterialProperties(eps_parallel=ε_n,
  eps_perpendicular=ε_t, 'z', omega)`` — the constructor's ``eps_parallel``
  is the component *along the optical axis*, i.e. the normal component for
  axis 'z'. ``_decay_constants`` then gives k_spp, κ_d, κ_m for propagation
  along x (k_spp² = k₀² ε_d ε_n (ε_t − ε_d)/(ε_t ε_n − ε_d²)).
- **Interior curl/divergence losses**: the metal-side ε spec is the diagonal
  ``(ε_t, ε_t, ε_n)`` — ε_t multiplies E_x, E_y and ε_n multiplies E_z in
  both ``∇×H = −iωε₀(ε·E)`` and ``∇·(ε·E) = 0``.
- **DisplacementAdapter**: the continuous quantity across z = 0 is the normal
  displacement ``D_z = ε₀ ε_n E_z`` (ε_n, *not* ε_t, because D_z couples to
  the zz tensor component), so the adapter divides the MLP's channel 2 by
  ε_n below the interface and by ε_d above it.
- **Metal-side loss preconditioning**: the curl-H residual penalises an Ê
  error component-wise as ``(2π|ε_component|)²``; the stiffness is set by the
  largest component ``max(|ε_t|, |ε_n|)`` and the metal weight is that value
  to a negative, phase-dependent exponent (see ``METAL_CURL_EXPONENT_ADAM`` /
  ``METAL_CURL_EXPONENT_LBFGS``).
- **Fourier band**: the dimensionless wavenumbers must fit inside the
  feature band (0.1, 40): silver κ̂_m ≈ 27.6 is the binding constraint;
  uniaxial k̂_spp ≈ 6.75, κ̂_d ≈ 2.46, κ̂_m ≈ 9.86 all fit comfortably.

Sign convention (as in ``src.models.loss_functions``): time dependence
``exp(-iωt)``, so ``∇×E = iωμ₀H``, ``∇×H = -iωε₀εᵣE`` and ``Im ε > 0`` lossy.

Non-dimensionalisation
----------------------
Exactly as in ``examples/validate_plane_wave.py``: coordinates are divided by
λ₀ and the fields are scaled so the network outputs O(1) quantities::

    x̂ = x / λ₀,     Ê = E / (η₀ H₀),     Ĥ = H / H₀

in which Maxwell's equations read ``∇̂×Ê = i·2π·Ĥ`` and
``∇̂×Ĥ = -i·2π·εᵣ·Ê``, i.e. ``MaxwellCurlLoss(frequency=2π, mu0=1, eps0=1)``
with the relative permittivity passed per point. With ``H₀ = 1`` the mode has
``|Ĥ| ~ 1`` and ``|Ê| ~ 1.03`` (silver) / ``~ 1.07`` (uniaxial), so both
residuals are O(1) and equally weighted. :class:`SPPPINN` wraps the trained
core so that it accepts coordinates in metres and returns SI fields; all
validation metrics are computed in SI units.

Geometry (SI), derived from the case's mode scales: x ∈ [0, 2λ_spp],
y ∈ [0, 0.2λ₀] (the mode is y-invariant), z ∈ [−4.4/κ_m, +1.2/κ_d]
(silver: −101 nm … 503 nm; uniaxial: −282 nm … 308 nm). Collocation points
are stratified in z: ~45% in the metal with |z| exponentially biased toward
the interface (scale 1/κ_m) and ~55% in air (scale 1/κ_d with a uniform
floor); a guard band of ±1 nm around z = 0 is excluded because the ε jump
makes the autograd curl meaningless there.

Usage::

    python examples/validate_spp.py [--case {silver,uniaxial}] [--epochs 4000]
                                    [--n-points 2048] [--lr 1e-3] [--seed 0]
                                    [--device cpu] [--lbfgs-steps 100]
                                    [--lbfgs-dtype {float64,float32}] [--quick]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

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
from src.models import (  # noqa: E402
    ElectromagneticPINN,
    MaxwellCurlLoss,
    MaxwellDivergenceLoss,
    TangentialContinuityLoss,
    to_complex,
)
from src.physics import MaxwellEquations  # noqa: E402
from src.physics.metamaterial import MetamaterialProperties  # noqa: E402

logger = logging.getLogger("validate_spp")

# --------------------------------------------------------------------------- physics
LAMBDA0 = 633e-9  # m, free-space wavelength
OMEGA = 2 * np.pi * C0 / LAMBDA0  # rad/s
K0 = OMEGA / C0  # rad/m
H0 = 1.0  # A/m, |H_y| at the interface

# Material cases: the lower half-space (z < 0) permittivity tensor is
# diag(eps_metal_t, eps_metal_t, eps_metal_n) — optical axis along z.
CASES: Dict[str, Dict[str, complex]] = {
    # silver at 633 nm (isotropic: eps_t = eps_n; Im > 0 lossy, e^{-iwt})
    "silver": {"eps_metal_t": -18.3 + 0.55j, "eps_metal_n": -18.3 + 0.55j, "eps_diel": 1.0},
    # type-II uniaxial metamaterial (matches tests/test_analytical_spp.py)
    "uniaxial": {"eps_metal_t": -4 + 0.2j, "eps_metal_n": 3 + 0.05j, "eps_diel": 1.0},
}

# Dimensionless frame: x̂ = x/λ₀ -> ω̂ = k̂₀ = 2π, μ̂₀ = ε̂₀ = 1
OMEGA_HAT = 2 * np.pi
E_SCALE = ETA0 * abs(H0)  # V/m per dimensionless unit (|E| of the mode ~ 1.03-1.07 η₀)
H_SCALE = abs(H0)  # A/m per dimensionless unit
FIELD_SCALE = torch.tensor([E_SCALE] * 3 + [H_SCALE] * 3, dtype=torch.float32).view(1, 6, 1)

# --------------------------------------------------------------------------- defaults
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2048
N_EPOCHS = 4000
LEARNING_RATE = 1e-3
# The plane-wave example used boundary weight 10 with no ramp; here the
# metal-side curl-H residual is stiff (|2π ε_m| ≈ 115 penalty factor on Ê
# errors for |ε_m| = 18.3), so with full physics from epoch 0 Adam collapses
# to the trivial E = H = 0 minimiser and never fits the anchor (verified:
# a weight-10 run plateaued at boundary MSE ≈ the anchor's mean square for
# 4000 epochs, rel L2 ≈ 1). Cure: stronger anchor + a physics-loss ramp.
BOUNDARY_WEIGHT = 100.0
DIVERGENCE_WEIGHT = 1.0
CONTINUITY_WEIGHT = 1.0
PHYSICS_RAMP_FRAC = 0.25  # interior physics weight ramps 0 -> 1 over this epoch fraction
# Metal-side physics rebalancing exponents: the curl-H residual contains
# i·2π·(ε·Ê), so an Ê error in the metal is penalised (2π max|ε_comp|)² ≈ 335×
# harder than in air (silver) and the collapse basin dominates. The interior
# losses are therefore computed per medium and the metal terms weighted by
# max(|ε_t|, |ε_n|)^(-exponent). Exponent 1.0 (weight 1/|ε_m|) is the safe
# anti-collapse setting for the Adam phase (verified: it reached the minimum
# tier where the unweighted recipe collapsed to E = H = 0). The converged
# network's metal curl residual stayed high there (0.79 rel) and the κ_m fit
# 15.8% low, so the L-BFGS *refinement* phase — where collapse is no longer a
# risk because the mode is already established — presses the metal curl term
# ~4.3× harder with exponent 0.5 (1/√|ε_m|). Measured on 8 float64 L-BFGS
# steps from the minimum-tier checkpoint: κ_m error 5.0% with exponent 0.5
# vs 19.1% with exponent 1.0; metal curl-E residual 0.46 vs 0.68.
METAL_CURL_EXPONENT_ADAM = 1.0
METAL_CURL_EXPONENT_LBFGS = 0.5
METAL_DIV_EXPONENT = 1.0
GUARD = 1e-9  # training guard band excluded around z = 0
VAL_GUARD = 2e-9  # validation guard band
METAL_FRACTION = 0.45  # fraction of collocation points in the metal
AIR_UNIFORM_FLOOR = 0.3  # fraction of air points sampled uniformly in z
CONTINUITY_OFFSET = 2e-9  # m, evaluation offset of the tangential-continuity loss
Y_MAX = 0.2 * LAMBDA0
QUICK_EPOCHS = 200
# 50 float64 outer steps (max_iter=20 evaluations each, ~18 s/step on CPU at
# 4096 points) cost ~6× the 40 float32 steps of the minimum-tier run and keep
# the full experiment inside a ~35 min budget.
LBFGS_STEPS = 50
LBFGS_POINTS_FACTOR = 2  # fixed L-BFGS collocation set = factor * n_points
LBFGS_DTYPE = torch.float64  # L-BFGS refinement precision (float32 was the residual floor)

FIGURES_DIR = REPO_ROOT / "figures" / "spp_validation"


def configure_case(case: str) -> None:
    """
    Select the material case and (re)derive every case-dependent module global:
    permittivities, mode constants (k_spp, κ_d, κ_m via
    :class:`MetamaterialProperties` — note ``eps_parallel`` = the *normal*
    component ε_n for optical axis 'z'), domain extents, the metal-side loss
    preconditioning weights and the output paths.
    """
    global CASE, EPS_METAL_T, EPS_METAL_N, EPS_DIEL, _MATERIAL
    global K_SPP, KAPPA_D, KAPPA_M, LAMBDA_SPP, DELTA_D, DELTA_M
    global X_MAX, Z_MIN, Z_MAX
    global METAL_CURL_WEIGHT_ADAM, METAL_CURL_WEIGHT_LBFGS, METAL_DIV_WEIGHT
    global MODEL_PATH, FIGURE_PREFIX

    if case not in CASES:
        raise ValueError(f"Unknown case {case!r}; choose from {sorted(CASES)}")
    CASE = case
    spec = CASES[case]
    EPS_METAL_T = complex(spec["eps_metal_t"])  # in-plane (xx, yy) component
    EPS_METAL_N = complex(spec["eps_metal_n"])  # normal (zz) component
    EPS_DIEL = complex(spec["eps_diel"]).real  # air; kept real

    # eps_parallel is the component along the optical axis = ε_n for axis 'z'.
    _MATERIAL = MetamaterialProperties(EPS_METAL_N, EPS_METAL_T, "z", omega=OMEGA)
    K_SPP, KAPPA_D, KAPPA_M = _MATERIAL._decay_constants(OMEGA, EPS_DIEL, "x")
    LAMBDA_SPP = 2 * np.pi / K_SPP.real
    DELTA_D = 1.0 / KAPPA_D.real  # penetration into the dielectric
    DELTA_M = 1.0 / KAPPA_M.real  # penetration into the metal/metamaterial

    # Domain (SI, metres), derived from the mode scales
    X_MAX = 2 * LAMBDA_SPP
    Z_MIN = -4.4 * DELTA_M  # silver: -100.7 nm; uniaxial: -282 nm
    Z_MAX = 1.2 * DELTA_D  # silver: 503 nm; uniaxial: 308 nm

    # Metal-side loss preconditioning (see the exponent constants' note).
    eps_stiff = max(abs(EPS_METAL_T), abs(EPS_METAL_N))
    METAL_CURL_WEIGHT_ADAM = eps_stiff**-METAL_CURL_EXPONENT_ADAM
    METAL_CURL_WEIGHT_LBFGS = eps_stiff**-METAL_CURL_EXPONENT_LBFGS
    METAL_DIV_WEIGHT = eps_stiff**-METAL_DIV_EXPONENT

    suffix = "" if case == "silver" else f"_{case}"
    MODEL_PATH = REPO_ROOT / "artifacts" / "models" / f"spp_validation{suffix}.pth"
    FIGURE_PREFIX = "" if case == "silver" else f"{case}_"


configure_case("silver")


# --------------------------------------------------------------------------- analytical
def analytical_fields_si(coords: torch.Tensor) -> torch.Tensor:
    """Analytical SPP ``(E, H)`` at SI ``coords`` in the network's ``[N, 6, 2]`` layout.

    The output real dtype follows ``coords.dtype`` (float32 pipeline by
    default; float64 during the L-BFGS refinement phase).
    """
    E, H = analytical_spp_fields(
        coords, OMEGA, EPS_METAL_T, EPS_METAL_N, eps_dielectric=EPS_DIEL, H0=H0
    )
    return torch.cat([complex_to_pinn_format(E), complex_to_pinn_format(H)], dim=1).to(
        coords.dtype if coords.is_floating_point() else torch.float32
    )


def analytical_fields_hat(coords_hat: torch.Tensor) -> torch.Tensor:
    """Analytical mode at dimensionless coords, in the core's scaled ``[N, 6, 2]`` units."""
    fields = analytical_fields_si(coords_hat * LAMBDA0)
    return fields / FIELD_SCALE.to(device=fields.device, dtype=fields.dtype)


class AnalyticalSPP(nn.Module):
    """Exact SI SPP mode as an ``nn.Module`` (coords in metres -> ``[N, 6, 2]``).

    Differentiable w.r.t. coords, so it can be pushed through the full
    validation pipeline as a convention self-check (CPU only: the analytical
    routine builds CPU tensors internally).
    """

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return analytical_fields_si(coords)


# --------------------------------------------------------------------------- network
class DisplacementAdapter(nn.Module):
    """
    Dimensionless field network with the interface's E_z discontinuity built in.

    The physical mode has ``E_z`` jumping at z = 0 while the normal
    displacement ``D_z = ε₀ ε_zz E_z`` is continuous; a continuous MLP cannot
    represent the jump, and its smoothed version fights the divergence/curl
    losses exactly where the sampling is densest. The wrapped MLP therefore
    represents the *continuous* quantity D̂_z on channel 2, and this adapter
    divides it by the local zz permittivity component per point, so
    ``forward(coords_hat)`` returns genuine ``(Ê, Ĥ)`` in the ``[N, 6, 2]``
    layout with the exact jump. All losses see this module.

    ``eps_below`` must be the **normal** (zz) component of the lower medium —
    ε_m for isotropic silver, ε_n (not ε_t) for the uniaxial case — because
    D_z couples only to ε_zz. The values are kept as python complex and cast
    to the working dtype per forward, so ``.to(float64)`` conversions of the
    module are safe.
    """

    def __init__(self, mlp: nn.Module, eps_below: complex, eps_above: complex):
        super().__init__()
        self.mlp = mlp
        self.eps_below = complex(eps_below)
        self.eps_above = complex(eps_above)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        out = self.mlp(coords)  # [N, 6, 2]; channel 2 carries D̂_z
        fields = torch.complex(out[..., 0], out[..., 1])  # [N, 6]
        eps = torch.where(
            coords[:, 2] < 0,
            torch.tensor(self.eps_below, dtype=fields.dtype, device=fields.device),
            torch.tensor(self.eps_above, dtype=fields.dtype, device=fields.device),
        )
        e_z = fields[:, 2] / eps
        fields = torch.cat([fields[:, :2], e_z.unsqueeze(1), fields[:, 3:]], dim=1)
        return torch.stack([fields.real, fields.imag], dim=-1)


class SPPPINN(nn.Module):
    """
    SI-unit wrapper around a dimensionless :class:`ElectromagneticPINN` core.

    ``forward(coords_m)`` accepts coordinates in metres and returns SI fields
    ``[N, 6, 2]``; ``core(coords_m / LAMBDA0)`` returns the dimensionless
    scaled fields ``(Ê, Ĥ)`` used for training.
    """

    def __init__(self, core: nn.Module, wavelength: float = LAMBDA0):
        super().__init__()
        self.core = core
        self.wavelength = float(wavelength)
        self.register_buffer("field_scale", FIELD_SCALE.clone())

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.core(coords / self.wavelength) * self.field_scale


def create_network(
    hidden_dims: Tuple[int, ...] = (128, 128, 128, 128),
    fourier_modes: int = 128,
    device: torch.device = DEVICE,
) -> SPPPINN:
    """
    Build the SPP PINN (complex-valued MLP with Fourier features).

    In the dimensionless frame the silver mode's wavenumbers are k̂_spp ≈ 6.5,
    κ̂_d ≈ 1.5 and κ̂_m ≈ 27.6, so the Fourier band is widened to (0.1, 40)
    rad per input unit (the default 20 would not resolve the metal-side
    decay). The uniaxial case (k̂_spp ≈ 6.75, κ̂_d ≈ 2.46, κ̂_m ≈ 9.86) fits
    inside the same band.
    """
    mlp = ElectromagneticPINN(
        spatial_dim=3,
        field_components=6,
        hidden_dims=list(hidden_dims),
        complex_valued=True,
        frequency=OMEGA,  # time-harmonic input layout (3 coordinates)
        use_fourier=True,
        fourier_modes=fourier_modes,
        fourier_k_range=(0.1, 40.0),
        activation_type="complex_tanh",
    )
    adapter = DisplacementAdapter(mlp, eps_below=EPS_METAL_N, eps_above=EPS_DIEL)
    return SPPPINN(adapter).to(device)


# --------------------------------------------------------------------------- permittivity
def metal_eps_diag() -> torch.Tensor:
    """Diagonal ``(3,)`` complex permittivity of the lower medium: (ε_t, ε_t, ε_n)."""
    return torch.tensor([EPS_METAL_T, EPS_METAL_T, EPS_METAL_N], dtype=torch.complex128)


def epsilon_tensor(coords: torch.Tensor) -> torch.Tensor:
    """
    Per-point relative-permittivity tensor ``(N, 3, 3)`` complex:
    ``diag(ε_t, ε_t, ε_n)`` where z < 0 (isotropic ε_m·I for the silver case),
    ε_d·I where z ≥ 0. Works in SI or dimensionless coordinates (only the sign
    of z matters); the values carry no autograd graph.
    """
    z = coords[:, 2].detach()
    dtype = torch.complex128 if coords.dtype == torch.float64 else torch.complex64
    eps_m_diag = metal_eps_diag().to(dtype=dtype, device=coords.device)
    eps_d_diag = torch.full((3,), complex(EPS_DIEL), dtype=dtype, device=coords.device)
    diag = torch.where(z.view(-1, 1) < 0, eps_m_diag, eps_d_diag)
    return torch.diag_embed(diag)


# --------------------------------------------------------------------------- sampling
def _sample_z_metal(n: int, guard: float, device: torch.device) -> torch.Tensor:
    """|z| ~ truncated Exp(scale=δ_m) on [guard, |Z_MIN|], returned negative."""
    depth = -Z_MIN - guard
    u = torch.rand(n, device=device)
    z_abs = guard - DELTA_M * torch.log1p(-u * (1.0 - math.exp(-depth / DELTA_M)))
    return -z_abs


def _sample_z_air(n: int, guard: float, device: torch.device) -> torch.Tensor:
    """z on [guard, Z_MAX]: truncated Exp(scale=δ_d) mixed with a uniform floor."""
    n_uniform = int(round(AIR_UNIFORM_FLOOR * n))
    n_exp = n - n_uniform
    span = Z_MAX - guard
    u = torch.rand(n_exp, device=device)
    z_exp = guard - DELTA_D * torch.log1p(-u * (1.0 - math.exp(-span / DELTA_D)))
    z_uni = guard + span * torch.rand(n_uniform, device=device)
    return torch.cat([z_exp, z_uni])


def sample_collocation_points(
    n_points: int, guard: float = GUARD, device: torch.device = DEVICE
) -> torch.Tensor:
    """
    Stratified interior points in SI metres, shape ``(n_points, 3)``,
    ``requires_grad``. ~45% metal / ~55% air, both z-strata biased toward the
    interface, with |z| < ``guard`` excluded.
    """
    n_metal = int(round(METAL_FRACTION * n_points))
    n_air = n_points - n_metal
    z = torch.cat(
        [_sample_z_metal(n_metal, guard, device), _sample_z_air(n_air, guard, device)]
    )
    x = torch.rand(n_points, device=device) * X_MAX
    y = torch.rand(n_points, device=device) * Y_MAX
    coords = torch.stack([x, y, z], dim=1)
    coords.requires_grad_(True)
    return coords


def _avoid_guard(z: torch.Tensor, guard: float) -> torch.Tensor:
    """Push |z| < guard onto ±guard (z = 0 goes to +guard)."""
    return torch.where(z.abs() < guard, torch.where(z < 0, -guard, guard), z)


def sample_boundary_points(
    n_points: int, guard: float = GUARD, device: torch.device = DEVICE
) -> torch.Tensor:
    """
    Points on the six faces of the SI domain, ``n_points // 6`` per face.
    On the x/y faces, z values inside the guard band are nudged onto ±guard.
    """
    per_face = max(1, n_points // 6)
    low = torch.tensor([0.0, 0.0, Z_MIN], device=device)
    high = torch.tensor([X_MAX, Y_MAX, Z_MAX], device=device)
    faces = []
    for axis in range(3):
        for value in (low[axis], high[axis]):
            pts = low + torch.rand(per_face, 3, device=device) * (high - low)
            pts[:, axis] = value
            if axis != 2:
                pts[:, 2] = _avoid_guard(pts[:, 2], guard)
            faces.append(pts)
    return torch.cat(faces, dim=0)


def sample_interface_points(
    n_points: int, device: torch.device = DEVICE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Points on z = 0 (SI metres) and their unit normals ẑ."""
    x = torch.rand(n_points, device=device) * X_MAX
    y = torch.rand(n_points, device=device) * Y_MAX
    coords = torch.stack([x, y, torch.zeros_like(x)], dim=1)
    normals = torch.zeros_like(coords)
    normals[:, 2] = 1.0
    return coords, normals


# --------------------------------------------------------------------------- training
def train(
    network: SPPPINN,
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
) -> Tuple[SPPPINN, Dict[str, list]]:
    """
    Train the dimensionless core with Maxwell curl + divergence residuals
    (piecewise ε) in the interior, tangential continuity across z = 0, and a
    soft Dirichlet (analytical-mode) term on the six boundary faces.

    The interior physics terms (curl, divergence, continuity) are multiplied
    by a ramp ``min(1, epoch / (physics_ramp_frac * n_epochs))`` so the anchor
    establishes the mode's amplitude before the stiff metal-side curl term can
    pin the network in the trivial E = H = 0 basin (see the note at
    ``BOUNDARY_WEIGHT``). The logged history stores the *unramped* component
    values; ``total`` is the ramped training objective.

    Phase 1: ``n_epochs`` of Adam (cosine-annealed LR) on freshly sampled
    points, metal curl weight ``METAL_CURL_WEIGHT_ADAM`` (anti-collapse).
    Phase 2 (if ``lbfgs_steps > 0``): L-BFGS refinement (full physics weight,
    metal curl weight raised to ``METAL_CURL_WEIGHT_LBFGS``)
    on a fixed set of ``LBFGS_POINTS_FACTOR * n_points`` interior points; each
    outer step runs up to 20 function evaluations with strong-Wolfe line
    search. When ``lbfgs_dtype`` is float64 the network, collocation set and
    anchor targets are promoted to double precision for this phase (float32
    was the plane-wave experiment's residual floor) and the network is
    converted back to float32 at the end.

    Returns the network (weights restored to the lowest-loss iterate) and a
    history dict with ``epoch``, ``total``, ``curl``, ``div``, ``continuity``,
    ``boundary``, ``lr``.
    """
    core = network.core
    curl_loss = MaxwellCurlLoss(frequency=OMEGA_HAT, mu0=1.0, eps0=1.0)
    div_loss = MaxwellDivergenceLoss()
    # The core consumes dimensionless coords, so the offset is dimensionless too.
    cont_loss = TangentialContinuityLoss(offset=CONTINUITY_OFFSET / LAMBDA0)
    eps_m_diag = metal_eps_diag()

    optimizer = torch.optim.Adam(core.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, n_epochs), eta_min=learning_rate * 1e-2
    )

    history: Dict[str, list] = {
        "epoch": [], "total": [], "curl": [], "div": [], "continuity": [], "boundary": [], "lr": [],
    }
    best_loss = float("inf")
    best_state = {k: v.detach().clone() for k, v in core.state_dict().items()}

    n_boundary = max(6, n_points // 2)
    n_interface = max(1, n_points // 4)

    ramp_epochs = max(1, int(physics_ramp_frac * n_epochs))

    def compute_losses(
        coords_air, coords_metal, iface_hat, normals, boundary_hat, target_hat,
        ramp=1.0, w_curl_m=None,
    ):
        # Per-medium interior losses; the metal side is preconditioned by a
        # phase-dependent curl weight and METAL_DIV_WEIGHT (see the exponent
        # constants): Adam uses the anti-collapse 1/|ε|, L-BFGS 1/√|ε|.
        if w_curl_m is None:
            w_curl_m = METAL_CURL_WEIGHT_ADAM
        l_curl = curl_loss.compute(
            network=core, coords=coords_air, epsilon=complex(EPS_DIEL)
        ) + w_curl_m * curl_loss.compute(
            network=core, coords=coords_metal, epsilon=eps_m_diag
        )
        l_div = div_loss.compute(
            network=core, coords=coords_air, epsilon=complex(EPS_DIEL)
        ) + METAL_DIV_WEIGHT * div_loss.compute(
            network=core, coords=coords_metal, epsilon=eps_m_diag
        )
        l_cont = cont_loss.compute(
            network=core, interface_coords=iface_hat, normal_vectors=normals
        )
        l_bc = torch.mean((core(boundary_hat) - target_hat) ** 2)
        total = (
            ramp * (l_curl + divergence_weight * l_div + continuity_weight * l_cont)
            + boundary_weight * l_bc
        )
        return total, l_curl, l_div, l_cont, l_bc

    def sample_all(n_int, n_bc, n_if, dtype=torch.float32):
        coords_hat = (
            (sample_collocation_points(n_int, device=device) / LAMBDA0).detach().to(dtype)
        )
        metal = coords_hat[:, 2] < 0
        coords_air = coords_hat[~metal].requires_grad_(True)
        coords_metal = coords_hat[metal].requires_grad_(True)
        iface, normals = sample_interface_points(n_if, device=device)
        iface_hat = (iface / LAMBDA0).to(dtype)
        normals = normals.to(dtype)
        boundary_hat = (sample_boundary_points(n_bc, device=device) / LAMBDA0).to(dtype)
        with torch.no_grad():
            target_hat = analytical_fields_hat(boundary_hat)
        return coords_air, coords_metal, iface_hat, normals, boundary_hat, target_hat

    core.train()
    t0 = time.perf_counter()
    for epoch in range(n_epochs):
        batch = sample_all(n_points, n_boundary, n_interface)
        ramp = min(1.0, (epoch + 1) / ramp_epochs)
        optimizer.zero_grad(set_to_none=True)
        loss, l_curl, l_div, l_cont, l_bc = compute_losses(*batch, ramp=ramp)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        history["epoch"].append(epoch)
        history["total"].append(loss_val)
        history["curl"].append(l_curl.item())
        history["div"].append(l_div.item())
        history["continuity"].append(l_cont.item())
        history["boundary"].append(l_bc.item())
        history["lr"].append(optimizer.param_groups[0]["lr"])

        # Totals from partially ramped epochs are not comparable to the full
        # objective, so track the best iterate only once the ramp is complete.
        if ramp >= 1.0 and loss_val < best_loss and math.isfinite(loss_val):
            best_loss = loss_val
            best_state = {k: v.detach().clone() for k, v in core.state_dict().items()}

        if epoch % log_every == 0 or epoch == n_epochs - 1:
            logger.info(
                "epoch %5d | total %.3e | curl %.3e | div %.3e | cont %.3e | bc %.3e | lr %.2e | %.0fs",
                epoch, loss_val, l_curl.item(), l_div.item(), l_cont.item(), l_bc.item(),
                optimizer.param_groups[0]["lr"], time.perf_counter() - t0,
            )

    if lbfgs_steps > 0:
        core.load_state_dict(best_state)
        if lbfgs_dtype == torch.float64:
            core.to(torch.float64)
            logger.info("L-BFGS phase in float64")
        batch = sample_all(
            LBFGS_POINTS_FACTOR * n_points,
            LBFGS_POINTS_FACTOR * n_boundary,
            LBFGS_POINTS_FACTOR * n_interface,
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
                *batch, w_curl_m=METAL_CURL_WEIGHT_LBFGS
            )
            loss.backward()
            parts.update(
                curl=l_curl.item(), div=l_div.item(), cont=l_cont.item(), bc=l_bc.item()
            )
            return loss

        for step in range(lbfgs_steps):
            loss_val = float(lbfgs.step(closure).detach())
            epoch = n_epochs + step
            history["epoch"].append(epoch)
            history["total"].append(loss_val)
            history["curl"].append(parts["curl"])
            history["div"].append(parts["div"])
            history["continuity"].append(parts["cont"])
            history["boundary"].append(parts["bc"])
            history["lr"].append(float("nan"))
            if loss_val < best_loss and math.isfinite(loss_val):
                best_loss = loss_val
                best_state = {k: v.detach().clone() for k, v in core.state_dict().items()}
            logger.info(
                "lbfgs %3d | total %.3e | curl %.3e | div %.3e | cont %.3e | bc %.3e | %.0fs",
                step, loss_val, parts["curl"], parts["div"], parts["cont"], parts["bc"],
                time.perf_counter() - t0,
            )
            if not math.isfinite(loss_val):
                logger.warning("L-BFGS produced a non-finite loss; stopping refinement")
                break

    core.load_state_dict(best_state)
    network.to(torch.float32)  # evaluation/serialisation dtype; no-op for float32 phases
    logger.info("restored best weights (loss %.3e)", best_loss)
    return network, history


# --------------------------------------------------------------------------- validation
def _relative_l2(pred: torch.Tensor, ref: torch.Tensor) -> float:
    return (
        torch.linalg.vector_norm(pred - ref) / torch.linalg.vector_norm(ref).clamp_min(1e-30)
    ).item()


def _line_along_x(z: float, n: int, device: torch.device) -> torch.Tensor:
    x = torch.linspace(0.0, X_MAX, n, device=device)
    return torch.stack(
        [x, torch.full_like(x, Y_MAX / 2), torch.full_like(x, z)], dim=1
    )


def estimate_k_spp(
    network: nn.Module, z: float = 50e-9, n_line: int = 512, device: torch.device = DEVICE
) -> Dict[str, float]:
    """Recovered Re k_spp from a linear fit to the unwrapped phase of H_y along x."""
    coords = _line_along_x(z, n_line, device)
    with torch.no_grad():
        _, H = to_complex(network(coords))
    hy = H[:, 1].cpu().numpy().astype(np.complex128)
    x = coords[:, 0].cpu().numpy().astype(np.float64)
    phase = np.unwrap(np.angle(hy))
    k_fit = float(np.polyfit(x, phase, 1)[0])
    return {
        "k_spp_fit": k_fit,
        "k_spp_rel_error": float(abs(k_fit - K_SPP.real) / K_SPP.real),
        "k_spp_analytical": float(K_SPP.real),
    }


def fit_decay_constants(
    network: nn.Module, x: Optional[float] = None, n_line: int = 200,
    guard: float = VAL_GUARD, device: torch.device = DEVICE,
) -> Dict[str, float]:
    """
    κ fits from ln|H_y| vs z on each side of the interface at fixed (x, y).

    Air: slope of ln|H_y| is −κ_d, so ``kappa_d_fit = −slope``; metal: slope is
    +κ_m. Positive fitted values mean the decay has the correct (bound) sign.
    """
    if x is None:
        x = 0.25 * X_MAX
    out: Dict[str, float] = {}
    for side, z_lo, z_hi, kappa_ref, sign, name in [
        ("air", guard, 0.9 * Z_MAX, KAPPA_D.real, -1.0, "kappa_d"),
        ("metal", 0.95 * Z_MIN, -guard, KAPPA_M.real, 1.0, "kappa_m"),
    ]:
        z = torch.linspace(z_lo, z_hi, n_line, device=device)
        coords = torch.stack(
            [torch.full_like(z, x), torch.full_like(z, Y_MAX / 2), z], dim=1
        )
        with torch.no_grad():
            _, H = to_complex(network(coords))
        log_hy = np.log(np.abs(H[:, 1].cpu().numpy().astype(np.complex128)) + 1e-30)
        slope = float(np.polyfit(z.cpu().numpy().astype(np.float64), log_hy, 1)[0])
        kappa_fit = sign * slope
        out[f"{name}_fit"] = kappa_fit
        out[f"{name}_fit_rel_error"] = float(abs(kappa_fit - kappa_ref) / kappa_ref)
        out[f"{name}_analytical"] = float(kappa_ref)
        out[f"decay_sign_correct_{side}"] = float(kappa_fit > 0)
    return out


def continuity_residuals(
    network: nn.Module, n_points: int = 2000, offset: float = VAL_GUARD,
    device: torch.device = DEVICE,
) -> Dict[str, float]:
    """Tangential continuity residual at z = ±offset, relative to the field RMS."""
    coords, normals = sample_interface_points(n_points, device=device)
    with torch.no_grad():
        E_p, H_p = to_complex(network(coords + offset * normals))
        E_m, H_m = to_complex(network(coords - offset * normals))
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


def validate(
    network: nn.Module, n_points: int = 20000, device: torch.device = DEVICE
) -> Dict[str, float]:
    """
    SI-unit validation metrics on ``n_points`` fresh stratified points
    (excluding the ±2 nm guard band around the interface).

    Keys: relative L2 vs the analytical mode (E and H, overall and per
    half-space), curl residual RMS / (k₀ · RMS field) per half-space (the
    curl-H residual uses the case's diagonal ε: (ε_t, ε_t, ε_n) in the metal),
    recovered k_spp (phase-slope fit at z = 50 nm), κ_d/κ_m decay fits,
    tangential-continuity residuals at z = ±2 nm, and field-scale summaries.
    """
    network.eval()
    coords = sample_collocation_points(n_points, guard=VAL_GUARD, device=device)
    fields = network(coords)
    E, H = to_complex(fields)

    maxwell = MaxwellEquations(OMEGA, mu0=MU0, eps0=EPS0)
    curl_E = maxwell.curl_operator(E, coords)
    curl_H = maxwell.curl_operator(H, coords)
    z = coords[:, 2].detach()
    metal = z < 0
    eps_m_row = metal_eps_diag().to(dtype=E.dtype, device=device).view(1, 3)
    eps_d_row = torch.full((1, 3), complex(EPS_DIEL), dtype=E.dtype, device=device)
    eps_diag = torch.where(metal.view(-1, 1), eps_m_row, eps_d_row)  # (N, 3)
    res_E = curl_E - 1j * OMEGA * MU0 * H
    res_H = curl_H + 1j * OMEGA * EPS0 * eps_diag * E

    with torch.no_grad():
        E_ref, H_ref = analytical_spp_fields(
            coords.detach(), OMEGA, EPS_METAL_T, EPS_METAL_N, eps_dielectric=EPS_DIEL, H0=H0
        )
        E_ref = E_ref.to(E.dtype)
        H_ref = H_ref.to(H.dtype)

        metrics: Dict[str, float] = {
            "rel_l2_E": _relative_l2(E, E_ref),
            "rel_l2_H": _relative_l2(H, H_ref),
            "rel_l2_total": _relative_l2(
                fields,
                torch.cat([complex_to_pinn_format(E_ref), complex_to_pinn_format(H_ref)], 1).to(
                    fields.dtype
                ),
            ),
        }
        for side, mask in (("air", ~metal), ("metal", metal)):
            metrics[f"rel_l2_E_{side}"] = _relative_l2(E[mask], E_ref[mask])
            metrics[f"rel_l2_H_{side}"] = _relative_l2(H[mask], H_ref[mask])
            E_rms = torch.sqrt(torch.mean(torch.sum(E[mask].abs() ** 2, 1))).clamp_min(1e-30)
            H_rms = torch.sqrt(torch.mean(torch.sum(H[mask].abs() ** 2, 1))).clamp_min(1e-30)
            rE = torch.linalg.vector_norm(res_E[mask], dim=1)
            rH = torch.linalg.vector_norm(res_H[mask], dim=1)
            metrics[f"curl_E_residual_rel_{side}"] = (
                torch.sqrt(torch.mean(rE**2)) / (K0 * E_rms)
            ).item()
            metrics[f"curl_H_residual_rel_{side}"] = (
                torch.sqrt(torch.mean(rH**2)) / (K0 * H_rms)
            ).item()

        E_mag = torch.linalg.vector_norm(E, dim=1)
        H_mag = torch.linalg.vector_norm(H, dim=1)
        metrics["mean_abs_E"] = E_mag.mean().item()
        metrics["mean_abs_H"] = H_mag.mean().item()
        E_rms_all = torch.sqrt(torch.mean(torch.sum(E.abs() ** 2, 1))).clamp_min(1e-30)
        H_rms_all = torch.sqrt(torch.mean(torch.sum(H.abs() ** 2, 1))).clamp_min(1e-30)
        metrics["impedance_ratio"] = ((E_rms_all / H_rms_all) / ETA0).item()

    metrics.update(estimate_k_spp(network, device=device))
    metrics.update(fit_decay_constants(network, device=device))
    metrics.update(continuity_residuals(network, device=device))
    return metrics


# --------------------------------------------------------------------------- plots
def visualize(
    network: nn.Module, history: Optional[Dict[str, list]], metrics: Dict[str, float],
    out_dir: Path = FIGURES_DIR, device: torch.device = DEVICE,
    prefix: Optional[str] = None,
) -> Dict[str, str]:
    """Write training-curve, x–z field-map, decay-profile and phase figures.

    File names are prefixed per case (silver: none; uniaxial: ``uniaxial_``).
    """
    if prefix is None:
        prefix = FIGURE_PREFIX
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    network.eval()
    nm = 1e9

    if history is not None and history["epoch"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for key in ("total", "curl", "div", "continuity", "boundary"):
            ax.semilogy(history["epoch"], history[key], label=key, linewidth=1)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss (dimensionless)")
        ax.set_title(f"SPP PINN training ({CASE})")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        p = out_dir / f"{prefix}training_history.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths["training_history"] = str(p)

    # Field map: Re H_y in the x-z plane at y = Y_MAX / 2
    nx, nz = 160, 120
    x = torch.linspace(0.0, X_MAX, nx, device=device)
    zv = torch.linspace(Z_MIN, Z_MAX, nz, device=device)
    X, Z = torch.meshgrid(x, zv, indexing="ij")
    coords = torch.stack(
        [X.flatten(), torch.full_like(X.flatten(), Y_MAX / 2), Z.flatten()], dim=1
    )
    with torch.no_grad():
        pred = network(coords)[:, 4, 0].reshape(nx, nz).cpu().numpy()  # Re H_y
        ref = analytical_fields_si(coords)[:, 4, 0].reshape(nx, nz).cpu().numpy()
    Xn = X.cpu().numpy() * nm
    Zn = Z.cpu().numpy() * nm
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
    fig.suptitle(f"SPP mode ({CASE}) in the x-z plane (y = Y_MAX/2); interface at z = 0")
    fig.tight_layout()
    p = out_dir / f"{prefix}field_maps.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["field_maps"] = str(p)

    # Decay profiles: ln|H_y|(z) on both sides with fitted lines
    x0 = 0.25 * X_MAX
    z_line = torch.linspace(Z_MIN, Z_MAX, 400, device=device)
    coords = torch.stack(
        [torch.full_like(z_line, x0), torch.full_like(z_line, Y_MAX / 2), z_line], dim=1
    )
    with torch.no_grad():
        _, Hp = to_complex(network(coords))
        _, Hr = analytical_spp_fields(
            coords, OMEGA, EPS_METAL_T, EPS_METAL_N, eps_dielectric=EPS_DIEL, H0=H0
        )
    zs = z_line.cpu().numpy() * nm
    ln_p = np.log(np.abs(Hp[:, 1].cpu().numpy().astype(np.complex128)) + 1e-30)
    ln_r = np.log(np.abs(Hr[:, 1].cpu().numpy()) + 1e-30)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(zs, ln_r, "k-", label="analytical")
    ax.plot(zs, ln_p, "r--", label="PINN")
    kd_fit, km_fit = metrics.get("kappa_d_fit"), metrics.get("kappa_m_fit")
    ln0 = float(ln_r[np.argmin(np.abs(zs))])
    if kd_fit is not None:
        za = zs[zs > 0]
        ax.plot(za, ln0 - kd_fit * za / nm, "b:", label=f"fit κ_d = {kd_fit:.3e} /m")
        zm = zs[zs < 0]
        ax.plot(zm, ln0 + km_fit * zm / nm, "g:", label=f"fit κ_m = {km_fit:.3e} /m")
    ax.axvline(0.0, color="k", lw=0.8)
    ax.set_xlabel("z [nm]")
    ax.set_ylabel("ln |H_y|")
    ax.set_title(
        f"Decay profiles ({CASE}) at x = {x0 * nm:.0f} nm "
        f"(κ_d ref {KAPPA_D.real:.3e}, κ_m ref {KAPPA_M.real:.3e} /m)"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / f"{prefix}decay_profiles.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["decay_profiles"] = str(p)

    # Phase of H_y along x at z = 50 nm
    coords = _line_along_x(50e-9, 512, device)
    with torch.no_grad():
        _, Hp = to_complex(network(coords))
        _, Hr = analytical_spp_fields(
            coords, OMEGA, EPS_METAL_T, EPS_METAL_N, eps_dielectric=EPS_DIEL, H0=H0
        )
    xs = coords[:, 0].cpu().numpy() * nm
    ph_p = np.unwrap(np.angle(Hp[:, 1].cpu().numpy().astype(np.complex128)))
    ph_r = np.unwrap(np.angle(Hr[:, 1].cpu().numpy()))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ph_r, "k-", label=f"analytical (Re k_spp = {K_SPP.real:.4e} /m)")
    ax.plot(xs, ph_p, "r--", label=f"PINN (fit {metrics.get('k_spp_fit', float('nan')):.4e} /m)")
    ax.set_xlabel("x [nm]")
    ax.set_ylabel("arg H_y [rad]")
    ax.set_title(
        f"Phase ({CASE}) along x at z = 50 nm; k_spp rel. error "
        f"{metrics.get('k_spp_rel_error', float('nan')):.2e}"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / f"{prefix}phase_profile.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["phase_profile"] = str(p)
    return paths


# --------------------------------------------------------------------------- main
def success_tier(m: Dict[str, float]) -> str:
    """Classify against the experiment's minimum / target / stretch criteria."""
    rel = max(m["rel_l2_E"], m["rel_l2_H"])
    k_err = m["k_spp_rel_error"]
    kappa_err = max(m["kappa_d_fit_rel_error"], m["kappa_m_fit_rel_error"])
    decay_ok = m["decay_sign_correct_air"] > 0 and m["decay_sign_correct_metal"] > 0
    wavelike = k_err < 0.2
    if rel < 5e-3 and k_err < 0.01 and kappa_err < 0.02:
        return "stretch"
    if rel < 0.05 and k_err < 0.01 and kappa_err < 0.10:
        return "target"
    if rel < 0.5 and decay_ok and wavelike:
        return "minimum"
    return "not met"


def write_metrics_json(
    path: Path, case: str, metrics: Dict, ref_metrics: Dict, figures: Dict[str, str]
) -> None:
    """Merge this case's results into ``metrics.json`` under a per-case key.

    Earlier revisions wrote a single flat ``{"metrics": ...}`` object (the
    silver case); such a legacy file is migrated to ``{"silver": {...}}`` so
    the other case's entry is never clobbered.
    """
    data: Dict = {}
    if path.exists():
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            data = {}
    if "metrics" in data and not any(k in data for k in CASES):
        data = {"silver": data}
    data[case] = {"metrics": metrics, "analytical_reference": ref_metrics, "figures": figures}
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--case", choices=sorted(CASES), default="silver", help="material case")
    p.add_argument("--epochs", type=int, default=N_EPOCHS)
    p.add_argument("--n-points", type=int, default=BATCH_SIZE, help="interior collocation points per epoch")
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=str(DEVICE))
    p.add_argument("--lbfgs-steps", type=int, default=LBFGS_STEPS, help="L-BFGS outer steps after Adam (0 disables)")
    p.add_argument("--lbfgs-dtype", choices=("float64", "float32"), default="float64", help="precision of the L-BFGS refinement phase")
    p.add_argument("--quick", action="store_true", help=f"smoke run: {QUICK_EPOCHS} epochs, 512 points, no L-BFGS")
    p.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    p.add_argument("--model-out", type=Path, default=None, help="checkpoint path (default: per-case under artifacts/models)")
    return p.parse_args(argv)


def main(argv=None) -> Dict[str, float]:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    configure_case(args.case)
    model_out = args.model_out if args.model_out is not None else MODEL_PATH
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    n_epochs, n_points, lbfgs_steps = (
        (QUICK_EPOCHS, 512, 0) if args.quick else (args.epochs, args.n_points, args.lbfgs_steps)
    )
    lbfgs_dtype = torch.float64 if args.lbfgs_dtype == "float64" else torch.float32

    logger.info(
        "case=%s: λ0 = %.0f nm, ε_t = %s, ε_n = %s, k_spp = %.4e%+.4ej /m (λ_spp = %.1f nm), "
        "κ_d = %.4e /m (δ_d = %.0f nm), κ_m = %.4e /m (δ_m = %.1f nm)",
        CASE, LAMBDA0 * 1e9, EPS_METAL_T, EPS_METAL_N, K_SPP.real, K_SPP.imag,
        LAMBDA_SPP * 1e9, KAPPA_D.real, DELTA_D * 1e9, KAPPA_M.real, DELTA_M * 1e9,
    )
    logger.info(
        "device=%s epochs=%d n_points=%d lr=%.1e lbfgs_steps=%d (%s) seed=%d "
        "domain z [%.0f, %.0f] nm, metal curl weight %.3f (adam) / %.3f (lbfgs), div %.3f",
        device, n_epochs, n_points, args.lr, lbfgs_steps, args.lbfgs_dtype, args.seed,
        Z_MIN * 1e9, Z_MAX * 1e9, METAL_CURL_WEIGHT_ADAM, METAL_CURL_WEIGHT_LBFGS,
        METAL_DIV_WEIGHT,
    )

    # Convention self-check: the analytical mode through the full validation pipeline.
    ref_metrics = validate(AnalyticalSPP().to(device), n_points=4000, device=device)
    logger.info(
        "analytical-mode self-check: rel_l2_E %.2e, curl_E air %.2e, curl_H metal %.2e, "
        "k_spp err %.2e, κ_d err %.2e, κ_m err %.2e, continuity_H %.2e",
        ref_metrics["rel_l2_E"], ref_metrics["curl_E_residual_rel_air"],
        ref_metrics["curl_H_residual_rel_metal"], ref_metrics["k_spp_rel_error"],
        ref_metrics["kappa_d_fit_rel_error"], ref_metrics["kappa_m_fit_rel_error"],
        ref_metrics["continuity_H_rel"],
    )

    network = create_network(device=device)
    logger.info("network parameters: %d", sum(p.numel() for p in network.parameters()))

    t0 = time.perf_counter()
    network, history = train(
        network, n_epochs=n_epochs, n_points=n_points, learning_rate=args.lr,
        device=device, lbfgs_steps=lbfgs_steps, lbfgs_dtype=lbfgs_dtype,
    )
    train_time = time.perf_counter() - t0
    logger.info("training time %.1f s", train_time)

    metrics = validate(network, device=device)
    metrics["train_time_s"] = train_time
    metrics["epochs"] = n_epochs
    metrics["n_points"] = n_points
    metrics["lbfgs_steps"] = lbfgs_steps
    metrics["lbfgs_dtype"] = args.lbfgs_dtype
    metrics["lr"] = args.lr
    metrics["seed"] = args.seed
    metrics["case"] = CASE
    metrics["final_loss"] = history["total"][-1]
    metrics["best_loss"] = min(history["total"])
    metrics["success_tier"] = success_tier(metrics)
    for k, v in metrics.items():
        logger.info("%-32s %s", k, f"{v:.4e}" if isinstance(v, float) else v)

    figures = visualize(network, history, metrics, out_dir=args.figures_dir, device=device)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": network.state_dict(),
            "config": {
                "case": CASE, "hidden_dims": [128] * 4, "fourier_modes": 128,
                "fourier_k_range": (0.1, 40.0), "wavelength": LAMBDA0,
                "eps_metal_t": [EPS_METAL_T.real, EPS_METAL_T.imag],
                "eps_metal_n": [EPS_METAL_N.real, EPS_METAL_N.imag],
                "eps_dielectric": EPS_DIEL,
                "H0": H0, "E_scale": E_SCALE, "H_scale": H_SCALE,
            },
            "metrics": metrics,
        },
        model_out,
    )
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_json(args.figures_dir / "metrics.json", CASE, metrics, ref_metrics, figures)
    logger.info("saved model to %s, figures + metrics.json to %s", model_out, args.figures_dir)
    logger.info("success tier: %s", metrics["success_tier"])
    return metrics


if __name__ == "__main__":
    main()
