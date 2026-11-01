"""
Frequency-Conditioned SPP PINN: Dispersion Recovery for a Uniaxial Metamaterial

Trains ONE network conditioned on frequency to recover the bound TM SPP mode of
the type-II uniaxial metamaterial case (optical axis z, in-plane
ε_t = −4 + 0.2j, normal ε_n = 3 + 0.05j, against ε_d = 1) across the band
ω ∈ [0.85, 1.15]·ω₀ with ω₀ = 2πc / 633 nm. This replaces per-ω retraining —
the design-tool seed experiment — and its headline result is the
PINN-recovered dispersion curve k_spp(ω) against the closed form
(:func:`src.analytical.analytical_spp_fields`, exact at any ω).

Idealisation (stated up front)
------------------------------
The permittivities are held **non-dispersive across the band** (ε(ω) = const).
Because every mode constant then scales linearly with ω,

    k_spp(ω) = (ω/c)·n_spp,   κ_d(ω) ∝ ω,   κ_m(ω) ∝ ω   (n_spp fixed),

the analytical dispersion curve is a straight line through the origin and the
mode family is self-similar (the field pattern rescales spatially with ω).
The network is *not* told any of this — it sees ω only as a normalised input
feature — so recovering the linear dispersion from physics + boundary anchors
is a genuine test of frequency conditioning, and the same machinery applies
unchanged to a dispersive ε(ω) table.

Recipe (inherited from ``examples/validate_spp.py``, the validated single-ω
experiment): dimensionless frame (coords/λ₀, fields/η₀H₀), the
``DisplacementAdapter`` making the E_z jump exact (its ε_zz divisor is
frequency-independent here), boundary anchor weight 100 with a physics ramp,
per-medium curl weighting (1/|ε| for Adam, 1/√|ε| for L-BFGS), Adam followed
by a float64 L-BFGS refinement.

Frequency conditioning
----------------------
* **Input**: 4 features — (x, y, z)/λ₀ plus ω̂ = (ω − ω₀)/(0.15 ω₀) ∈ [−1, 1].
  Fourier features are applied to the *spatial* part only (the mode's spatial
  wavenumbers k̂_spp ∈ [5.7, 7.8], κ̂_m ∈ [8.4, 11.3] need them; the ω
  dependence is smooth, so ω̂ is appended raw to the feature vector). A
  4-input ``ElectromagneticPINN`` would push 40 rad/unit Fourier modes onto
  ω̂ too, so :class:`OmegaConditionedCore` wraps ``FourierEMFeatures`` +
  ``ElectromagneticPINN(use_fourier=False)`` instead.
* **Per-ω physics without per-ω loss objects**: in the dimensionless frame
  the residuals are ``∇̂×Ê = i·2π·(ω/ω₀)·Ĥ`` and
  ``∇̂×Ĥ = −i·2π·(ω/ω₀)·ε·Ê``. Both ω factors fold into the *material*
  arguments of a single ``MaxwellCurlLoss(frequency=2π)``: pass
  ``mu_r = ω/ω₀`` per row and ``epsilon = (ω/ω₀)·ε`` per row. One
  concatenated batch therefore mixes frequencies at the autograd cost of the
  single-ω recipe (verified to machine precision on the analytical mode).
* **Batching**: each epoch samples ``N_FREQ_SUB = 4`` fresh uniform ω's, one
  per sub-block of the batch, so the network sees the band densely; L-BFGS
  refines on a fixed float64 set spanning 5 frequencies including both ends.

Geometry is sized by the worst case over the band (all at ω_min = 0.85 ω₀,
where the mode is largest): x ∈ [0, 2λ_spp(ω_min)] = [0, 1387 nm],
z ∈ [−3.5/κ_m(ω_min), +1.2/κ_d(ω_min)] = [−264, +363 nm], y ∈ [0, 0.2λ₀].
Collocation z-strata use the sampled ω's own decay scales.

Validation (the headline): at 9 frequencies spanning the band (both ends
included; the 4 odd grid points are strictly held out from the L-BFGS set),
recover Re k_spp by a phase-slope fit at z = 50 nm (plus Im k_spp from the
amplitude slope, reported unscored), κ_d/κ_m by decay fits, and rel L2 vs the
analytical mode. The analytical mode itself is pushed through the identical
pipeline at 3 frequencies as a convention self-check. Success tiers:
minimum = bound mode at all 9 ω and rel L2 < 0.5 everywhere; target =
rel L2 < 0.1 at all ω and k_spp within 1% across the band; stretch =
rel L2 < 0.02 and k_spp within 0.2%.

Usage::

    python examples/validate_spp_dispersion.py [--epochs 5000] [--n-points 2048]
        [--lr 1e-3] [--seed 0] [--device cpu] [--lbfgs-steps 120]
        [--lbfgs-dtype {float64,float32}] [--quick]
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
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

from examples.validate_spp import DisplacementAdapter  # noqa: E402
from src.analytical import analytical_spp_fields, complex_to_pinn_format  # noqa: E402
from src.constants import C0, EPS0, ETA0, MU0  # noqa: E402
from src.experiments import (  # noqa: E402
    LinearFeature,
    TrainingConfig,
    add_core_args,
    add_output_args,
    banded_success_tier,
    load_core_checkpoint,
    measurement,
    relative_l2,
    run_training,
    write_checkpoint,
    write_json_report,
)
from src.models import (  # noqa: E402
    ElectromagneticPINN,
    FourierEMFeatures,
    MaxwellCurlLoss,
    MaxwellDivergenceLoss,
    TangentialContinuityLoss,
    to_complex,
)
from src.physics import MaxwellEquations  # noqa: E402
from src.physics.metamaterial import MetamaterialProperties  # noqa: E402

logger = logging.getLogger("validate_spp_dispersion")

# --------------------------------------------------------------------------- physics
LAMBDA0 = 633e-9  # m, band-centre free-space wavelength
OMEGA0 = 2 * np.pi * C0 / LAMBDA0  # rad/s, band centre
BAND_HALF_WIDTH = 0.15  # band = [1 - w, 1 + w]·ω₀
OMEGA_MIN = (1 - BAND_HALF_WIDTH) * OMEGA0
OMEGA_MAX = (1 + BAND_HALF_WIDTH) * OMEGA0
H0 = 1.0  # A/m

# Uniaxial type-II metamaterial (optical axis z), NON-DISPERSIVE across the band.
EPS_T = -4 + 0.2j  # in-plane (xx, yy)
EPS_N = 3 + 0.05j  # normal (zz)
EPS_D = 1.0  # dielectric half-space (air)

# eps_parallel = component along the optical axis = the normal component for axis 'z'.
_MATERIAL = MetamaterialProperties(EPS_N, EPS_T, "z")


def mode_constants(omega: float) -> Tuple[complex, complex, complex]:
    """Analytical ``(k_spp, κ_d, κ_m)`` at ``omega`` (bound-mode branches)."""
    return _MATERIAL.decay_constants(omega, EPS_D, "x")


#: The frequency feature: ω̂ = (ω − ω₀)/(0.15 ω₀) ∈ [−1, 1] across the band.
OMEGA_FEATURE = LinearFeature.centred(OMEGA0, BAND_HALF_WIDTH * OMEGA0)


def omega_hat(omega: float) -> float:
    """Normalised frequency feature ω̂ = (ω − ω₀)/(0.15 ω₀) ∈ [−1, 1] on the band."""
    return OMEGA_FEATURE.to_hat(omega)


def omega_from_hat(w_hat: float) -> float:
    """Inverse of :func:`omega_hat`."""
    return OMEGA_FEATURE.from_hat(w_hat)


# Dimensionless frame: x̂ = x/λ₀. The loss reference frequency is k̂₀(ω₀) = 2π;
# the per-row factor ω/ω₀ is folded into the material arguments (see module doc).
OMEGA_HAT_REF = 2 * np.pi
E_SCALE = ETA0 * abs(H0)
H_SCALE = abs(H0)
FIELD_SCALE = torch.tensor([E_SCALE] * 3 + [H_SCALE] * 3, dtype=torch.float32).view(1, 6, 1)

# --------------------------------------------------------------------------- domain
# Worst case over the band: the mode is largest at ω_min (λ_spp = 693 nm,
# δ_m = 75.5 nm, δ_d = 302 nm), so the domain sized there contains every ω's mode.
_K_SPP_MIN, _KAPPA_D_MIN, _KAPPA_M_MIN = mode_constants(OMEGA_MIN)
X_MAX = 2 * (2 * np.pi / _K_SPP_MIN.real)  # 2 λ_spp(ω_min) ≈ 1387 nm
Z_MIN = -3.5 / _KAPPA_M_MIN.real  # ≈ −264 nm
Z_MAX = 1.2 / _KAPPA_D_MIN.real  # ≈ +363 nm
Y_MAX = 0.2 * LAMBDA0

# --------------------------------------------------------------------------- defaults
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2048
N_EPOCHS = 5000
LEARNING_RATE = 1e-3
N_FREQ_SUB = 4  # fresh ω's per epoch, one per sub-block
BOUNDARY_WEIGHT = 100.0
DIVERGENCE_WEIGHT = 1.0
CONTINUITY_WEIGHT = 1.0
PHYSICS_RAMP_FRAC = 0.25
# Metal-side loss preconditioning, identical to the single-ω recipe (ε is fixed
# across the band, so the weights are scalars): anti-collapse 1/|ε| during Adam,
# 1/√|ε| during the L-BFGS refinement.
EPS_STIFF = max(abs(EPS_T), abs(EPS_N))
METAL_CURL_WEIGHT_ADAM = EPS_STIFF**-1.0
METAL_CURL_WEIGHT_LBFGS = EPS_STIFF**-0.5
METAL_DIV_WEIGHT = EPS_STIFF**-1.0
GUARD = 1e-9
VAL_GUARD = 2e-9
METAL_FRACTION = 0.45
AIR_UNIFORM_FLOOR = 0.3
CONTINUITY_OFFSET = 2e-9
QUICK_EPOCHS = 200
LBFGS_STEPS = 120
LBFGS_POINTS_FACTOR = 2
LBFGS_DTYPE = torch.float64
# Fixed L-BFGS refinement set: 5 frequencies spanning the band incl. both ends.
LBFGS_OMEGAS = tuple(f * OMEGA0 for f in (0.85, 0.925, 1.0, 1.075, 1.15))
# Validation grid: 9 frequencies incl. both ends. The odd grid points (0.8875,
# 0.9625, 1.0375, 1.1125)·ω₀ are strictly held out from the L-BFGS set (Adam
# samples the continuum, so no discrete ω is ever trained more than sampled).
VALIDATION_OMEGAS = tuple(f * OMEGA0 for f in np.linspace(0.85, 1.15, 9))
SELF_CHECK_OMEGAS = tuple(f * OMEGA0 for f in (0.85, 1.0, 1.15))

FIGURES_DIR = REPO_ROOT / "figures" / "spp_dispersion"
MODEL_PATH = REPO_ROOT / "artifacts" / "models" / "spp_dispersion.pth"


# --------------------------------------------------------------------------- analytical
def analytical_fields_si(coords: torch.Tensor, omega: float) -> torch.Tensor:
    """Analytical SPP ``(E, H)`` at SI ``coords`` and ``omega``, ``[N, 6, 2]`` layout."""
    E, H = analytical_spp_fields(coords, omega, EPS_T, EPS_N, eps_dielectric=EPS_D, H0=H0)
    return torch.cat([complex_to_pinn_format(E), complex_to_pinn_format(H)], dim=1).to(
        coords.dtype if coords.is_floating_point() else torch.float32
    )


def analytical_fields_hat(coords_hat: torch.Tensor, omega: float) -> torch.Tensor:
    """Analytical mode at dimensionless coords, in the core's scaled units."""
    fields = analytical_fields_si(coords_hat * LAMBDA0, omega)
    return fields / FIELD_SCALE.to(device=fields.device, dtype=fields.dtype)


class AnalyticalDispersionSPP(nn.Module):
    """Exact SI SPP mode with the same ``(coords, omega)`` interface as the PINN.

    ``forward(coords_m, omega)`` and ``at_omega(omega)`` mirror
    :class:`SPPDispersionPINN`, so the full validation pipeline runs on the
    exact mode as a convention self-check at any probe frequency (this is the
    "analytical network that consumes the ω feature").
    """

    def forward(self, coords: torch.Tensor, omega: float) -> torch.Tensor:
        return analytical_fields_si(coords, omega)

    def at_omega(self, omega: float) -> nn.Module:
        return _FixedOmega(self, omega)


# --------------------------------------------------------------------------- network
class OmegaConditionedCore(nn.Module):
    """
    Dimensionless field network conditioned on frequency.

    Consumes ``[x̂, ŷ, ẑ, ω̂]`` of shape ``(N, 4)``: Fourier features are applied
    to the spatial columns only (band (0.1, 40) rad/unit covers κ̂_m up to 11.3
    at ω_max) and the raw ω̂ feature is appended before the complex MLP — the ω
    dependence of the mode is smooth, so it needs no Fourier encoding, and
    keeping it out of the random Fourier directions avoids spurious 40 rad/unit
    oscillations along the frequency axis. Returns ``(N, 6, 2)``.
    """

    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (128, 128, 128, 128),
        fourier_modes: int = 128,
        fourier_k_range: Tuple[float, float] = (0.1, 40.0),
    ):
        super().__init__()
        self.fourier = FourierEMFeatures(3, fourier_modes, fourier_k_range, include_dc=True)
        self.mlp = ElectromagneticPINN(
            spatial_dim=self.fourier.output_dim + 1,  # spatial features + ω̂
            field_components=6,
            hidden_dims=list(hidden_dims),
            complex_valued=True,
            frequency=OMEGA0,  # selects the time-harmonic input layout (no +1 column)
            use_fourier=False,  # spatial Fourier encoding is done here instead
            activation_type="complex_tanh",
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        features = self.fourier(coords[:, :3])
        return self.mlp(torch.cat([features, coords[:, 3:4]], dim=1))


class OmegaColumnNet(nn.Module):
    """
    3-column spatial view of a 4-input core with a fixed per-row ω̂ column.

    The differential-operator losses require ``coords`` with at most 3 columns
    (``src.physics.differential_ops`` raises on more), so this wrapper appends
    the stored ω̂ column — aligned row-for-row with the batch — inside the
    forward. Gradients flow through the spatial columns only, which is exactly
    the curl/divergence semantics (no ∂/∂ω enters Maxwell's equations).
    """

    def __init__(self, core: nn.Module, omega_hat_col: torch.Tensor):
        super().__init__()
        self.core = core
        self.omega_hat_col = omega_hat_col  # (N, 1), no grad

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.core(torch.cat([coords, self.omega_hat_col.to(coords.dtype)], dim=1))


class _FixedOmega(nn.Module):
    """SI 3-column module at a fixed ω: ``forward(coords_m) -> [N, 6, 2]``."""

    def __init__(self, parent: nn.Module, omega: float):
        super().__init__()
        self.parent = parent
        self.omega = float(omega)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.parent(coords, self.omega)


class SPPDispersionPINN(nn.Module):
    """
    SI-unit wrapper around the dimensionless ω-conditioned core.

    ``forward(coords_m, omega)`` accepts SI coordinates and a frequency and
    returns SI fields ``[N, 6, 2]``; ``at_omega(omega)`` returns a 3-column SI
    module for the (single-frequency) validation pipeline. ``core`` consumes
    ``[x̂, ŷ, ẑ, ω̂]`` and returns the dimensionless scaled fields.
    """

    def __init__(self, core: nn.Module, wavelength: float = LAMBDA0):
        super().__init__()
        self.core = core
        self.wavelength = float(wavelength)
        self.register_buffer("field_scale", FIELD_SCALE.clone())

    def forward(self, coords: torch.Tensor, omega: float) -> torch.Tensor:
        coords_hat = coords / self.wavelength
        w = coords_hat.new_full((coords_hat.shape[0], 1), omega_hat(omega))
        scale = self.field_scale.to(coords_hat.dtype)
        return self.core(torch.cat([coords_hat, w], dim=1)) * scale

    def at_omega(self, omega: float) -> nn.Module:
        return _FixedOmega(self, omega)


def create_network(
    hidden_dims: Tuple[int, ...] = (128, 128, 128, 128),
    fourier_modes: int = 128,
    device: torch.device = DEVICE,
) -> SPPDispersionPINN:
    """Build the ω-conditioned SPP PINN (Fourier spatial features + ω̂ + adapter).

    The ``DisplacementAdapter`` divisor is the zz permittivity component (ε_n
    below the interface, ε_d above) — frequency-independent here because ε is
    non-dispersive across the band. It indexes z at column 2, so it works
    unchanged on 4-column ``[x̂, ŷ, ẑ, ω̂]`` inputs.
    """
    mlp = OmegaConditionedCore(hidden_dims=hidden_dims, fourier_modes=fourier_modes)
    adapter = DisplacementAdapter(mlp, eps_below=EPS_N, eps_above=EPS_D)
    return SPPDispersionPINN(adapter).to(device)


# --------------------------------------------------------------------------- permittivity
def metal_eps_diag() -> torch.Tensor:
    """Diagonal ``(3,)`` complex permittivity of the lower medium: (ε_t, ε_t, ε_n)."""
    return torch.tensor([EPS_T, EPS_T, EPS_N], dtype=torch.complex128)


def interior_material_args(
    omega_hat_col: torch.Tensor, metal: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-row ``(epsilon, mu_r)`` arguments for ``MaxwellCurlLoss(frequency=2π)``.

    In the dimensionless frame the curl residuals at frequency ω read
    ``∇̂×Ê − i·2π·(ω/ω₀)·Ĥ`` and ``∇̂×Ĥ + i·2π·(ω/ω₀)·ε·Ê``; with the loss's
    reference frequency fixed at 2π the ratio r = ω/ω₀ = 1 + 0.15·ω̂ folds into
    ``mu_r = r`` (per row) and ``epsilon = r·ε`` (per row), letting one batch
    mix frequencies. Exact — verified to float64 precision on the analytical
    mode.
    """
    r = 1.0 + BAND_HALF_WIDTH * omega_hat_col.reshape(-1)  # ω/ω₀ per row
    r64 = r.detach().to(torch.float64)
    if metal:
        diag = metal_eps_diag()
    else:
        diag = torch.full((3,), complex(EPS_D), dtype=torch.complex128)
    eps_rows = r64.view(-1, 1, 1) * torch.diag_embed(diag).to(r64.device)
    return eps_rows, r


# --------------------------------------------------------------------------- sampling
def _sample_z_metal(n: int, delta_m: float, guard: float, device: torch.device) -> torch.Tensor:
    """|z| ~ truncated Exp(scale=δ_m(ω)) on [guard, |Z_MIN|], returned negative."""
    depth = -Z_MIN - guard
    u = torch.rand(n, device=device)
    z_abs = guard - delta_m * torch.log1p(-u * (1.0 - math.exp(-depth / delta_m)))
    return -z_abs


def _sample_z_air(n: int, delta_d: float, guard: float, device: torch.device) -> torch.Tensor:
    """z on [guard, Z_MAX]: truncated Exp(scale=δ_d(ω)) mixed with a uniform floor."""
    n_uniform = int(round(AIR_UNIFORM_FLOOR * n))
    n_exp = n - n_uniform
    span = Z_MAX - guard
    u = torch.rand(n_exp, device=device)
    z_exp = guard - delta_d * torch.log1p(-u * (1.0 - math.exp(-span / delta_d)))
    z_uni = guard + span * torch.rand(n_uniform, device=device)
    return torch.cat([z_exp, z_uni])


def sample_collocation_points(
    n_points: int, omega: float, guard: float = GUARD, device: torch.device = DEVICE
) -> torch.Tensor:
    """
    Stratified interior points (SI metres, ``(n, 3)``, requires_grad) for one ω.

    The domain extents are the fixed worst-case ones; the z-strata exponential
    scales are that ω's own penetration depths, so every frequency's near-field
    is sampled densely.
    """
    _, kappa_d, kappa_m = mode_constants(omega)
    n_metal = int(round(METAL_FRACTION * n_points))
    n_air = n_points - n_metal
    z = torch.cat(
        [
            _sample_z_metal(n_metal, 1.0 / kappa_m.real, guard, device),
            _sample_z_air(n_air, 1.0 / kappa_d.real, guard, device),
        ]
    )
    x = torch.rand(n_points, device=device) * X_MAX
    y = torch.rand(n_points, device=device) * Y_MAX
    coords = torch.stack([x, y, z], dim=1)
    coords.requires_grad_(True)
    return coords


def _avoid_guard(z: torch.Tensor, guard: float) -> torch.Tensor:
    return torch.where(z.abs() < guard, torch.where(z < 0, -guard, guard), z)


def sample_boundary_points(
    n_points: int, guard: float = GUARD, device: torch.device = DEVICE
) -> torch.Tensor:
    """Points on the six faces of the SI domain, ``n_points // 6`` per face."""
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


def sample_training_batch(
    n_int: int,
    n_bc: int,
    n_if: int,
    omegas: Sequence[float],
    device: torch.device = DEVICE,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """
    One multi-frequency training batch: per-ω blocks concatenated.

    Interior points are split into air/metal with aligned ω̂ columns and the
    per-row material arguments for the folded-frequency curl loss; boundary
    anchors are the analytical mode at each block's own ω.
    """
    n_om = len(omegas)
    int_blocks, bc_blocks, tgt_blocks, if_blocks, nrm_blocks = [], [], [], [], []
    w_int, w_bc, w_if = [], [], []
    for i, omega in enumerate(omegas):
        ni = n_int // n_om + (n_int % n_om if i == n_om - 1 else 0)
        nb = max(6, n_bc // n_om)
        nf = max(1, n_if // n_om)
        w = omega_hat(omega)

        c = (sample_collocation_points(ni, omega, device=device) / LAMBDA0).detach().to(dtype)
        int_blocks.append(c)
        w_int.append(torch.full((c.shape[0], 1), w, dtype=dtype, device=device))

        b = (sample_boundary_points(nb, device=device) / LAMBDA0).to(dtype)
        bc_blocks.append(b)
        w_bc.append(torch.full((b.shape[0], 1), w, dtype=dtype, device=device))
        with torch.no_grad():
            tgt_blocks.append(analytical_fields_hat(b, omega).to(dtype))

        f, nrm = sample_interface_points(nf, device=device)
        if_blocks.append((f / LAMBDA0).to(dtype))
        nrm_blocks.append(nrm.to(dtype))
        w_if.append(torch.full((f.shape[0], 1), w, dtype=dtype, device=device))

    coords = torch.cat(int_blocks)
    w_col = torch.cat(w_int)
    metal = coords[:, 2] < 0
    coords_air = coords[~metal].requires_grad_(True)
    coords_metal = coords[metal].requires_grad_(True)
    w_air, w_metal = w_col[~metal], w_col[metal]
    eps_air, mu_air = interior_material_args(w_air, metal=False)
    eps_metal, mu_metal = interior_material_args(w_metal, metal=True)
    return {
        "coords_air": coords_air,
        "coords_metal": coords_metal,
        "w_air": w_air,
        "w_metal": w_metal,
        "eps_air": eps_air,
        "mu_air": mu_air,
        "eps_metal": eps_metal,
        "mu_metal": mu_metal,
        "boundary": torch.cat(bc_blocks),
        "w_bc": torch.cat(w_bc),
        "target": torch.cat(tgt_blocks),
        "iface": torch.cat(if_blocks),
        "normals": torch.cat(nrm_blocks),
        "w_if": torch.cat(w_if),
    }


# --------------------------------------------------------------------------- training
#: Atomic best-weights checkpointing (see :func:`src.experiments.write_checkpoint`).
#: A run takes ~1 h on CPU; ``--resume`` reloads the core weights and continues.
_write_checkpoint = write_checkpoint
load_checkpoint_into = load_core_checkpoint


def train(
    network: SPPDispersionPINN,
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
) -> Tuple[SPPDispersionPINN, Dict[str, list]]:
    """
    Train the ω-conditioned core: Maxwell curl + divergence residuals with the
    per-row folded frequency, tangential continuity across z = 0, and the
    analytical-mode anchor on the six faces at each block's own ω.

    Phase 1 draws ``N_FREQ_SUB`` fresh uniform ω's over the band each epoch, one
    per sub-block, with the metal curl weight at its anti-collapse 1/|ε|. Phase 2
    refines on a fixed batch spanning :data:`LBFGS_OMEGAS`, with that weight
    raised to 1/√|ε|. The schedule around the physics — ramp, best-iterate
    tracking, checkpointing — is :func:`src.experiments.run_training`.
    """
    core = network.core
    curl_loss = MaxwellCurlLoss(frequency=OMEGA_HAT_REF, mu0=1.0, eps0=1.0)
    div_loss = MaxwellDivergenceLoss()
    cont_loss = TangentialContinuityLoss(offset=CONTINUITY_OFFSET / LAMBDA0)
    eps_m_diag = metal_eps_diag()

    def compute_losses(batch: Dict[str, torch.Tensor], ramp=1.0, w_curl_m=None):
        if w_curl_m is None:
            w_curl_m = METAL_CURL_WEIGHT_ADAM
        # The per-row ω/ω₀ is folded into (epsilon, mu_r); see interior_material_args.
        l_curl = curl_loss.compute(
            network=OmegaColumnNet(core, batch["w_air"]),
            coords=batch["coords_air"],
            epsilon=batch["eps_air"],
            mu_r=batch["mu_air"],
        ) + w_curl_m * curl_loss.compute(
            network=OmegaColumnNet(core, batch["w_metal"]),
            coords=batch["coords_metal"],
            epsilon=batch["eps_metal"],
            mu_r=batch["mu_metal"],
        )
        # ∇·(εE) = 0 and ∇·H = 0 carry no ω factor: pass the unscaled ε.
        l_div = div_loss.compute(
            network=OmegaColumnNet(core, batch["w_air"]),
            coords=batch["coords_air"],
            epsilon=complex(EPS_D),
        ) + METAL_DIV_WEIGHT * div_loss.compute(
            network=OmegaColumnNet(core, batch["w_metal"]),
            coords=batch["coords_metal"],
            epsilon=eps_m_diag,
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
        """L-BFGS: the fixed band spanned by :data:`LBFGS_OMEGAS`."""
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
        lbfgs_loss_kwargs={"w_curl_m": METAL_CURL_WEIGHT_LBFGS},
        checkpoint_path=checkpoint_path,
        lbfgs_note=f" on {len(LBFGS_OMEGAS)} fixed frequencies",
    )


# --------------------------------------------------------------------------- validation
#: ``‖pred − ref‖ / ‖ref‖`` — see :func:`src.experiments.relative_l2`.
_relative_l2 = relative_l2


def _line_along_x(z: float, n: int, device: torch.device) -> torch.Tensor:
    x = torch.linspace(0.0, X_MAX, n, device=device)
    return torch.stack([x, torch.full_like(x, Y_MAX / 2), torch.full_like(x, z)], dim=1)


def estimate_k_spp(
    net3: nn.Module, omega: float, z: float = 50e-9, n_line: int = 512,
    device: torch.device = DEVICE,
) -> Dict[str, float]:
    """Re k_spp from the phase slope of H_y along x (and Im k_spp from ln|H_y|)."""
    k_ref = mode_constants(omega)[0]
    coords = _line_along_x(z, n_line, device)
    with torch.no_grad():
        _, H = to_complex(net3(coords))
    hy = H[:, 1].cpu().numpy().astype(np.complex128)
    x = coords[:, 0].cpu().numpy().astype(np.float64)
    k_fit = float(np.polyfit(x, np.unwrap(np.angle(hy)), 1)[0])
    k_imag_fit = -float(np.polyfit(x, np.log(np.abs(hy) + 1e-30), 1)[0])
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
    net3: nn.Module, omega: float, x: Optional[float] = None, n_line: int = 200,
    guard: float = VAL_GUARD, device: torch.device = DEVICE,
) -> Dict[str, float]:
    """κ fits from ln|H_y| vs z on each side of the interface at fixed (x, y)."""
    _, kappa_d, kappa_m = mode_constants(omega)
    if x is None:
        x = 0.25 * X_MAX
    return measurement.fit_decay_constants(
        net3, kappa_d.real, kappa_m.real,
        x=x, y=Y_MAX / 2, z_min=Z_MIN, z_max=Z_MAX, guard=guard,
        n_line=n_line, device=device,
    )


def continuity_residuals(
    net3: nn.Module, n_points: int = 2000, offset: float = VAL_GUARD,
    device: torch.device = DEVICE,
) -> Dict[str, float]:
    """Tangential continuity residual at z = ±offset, relative to the field RMS."""
    coords, normals = sample_interface_points(n_points, device=device)
    return measurement.continuity_residuals(net3, coords, normals, offset)


def validate_at_omega(
    model: nn.Module, omega: float, n_points: int = 8000, device: torch.device = DEVICE
) -> Dict[str, float]:
    """
    SI-unit validation of ``model.at_omega(omega)`` on fresh stratified points.

    Keys: rel L2 vs the analytical mode (overall and per half-space), curl
    residual RMS / (k₀·RMS field) per half-space, k_spp phase/amplitude fits,
    κ_d/κ_m decay fits, tangential-continuity residuals, field-scale summaries.
    """
    net3 = model.at_omega(omega)
    net3.eval()
    k0 = omega / C0
    coords = sample_collocation_points(n_points, omega, guard=VAL_GUARD, device=device)
    fields = net3(coords)
    E, H = to_complex(fields)

    maxwell = MaxwellEquations(omega, mu0=MU0, eps0=EPS0)
    curl_E = maxwell.curl_operator(E, coords)
    curl_H = maxwell.curl_operator(H, coords)
    z = coords[:, 2].detach()
    metal = z < 0
    eps_m_row = metal_eps_diag().to(dtype=E.dtype, device=device).view(1, 3)
    eps_d_row = torch.full((1, 3), complex(EPS_D), dtype=E.dtype, device=device)
    eps_diag = torch.where(metal.view(-1, 1), eps_m_row, eps_d_row)
    res_E = curl_E - 1j * omega * MU0 * H
    res_H = curl_H + 1j * omega * EPS0 * eps_diag * E

    with torch.no_grad():
        ref = analytical_fields_si(coords.detach(), omega)
        E_ref, H_ref = to_complex(ref.to(fields.dtype))

        metrics: Dict[str, float] = {
            "omega": float(omega),
            "omega_over_omega0": float(omega / OMEGA0),
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
    metrics.update(continuity_residuals(net3, device=device))
    return metrics


def validate_band(
    model: nn.Module,
    omegas: Sequence[float] = VALIDATION_OMEGAS,
    n_points: int = 8000,
    device: torch.device = DEVICE,
) -> Dict[str, Dict[str, float]]:
    """Run :func:`validate_at_omega` at each ω; keys are ``f"{ω/ω₀:.4f}"``."""
    return {
        f"{omega / OMEGA0:.4f}": validate_at_omega(model, omega, n_points, device)
        for omega in omegas
    }


def summarise(per_freq: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Band-level worst/median statistics and the success tier."""
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
    k_spp < 1% at every ω), stretch (rel L2 < 0.02 and k_spp < 0.2%)."""
    return banded_success_tier(summary, stretch=(0.02, 0.002), target=(0.1, 0.01))


# --------------------------------------------------------------------------- plots
def plot_dispersion(
    per_freq: Dict[str, Dict[str, float]], out_dir: Path = FIGURES_DIR,
) -> str:
    """Headline figure: PINN k_spp(ω) over the analytical dispersion + κ + rel L2."""
    out_dir.mkdir(parents=True, exist_ok=True)
    om = np.array([m["omega"] for m in per_freq.values()])
    order = np.argsort(om)
    om = om[order]
    vals = list(per_freq.values())
    get = lambda key: np.array([vals[i][key] for i in order])  # noqa: E731

    om_dense = np.linspace(OMEGA_MIN, OMEGA_MAX, 200)
    modes = [mode_constants(w) for w in om_dense]
    k_dense = np.array([m[0].real for m in modes])
    kd_dense = np.array([m[1].real for m in modes])
    km_dense = np.array([m[2].real for m in modes])
    x_dense = om_dense / OMEGA0
    x_pts = om / OMEGA0

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    ax.plot(x_dense, k_dense / 1e7, "k-", lw=1.5, label="analytical (non-dispersive ε)")
    ax.plot(x_pts, get("k_spp_fit") / 1e7, "ro", ms=6, mfc="none", mew=1.5,
            label="PINN phase-slope fit")
    ax.set_xlabel(r"$\omega / \omega_0$")
    ax.set_ylabel(r"Re $k_{\mathrm{spp}}$ [$10^7$ /m]")
    ax.set_title("SPP dispersion: one ω-conditioned PINN")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax2 = ax.twinx()
    ax2.semilogy(x_pts, 100 * get("k_spp_rel_error"), "b^", ms=4, alpha=0.6)
    ax2.set_ylabel(r"$|k_{\mathrm{fit}} - k|/k$ [%]", color="b", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="b", labelsize=7)

    ax = axes[1]
    ax.plot(x_dense, kd_dense / 1e6, "k-", lw=1.2, label=r"analytical $\kappa_d$")
    ax.plot(x_dense, km_dense / 1e6, "k--", lw=1.2, label=r"analytical $\kappa_m$")
    ax.plot(x_pts, get("kappa_d_fit") / 1e6, "bo", ms=6, mfc="none", label=r"PINN $\kappa_d$")
    ax.plot(x_pts, get("kappa_m_fit") / 1e6, "rs", ms=6, mfc="none", label=r"PINN $\kappa_m$")
    ax.set_xlabel(r"$\omega / \omega_0$")
    ax.set_ylabel(r"decay constant [$10^6$ /m]")
    ax.set_title("Transverse decay constants")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.semilogy(x_pts, get("rel_l2_E"), "o-", label="rel L2 E")
    ax.semilogy(x_pts, get("rel_l2_H"), "s-", label="rel L2 H")
    for f in LBFGS_OMEGAS:
        ax.axvline(f / OMEGA0, color="k", alpha=0.15, lw=4)
    ax.set_xlabel(r"$\omega / \omega_0$")
    ax.set_ylabel("relative L2 error vs analytical")
    ax.set_title("Field error across the band\n(grey bands: L-BFGS refinement frequencies)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    fig.suptitle(
        rf"Uniaxial metamaterial SPP, $\epsilon_t$ = {EPS_T}, $\epsilon_n$ = {EPS_N}, "
        rf"band [{1 - BAND_HALF_WIDTH:.2f}, {1 + BAND_HALF_WIDTH:.2f}]$\,\omega_0$"
    )
    fig.tight_layout()
    p = out_dir / "dispersion.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


def plot_field_maps(
    model: nn.Module, omega: float, out_dir: Path = FIGURES_DIR,
    device: torch.device = DEVICE,
) -> str:
    """Re H_y in the x-z plane at one ω: PINN vs analytical vs difference."""
    out_dir.mkdir(parents=True, exist_ok=True)
    net3 = model.at_omega(omega)
    net3.eval()
    nm = 1e9
    nx, nz = 160, 120
    x = torch.linspace(0.0, X_MAX, nx, device=device)
    zv = torch.linspace(Z_MIN, Z_MAX, nz, device=device)
    X, Z = torch.meshgrid(x, zv, indexing="ij")
    coords = torch.stack(
        [X.flatten(), torch.full_like(X.flatten(), Y_MAX / 2), Z.flatten()], dim=1
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
    fig.suptitle(
        f"SPP mode at ω = {omega / OMEGA0:.4f} ω₀ (λ₀ = {2 * np.pi * C0 / omega * 1e9:.0f} nm), "
        "x-z plane; interface at z = 0"
    )
    fig.tight_layout()
    p = out_dir / f"field_maps_omega_{omega / OMEGA0:.4f}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


def plot_history(history: Dict[str, list], out_dir: Path = FIGURES_DIR) -> str:
    """Training-curve figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for key in ("total", "curl", "div", "continuity", "boundary"):
        ax.semilogy(history["epoch"], history[key], label=key, linewidth=1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (dimensionless)")
    ax.set_title("ω-conditioned SPP PINN training")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = out_dir / "training_history.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


# --------------------------------------------------------------------------- main
def write_metrics_json(
    path: Path,
    per_freq: Dict,
    summary: Dict,
    self_check: Dict,
    figures: Dict[str, str],
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
    add_output_args(
        p, quick_epochs=QUICK_EPOCHS, figures_dir=FIGURES_DIR, model_out=MODEL_PATH,
    )
    return p.parse_args(argv)


def main(argv=None) -> Dict:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    n_epochs, n_points, lbfgs_steps = (
        (QUICK_EPOCHS, 512, 0) if args.quick else (args.epochs, args.n_points, args.lbfgs_steps)
    )
    n_val_points = 2000 if args.quick else 8000
    lbfgs_dtype = torch.float64 if args.lbfgs_dtype == "float64" else torch.float32

    k_lo, kd_lo, km_lo = mode_constants(OMEGA_MIN)
    k_hi, kd_hi, km_hi = mode_constants(OMEGA_MAX)
    logger.info(
        "band [%.2f, %.2f] ω₀ (ω₀ = 2πc/633 nm), ε_t = %s, ε_n = %s, ε_d = %s (non-dispersive)",
        1 - BAND_HALF_WIDTH, 1 + BAND_HALF_WIDTH, EPS_T, EPS_N, EPS_D,
    )
    logger.info(
        "mode ranges: k_spp [%.3e, %.3e] /m (k̂ [%.2f, %.2f]), κ_d [%.3e, %.3e] /m, "
        "κ_m [%.3e, %.3e] /m; domain x [0, %.0f] nm, z [%.0f, %.0f] nm",
        k_lo.real, k_hi.real, k_lo.real * LAMBDA0, k_hi.real * LAMBDA0,
        kd_lo.real, kd_hi.real, km_lo.real, km_hi.real,
        X_MAX * 1e9, Z_MIN * 1e9, Z_MAX * 1e9,
    )
    logger.info(
        "device=%s epochs=%d n_points=%d (%d ω/epoch) lr=%.1e lbfgs_steps=%d (%s, %d fixed ω) "
        "seed=%d",
        device, n_epochs, n_points, N_FREQ_SUB, args.lr, lbfgs_steps, args.lbfgs_dtype,
        len(LBFGS_OMEGAS), args.seed,
    )

    # Convention self-check: the exact mode through the identical pipeline.
    self_check_omegas = SELF_CHECK_OMEGAS[:2] if args.quick else SELF_CHECK_OMEGAS
    self_check = validate_band(
        AnalyticalDispersionSPP(), self_check_omegas, n_points=4000, device=device
    )
    for key, m in self_check.items():
        logger.info(
            "self-check ω = %s ω₀: rel_l2_E %.2e, k_spp err %.2e, κ_d err %.2e, κ_m err %.2e",
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
            logger.warning("--resume given but %s does not exist; training fresh",
                           checkpoint_path)
    network, history = train(
        network, n_epochs=n_epochs, n_points=n_points, learning_rate=args.lr,
        device=device, lbfgs_steps=lbfgs_steps, lbfgs_dtype=lbfgs_dtype,
        checkpoint_path=checkpoint_path,
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
    logger.info("%8s | %10s | %10s | %10s | %10s | %10s", "ω/ω₀", "rel_l2_E", "rel_l2_H",
                "k_spp err", "κ_d err", "κ_m err")
    for key, m in per_freq.items():
        logger.info(
            "%8s | %10.3e | %10.3e | %10.3e | %10.3e | %10.3e",
            key, m["rel_l2_E"], m["rel_l2_H"], m["k_spp_rel_error"],
            m["kappa_d_fit_rel_error"], m["kappa_m_fit_rel_error"],
        )
    for k, v in summary.items():
        logger.info("%-28s %s", k, f"{v:.4e}" if isinstance(v, float) else v)

    figures = {
        "dispersion": plot_dispersion(per_freq, out_dir=args.figures_dir),
        "field_maps": plot_field_maps(network, OMEGA_MAX, out_dir=args.figures_dir,
                                      device=device),
        "training_history": plot_history(history, out_dir=args.figures_dir),
    }

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": network.state_dict(),
            "config": {
                "hidden_dims": [128] * 4, "fourier_modes": 128,
                "fourier_k_range": (0.1, 40.0), "wavelength": LAMBDA0,
                "band_half_width": BAND_HALF_WIDTH, "omega0": OMEGA0,
                "eps_t": [EPS_T.real, EPS_T.imag], "eps_n": [EPS_N.real, EPS_N.imag],
                "eps_dielectric": EPS_D, "H0": H0,
                "E_scale": E_SCALE, "H_scale": H_SCALE,
                "domain": {"x_max": X_MAX, "y_max": Y_MAX, "z_min": Z_MIN, "z_max": Z_MAX},
            },
            "summary": summary,
        },
        args.model_out,
    )
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    run_info = {
        "lbfgs_omegas_over_omega0": [f / OMEGA0 for f in LBFGS_OMEGAS],
        "validation_omegas_over_omega0": [f / OMEGA0 for f in VALIDATION_OMEGAS],
        "n_freq_sub": N_FREQ_SUB, "lbfgs_dtype": args.lbfgs_dtype, "quick": bool(args.quick),
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
