r"""
A PINN on the **Real Ag/Silica Multilayer** — Does Keeping the Layers Beat
Homogenising Them?

Every metamaterial result in this project so far has replaced the Ag/silica
multilayer by a single homogeneous uniaxial medium ``(ε_t, ε_n)``. That
substitution is *wrong by a measurable amount*: ``examples/emt_validity.py``
shows the error is first order in the layer period — a surface term left by
truncating the periodic stack at the air interface — and at a 30 nm period it
puts the homogenised ``Im k_spp`` ~25 % and ``Re k_spp`` ~2.3 % off the truth.

This experiment stops homogenising. The PINN is trained on the **actual layered
ε(z)**, at a single fixed frequency, and the question it answers is the obvious
one: *does the PINN land closer to the transfer-matrix truth than the
effective-medium prediction does?* The transfer matrix (:mod:`src.transfer_matrix`,
validated to 5e-17 against Fresnel and 3.6e-19 against the single-interface SPP)
is the ground truth for both ``k_spp`` and the field profile.

The methodological point: the displacement adapter generalises for free
--------------------------------------------------------------------------
The recipe that won the single-interface and dispersive-HMM runs
(``examples/validate_spp.py``, ``examples/validate_hmm_dispersion.py``) contains
one structural idea — the **DisplacementAdapter**. The network emits a
*continuous* normal displacement ``D̂_z`` on channel 2 and the adapter divides it
by the local ``ε_zz``, so the ``E_z`` discontinuity at the interface is exact by
construction instead of being smoothed by a continuous MLP.

That idea does not care how many interfaces there are. For a TM mode
``F(x, z) = F(z) e^{i k_x x}`` in a planar stack, ``∇ × H = −i ω ε₀ ε E`` gives

    E_z = −k_x H_y / (ω ε₀ ε(z))        ⟹      D_z = ε₀ ε E_z = −k_x H_y / ω,

in which **no ε survives**. The interface matrix ``D(j → j+1)`` of the transfer
matrix is precisely the statement that ``H_y`` and ``E_x`` are continuous at
every boundary; ``D_z`` inherits that continuity from ``H_y`` alone. So across
all thirteen interfaces of this stack the only discontinuous field component is
``E_z``, and it is exactly ``D_z/ε(z)`` — the same construction, unchanged, with
``ε(z)`` swapped from a two-valued step for the piecewise-constant profile of
the real stack. (Asserted, not assumed: see ``TestModeFieldProfile`` in
``tests/test_transfer_matrix.py``, which checks ``H_y``, ``E_x`` and ``D_z``
continuity and the exact ``ε_below/ε_above`` jump of ``E_z`` at every interface
of this very stack, and ``tests/examples/test_validate_multilayer.py``, which
checks the adapter divides by the *correct layer's* ε.)

Structure, and why this period
------------------------------
Ag (Drude, ε_∞ = 3.7, ħω_p = 9.1 eV, ħγ = 0.018 eV) / silica (ε = 2.25), metal
fill ``f = 0.30``, air above, **λ₀ = 633 nm fixed** (one hard thing at a time —
the frequency conditioning of ``validate_hmm_dispersion.py`` is deliberately not
also switched on). Termination: **metal**, i.e. a full ``f a`` silver layer
against the air — the natural stack and the ``'metal'`` case of
``examples/emt_validity.multilayer_stack``, which is the one whose EMT error is
large. A semi-infinite silver substrate closes the stack from below, exactly as
in that study, so the mode is strictly bound on both sides.

Period ``a`` = 30 nm, ``N`` = 6 periods (180 nm of stack). Both numbers come
from the tractability probe and a convergence scan reported in the results doc:

* ``a`` = 30 nm is where EMT is dramatically wrong (Re 2.3 %, Im 25 %) while the
  layers are still numerically resolvable — 9 nm of Ag and 21 nm of silica.
  ``a`` = 10 nm makes the discrepancy subtler in ``Re k`` (0.9 %) and needs a
  3× wider Fourier band; ``a`` = 60 nm resolves easily but is far outside any
  homogenisation regime, so beating EMT there proves less.
* ``N`` = 6 is where the finite stack has converged to the semi-infinite answer:
  ``n_eff`` changes by 0.01 % between ``N`` = 6 and ``N`` = 16, so the truth the
  PINN is compared against is the *homogenisation* error, not a truncation
  artefact. Fewer periods would be cheaper but would move the target.

Resolution — the two choices the layering forces
------------------------------------------------
The field now carries the layer periodicity, so both the feature band and the
collocation sampling have to be set from ``a`` rather than inherited.

* **Fourier band.** The spatial frequency of the stack is ``2π/a = 21.1 k₀`` at
  this period — the 1-D dispersion run's band of ``(0.1, 8)·k₀`` would not even
  reach the fundamental. The band here goes to ``3 × 2π/a = 63.3`` in
  ``k₀``-scaled units: ``H_y`` and ``E_x`` are only *C⁰* at each interface (their
  ``z``-derivatives jump with ε), and representing a kink needs harmonics well
  past the fundamental. It is also **anisotropic**, which the isotropic
  :class:`src.models.FourierEMFeatures` cannot express: along ``x`` the field
  contains one wavenumber, ``k_spp/k₀ ≈ 1.06``, so a 63-rad/unit feature along
  ``x`` would only invent oscillations the solution does not have. See
  :class:`LayeredFourierFeatures`.
* **Collocation sampling in z.** The thinnest layer is ``f a`` = 9 nm out of a
  568 nm domain. Exponential stratification toward the interface (the
  single-interface recipe) would starve the deep layers, so the stack region is
  sampled **uniformly** with only a light near-interface bias: at the default
  2048 interior points that is ≈ 45 points in every 9 nm silver layer and ≈ 105
  in every 21 nm silica layer, i.e. many points per layer even in the deepest
  period. ``sample_collocation_hat`` reports the count.

Ground truth and the anchor
---------------------------
``k_spp`` and the field profile both come from the transfer matrix:
:func:`src.transfer_matrix.find_mode` for the complex root and
:func:`src.transfer_matrix.mode_field_profile` for ``H_y(z)``, ``E_x(z)``,
``E_z(z)`` reconstructed from the layer amplitudes. The analytical
single-interface SPP is **not** the truth here, so the soft-Dirichlet anchor on
the six faces of the domain is the *TMM* profile.

Validation
----------
* The headline: ``Re k_spp`` and ``Im k_spp`` from the PINN vs the TMM truth vs
  the EMT prediction, with both error ratios stated explicitly.
* rel L2 of the field vs the TMM profile, overall and per region (inside the
  stack / in the air / in the substrate).
* A field-profile figure with ``|H_y|(z)`` and ``E_z(z)`` from all three, layer
  boundaries marked — the ``E_z`` sawtooth is the visual payoff.
* The TMM profile itself pushed through the identical pipeline as a self-check.

Tiers: minimum = bound mode recovered, rel L2 < 0.5, correct qualitative layer
structure; target = rel L2 < 0.1 and ``|k_spp`` error| smaller than EMT's;
stretch = rel L2 < 0.03 and ``k_spp`` within 1 % of TMM.

Usage::

    python examples/validate_multilayer.py [--epochs 4000] [--n-points 2048]
        [--lr 1e-3] [--seed 0] [--device cpu] [--lbfgs-steps 60]
        [--period-nm 30] [--n-periods 6] [--probe-epochs 800] [--probe-only]
        [--skip-probe] [--resume] [--quick]
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
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

from examples.emt_validity import (  # noqa: E402
    EPS_D as EPS_AIR,
)
from examples.emt_validity import (  # noqa: E402
    EPS_D2,
    FILL_FRACTION,
    emt_wavevector,
    multilayer_stack,
)
from src.analytical import analytical_spp_fields, complex_to_pinn_format  # noqa: E402
from src.constants import C0, EPS0, ETA0, MU0  # noqa: E402
from src.effective_medium import (  # noqa: E402
    drude_permittivity,
    hmm_permittivities,
    omega_from_wavelength,
)
from src.experiments import (  # noqa: E402
    LayeredAdapter,
    TrainingConfig,
    add_core_args,
    add_output_args,
    load_core_checkpoint,
    measurement,
    relative_l2,
    run_training,
    write_checkpoint,
    write_json_report,
)
from src.models import (  # noqa: E402
    ElectromagneticPINN,
    MaxwellCurlLoss,
    MaxwellDivergenceLoss,
    TangentialContinuityLoss,
    to_complex,
)
from src.physics import MaxwellEquations  # noqa: E402
from src.transfer_matrix import (  # noqa: E402
    find_mode,
    layer_boundaries,
    mode_field_profile,
    permittivity_profile,
)

logger = logging.getLogger("validate_multilayer")

# --------------------------------------------------------------------------- design point
LAMBDA0 = 633e-9
OMEGA = float(omega_from_wavelength(LAMBDA0))
K0 = OMEGA / C0
H0 = 1.0  # A/m at the top (stack/air) interface

PERIOD = 30e-9
N_PERIODS = 6
TERMINATION = "metal"
SUBSTRATE_PAD = 30e-9
"""Silver substrate included *inside* the PINN domain, below the stack.

The transfer-matrix stack is closed by a semi-infinite silver substrate; the
PINN domain has to stop somewhere, so it stops 30 nm into that substrate — about
1.3 e-foldings of the silver decay length at this ``k_x`` (23.1 nm), which puts
the bottom anchor face safely inside a homogeneous medium rather than on a layer
boundary.
"""

EPS_AG = complex(drude_permittivity(OMEGA))

# --------------------------------------------------------------------------- scaled frame
# x̂ = k₀ x makes Maxwell's curl equations frequency-free: ∇̂×Ê = i Ĥ and
# ∇̂×Ĥ = −i ε Ê with Ê = E/(η₀H₀), Ĥ = H/H₀ (as in validate_hmm_dispersion.py).
OMEGA_HAT = 1.0
E_SCALE = ETA0 * abs(H0)
H_SCALE = abs(H0)
FIELD_SCALE = torch.tensor([E_SCALE] * 3 + [H_SCALE] * 3, dtype=torch.float32).view(1, 6, 1)

# --------------------------------------------------------------------------- defaults
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2048
N_EPOCHS = 4000
LEARNING_RATE = 1e-3
BOUNDARY_WEIGHT = 100.0
DIVERGENCE_WEIGHT = 1.0
CONTINUITY_WEIGHT = 1.0
PHYSICS_RAMP_FRAC = 0.25
METAL_CURL_EXPONENT_ADAM = 1.0
METAL_CURL_EXPONENT_LBFGS = 0.5
METAL_DIV_EXPONENT = 1.0
QUICK_EPOCHS = 150
LBFGS_STEPS = 60
LBFGS_POINTS_FACTOR = 2
LBFGS_DTYPE = torch.float64
PROBE_EPOCHS = 800
PROBE_POINTS = 4096

GUARD = 0.5e-9
"""Half-width of the excluded band around **every** interface (m).

``ε`` jumps there, so the autograd ``∂_z(ε E_x)`` inside the divergence residual
is a delta function rather than a number. 0.5 nm removes 11 % of each 9 nm
silver layer — the smallest band that stays clear of float32 rounding on the
scaled coordinates, and small enough to leave the thin layers well sampled.
"""
VAL_GUARD = 1.0e-9
CONTINUITY_OFFSET = 0.5e-9
"""Evaluation offset of the tangential-continuity loss (m).

Two constraints, and the *smaller* one binds. It must stay inside the layer on
each side (trivial: 0.5 nm against a 9 nm layer), but it must also be small
against the field's own variation scale, because the loss compares the field at
``±offset`` and the exact solution is **not** equal at those two points: ``H_y``
varies by ``≈ 2 κ δ`` across the gap, which is a residual the true mode cannot
avoid. With ``κ_Ag ≈ 4.3e7`` /m the floor is ~4 % at 0.5 nm (and would be ~13 %
at the 1.5 nm one might pick from the layer thickness alone). Every measured
continuity residual in this experiment has to be read against that floor.
"""

X_PERIODS = 2.0  # x ∈ [0, 2 λ_spp]
Z_AIR_DEPTHS = 1.2  # z ≤ 1.2 / Re κ_d
Y_WAVELENGTHS = 0.2  # thin: the mode is y-invariant
METAL_FRACTION = 0.55  # fraction of collocation points below z = 0
STACK_UNIFORM_FLOOR = 0.7  # of those, the fraction sampled uniformly in z
AIR_UNIFORM_FLOOR = 0.3

FOURIER_Z_MODES = 64
FOURIER_X_MODES = 24
FOURIER_MIX_MODES = 40
FOURIER_K_MIN = 0.1
FOURIER_KZ_HARMONICS = 3.0
"""Fourier band along ``z``, as a multiple of the stack's own ``2π/a``.

``H_y`` and ``E_x`` are continuous but kinked at every interface, so the band
must reach past the fundamental to represent the kinks; three harmonics is the
smallest multiple for which the probe reaches the accuracy the anchor needs.
"""
FOURIER_KX_MAX = 6.0
"""Fourier band along ``x`` and ``y``, in ``k₀``-scaled units.

The mode contains exactly one ``x`` wavenumber, ``k_spp/k₀ ≈ 1.06``; the band is
kept low deliberately, because high-frequency features along ``x`` can only
manufacture oscillations that the solution does not have.
"""
HIDDEN_DIMS = (128, 128, 128, 128)

FIGURES_DIR = REPO_ROOT / "figures" / "multilayer"
MODEL_PATH = REPO_ROOT / "artifacts" / "models" / "multilayer.pth"


# ============================================================ geometry & truth
class Structure:
    """
    The Ag/silica stack, its transfer-matrix mode and every derived length.

    Built once at import (and rebuilt by :func:`configure_structure` when
    ``--period-nm`` / ``--n-periods`` change), because the domain, the Fourier
    band, the sampling strata and the anchor all descend from it.

    Attributes:
        period, n_periods: The layer period ``a`` (m) and the bilayer count.
        eps_layers, thicknesses: The stack in the increasing-``z`` ordering of
            :mod:`src.transfer_matrix`; ``eps_layers[0]`` is the semi-infinite
            silver substrate, ``eps_layers[-1]`` the air superstrate.
        z0: Position of the lowest interface, chosen so the **topmost**
            interface (stack against air) sits at ``z = 0``.
        boundaries: All ``len(eps_layers) - 1`` interface positions (m).
        k_tmm: The exact complex ``k_spp`` of this stack.
        k_emt: The homogenised prediction — the number this experiment is
            trying to beat.
        kappa_d: Air-side decay constant of the true mode (1/m).
    """

    def __init__(self, period: float, n_periods: int, termination: str = TERMINATION):
        self.period = float(period)
        self.n_periods = int(n_periods)
        self.termination = termination
        self.eps_layers, self.thicknesses = multilayer_stack(
            self.period, self.n_periods, FILL_FRACTION, EPS_AG, EPS_D2, termination
        )
        self.z0 = -float(sum(self.thicknesses))
        self.boundaries = layer_boundaries(self.thicknesses, self.z0)
        self.k_emt = complex(emt_wavevector(OMEGA, FILL_FRACTION, EPS_D2))
        root = find_mode(self.k_emt, K0, self.eps_layers, self.thicknesses)
        if root is None:
            raise RuntimeError(
                f"no bound mode found for a = {self.period * 1e9:.1f} nm, N = {self.n_periods}"
            )
        self.k_tmm = complex(root)
        self.kappa_d = complex(np.sqrt(self.k_tmm**2 - EPS_AIR * K0**2))
        if self.kappa_d.real < 0:
            self.kappa_d = -self.kappa_d

        # Domain (SI). The stack spans [z0, 0]; the substrate pad extends below.
        self.z_min = self.z0 - SUBSTRATE_PAD
        self.z_max = Z_AIR_DEPTHS / self.kappa_d.real
        self.lambda_spp = 2.0 * np.pi / self.k_tmm.real
        self.x_max = X_PERIODS * self.lambda_spp
        self.y_max = Y_WAVELENGTHS * LAMBDA0

        # EMT twin, for the three-way comparison and the overlay figure.
        self.eps_t, self.eps_n = (complex(v) for v in hmm_permittivities(
            OMEGA, FILL_FRACTION, EPS_D2
        ))

        # Bloch envelope of the true mode inside the stack, read off |H_y| at
        # the interfaces (which removes the intra-period ripple). It sets the
        # near-interface sampling bias and is the layered analogue of κ_m.
        envelope = self.profile(self.boundaries)
        self.envelope_kappa = float(
            np.polyfit(self.boundaries, np.log(np.abs(envelope.H_y)), 1)[0]
        )
        self.envelope_kappa_hat = self.envelope_kappa / K0

    # -- scaled-frame views ------------------------------------------------
    @property
    def boundaries_hat(self) -> np.ndarray:
        return self.boundaries * K0

    @property
    def domain_hat(self) -> Tuple[float, float, float, float]:
        """``(x̂_max, ŷ_max, ẑ_min, ẑ_max)`` in ``k₀``-scaled units."""
        return (self.x_max * K0, self.y_max * K0, self.z_min * K0, self.z_max * K0)

    @property
    def kz_band_hat(self) -> float:
        """Top of the ``z`` Fourier band: ``FOURIER_KZ_HARMONICS × (2π/a)/k₀``."""
        return FOURIER_KZ_HARMONICS * 2.0 * np.pi / (self.period * K0)

    def material_group(self, z: np.ndarray) -> np.ndarray:
        """0 = silver, 1 = silica, 2 = air, per ``z`` (SI metres)."""
        eps = permittivity_profile(z, self.eps_layers, self.thicknesses, z0=self.z0)
        group = np.full(eps.shape, 1, dtype=np.int64)
        group[np.isclose(eps, EPS_AG)] = 0
        group[np.isclose(eps, complex(EPS_AIR))] = 2
        return group

    def profile(self, z: np.ndarray):
        """TMM :class:`~src.transfer_matrix.ModeFieldProfile` at SI ``z``."""
        return mode_field_profile(
            self.k_tmm, K0, self.eps_layers, self.thicknesses, z,
            z0=self.z0, H0=H0, h0_at=0.0, omega=OMEGA,
        )

    def summary(self) -> Dict[str, object]:
        return {
            "wavelength_nm": LAMBDA0 * 1e9,
            "period_nm": self.period * 1e9,
            "n_periods": self.n_periods,
            "termination": self.termination,
            "fill_fraction": FILL_FRACTION,
            "metal_thickness_nm": FILL_FRACTION * self.period * 1e9,
            "dielectric_thickness_nm": (1.0 - FILL_FRACTION) * self.period * 1e9,
            "n_interfaces": int(self.boundaries.size),
            "eps_ag": [EPS_AG.real, EPS_AG.imag],
            "eps_silica": EPS_D2,
            "eps_air": EPS_AIR,
            "eps_t_emt": [self.eps_t.real, self.eps_t.imag],
            "eps_n_emt": [self.eps_n.real, self.eps_n.imag],
            "k_tmm": [self.k_tmm.real, self.k_tmm.imag],
            "k_emt": [self.k_emt.real, self.k_emt.imag],
            "n_eff_tmm": self.k_tmm.real / K0,
            "n_eff_emt": self.k_emt.real / K0,
            "emt_error_re": abs(self.k_emt.real - self.k_tmm.real) / abs(self.k_tmm.real),
            "emt_error_im": abs(self.k_emt.imag - self.k_tmm.imag) / abs(self.k_tmm.imag),
            "domain_nm": {
                "x": [0.0, self.x_max * 1e9],
                "y": [0.0, self.y_max * 1e9],
                "z": [self.z_min * 1e9, self.z_max * 1e9],
            },
            "kz_band_hat": self.kz_band_hat,
            "layer_wavenumber_over_k0": 2.0 * np.pi / (self.period * K0),
        }


STRUCT = Structure(PERIOD, N_PERIODS)


def configure_structure(period: float, n_periods: int) -> Structure:
    """Rebuild the module-level :data:`STRUCT` for a different ``(a, N)``."""
    global STRUCT, PERIOD, N_PERIODS
    PERIOD, N_PERIODS = float(period), int(n_periods)
    STRUCT = Structure(PERIOD, N_PERIODS)
    return STRUCT


# ------------------------------------------------------------------ reference fields
def _phase(x: np.ndarray, k_x: complex) -> np.ndarray:
    return np.exp(1j * k_x * x)


class TMMFieldModule(nn.Module):
    r"""
    The transfer-matrix mode as a **differentiable** torch module.

    ``forward(coords_m) -> [N, 6, 2]`` in SI units, evaluating

        H_y(x, z) = [A_j e^{+i k_j ζ} + B_j e^{−i k_j ζ}] e^{i k_x x}

    with the per-medium amplitudes, ``k_z`` and reference planes taken once from
    :func:`src.transfer_matrix.mode_field_profile` and then re-evaluated in
    torch. Doing the evaluation in torch rather than numpy matters for one
    reason: the self-check has to pass through the *identical* pipeline, and the
    curl-residual part of that pipeline differentiates the module. A numpy
    round-trip detaches the graph and would report a curl residual of ~1
    (i.e. ``|∇×E| = 0`` against a non-zero ``ωμ₀H``) for a field that is exact.

    The medium lookup uses ``torch.bucketize(..., right=True)``, matching
    :func:`src.transfer_matrix.layer_index_at`; it is piecewise constant, so
    detaching it is correct rather than a compromise.
    """

    def __init__(self, struct: Optional[Structure] = None):
        super().__init__()
        struct = struct or STRUCT
        prof = struct.profile(np.asarray([0.0]))
        n_media = len(struct.eps_layers)
        reference = struct.boundaries[np.clip(np.arange(n_media) - 1, 0, None)]
        # Plain tensors, not buffers — see LayeredDisplacementAdapter for why a
        # complex table must not be exposed to ``Module.to(dtype)``.
        self.amplitudes = torch.as_tensor(prof.amplitudes, dtype=torch.complex128)
        self.k_z = torch.as_tensor(prof.k_z, dtype=torch.complex128)
        self.eps_values = torch.as_tensor(
            np.asarray([complex(e) for e in struct.eps_layers]), dtype=torch.complex128
        )
        self.boundaries = torch.as_tensor(struct.boundaries, dtype=torch.float64)
        self.reference = torch.as_tensor(reference, dtype=torch.float64)
        self.k_x = complex(struct.k_tmm)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        device = coords.device
        x = coords[:, 0].to(torch.float64)
        z = coords[:, 2].to(torch.float64)
        idx = torch.bucketize(
            z.detach().contiguous(), self.boundaries.to(device), right=True
        )
        zeta = (z - self.reference.to(device)[idx]).to(torch.complex128)
        k_z = self.k_z.to(device)[idx]
        eps = self.eps_values.to(device)[idx]
        amps = self.amplitudes.to(device)
        up = amps[idx, 0] * torch.exp(1j * k_z * zeta)
        down = amps[idx, 1] * torch.exp(-1j * k_z * zeta)
        phase = torch.exp(1j * self.k_x * x.to(torch.complex128))
        h_y = (up + down) * phase
        e_x = (k_z / eps) * (up - down) * phase / (OMEGA * EPS0)
        e_z = -self.k_x * h_y / (OMEGA * EPS0 * eps)
        zeros = torch.zeros_like(h_y)
        fields = torch.stack([e_x, zeros, e_z, zeros, h_y, zeros], dim=1)
        dtype = coords.dtype if coords.is_floating_point() else torch.float32
        return torch.stack([fields.real, fields.imag], dim=-1).to(dtype)


def tmm_field_module(struct: Optional[Structure] = None) -> TMMFieldModule:
    """The cached :class:`TMMFieldModule` for ``struct`` (built on first use)."""
    struct = struct or STRUCT
    module = getattr(struct, "_tmm_module", None)
    if module is None:
        module = TMMFieldModule(struct)
        struct._tmm_module = module  # type: ignore[attr-defined]
    return module


def tmm_fields_si(coords: torch.Tensor, struct: Optional[Structure] = None) -> torch.Tensor:
    """
    The transfer-matrix mode at SI ``coords``, in the network's ``[N, 6, 2]`` layout.

    ``F(x, z) = F(z) e^{i k_x x}`` with ``F(z)`` from
    :func:`src.transfer_matrix.mode_field_profile`. This is the ground truth for
    the anchor **and** for every field metric — the analytical single-interface
    SPP is not the truth for a layered stack.
    """
    return tmm_field_module(struct)(coords)


def tmm_fields_hat(coords_hat: torch.Tensor, struct: Optional[Structure] = None) -> torch.Tensor:
    """The TMM mode at ``k₀``-scaled coords, in the core's scaled field units."""
    fields = tmm_fields_si(coords_hat / K0, struct)
    return fields / FIELD_SCALE.to(device=fields.device, dtype=fields.dtype)


def emt_fields_si(coords: torch.Tensor, struct: Optional[Structure] = None) -> torch.Tensor:
    """
    The **homogenised** mode at SI ``coords``: the uniaxial half-space
    ``diag(ε_t, ε_t, ε_n)`` below ``z = 0``, air above. This is the field every
    previous experiment in the project treated as the answer.
    """
    struct = struct or STRUCT
    E, H = analytical_spp_fields(
        coords, OMEGA, struct.eps_t, struct.eps_n, eps_dielectric=EPS_AIR, H0=H0
    )
    return torch.cat([complex_to_pinn_format(E), complex_to_pinn_format(H)], dim=1).to(
        coords.dtype if coords.is_floating_point() else torch.float32
    )


class EMTFieldModule(nn.Module):
    """
    The **homogenised** mode with the PINN's SI interface.

    Pushed through the same :func:`validate` as the PINN, it turns "how wrong is
    EMT?" into the same numbers the PINN is scored on — the baseline for both
    the field metric and the ``k_spp`` comparison.
    """

    def __init__(self, struct: Optional[Structure] = None):
        super().__init__()
        self.struct = struct or STRUCT

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return emt_fields_si(coords, self.struct)


# ================================================================== network
class LayeredFourierFeatures(nn.Module):
    r"""
    Anisotropic Fourier features sized by the layer period.

    :class:`src.models.FourierEMFeatures` samples wave-vector *directions*
    isotropically, which is wrong for a layered stack: the required band along
    ``z`` (``3 × 2π/a = 63`` in ``k₀`` units at a 30 nm period) is sixty times
    the band the field actually contains along ``x`` (one wavenumber,
    ``k_spp/k₀ ≈ 1.06``). Isotropic sampling would spend most of its modes on
    high-frequency oscillation along ``x`` that the solution does not have —
    capacity wasted at best, spurious ripple at worst.

    The modes are therefore split three ways:

    * ``n_z`` **axis-aligned in z**, log-spaced over ``[k_min, k_z_max]`` — these
      carry the layer structure;
    * ``n_x`` **axis-aligned in x**, log-spaced over ``[k_min, k_xy_max]`` — the
      propagation phase;
    * ``n_mix`` oblique modes with ``k_x, k_y`` uniform in ``[−k_xy_max,
      k_xy_max]`` and ``|k_z|`` log-spaced to ``k_z_max`` with a random sign,
      so the encoding is not a strict tensor product.

    Each mode contributes a sine and a cosine; with ``include_dc`` the raw
    coordinates are passed through too.

    Args:
        k_z_max: Top of the ``z`` band (rad per scaled unit).
        k_xy_max: Top of the in-plane band.
        n_z, n_x, n_mix: Mode counts.
        k_min: Bottom of both log-spaced bands.
        include_dc: Append the raw coordinates.
        generator: Optional RNG for reproducible mode directions.
    """

    def __init__(
        self,
        k_z_max: float,
        k_xy_max: float = FOURIER_KX_MAX,
        n_z: int = FOURIER_Z_MODES,
        n_x: int = FOURIER_X_MODES,
        n_mix: int = FOURIER_MIX_MODES,
        k_min: float = FOURIER_K_MIN,
        include_dc: bool = True,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        self.include_dc = bool(include_dc)
        self.k_z_max = float(k_z_max)
        self.k_xy_max = float(k_xy_max)

        kz = torch.logspace(math.log10(k_min), math.log10(k_z_max), n_z, dtype=torch.float32)
        kx = torch.logspace(math.log10(k_min), math.log10(k_xy_max), n_x, dtype=torch.float32)
        zero_z, zero_x = torch.zeros(n_z), torch.zeros(n_x)
        z_modes = torch.stack([zero_z, zero_z, kz], dim=1)
        x_modes = torch.stack([kx, zero_x, zero_x], dim=1)

        kw = {"generator": generator} if generator is not None else {}
        mix_kz = torch.logspace(
            math.log10(k_min), math.log10(k_z_max), n_mix, dtype=torch.float32
        ) * torch.where(torch.rand(n_mix, **kw) < 0.5, -1.0, 1.0)
        mix_kx = (2.0 * torch.rand(n_mix, **kw) - 1.0) * k_xy_max
        mix_ky = (2.0 * torch.rand(n_mix, **kw) - 1.0) * k_xy_max
        mix_modes = torch.stack([mix_kx, mix_ky, mix_kz], dim=1)

        self.register_buffer("k_vectors", torch.cat([z_modes, x_modes, mix_modes], dim=0))

    @property
    def num_modes(self) -> int:
        return int(self.k_vectors.shape[0])

    @property
    def output_dim(self) -> int:
        return 2 * self.num_modes + (3 if self.include_dc else 0)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        projection = coords @ self.k_vectors.to(coords.dtype).T
        features = [torch.sin(projection), torch.cos(projection)]
        if self.include_dc:
            features.append(coords)
        return torch.cat(features, dim=1)


#: The displacement adapter with the **stack's** piecewise ``ε_zz(z)``.
#:
#: The same construction as the single-interface run, with the divisor looked up
#: in the real layer profile instead of being one of two constants — which is
#: the whole claim of the experiment, since ``D_z`` is continuous across *every*
#: interface (see the module docstring). :class:`src.experiments.LayeredAdapter`
#: carries the details, including why its ε table is deliberately not a
#: registered buffer: the float64 L-BFGS promotion would strip ``Im ε_Ag``, and
#: with it the imaginary part of ``k_spp`` this experiment exists to measure.
LayeredDisplacementAdapter = LayeredAdapter


class MultilayerCore(nn.Module):
    """Scaled field network: anisotropic Fourier features then a complex MLP."""

    def __init__(
        self,
        k_z_max: float,
        hidden_dims: Tuple[int, ...] = HIDDEN_DIMS,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        self.fourier = LayeredFourierFeatures(k_z_max, generator=generator)
        self.mlp = ElectromagneticPINN(
            spatial_dim=self.fourier.output_dim,
            field_components=6,
            hidden_dims=list(hidden_dims),
            complex_valued=True,
            frequency=OMEGA,  # selects the time-harmonic input layout
            use_fourier=False,  # the encoding above replaces it
            activation_type="complex_tanh",
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.fourier(coords))


class MultilayerPINN(nn.Module):
    """SI wrapper: ``forward(coords_m) -> [N, 6, 2]`` in V/m and A/m."""

    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core
        self.register_buffer("field_scale", FIELD_SCALE.clone())

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        scale = self.field_scale.to(coords.dtype)
        return self.core(coords * K0) * scale


def create_network(
    struct: Optional[Structure] = None,
    hidden_dims: Tuple[int, ...] = HIDDEN_DIMS,
    device: torch.device = DEVICE,
    generator: Optional[torch.Generator] = None,
) -> MultilayerPINN:
    """Build the layered SPP PINN (anisotropic Fourier + MLP + layered adapter)."""
    struct = struct or STRUCT
    core = MultilayerCore(struct.kz_band_hat, hidden_dims, generator=generator)
    adapter = LayeredDisplacementAdapter(core, struct.boundaries_hat, struct.eps_layers)
    return MultilayerPINN(adapter).to(device)


# ================================================================== sampling
def _push_off_boundaries(
    z_hat: torch.Tensor, boundaries_hat: torch.Tensor, guard_hat: float
) -> torch.Tensor:
    """Move every point within ``guard_hat`` of an interface onto ``±guard_hat``."""
    delta = z_hat.unsqueeze(1) - boundaries_hat.to(z_hat.dtype).unsqueeze(0)
    nearest = delta.abs().argmin(dim=1)
    offset = delta.gather(1, nearest.unsqueeze(1)).squeeze(1)
    sign = torch.where(offset < 0, -1.0, 1.0).to(z_hat.dtype)
    pushed = boundaries_hat.to(z_hat.dtype)[nearest] + sign * guard_hat
    return torch.where(offset.abs() < guard_hat, pushed, z_hat)


def sample_collocation_hat(
    n_points: int,
    struct: Optional[Structure] = None,
    guard: float = GUARD,
    device: torch.device = DEVICE,
) -> torch.Tensor:
    r"""
    Interior collocation points in ``k₀``-scaled units, stratified for the layers.

    ``METAL_FRACTION`` of the points go below ``z = 0`` (the stack plus the
    substrate pad) and the rest into the air.

    * **Below**: ``STACK_UNIFORM_FLOOR`` of them **uniform in z**, the rest
      exponentially biased toward the top interface with the mode's own scale.
      The uniform floor is the point: the field decays by ``e^{-3}`` across the
      stack, so a purely exponential rule would leave the deepest periods with a
      handful of points and the network free to invent whatever it liked there.
      Uniform sampling puts ≈ ``0.55·0.7·n·(f a)/(N a + pad)`` points in *every*
      silver layer — 45 of 2048 at the defaults.
    * **Above**: exponential with scale ``1/κ_d`` plus a uniform floor, as in the
      single-interface recipe.

    Points closer than ``guard`` to any interface are pushed clear (see
    :data:`GUARD`).
    """
    struct = struct or STRUCT
    x_max, y_max, z_min, z_max = struct.domain_hat
    guard_hat = guard * K0
    bounds = torch.as_tensor(struct.boundaries_hat, dtype=torch.float32, device=device)

    n_below = int(round(METAL_FRACTION * n_points))
    n_above = n_points - n_below

    n_uni = int(round(STACK_UNIFORM_FLOOR * n_below))
    z_uni = z_min + (0.0 - z_min) * torch.rand(n_uni, device=device)
    # The remainder is biased toward z = 0 with the stack's *own* Bloch envelope
    # scale (measured from the TMM profile, not from an EMT surrogate).
    depth = 1.0 / struct.envelope_kappa_hat
    n_exp = n_below - n_uni
    u = torch.rand(n_exp, device=device)
    span = -z_min
    z_exp = -(guard_hat - depth * torch.log1p(-u * (1.0 - math.exp(-span / depth))))
    z_below = torch.cat([z_uni, z_exp]).clamp(min=z_min, max=-guard_hat)

    n_air_uni = int(round(AIR_UNIFORM_FLOOR * n_above))
    n_air_exp = n_above - n_air_uni
    depth_air = K0 / struct.kappa_d.real
    u = torch.rand(n_air_exp, device=device)
    air_span = z_max - guard_hat
    z_air_exp = guard_hat - depth_air * torch.log1p(
        -u * (1.0 - math.exp(-air_span / depth_air))
    )
    z_air_uni = guard_hat + air_span * torch.rand(n_air_uni, device=device)
    z_above = torch.cat([z_air_exp, z_air_uni]).clamp(min=guard_hat, max=z_max)

    z = _push_off_boundaries(torch.cat([z_below, z_above]), bounds, guard_hat)
    x = torch.rand(n_points, device=device) * x_max
    y = torch.rand(n_points, device=device) * y_max
    return torch.stack([x, y, z], dim=1)


def sample_collocation_si(
    n_points: int,
    struct: Optional[Structure] = None,
    guard: float = GUARD,
    device: torch.device = DEVICE,
) -> torch.Tensor:
    """:func:`sample_collocation_hat` in metres, with ``requires_grad`` set."""
    coords = sample_collocation_hat(n_points, struct, guard, device) / K0
    coords.requires_grad_(True)
    return coords


def sample_boundary_hat(
    n_points: int,
    struct: Optional[Structure] = None,
    guard: float = GUARD,
    device: torch.device = DEVICE,
) -> torch.Tensor:
    """Points on the six faces of the scaled box, ``n_points // 6`` per face."""
    struct = struct or STRUCT
    x_max, y_max, z_min, z_max = struct.domain_hat
    guard_hat = guard * K0
    bounds = torch.as_tensor(struct.boundaries_hat, dtype=torch.float32, device=device)
    per_face = max(1, n_points // 6)
    low = torch.tensor([0.0, 0.0, z_min], device=device)
    high = torch.tensor([x_max, y_max, z_max], device=device)
    faces = []
    for axis in range(3):
        for value in (low[axis], high[axis]):
            pts = low + torch.rand(per_face, 3, device=device) * (high - low)
            pts[:, axis] = value
            if axis != 2:
                pts[:, 2] = _push_off_boundaries(pts[:, 2], bounds, guard_hat)
            faces.append(pts)
    return torch.cat(faces, dim=0)


def sample_interface_hat(
    n_points: int, struct: Optional[Structure] = None, device: torch.device = DEVICE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Points spread over **all** the stack's interfaces, and their unit normals ẑ.

    Every boundary gets the same share, so the continuity constraint is applied
    thirteen times over rather than only at the air interface.
    """
    struct = struct or STRUCT
    x_max, y_max, _, _ = struct.domain_hat
    bounds = torch.as_tensor(struct.boundaries_hat, dtype=torch.float32, device=device)
    which = torch.randint(0, bounds.numel(), (n_points,), device=device)
    coords = torch.stack(
        [
            torch.rand(n_points, device=device) * x_max,
            torch.rand(n_points, device=device) * y_max,
            bounds[which],
        ],
        dim=1,
    )
    normals = torch.zeros_like(coords)
    normals[:, 2] = 1.0
    return coords, normals


def points_per_layer(n_points: int, struct: Optional[Structure] = None, trials: int = 8) -> Dict:
    """
    Mean collocation points landing in each finite layer — the resolution claim.

    Returned as a dict with the per-layer means and the worst (thinnest, deepest)
    layer, so the docstring's "several points per layer" is a measured number
    rather than an assertion.
    """
    struct = struct or STRUCT
    bounds = struct.boundaries
    counts = np.zeros(bounds.size - 1)
    for _ in range(trials):
        z = (sample_collocation_hat(n_points, struct, device=torch.device("cpu"))[:, 2]
             / K0).cpu().numpy()
        counts += np.histogram(z, bins=bounds)[0]
    counts /= trials
    groups = struct.material_group(0.5 * (bounds[:-1] + bounds[1:]))
    return {
        "per_layer": counts.tolist(),
        "min_per_layer": float(counts.min()),
        "mean_per_metal_layer": float(counts[groups == 0].mean()) if np.any(groups == 0) else 0.0,
        "mean_per_dielectric_layer": (
            float(counts[groups == 1].mean()) if np.any(groups == 1) else 0.0
        ),
        "n_points": int(n_points),
    }


# ================================================================== training
def _material_masks(coords_hat: torch.Tensor, struct: Structure) -> Dict[str, torch.Tensor]:
    """Boolean masks splitting scaled points into silver / silica / air."""
    z = (coords_hat[:, 2].detach().cpu().double().numpy()) / K0
    group = struct.material_group(z)
    device = coords_hat.device
    return {
        "metal": torch.as_tensor(group == 0, device=device),
        "dielectric": torch.as_tensor(group == 1, device=device),
        "air": torch.as_tensor(group == 2, device=device),
    }


#: Atomic best-weights checkpointing (see :func:`src.experiments.write_checkpoint`).
_write_checkpoint = write_checkpoint
load_checkpoint_into = load_core_checkpoint


def probe_representability(
    struct: Optional[Structure] = None,
    n_epochs: int = PROBE_EPOCHS,
    n_points: int = PROBE_POINTS,
    learning_rate: float = 2e-3,
    device: torch.device = DEVICE,
    seed: int = 0,
    log_every: int = 100,
) -> Dict[str, float]:
    r"""
    **Tractability probe.** Can a network of this capacity even *represent* the
    TMM field of this stack, by direct supervised regression?

    No physics loss, no anchor — just least squares against
    :func:`tmm_fields_si` on fresh interior points. If the answer is no, no
    choice of PINN loss can rescue it and the honest move is to reduce ``N``,
    widen ``a`` or grow the network. Because it isolates *representation* from
    *optimisation under a physics objective*, its rel L2 at matched-ish budget
    is a floor to judge the full run against — but note it is a *moving* floor:
    at 10x the epochs it reaches 1.4e-3, past the full PINN (see
    ``figures/ablation/supervised_baseline.json`` and the 2026-09-02 ablation
    doc). It fits the answer; the PINN solves the problem.

    Returns:
        Dict with the final and best rel L2 (overall and per region), the loss
        history and the wall time.
    """
    struct = struct or STRUCT
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    network = create_network(struct, device=device, generator=generator)
    core = network.core
    optimizer = torch.optim.Adam(core.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, n_epochs), eta_min=learning_rate * 1e-2
    )

    history: List[float] = []
    t0 = time.perf_counter()
    for epoch in range(n_epochs):
        coords = sample_collocation_hat(n_points, struct, device=device)
        with torch.no_grad():
            target = tmm_fields_hat(coords, struct)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((core(coords) - target) ** 2)
        loss.backward()
        optimizer.step()
        scheduler.step()
        history.append(loss.item())
        if epoch % log_every == 0 or epoch == n_epochs - 1:
            logger.info("probe %5d | mse %.4e | %.0fs", epoch, loss.item(),
                        time.perf_counter() - t0)

    coords = sample_collocation_si(4 * n_points, struct, guard=VAL_GUARD, device=device)
    with torch.no_grad():
        pred = network(coords)
        ref = tmm_fields_si(coords, struct)
        z = coords[:, 2].detach()
        regions = {
            "stack": (z >= struct.z0) & (z < 0.0),
            "air": z >= 0.0,
            "substrate": z < struct.z0,
        }
        out = {
            "probe_rel_l2": _relative_l2(pred, ref),
            "probe_final_mse": history[-1],
            "probe_best_mse": float(min(history)),
            "probe_epochs": float(n_epochs),
            "probe_time_s": time.perf_counter() - t0,
        }
        for name, mask in regions.items():
            out[f"probe_rel_l2_{name}"] = _relative_l2(pred[mask], ref[mask])
    out["probe_history"] = history  # type: ignore[assignment]
    return out


def train(
    network: MultilayerPINN,
    struct: Optional[Structure] = None,
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
) -> Tuple[MultilayerPINN, Dict[str, list]]:
    """
    Train the layered SPP PINN: the recipe of ``examples/validate_spp.py``, with
    the stack's ε(z) everywhere it used a two-valued step.

    Interior curl and divergence residuals are computed **per material** (silver,
    silica, air) rather than per half-space, because ε now takes three values;
    the silver terms carry the anti-collapse preconditioner ``|ε_Ag|^{-p}`` (the
    curl-H residual penalises an Ê error in silver ``|ε_Ag|² ≈ 320×`` harder than
    in air, and without the reweighting Adam falls into the trivial ``E = H = 0``
    basin). Tangential continuity is imposed on *all* interfaces, and the soft
    Dirichlet anchor on the six faces is the **TMM** profile. Phase 2 raises the
    silver curl weight to ``|ε_Ag|^{-1/2}``.

    The schedule around the physics — ramp, best-iterate tracking, float64
    L-BFGS, checkpointing — is :func:`src.experiments.run_training`.
    """
    struct = struct or STRUCT
    core = network.core
    curl_loss = MaxwellCurlLoss(frequency=OMEGA_HAT, mu0=1.0, eps0=1.0)
    div_loss = MaxwellDivergenceLoss()
    cont_loss = TangentialContinuityLoss(offset=CONTINUITY_OFFSET * K0)

    w_curl_adam = abs(EPS_AG) ** -METAL_CURL_EXPONENT_ADAM
    w_curl_lbfgs = abs(EPS_AG) ** -METAL_CURL_EXPONENT_LBFGS
    w_div_metal = abs(EPS_AG) ** -METAL_DIV_EXPONENT
    eps_by_group = (EPS_AG, complex(EPS_D2), complex(EPS_AIR))

    def compute_losses(batch, ramp: float = 1.0, w_curl_m: Optional[float] = None):
        if w_curl_m is None:
            w_curl_m = w_curl_adam
        weights = (w_curl_m, 1.0, 1.0)
        div_weights = (w_div_metal, 1.0, 1.0)
        l_curl = batch["zero"].clone()
        l_div = batch["zero"].clone()
        for name, eps, wc, wd in zip(
            ("metal", "dielectric", "air"), eps_by_group, weights, div_weights, strict=True
        ):
            coords = batch[f"coords_{name}"]
            if coords.shape[0] == 0:
                continue
            l_curl = l_curl + wc * curl_loss.compute(network=core, coords=coords, epsilon=eps)
            l_div = l_div + wd * div_loss.compute(network=core, coords=coords, epsilon=eps)
        l_cont = cont_loss.compute(
            network=core, interface_coords=batch["iface"], normal_vectors=batch["normals"]
        )
        l_bc = torch.mean((core(batch["boundary"]) - batch["target"]) ** 2)
        total = (
            ramp * (l_curl + divergence_weight * l_div + continuity_weight * l_cont)
            + boundary_weight * l_bc
        )
        return total, l_curl, l_div, l_cont, l_bc

    def sample_batch(n_int, n_bc, n_if, dtype=torch.float32):
        coords = sample_collocation_hat(n_int, struct, device=device).detach().to(dtype)
        masks = _material_masks(coords, struct)
        batch = {
            f"coords_{name}": coords[mask].clone().requires_grad_(True)
            for name, mask in masks.items()
        }
        iface, normals = sample_interface_hat(n_if, struct, device=device)
        batch["iface"] = iface.to(dtype)
        batch["normals"] = normals.to(dtype)
        boundary = sample_boundary_hat(n_bc, struct, device=device).to(dtype)
        batch["boundary"] = boundary
        with torch.no_grad():
            batch["target"] = tmm_fields_hat(boundary, struct).to(dtype)
        batch["zero"] = torch.zeros((), dtype=dtype, device=device)
        return batch

    return run_training(
        network,
        TrainingConfig(
            n_epochs=n_epochs,
            n_points=n_points,
            n_boundary=max(6, n_points // 2),
            n_interface=max(1, n_points // 4),
            learning_rate=learning_rate,
            physics_ramp_frac=physics_ramp_frac,
            lbfgs_steps=lbfgs_steps,
            lbfgs_dtype=lbfgs_dtype,
            lbfgs_points_factor=LBFGS_POINTS_FACTOR,
            log_every=log_every,
        ),
        sample_batch,
        compute_losses,
        logger,
        lbfgs_loss_kwargs={"w_curl_m": w_curl_lbfgs},
        checkpoint_path=checkpoint_path,
    )


# ================================================================== validation
#: ``‖pred − ref‖ / ‖ref‖`` — see :func:`src.experiments.relative_l2`.
_relative_l2 = relative_l2


K_PROBE_HEIGHTS = tuple(np.linspace(0.05, 0.95, 20))
"""Air-side heights (as a fraction of ``z_max``) the ``k_spp`` fit is averaged over."""


def estimate_k_spp(
    net3: nn.Module,
    struct: Optional[Structure] = None,
    n_line: int = 512,
    heights: Sequence[float] = K_PROBE_HEIGHTS,
    device: torch.device = DEVICE,
) -> Dict[str, float]:
    r"""
    ``Re k_spp`` from the phase slope of ``H_y`` along ``x``, ``Im k_spp`` from
    ``−d ln|H_y|/dx`` — **the headline measurement**.

    The probe lines sit in the air, well clear of the layers, where
    ``H_y ∝ e^{i k_x x}`` with no ``z`` structure to confuse the fit.

    **Why many lines, and why weighted.** ``Re k_spp`` is read off a phase that
    advances by ``2·2π`` across the box: a well-conditioned fit, and in practice
    it agrees to 0.001–0.02 % at *every* height. ``Im k_spp`` is a different
    animal. Over the ``2 λ_spp`` box the true mode's amplitude falls by only
    ``1 − e^{−2 Im k λ_spp} ≈ 1 %``, so a 0.2 % ripple in the network's
    ``x``-dependence is a tens-of-per-cent error in the loss — and the field
    error in the air is exactly of that order. Single-line fits therefore
    scatter by ±17 % even when the mode is right.

    The fit is run at all of :data:`K_PROBE_HEIGHTS` and averaged **weighted by
    each line's mean ``|H_y|``**, because the log-fit is better conditioned
    where the signal is larger. The scatter is not hidden: ``k_spp_im_std``,
    ``k_spp_im_spread`` and ``k_spp_im_worst_rel_error`` are all reported, and
    the headline has to be read with them.

    The fit is only valid in the air. Inside the stack ``H_y`` carries the layer
    structure and the amplitudes are e-foldings smaller; a log-slope fit there
    returns nonsense (measured: −158 % to +12 % at the interfaces), which is why
    every probe line sits above ``z = 0``.

    Both the TMM truth and the EMT prediction are reported alongside, with the
    error ratios that decide the experiment.
    """
    struct = struct or STRUCT
    x = torch.linspace(0.0, struct.x_max, n_line, device=device)
    xs = x.cpu().numpy().astype(np.float64)
    re_fits, im_fits, weights = [], [], []
    for height in heights:
        coords = torch.stack(
            [x, torch.full_like(x, struct.y_max / 2),
             torch.full_like(x, height * struct.z_max)], dim=1
        )
        with torch.no_grad():
            _, H = to_complex(net3(coords))
        hy = H[:, 1].cpu().numpy().astype(np.complex128)
        re_fits.append(float(np.polyfit(xs, np.unwrap(np.angle(hy)), 1)[0]))
        im_fits.append(-float(np.polyfit(xs, np.log(np.abs(hy) + 1e-300), 1)[0]))
        weights.append(float(np.abs(hy).mean()))
    w = np.asarray(weights, dtype=float)
    w = w / max(w.sum(), 1e-300)
    k_re = float(np.average(re_fits, weights=w))
    k_im = float(np.average(im_fits, weights=w))

    tmm, emt = struct.k_tmm, struct.k_emt
    err_re = abs(k_re - tmm.real) / abs(tmm.real)
    err_im = abs(k_im - tmm.imag) / abs(tmm.imag)
    emt_re = abs(emt.real - tmm.real) / abs(tmm.real)
    emt_im = abs(emt.imag - tmm.imag) / abs(tmm.imag)
    return {
        "k_spp_re_pinn": k_re,
        "k_spp_im_pinn": k_im,
        "k_spp_n_probe_lines": float(len(im_fits)),
        "k_spp_re_spread": float(np.ptp(re_fits)),
        "k_spp_re_worst_rel_error": float(
            np.max(np.abs(np.asarray(re_fits) - tmm.real)) / abs(tmm.real)
        ),
        "k_spp_im_std": float(np.std(im_fits)),
        "k_spp_im_spread": float(np.ptp(im_fits)),
        "k_spp_im_spread_over_tmm": float(np.ptp(im_fits) / abs(tmm.imag)),
        "k_spp_im_std_over_tmm": float(np.std(im_fits) / abs(tmm.imag)),
        # The worst single line, so the headline can never be read as luck.
        "k_spp_im_worst_rel_error": float(
            np.max(np.abs(np.asarray(im_fits) - tmm.imag)) / abs(tmm.imag)
        ),
        "k_spp_re_tmm": tmm.real,
        "k_spp_im_tmm": tmm.imag,
        "k_spp_re_emt": emt.real,
        "k_spp_im_emt": emt.imag,
        "n_eff_pinn": k_re / K0,
        "n_eff_tmm": tmm.real / K0,
        "n_eff_emt": emt.real / K0,
        "k_spp_re_rel_error": err_re,
        "k_spp_im_rel_error": err_im,
        "emt_re_rel_error": emt_re,
        "emt_im_rel_error": emt_im,
        # > 1 means the PINN beat the homogenised model on that component.
        "re_error_ratio_emt_over_pinn": emt_re / max(err_re, 1e-30),
        "im_error_ratio_emt_over_pinn": emt_im / max(err_im, 1e-30),
        "beats_emt_re": float(err_re < emt_re),
        "beats_emt_im": float(err_im < emt_im),
    }


def fit_air_decay(
    net3: nn.Module,
    struct: Optional[Structure] = None,
    n_line: int = 256,
    guard: float = VAL_GUARD,
    device: torch.device = DEVICE,
) -> Dict[str, float]:
    """``κ_d`` from ``ln|H_y|`` vs ``z`` in the air, and the sign that says *bound*."""
    struct = struct or STRUCT
    z = torch.linspace(guard, 0.9 * struct.z_max, n_line, device=device)
    coords = torch.stack(
        [torch.full_like(z, 0.25 * struct.x_max), torch.full_like(z, struct.y_max / 2), z], dim=1
    )
    with torch.no_grad():
        _, H = to_complex(net3(coords))
    log_hy = np.log(np.abs(H[:, 1].cpu().numpy().astype(np.complex128)) + 1e-300)
    slope = float(np.polyfit(z.cpu().numpy().astype(np.float64), log_hy, 1)[0])
    kappa = -slope
    return {
        "kappa_d_fit": kappa,
        "kappa_d_tmm": struct.kappa_d.real,
        "kappa_d_rel_error": abs(kappa - struct.kappa_d.real) / struct.kappa_d.real,
        "bound_in_air": float(kappa > 0.0),
    }


def stack_decay(
    net3: nn.Module,
    struct: Optional[Structure] = None,
    n_line: int = 256,
    device: torch.device = DEVICE,
) -> Dict[str, float]:
    """
    Envelope decay into the stack, from the ``|H_y|`` values at the *interfaces*.

    Sampling at the interfaces removes the intra-period ripple, leaving the
    Bloch envelope whose slope is the effective ``κ`` the homogenised model
    would call ``κ_m``.
    """
    struct = struct or STRUCT
    z_np = struct.boundaries.copy()
    z = torch.as_tensor(z_np, dtype=torch.float32, device=device)
    coords = torch.stack(
        [torch.full_like(z, 0.25 * struct.x_max), torch.full_like(z, struct.y_max / 2), z], dim=1
    )
    with torch.no_grad():
        _, H = to_complex(net3(coords))
    log_hy = np.log(np.abs(H[:, 1].cpu().numpy().astype(np.complex128)) + 1e-300)
    slope = float(np.polyfit(z_np, log_hy, 1)[0])
    ref = struct.profile(z_np)
    slope_ref = float(np.polyfit(z_np, np.log(np.abs(ref.H_y)), 1)[0])
    return {
        "kappa_stack_fit": slope,
        "kappa_stack_tmm": slope_ref,
        "kappa_stack_rel_error": abs(slope - slope_ref) / abs(slope_ref),
        "bound_in_stack": float(slope > 0.0),
    }


def layer_contrast(
    net3: nn.Module,
    struct: Optional[Structure] = None,
    n_per_layer: int = 40,
    device: torch.device = DEVICE,
) -> Dict[str, float]:
    r"""
    Is the ``E_z`` **sawtooth** actually there? — the "correct qualitative layer
    structure" of the minimum tier, as a number.

    ``D_z`` is continuous, so ``E_z = D_z/ε`` is larger in silica than in silver
    by roughly ``|ε_Ag|/ε_silica ≈ 8``. The contrast is measured as the ratio of
    the mean ``|E_z|`` over silica points to that over silver points, both taken
    on the same interior grid inside the stack, and compared with the TMM's own
    contrast.
    """
    struct = struct or STRUCT
    bounds = struct.boundaries
    inside = bounds[bounds <= 0.0]
    zs = []
    for lo, hi in zip(inside[:-1], inside[1:], strict=True):
        zs.append(np.linspace(lo, hi, n_per_layer + 2)[1:-1])
    z_np = np.concatenate(zs)
    group = struct.material_group(z_np)
    z = torch.as_tensor(z_np, dtype=torch.float32, device=device)
    coords = torch.stack(
        [torch.full_like(z, 0.25 * struct.x_max), torch.full_like(z, struct.y_max / 2), z], dim=1
    )
    with torch.no_grad():
        E, _ = to_complex(net3(coords))
    ez = E[:, 2].cpu().numpy()
    ez_ref = struct.profile(z_np).E_z * _phase(np.full_like(z_np, 0.25 * struct.x_max),
                                               struct.k_tmm)

    def contrast(values: np.ndarray) -> float:
        metal = np.abs(values[group == 0]).mean()
        diel = np.abs(values[group == 1]).mean()
        return float(diel / max(metal, 1e-300))

    got, want = contrast(ez), contrast(ez_ref)
    return {
        "ez_layer_contrast_pinn": got,
        "ez_layer_contrast_tmm": want,
        "ez_layer_contrast_rel_error": abs(got - want) / abs(want),
        "ez_layer_contrast_expected": abs(EPS_AG) / EPS_D2,
    }


def continuity_residuals(
    net3: nn.Module,
    struct: Optional[Structure] = None,
    n_points: int = 2000,
    offset: float = CONTINUITY_OFFSET,
    device: torch.device = DEVICE,
) -> Dict[str, float]:
    """
    Tangential continuity residual across all interfaces, relative to the field RMS.

    Read against the ``≈ 2 κ δ`` floor described at :data:`CONTINUITY_OFFSET`:
    the exact TMM mode scores ~0.03 on this metric, not 0.
    """
    struct = struct or STRUCT
    coords_hat, normals = sample_interface_hat(n_points, struct, device=device)
    coords = coords_hat / K0
    off = offset
    return measurement.continuity_residuals(net3, coords, normals, off)


def validate(
    net3: nn.Module,
    struct: Optional[Structure] = None,
    n_points: int = 20000,
    device: torch.device = DEVICE,
) -> Dict[str, float]:
    """
    SI-unit validation of a network against the **transfer-matrix** mode.

    Keys: rel L2 vs the TMM field (overall, per component, and per region —
    inside the stack, in the air, in the silver substrate), curl residuals per
    material, the three-way ``k_spp`` comparison (:func:`estimate_k_spp`), the
    air-side and in-stack decay fits, the ``E_z`` layer contrast and the
    tangential-continuity residuals. The EMT field's own rel L2 against the TMM
    is included so the field comparison, like the ``k_spp`` one, has a baseline.
    """
    struct = struct or STRUCT
    net3.eval()
    coords = sample_collocation_si(n_points, struct, guard=VAL_GUARD, device=device)
    fields = net3(coords)
    E, H = to_complex(fields)

    maxwell = MaxwellEquations(OMEGA, mu0=MU0, eps0=EPS0)
    curl_E = maxwell.curl_operator(E, coords)
    curl_H = maxwell.curl_operator(H, coords)
    z = coords[:, 2].detach()
    z_np = z.cpu().double().numpy()
    group = struct.material_group(z_np)
    eps_np = permittivity_profile(z_np, struct.eps_layers, struct.thicknesses, z0=struct.z0)
    eps_row = torch.as_tensor(eps_np, dtype=E.dtype, device=device).view(-1, 1)
    res_E = curl_E - 1j * OMEGA * MU0 * H
    res_H = curl_H + 1j * OMEGA * EPS0 * eps_row * E

    with torch.no_grad():
        ref = tmm_fields_si(coords.detach(), struct).to(fields.dtype)
        E_ref, H_ref = to_complex(ref)
        emt = emt_fields_si(coords.detach(), struct).to(fields.dtype)

        metrics: Dict[str, float] = {
            "rel_l2_total": _relative_l2(fields, ref),
            "rel_l2_E": _relative_l2(E, E_ref),
            "rel_l2_H": _relative_l2(H, H_ref),
            "rel_l2_Ez": _relative_l2(E[:, 2], E_ref[:, 2]),
            "rel_l2_Ex": _relative_l2(E[:, 0], E_ref[:, 0]),
            # The baseline the field comparison is against: how far the
            # homogenised field itself is from the truth on the same points.
            "rel_l2_total_emt_vs_tmm": _relative_l2(emt, ref),
        }
        regions = {
            "stack": torch.as_tensor((z_np >= struct.z0) & (z_np < 0.0), device=device),
            "air": torch.as_tensor(z_np >= 0.0, device=device),
            "substrate": torch.as_tensor(z_np < struct.z0, device=device),
        }
        for name, mask in regions.items():
            if int(mask.sum()) == 0:
                continue
            metrics[f"rel_l2_{name}"] = _relative_l2(fields[mask], ref[mask])
            metrics[f"rel_l2_E_{name}"] = _relative_l2(E[mask], E_ref[mask])
            metrics[f"rel_l2_H_{name}"] = _relative_l2(H[mask], H_ref[mask])
            metrics[f"rel_l2_{name}_emt_vs_tmm"] = _relative_l2(emt[mask], ref[mask])
        for name, code in (("metal", 0), ("dielectric", 1), ("air", 2)):
            mask = torch.as_tensor(group == code, device=device)
            if int(mask.sum()) == 0:
                continue
            E_rms = torch.sqrt(torch.mean(torch.sum(E[mask].abs() ** 2, 1))).clamp_min(1e-30)
            H_rms = torch.sqrt(torch.mean(torch.sum(H[mask].abs() ** 2, 1))).clamp_min(1e-30)
            rE = torch.linalg.vector_norm(res_E[mask], dim=1)
            rH = torch.linalg.vector_norm(res_H[mask], dim=1)
            metrics[f"curl_E_residual_rel_{name}"] = (
                torch.sqrt(torch.mean(rE**2)) / (K0 * E_rms)
            ).item()
            metrics[f"curl_H_residual_rel_{name}"] = (
                torch.sqrt(torch.mean(rH**2)) / (K0 * H_rms)
            ).item()
        E_rms_all = torch.sqrt(torch.mean(torch.sum(E.abs() ** 2, 1))).clamp_min(1e-30)
        H_rms_all = torch.sqrt(torch.mean(torch.sum(H.abs() ** 2, 1))).clamp_min(1e-30)
        metrics["impedance_ratio"] = ((E_rms_all / H_rms_all) / ETA0).item()

    metrics.update(estimate_k_spp(net3, struct, device=device))
    metrics.update(fit_air_decay(net3, struct, device=device))
    metrics.update(stack_decay(net3, struct, device=device))
    metrics.update(layer_contrast(net3, struct, device=device))
    metrics.update(continuity_residuals(net3, struct, device=device))
    metrics["success_tier"] = success_tier(metrics)  # type: ignore[assignment]
    return metrics


def success_tier(metrics: Dict[str, float]) -> str:
    """
    minimum — bound mode recovered, rel L2 < 0.5, ``E_z`` layer contrast within
    50 % of the TMM's (the "correct qualitative layer structure");
    target — rel L2 < 0.1 **and** both ``k_spp`` components closer to the TMM
    than the EMT prediction is;
    stretch — rel L2 < 0.03 and both ``k_spp`` components within 1 % of the TMM.
    """
    rel = max(metrics["rel_l2_E"], metrics["rel_l2_H"])
    bound = metrics["bound_in_air"] > 0 and metrics["bound_in_stack"] > 0
    layered = metrics["ez_layer_contrast_rel_error"] < 0.5
    beats = metrics["beats_emt_re"] > 0 and metrics["beats_emt_im"] > 0
    within_1pct = metrics["k_spp_re_rel_error"] < 0.01 and metrics["k_spp_im_rel_error"] < 0.01
    if bound and layered and rel < 0.03 and within_1pct:
        return "stretch"
    if bound and layered and rel < 0.1 and beats:
        return "target"
    if bound and layered and rel < 0.5:
        return "minimum"
    return "not met"


# ================================================================== plotting
_C_PINN = "#D55E00"
_C_TMM = "#000000"
_C_EMT = "#0072B2"


def _mark_layers(ax, struct: Structure, label: bool = False) -> None:
    """Shade the silver layers and draw every interface."""
    bounds = struct.boundaries
    groups = struct.material_group(0.5 * (bounds[:-1] + bounds[1:]))
    for (lo, hi), g in zip(zip(bounds[:-1], bounds[1:], strict=True), groups, strict=True):
        if g == 0:
            ax.axvspan(lo * 1e9, hi * 1e9, color="0.82", lw=0, zorder=0)
    for b in bounds:
        ax.axvline(b * 1e9, color="0.55", lw=0.5, ls=":", zorder=1)
    if label:
        ax.annotate("Ag layers shaded", xy=(0.02, 0.95), xycoords="axes fraction",
                    fontsize=8, color="0.35", va="top")


def plot_field_profiles(
    net3: nn.Module, struct: Structure, out_dir: Path, device: torch.device = DEVICE
) -> str:
    """The payoff figure: ``|H_y|(z)`` and ``E_z(z)`` from PINN, TMM and EMT."""
    out_dir.mkdir(parents=True, exist_ok=True)
    z_np = np.linspace(struct.z_min, struct.z_max, 3000)
    x_probe = 0.0
    z = torch.as_tensor(z_np, dtype=torch.float32, device=device)
    coords = torch.stack(
        [torch.full_like(z, x_probe), torch.full_like(z, struct.y_max / 2), z], dim=1
    )
    with torch.no_grad():
        E, H = to_complex(net3(coords))
        E_emt, H_emt = to_complex(emt_fields_si(coords))
    hy = H[:, 1].cpu().numpy()
    ez = E[:, 2].cpu().numpy()
    prof = struct.profile(z_np)
    phase = _phase(np.full_like(z_np, x_probe), struct.k_tmm)
    hy_t, ez_t = prof.H_y * phase, prof.E_z * phase
    hy_e = H_emt[:, 1].cpu().numpy()
    ez_e = E_emt[:, 2].cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    zn = z_np * 1e9

    ax = axes[0, 0]
    for data, colour, style, label in (
        (hy_t, _C_TMM, "-", "TMM (truth)"),
        (hy_e, _C_EMT, "--", "EMT (homogenised)"),
        (hy, _C_PINN, "-", "PINN (layered)"),
    ):
        ax.semilogy(zn, np.abs(data), color=colour, ls=style, lw=1.8 if colour != _C_PINN else 1.6,
                    label=label)
    _mark_layers(ax, struct, label=True)
    ax.set_ylabel(r"$|H_y|$  [A/m]")
    ax.set_title("(a) magnetic field through the stack")
    ax.set_ylim(max(np.abs(hy_t).min() * 0.3, 1e-6), np.abs(hy_t).max() * 2)

    ax = axes[0, 1]
    for data, colour, style, label in (
        (ez_t, _C_TMM, "-", "TMM"),
        (ez_e, _C_EMT, "--", "EMT"),
        (ez, _C_PINN, "-", "PINN"),
    ):
        ax.plot(zn, data.real, color=colour, ls=style, lw=1.8 if colour != _C_PINN else 1.6,
                label=label)
    _mark_layers(ax, struct)
    ax.set_xlim(struct.z0 * 1e9 - 10.0, 60.0)
    ax.set_ylabel(r"Re $E_z$  [V/m]")
    ax.set_title(r"(b) the $E_z$ sawtooth — $D_z$ continuous, $\epsilon(z)$ piecewise")

    ax = axes[1, 0]
    ax.plot(zn, np.abs(E[:, 0].cpu().numpy()), color=_C_PINN, lw=1.6, label="PINN")
    ax.plot(zn, np.abs(prof.E_x * phase), color=_C_TMM, lw=1.8, ls="-", label="TMM")
    ax.plot(zn, np.abs(E_emt[:, 0].cpu().numpy()), color=_C_EMT, lw=1.8, ls="--", label="EMT")
    _mark_layers(ax, struct)
    ax.set_yscale("log")
    ax.set_ylabel(r"$|E_x|$  [V/m]")
    ax.set_title("(c) tangential electric field (continuous everywhere)")

    ax = axes[1, 1]
    scale = np.abs(np.concatenate([hy_t, ez_t / ETA0])).max()
    ax.semilogy(zn, np.abs(hy - hy_t) / scale, color=_C_PINN, lw=1.4, label=r"PINN $H_y$ error")
    ax.semilogy(zn, np.abs(ez - ez_t) / (ETA0 * scale), color="#009E73", lw=1.4,
                label=r"PINN $E_z$ error")
    ax.semilogy(zn, np.abs(hy_e - hy_t) / scale, color=_C_EMT, lw=1.4, ls="--",
                label=r"EMT $H_y$ error")
    _mark_layers(ax, struct)
    ax.set_ylabel("|field − TMM| / max|TMM|")
    ax.set_title("(d) pointwise error against the transfer matrix")

    for ax in axes.ravel():
        ax.set_xlabel("z [nm]")
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8.5, loc="best")
    fig.suptitle(
        rf"Ag/silica multilayer SPP at $\lambda_0$ = {LAMBDA0 * 1e9:.0f} nm: "
        rf"$a$ = {struct.period * 1e9:.0f} nm, $N$ = {struct.n_periods}, "
        rf"$f$ = {FILL_FRACTION:g}, metal-terminated against air"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = out_dir / "field_profiles.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_k_comparison(metrics: Dict[str, float], struct: Structure, out_dir: Path) -> str:
    """The money plot: PINN, TMM and EMT ``k_spp`` on one axis, per component."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax_re, ax_im) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    n_lines = int(metrics.get("k_spp_n_probe_lines", 1))
    for ax, values, spread, label, unit in (
        (ax_re,
         (metrics["k_spp_re_tmm"] / K0, metrics["k_spp_re_emt"] / K0,
          metrics["k_spp_re_pinn"] / K0),
         metrics.get("k_spp_re_spread", 0.0) / K0,
         r"Re $k_{\mathrm{spp}} / k_0$", ""),
        (ax_im,
         (metrics["k_spp_im_tmm"] * 1e-3, metrics["k_spp_im_emt"] * 1e-3,
          metrics["k_spp_im_pinn"] * 1e-3),
         metrics.get("k_spp_im_std", 0.0) * 1e-3,
         r"Im $k_{\mathrm{spp}}$  [$10^3$ m$^{-1}$]", ""),
    ):
        tmm, emt, pinn = values
        ax.axhline(tmm, color=_C_TMM, lw=2.0, label="TMM (truth)")
        ax.plot([0], [emt], "s", ms=13, color=_C_EMT, label="EMT (homogenised)")
        # The PINN point carries the 1σ scatter of the fit across the probe
        # lines: for Im k that scatter is the honest measure of how well the
        # number is determined, and it is much larger than the error itself.
        ax.errorbar([1], [pinn], yerr=[spread], fmt="o", ms=13, color=_C_PINN,
                    ecolor=_C_PINN, elinewidth=1.6, capsize=6,
                    label=rf"PINN (layered), $\pm1\sigma$ over {n_lines} probe lines")
        for xpos, value, colour in ((0, emt, _C_EMT), (1, pinn, _C_PINN)):
            ax.annotate("", xy=(xpos, value), xytext=(xpos, tmm),
                        arrowprops={"arrowstyle": "<->", "color": colour, "lw": 1.2})
        ax.set_xlim(-0.6, 1.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["EMT", "PINN"])
        ax.set_ylabel(label + unit)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=9, loc="best")

    ax_re.set_title(
        "Re $k_{{spp}}$:  EMT off by {:.2f}%,  PINN off by {:.2f}%\n"
        "(EMT error / PINN error = {:.2f}×)".format(
            100 * metrics["emt_re_rel_error"], 100 * metrics["k_spp_re_rel_error"],
            metrics["re_error_ratio_emt_over_pinn"],
        ),
        fontsize=10,
    )
    ax_im.set_title(
        "Im $k_{{spp}}$ (loss):  EMT off by {:.1f}%,  PINN off by {:.1f}%\n"
        "(EMT error / PINN error = {:.2f}×;  1σ across probe lines = {:.0f}%)".format(
            100 * metrics["emt_im_rel_error"], 100 * metrics["k_spp_im_rel_error"],
            metrics["im_error_ratio_emt_over_pinn"],
            100 * metrics.get("k_spp_im_std_over_tmm", 0.0),
        ),
        fontsize=10,
    )
    fig.suptitle(
        rf"Does the layered PINN beat the homogenised model? "
        rf"($a$ = {struct.period * 1e9:.0f} nm, $N$ = {struct.n_periods}, "
        rf"$\lambda_0$ = {LAMBDA0 * 1e9:.0f} nm)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = out_dir / "k_spp_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_field_map(
    net3: nn.Module, struct: Structure, out_dir: Path, device: torch.device = DEVICE
) -> str:
    """``Re H_y`` in the x-z plane: PINN, TMM and their difference."""
    out_dir.mkdir(parents=True, exist_ok=True)
    nx, nz = 180, 260
    x = torch.linspace(0.0, struct.x_max, nx, device=device)
    zv = torch.linspace(struct.z_min, struct.z_max, nz, device=device)
    X, Z = torch.meshgrid(x, zv, indexing="ij")
    coords = torch.stack(
        [X.flatten(), torch.full_like(X.flatten(), struct.y_max / 2), Z.flatten()], dim=1
    )
    with torch.no_grad():
        pred = net3(coords)[:, 4, 0].reshape(nx, nz).cpu().numpy()
        ref = tmm_fields_si(coords, struct)[:, 4, 0].reshape(nx, nz).cpu().numpy()
    Xn, Zn = X.cpu().numpy() * 1e9, Z.cpu().numpy() * 1e9
    vmax = np.abs(ref).max()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0), sharey=True)
    for ax, (data, title, lim) in zip(
        axes,
        [
            (pred, r"PINN Re $H_y$ [A/m]", vmax),
            (ref, r"TMM Re $H_y$ [A/m]", vmax),
            (pred - ref, "difference", np.abs(pred - ref).max() + 1e-30),
        ],
        strict=True,
    ):
        im = ax.pcolormesh(Xn, Zn, data, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
        for b in struct.boundaries:
            ax.axhline(b * 1e9, color="k", lw=0.4, ls=":")
        ax.set_title(title)
        ax.set_xlabel("x [nm]")
        fig.colorbar(im, ax=ax)
    axes[0].set_ylabel("z [nm]")
    fig.suptitle(
        f"Layered SPP, x-z plane; {struct.boundaries.size} interfaces marked "
        f"(a = {struct.period * 1e9:.0f} nm, N = {struct.n_periods})"
    )
    fig.tight_layout()
    path = out_dir / "field_map.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_history(
    history: Dict[str, list], probe_history: Optional[List[float]], out_dir: Path
) -> str:
    """Training curves, Adam and L-BFGS on their own axes, plus the probe."""
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = ("total", "curl", "div", "continuity", "boundary")
    epoch = np.asarray(history["epoch"], dtype=float)
    is_lbfgs = np.isnan(np.asarray(history["lr"], dtype=float))
    n_lbfgs = int(is_lbfgs.sum())
    n_panels = 2 + (1 if n_lbfgs else 0) if probe_history else 1 + (1 if n_lbfgs else 0)

    fig, axes = plt.subplots(1, n_panels, figsize=(5.6 * n_panels, 5.0), squeeze=False)
    axes = list(axes[0])
    idx = 0
    if probe_history:
        axes[idx].semilogy(probe_history, color=_C_PINN, lw=1)
        axes[idx].set_xlabel("probe epoch")
        axes[idx].set_ylabel("supervised MSE (scaled units)")
        axes[idx].set_title("Probe: can the network represent the TMM field?", fontsize=10)
        axes[idx].grid(alpha=0.3, which="both")
        idx += 1
    for key in keys:
        axes[idx].semilogy(epoch[~is_lbfgs], np.asarray(history[key], float)[~is_lbfgs],
                           label=key, lw=1)
    axes[idx].set_xlabel("Adam epoch")
    axes[idx].set_ylabel("loss (dimensionless, k₀-scaled frame)")
    axes[idx].set_title("Phase 1: Adam (cosine LR)", fontsize=10)
    axes[idx].grid(alpha=0.3, which="both")
    axes[idx].legend(fontsize=8)
    idx += 1
    if n_lbfgs:
        steps = np.arange(n_lbfgs, dtype=float)
        for key in keys:
            axes[idx].semilogy(steps, np.asarray(history[key], float)[is_lbfgs], label=key, lw=1)
        axes[idx].set_xlabel("L-BFGS step")
        axes[idx].set_title("Phase 2: float64 L-BFGS refinement", fontsize=10)
        axes[idx].grid(alpha=0.3, which="both")
        axes[idx].legend(fontsize=8)
    fig.suptitle("Layered Ag/silica SPP PINN training")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = out_dir / "training_history.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


# ================================================================== main
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_core_args(
        p, epochs=N_EPOCHS, n_points=BATCH_SIZE, lr=LEARNING_RATE,
        device=str(DEVICE), lbfgs_steps=LBFGS_STEPS,
    )
    p.add_argument("--period-nm", type=float, default=PERIOD * 1e9)
    p.add_argument("--n-periods", type=int, default=N_PERIODS)
    p.add_argument("--probe-epochs", type=int, default=PROBE_EPOCHS)
    p.add_argument("--probe-only", action="store_true",
                   help="run the tractability probe and stop")
    p.add_argument("--skip-probe", action="store_true")
    p.add_argument("--eval-only", action="store_true",
                   help="load <model-out> (or its .partial.pth) and re-run validation, "
                        "figures and metrics.json without training")
    add_output_args(
        p, quick_epochs=QUICK_EPOCHS, figures_dir=FIGURES_DIR, model_out=MODEL_PATH,
        quick_extra=", tiny probe",
    )
    return p.parse_args(argv)


def main(argv=None) -> Dict:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    struct = configure_structure(args.period_nm * 1e-9, args.n_periods)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    n_epochs, n_points, lbfgs_steps = (
        (QUICK_EPOCHS, 512, 0) if args.quick else (args.epochs, args.n_points, args.lbfgs_steps)
    )
    probe_epochs = 40 if args.quick else args.probe_epochs
    probe_points = 512 if args.quick else PROBE_POINTS
    n_val_points = 2000 if args.quick else 20000
    lbfgs_dtype = torch.float64 if args.lbfgs_dtype == "float64" else torch.float32

    info = struct.summary()
    logger.info(
        "Ag/silica multilayer: a = %.1f nm (%.1f nm Ag + %.1f nm SiO2), N = %d, %s-terminated, "
        "%d interfaces",
        info["period_nm"], info["metal_thickness_nm"], info["dielectric_thickness_nm"],
        struct.n_periods, struct.termination, info["n_interfaces"],
    )
    logger.info(
        "TMM  k_spp = %.6e + %.4ej  (n_eff %.5f) | EMT k_spp = %.6e + %.4ej (n_eff %.5f)",
        struct.k_tmm.real, struct.k_tmm.imag, info["n_eff_tmm"],
        struct.k_emt.real, struct.k_emt.imag, info["n_eff_emt"],
    )
    logger.info(
        "EMT error to beat: Re %.3f%%, Im %.2f%%",
        100 * info["emt_error_re"], 100 * info["emt_error_im"],
    )
    logger.info(
        "domain z [%.1f, %.1f] nm, x [0, %.1f] nm; layer wavenumber 2pi/a = %.1f k0, "
        "Fourier z-band to %.1f",
        info["domain_nm"]["z"][0], info["domain_nm"]["z"][1], info["domain_nm"]["x"][1],
        info["layer_wavenumber_over_k0"], info["kz_band_hat"],
    )
    resolution = points_per_layer(n_points, struct)
    logger.info(
        "sampling: %.0f points per Ag layer, %.0f per silica layer, %.0f in the thinnest "
        "(n_points = %d)",
        resolution["mean_per_metal_layer"], resolution["mean_per_dielectric_layer"],
        resolution["min_per_layer"], n_points,
    )

    probe: Dict[str, float] = {}
    if not args.skip_probe:
        logger.info("--- tractability probe: supervised fit of the TMM field ---")
        probe = probe_representability(
            struct, n_epochs=probe_epochs, n_points=probe_points, device=device, seed=args.seed
        )
        logger.info(
            "probe rel L2 %.4e  (stack %.4e, air %.4e, substrate %.4e) in %.0fs",
            probe["probe_rel_l2"], probe["probe_rel_l2_stack"], probe["probe_rel_l2_air"],
            probe["probe_rel_l2_substrate"], probe["probe_time_s"],
        )
        if args.probe_only:
            return {"structure": info, "probe": probe, "resolution": resolution}

    self_check = validate(TMMFieldModule(struct), struct, n_points=4000, device=device)
    logger.info(
        "self-check (TMM field through the pipeline): rel L2 %.2e, k_spp Re err %.2e, "
        "Im err %.2e, kappa_d err %.2e",
        self_check["rel_l2_total"], self_check["k_spp_re_rel_error"],
        self_check["k_spp_im_rel_error"], self_check["kappa_d_rel_error"],
    )
    emt_check = validate(EMTFieldModule(struct), struct, n_points=4000, device=device)
    logger.info(
        "EMT baseline through the same pipeline: rel L2 %.3f (stack %.3f, air %.3f)",
        emt_check["rel_l2_total"], emt_check.get("rel_l2_stack", float("nan")),
        emt_check.get("rel_l2_air", float("nan")),
    )

    generator = torch.Generator().manual_seed(args.seed)
    network = create_network(struct, device=device, generator=generator)
    logger.info("network parameters: %d", sum(p.numel() for p in network.parameters()))

    checkpoint_path = args.model_out.with_suffix(".partial.pth")
    if args.eval_only:
        # Re-score an existing run: the final artefact if it is there, otherwise
        # the atomic training checkpoint. Nothing is trained and nothing is
        # re-seeded, so the metrics describe exactly the saved weights.
        history = {k: [] for k in
                   ("epoch", "total", "curl", "div", "continuity", "boundary", "lr")}
        if args.model_out.exists():
            blob = torch.load(args.model_out, map_location="cpu", weights_only=False)
            network.load_state_dict(blob["state_dict"])
            train_time = float(blob.get("metrics", {}).get("train_time_s", float("nan")))
            history = blob.get("history") or history
            logger.info("evaluating the saved model %s", args.model_out)
        elif checkpoint_path.exists():
            logger.info("evaluating the checkpoint %s (loss %.3e)", checkpoint_path,
                        load_checkpoint_into(network, checkpoint_path))
            train_time = float("nan")
        else:
            raise FileNotFoundError(f"--eval-only: neither {args.model_out} nor {checkpoint_path}")
        network.to(device)
    else:
        if args.resume:
            if checkpoint_path.exists():
                logger.info("resuming from %s (loss %.3e)", checkpoint_path,
                            load_checkpoint_into(network, checkpoint_path))
            else:
                logger.warning("--resume given but %s does not exist; training fresh",
                               checkpoint_path)

        t0 = time.perf_counter()
        network, history = train(
            network, struct, n_epochs=n_epochs, n_points=n_points, learning_rate=args.lr,
            device=device, lbfgs_steps=lbfgs_steps, lbfgs_dtype=lbfgs_dtype,
            checkpoint_path=checkpoint_path,
        )
        train_time = time.perf_counter() - t0
        logger.info("training time %.1f s", train_time)

    metrics = validate(network, struct, n_points=n_val_points, device=device)
    metrics.update(
        train_time_s=train_time, epochs=float(n_epochs), n_points=float(n_points),
        lbfgs_steps=float(lbfgs_steps), lr=args.lr, seed=float(args.seed),
        final_loss=history["total"][-1] if history["total"] else float("nan"),
        best_loss=min(history["total"]) if history["total"] else float("nan"),
    )
    if probe:
        metrics["probe_rel_l2"] = probe["probe_rel_l2"]

    for key in sorted(metrics):
        value = metrics[key]
        logger.info("%-34s %s", key, f"{value:.5e}" if isinstance(value, float) else value)

    figures = {
        "field_profiles": plot_field_profiles(network, struct, args.figures_dir, device),
        "k_spp_comparison": plot_k_comparison(metrics, struct, args.figures_dir),
        "field_map": plot_field_map(network, struct, args.figures_dir, device),
        "training_history": plot_history(
            history, probe.get("probe_history"), args.figures_dir  # type: ignore[arg-type]
        ) if history["total"] else str(args.figures_dir / "training_history.png"),
    }

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": network.state_dict(),
            "history": history,
            "config": {
                "hidden_dims": list(HIDDEN_DIMS),
                "fourier": {
                    "n_z": FOURIER_Z_MODES, "n_x": FOURIER_X_MODES, "n_mix": FOURIER_MIX_MODES,
                    "k_min": FOURIER_K_MIN, "k_z_max": struct.kz_band_hat,
                    "k_xy_max": FOURIER_KX_MAX, "kz_harmonics": FOURIER_KZ_HARMONICS,
                },
                "structure": info,
                "H0": H0, "E_scale": E_SCALE, "H_scale": H_SCALE,
                "input_scaling": "coords * k0, k0 = omega / c",
                "guard_nm": GUARD * 1e9,
                "continuity_offset_nm": CONTINUITY_OFFSET * 1e9,
            },
            "metrics": metrics,
        },
        args.model_out,
    )
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    probe_record = {k: v for k, v in probe.items() if k != "probe_history"}
    write_json_report(args.figures_dir / "metrics.json", {
        "structure": info,
        "resolution": resolution,
        "probe": probe_record,
        "metrics": metrics,
        "tmm_self_check": self_check,
        "emt_baseline": emt_check,
        "figures": figures,
        "run_info": {
            "epochs": n_epochs, "n_points": n_points, "lbfgs_steps": lbfgs_steps,
            "lbfgs_dtype": args.lbfgs_dtype, "seed": args.seed,
            "quick": bool(args.quick), "device": str(device),
            "eval_only": bool(args.eval_only),
        },
    })
    logger.info("saved model to %s, figures + metrics.json to %s", args.model_out,
                args.figures_dir)
    logger.info(
        "HEADLINE  Re k/k0: TMM %.5f | EMT %.5f (%.2f%%) | PINN %.5f (%.2f%%) -> EMT/PINN %.2fx",
        metrics["n_eff_tmm"], metrics["n_eff_emt"], 100 * metrics["emt_re_rel_error"],
        metrics["n_eff_pinn"], 100 * metrics["k_spp_re_rel_error"],
        metrics["re_error_ratio_emt_over_pinn"],
    )
    logger.info(
        "HEADLINE  Im k   : TMM %.4e | EMT %.4e (%.1f%%) | PINN %.4e (%.1f%%) -> EMT/PINN %.2fx",
        metrics["k_spp_im_tmm"], metrics["k_spp_im_emt"], 100 * metrics["emt_im_rel_error"],
        metrics["k_spp_im_pinn"], 100 * metrics["k_spp_im_rel_error"],
        metrics["im_error_ratio_emt_over_pinn"],
    )
    logger.info("success tier: %s", metrics["success_tier"])
    return {
        "structure": info, "metrics": metrics, "probe": probe_record,
        "self_check": self_check, "emt_baseline": emt_check, "history": history,
    }


if __name__ == "__main__":
    main()
