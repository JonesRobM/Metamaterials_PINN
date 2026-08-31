"""
Benchmark of the anisotropic permittivity-tensor path against exact uniaxial
plane waves.

Validates :meth:`src.physics.MaxwellEquations.total_residual` and
:class:`src.models.loss_functions.MaxwellCurlLoss` with a full ``(N, 3, 3)``
permittivity tensor, using the closed-form ordinary and extraordinary plane
waves of a homogeneous uniaxial crystal (optical axis along ``z``,
``eps = diag(eps_t, eps_t, eps_n)`` relative).

Derivation (``exp(-iwt)`` convention, fields ~ ``exp(ik.r)`` so ``curl -> ik x``):

* Faraday ``curl E = i w mu0 H`` gives ``H = (k x E) / (w mu0)``.
* Substituting into Ampere ``curl H = -i w eps0 eps E`` yields the wave
  equation ``k (k.E) - k^2 E = -k0^2 eps E`` with ``k0 = w / c``.
* **Ordinary wave** (``E`` along ``y``, perpendicular to both ``k`` and the
  optical axis): ``k.E = 0`` and ``eps E = eps_t E``, so ``k^2 = eps_t k0^2``
  for every propagation angle.
* **Extraordinary wave**: take ``k = k(theta) (sin t, 0, cos t)`` in the x-z
  plane and build ``E`` from the *transverse* displacement field
  ``D = D0 (cos t, 0, -sin t) exp(ik.r)`` (``D`` is perpendicular to ``k``;
  ``E = eps^{-1} D / eps0`` is **not**), i.e. ``E0 = (cos t / eps_t, 0,
  -sin t / eps_n)`` up to normalisation. Inserting ``E0`` into the wave
  equation, both the x and z components reduce to the same dispersion relation

      ``k(theta)^2 (cos^2 t / eps_t + sin^2 t / eps_n) = k0^2``,

  i.e. ``1 / n_e(theta)^2 = cos^2 t / eps_t + sin^2 t / eps_n`` with
  ``n_e = k / k0``. Limits: at ``theta = 0`` (propagation along the optical
  axis) ``E`` is transverse to the axis and sees ``eps_t``, so
  ``k = sqrt(eps_t) k0`` (degenerate with the ordinary wave); at
  ``theta = 90 deg`` ``E`` lies along the optical axis and sees ``eps_n``,
  so ``k = sqrt(eps_n) k0``.

The derivation uses only bilinear (not conjugating) vector algebra, so it
holds verbatim for complex ``eps_t``, ``eps_n`` with a complex ``k(theta)``
of real direction (a homogeneous wave in an absorbing crystal). This is
verified independently in :func:`test_analytic_construction_self_consistent`
with a ten-line ``ik x`` implementation before any repository operator is
trusted, so a failure of the later tests indicts the repository code, not the
benchmark fields.
"""

from __future__ import annotations

import cmath
import math

import pytest
import torch
import torch.nn as nn

from src.constants import EPS0, MU0
from src.models.field_format import join_complex
from src.models.loss_functions import MaxwellCurlLoss
from src.physics.maxwell_equations import MaxwellEquations
from tests.conftest import K0, OMEGA, sample_coords

EPS_T_LOSSLESS = 2.25
EPS_N_LOSSLESS = 4.0
EPS_T_LOSSY = 2.25 + 0.01j
EPS_N_LOSSY = 4.0 + 0.02j

EPS_CASES = [
    pytest.param(EPS_T_LOSSLESS, EPS_N_LOSSLESS, id="lossless"),
    pytest.param(EPS_T_LOSSY, EPS_N_LOSSY, id="lossy"),
]

REL_TOL = 1e-9  # float64; observed residuals are ~1e-15 relative


class UniaxialPlaneWaveNet(nn.Module):
    """Exact plane wave ``E = E0 exp(ik.r)``, ``H = (k x E) / (w mu0)`` as a PINN.

    Unlike :class:`tests.conftest.PlaneWaveNetwork` this accepts a **complex**
    wavevector, as needed for the lossy uniaxial medium. Returns ``(N, 6, 2)``.
    """

    def __init__(self, k_vec, E0, omega: float = OMEGA):
        super().__init__()
        self.register_buffer("k_vec", torch.as_tensor(k_vec, dtype=torch.complex128))
        self.register_buffer("E0", torch.as_tensor(E0, dtype=torch.complex128))
        self.omega = float(omega)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        phase = coords.to(self.k_vec.dtype) @ self.k_vec
        E = self.E0.unsqueeze(0) * torch.exp(1j * phase).unsqueeze(1)
        H = torch.linalg.cross(self.k_vec.unsqueeze(0).expand_as(E), E, dim=1) / (
            self.omega * MU0
        )
        return join_complex(E, H)


def extraordinary_wave(theta: float, eps_t: complex, eps_n: complex):
    """Return ``(k_vec, E0)`` of the extraordinary wave at angle ``theta``."""
    n = cmath.sqrt(1.0 / (math.cos(theta) ** 2 / eps_t + math.sin(theta) ** 2 / eps_n))
    if n.real < 0:  # forward-propagating branch (Im(n) >= 0 follows for Im(eps) >= 0)
        n = -n
    k_vec = [n * K0 * math.sin(theta), 0.0, n * K0 * math.cos(theta)]
    E0 = [math.cos(theta) / eps_t, 0.0, -math.sin(theta) / eps_n]
    return k_vec, E0


def ordinary_wave(theta: float, eps_t: complex):
    """Return ``(k_vec, E0)`` of the ordinary wave (``E`` along ``y``)."""
    n = cmath.sqrt(eps_t)
    k_vec = [n * K0 * math.sin(theta), 0.0, n * K0 * math.cos(theta)]
    return k_vec, [0.0, 1.0, 0.0]


def eps_diag(eps_t: complex, eps_n: complex) -> torch.Tensor:
    return torch.tensor([eps_t, eps_t, eps_n], dtype=torch.complex128)


def assert_total_residual_small(net: UniaxialPlaneWaveNet, eps_t: complex, eps_n: complex):
    """Run ``MaxwellEquations.total_residual`` and assert per-block relative smallness."""
    maxwell = MaxwellEquations(OMEGA)
    coords = sample_coords(n=48)
    out = net(coords)
    E = torch.complex(out[:, :3, 0], out[:, :3, 1])
    H = torch.complex(out[:, 3:, 0], out[:, 3:, 1])
    eps_tensor = torch.diag_embed(eps_diag(eps_t, eps_n)).unsqueeze(0).expand(
        coords.shape[0], -1, -1
    )
    res = maxwell.total_residual(E, H, coords, eps_tensor)

    eps_E = torch.einsum("nij,nj->ni", eps_tensor, E)
    k_mag = net.k_vec.abs().max()
    scale_curl_E = (K0 * E.abs().max()).item()
    scale_curl_H = (OMEGA * EPS0 * eps_E.abs().max()).item()
    scale_div_E = (k_mag * eps_E.abs().max()).item()
    scale_div_H = (k_mag * H.abs().max()).item()

    assert res[:, 0:6].abs().max().item() / scale_curl_E < REL_TOL
    assert res[:, 6:12].abs().max().item() / scale_curl_H < REL_TOL
    assert res[:, 12:14].abs().max().item() / scale_div_E < REL_TOL
    assert res[:, 14:16].abs().max().item() / scale_div_H < REL_TOL


def curl_loss(net: UniaxialPlaneWaveNet, epsilon) -> float:
    return MaxwellCurlLoss(OMEGA).compute(net, sample_coords(n=48), epsilon=epsilon).item()


def curl_loss_scale(net: UniaxialPlaneWaveNet) -> float:
    """Squared magnitude scale of the curl residuals, ``(k0 |E|)^2``."""
    E0_max = net.E0.abs().max().item()
    return (K0 * E0_max) ** 2


# --------------------------------------------------------------------------- 0
def test_analytic_construction_self_consistent():
    """Verify the benchmark fields by direct ``ik x`` algebra (no repo code).

    Checks ``ik x E0 - i w mu0 H0 = 0`` and ``ik x H0 + i w eps0 eps E0 = 0``
    plus transversality of ``D`` and ``H``, for ordinary and extraordinary
    waves, lossless and lossy. A later failure therefore indicts the
    repository operators, not this construction.
    """

    def cross(a, b):
        return torch.tensor(
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ],
            dtype=torch.complex128,
        )

    for eps_t, eps_n in [(EPS_T_LOSSLESS, EPS_N_LOSSLESS), (EPS_T_LOSSY, EPS_N_LOSSY)]:
        eps = torch.diag(eps_diag(eps_t, eps_n))
        waves = [ordinary_wave(math.radians(30), eps_t)] + [
            extraordinary_wave(math.radians(d), eps_t, eps_n) for d in (0, 30, 45, 60, 90)
        ]
        for k_vec, E0 in waves:
            k = torch.tensor(k_vec, dtype=torch.complex128)
            E = torch.tensor(E0, dtype=torch.complex128)
            H = cross(k, E) / (OMEGA * MU0)
            eps_E = eps @ E
            r_faraday = 1j * cross(k, E) - 1j * OMEGA * MU0 * H
            r_ampere = 1j * cross(k, H) + 1j * OMEGA * EPS0 * eps_E
            assert r_faraday.abs().max() / (K0 * E.abs().max()) < 1e-12
            assert r_ampere.abs().max() / (OMEGA * EPS0 * eps_E.abs().max()) < 1e-12
            assert (k @ eps_E).abs() / (k.abs().max() * eps_E.abs().max()) < 1e-12
            assert (k @ H).abs() / (k.abs().max() * H.abs().max()) < 1e-12


# --------------------------------------------------------------------------- 1
@pytest.mark.parametrize("eps_t,eps_n", EPS_CASES)
def test_ordinary_wave_residuals_vanish(eps_t, eps_n):
    """Ordinary wave at 30 deg: k^2 = eps_t k0^2, E perpendicular to axis and k."""
    k_vec, E0 = ordinary_wave(math.radians(30), eps_t)
    net = UniaxialPlaneWaveNet(k_vec, E0)
    assert_total_residual_small(net, eps_t, eps_n)
    assert curl_loss(net, eps_diag(eps_t, eps_n)) < REL_TOL * curl_loss_scale(net)


# --------------------------------------------------------------------------- 2
@pytest.mark.parametrize("deg", [0, 30, 60, 90])
@pytest.mark.parametrize("eps_t,eps_n", EPS_CASES)
def test_extraordinary_wave_residuals_vanish(deg, eps_t, eps_n):
    """Extraordinary wave satisfies both curl equations with the tensor eps."""
    k_vec, E0 = extraordinary_wave(math.radians(deg), eps_t, eps_n)
    net = UniaxialPlaneWaveNet(k_vec, E0)
    assert_total_residual_small(net, eps_t, eps_n)
    assert curl_loss(net, eps_diag(eps_t, eps_n)) < REL_TOL * curl_loss_scale(net)


@pytest.mark.parametrize("eps_t,eps_n", EPS_CASES)
def test_extraordinary_dispersion_limits(eps_t, eps_n):
    """k(0) = sqrt(eps_t) k0 (E transverse to axis), k(90) = sqrt(eps_n) k0 (E along axis)."""
    k_axis, E0_axis = extraordinary_wave(0.0, eps_t, eps_n)
    k_perp, E0_perp = extraordinary_wave(math.pi / 2, eps_t, eps_n)
    k_axis_mag = torch.tensor(k_axis, dtype=torch.complex128).norm()
    k_perp_mag = torch.tensor(k_perp, dtype=torch.complex128).norm()
    assert abs(k_axis_mag.item() - abs(cmath.sqrt(eps_t)) * K0) < 1e-9 * K0
    assert abs(k_perp_mag.item() - abs(cmath.sqrt(eps_n)) * K0) < 1e-9 * K0
    # polarisation limits: E transverse to the axis at 0 deg, along it at 90 deg
    # (math.cos(pi/2) is ~1e-17, not exactly zero, hence the tolerances)
    assert abs(E0_axis[2]) < 1e-15
    assert abs(E0_perp[0]) < 1e-15 and abs(E0_perp[2]) > 0.1


# --------------------------------------------------------------------------- 3
def test_isotropic_mismatch_is_detected():
    """Negative control: the extraordinary wave must NOT satisfy isotropic eps_t.

    With eps = eps_t 1, the Ampere residual is i w eps0 (eps_t - eps_n) E_z
    plus the dispersion mismatch, so the loss must be orders of magnitude
    above the matched-tensor loss.
    """
    k_vec, E0 = extraordinary_wave(math.radians(45), EPS_T_LOSSLESS, EPS_N_LOSSLESS)
    net = UniaxialPlaneWaveNet(k_vec, E0)
    matched = curl_loss(net, eps_diag(EPS_T_LOSSLESS, EPS_N_LOSSLESS))
    mismatched = curl_loss(net, EPS_T_LOSSLESS)  # scalar isotropic
    assert mismatched > 1e3 * matched
    # Also large on the natural scale of the Ampere residual, (w eps0 |eps E|)^2
    ampere_scale = (OMEGA * EPS0 * EPS_N_LOSSLESS * net.E0.abs().max().item()) ** 2
    assert mismatched > 1e-3 * ampere_scale


# --------------------------------------------------------------------------- 4
def test_epsilon_format_equivalence():
    """(3,), (3, 3) and (N, 3, 3) permittivity specs give the same loss."""
    k_vec, E0 = extraordinary_wave(math.radians(30), EPS_T_LOSSLESS, EPS_N_LOSSLESS)
    net = UniaxialPlaneWaveNet(k_vec, E0)
    n = 48
    diag = torch.tensor([EPS_T_LOSSLESS, EPS_T_LOSSLESS, EPS_N_LOSSLESS], dtype=torch.float64)
    losses = [
        curl_loss(net, diag),
        curl_loss(net, torch.diag(diag)),
        curl_loss(net, torch.diag(diag).unsqueeze(0).expand(n, -1, -1)),
    ]
    assert losses[0] == pytest.approx(losses[1], rel=1e-12, abs=0.0)
    assert losses[0] == pytest.approx(losses[2], rel=1e-12, abs=0.0)
