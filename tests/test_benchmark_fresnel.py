"""Benchmark of the interface machinery against the exact Fresnel solution.

Unlike the self-consistency tests elsewhere in the suite (same field on both
sides), these tests build the *exact* analytical solution for a planar
dielectric interface and check that the repository's boundary-condition
residuals vanish for it -- and, crucially, that they do NOT vanish when the
reflection coefficient is detuned by 5%.

Physics derivation (verified independently with complex ik-algebra before any
repo code was used):

Convention
    Time dependence ``e^{-i omega t}``; plane waves ``F = F0 e^{i k.r}``.
    Maxwell in a source-free non-magnetic medium of relative permittivity eps:

        i k x E =  i omega mu0 H      =>  H = (k x E) / (omega mu0)
        i k x H = -i omega eps0 eps E =>  E = -(k x H) / (omega eps0 eps)

    Lossy media have ``Im(eps) > 0`` (decaying waves for the +z branch below).

Geometry
    Interface at z = 0 with normal +z; medium 1 (eps1) fills z < 0, medium 2
    (eps2) fills z > 0. Plane of incidence is x-z. The tangential wavevector
    ``kx = n1 k0 sin(theta_i)`` is conserved (Snell's law); on each side

        k_jz = sqrt(eps_j k0^2 - kx^2),   branch Im(k_jz) >= 0
        (and Re >= 0 when purely real),

    which handles real and complex eps2 (complex Snell) uniformly.
    Incident  k_i = (kx, 0, +k1z); reflected k_r = (kx, 0, -k1z);
    transmitted k_t = (kx, 0, +k2z).

TE (s-polarisation), E along y
    E_y continuity:            1 + r_s = t_s
    H_x = -k_z E_y/(omega mu0) continuity:  k1z (1 - r_s) = k2z t_s

        r_s = (k1z - k2z)/(k1z + k2z)
            = (n1 cos th_i - n2 cos th_t)/(n1 cos th_i + n2 cos th_t)
        t_s = 1 + r_s = 2 n1 cos th_i / (n1 cos th_i + n2 cos th_t)

TM (p-polarisation), H along y -- H-AMPLITUDE CONVENTION
    We define r_p and t_p as ratios of the *H_y* amplitudes (incident H_y = 1
    reference). E follows from ``E = -(k x H)/(omega eps0 eps)``, i.e.
    ``E_x = +k_z H_y/(omega eps0 eps)`` and ``E_z = -kx H_y/(omega eps0 eps)``.

    H_y continuity:            1 + r_p = t_p
    E_x continuity:            (k1z/eps1)(1 - r_p) = (k2z/eps2) t_p

        r_p = (eps2 k1z - eps1 k2z)/(eps2 k1z + eps1 k2z)
            = (n2 cos th_i - n1 cos th_t)/(n2 cos th_i + n1 cos th_t)
        t_p = 1 + r_p = 2 n2 cos th_i / (n2 cos th_i + n1 cos th_t)

    This is the Born & Wolf sign convention: at normal incidence
    r_p = +(n2-n1)/(n2+n1) = -r_s (the classic sign pitfall). The usual
    E-amplitude transmission is recovered as t_p^E = (n1/n2) t_p; here every
    field is constructed directly from the H amplitudes so no E-ratio
    convention ever enters. Brewster: r_p = 0 at th_B = atan(n2/n1).

Consequences checked below
    * tangential E, tangential H, normal D = eps0 eps E and normal B = mu0 H
      are continuous at z = 0 (for TM, normal D continuity is equivalent to
      H_y continuity since D_z = -kx H_y/omega on both sides);
    * normal E is *discontinuous* for TM at oblique incidence (jump by
      eps1/eps2);
    * each side's field satisfies Maxwell in its own medium exactly;
    * everything is bilinear in the amplitudes, so the same algebra closes
      for complex eps2 (verified independently to ~1e-13 relative).

Scales: fields are normalised to |E_incident| = 1, so |H| ~ 1/eta0. All
residual assertions are made relative to the appropriate field scale. The
wavelength is LAM0 = 1 m so that the loss classes' evaluation offsets
(1e-9 .. 1e-6 m) are << lambda and the leading offset error ~ (2 k delta)^2
stays far below the perturbation floors.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.constants import C0, EPS0, ETA0, MU0
from src.models.field_format import join_complex
from src.models.loss_functions import (
    InterfaceBoundaryLoss,
    MaxwellCurlLoss,
    TangentialContinuityLoss,
)
from src.physics.boundary_conditions import BoundaryConditions

LAM0 = 1.0  # free-space wavelength (m) -- huge vs the 1e-9..1e-6 m offsets
K0 = 2 * np.pi / LAM0
OMEGA = C0 * K0

EPS1 = 1.0
EPS2 = 2.25
EPS2_LOSSY = 2.25 + 0.1j

ESCALE = 1.0          # |E| scale (normalised incident wave)
HSCALE = 1.0 / ETA0   # |H| scale for |E| = 1

ANGLES_DEG = [0.0, 30.0, 55.0]
POLS = ["TE", "TM"]

_YHAT = np.array([0.0, 1.0, 0.0], dtype=complex)


# --------------------------------------------------------------------------
# Exact Fresnel construction (independent of all repo physics code)
# --------------------------------------------------------------------------
def _kz(eps: complex, kx: float) -> complex:
    """z-wavenumber with the decaying/outgoing branch Im >= 0 (Re >= 0 if real)."""
    v = complex(np.sqrt(eps * K0**2 - kx**2 + 0j))
    if v.imag < 0 or (v.imag == 0 and v.real < 0):
        v = -v
    return v


def fresnel_waves(eps1, eps2, theta_i_rad, pol, r_scale=1.0):
    """Exact plane-wave decomposition of the two-medium Fresnel solution.

    Returns ``(waves1, waves2, r, t)`` where each waves list holds
    ``(k, E0, H0)`` torch complex128 triples: medium 1 = incident + reflected,
    medium 2 = transmitted. ``r_scale`` multiplies the reflected amplitude
    only (``t`` stays exact) to produce a controlled violation of the
    interface conditions.
    """
    n1 = float(np.sqrt(eps1).real)
    kx = n1 * K0 * np.sin(theta_i_rad)
    k1z, k2z = _kz(eps1, kx), _kz(eps2, kx)
    ki = np.array([kx, 0.0, k1z], dtype=complex)
    kr = np.array([kx, 0.0, -k1z], dtype=complex)
    kt = np.array([kx, 0.0, k2z], dtype=complex)

    if pol == "TE":
        r = (k1z - k2z) / (k1z + k2z)
        t = 1.0 + r

        def wave(k, amp):
            E0 = amp * _YHAT
            H0 = np.cross(k, E0) / (OMEGA * MU0)
            return k, E0, H0

        waves1 = [wave(ki, 1.0), wave(kr, r * r_scale)]
        waves2 = [wave(kt, t)]
    elif pol == "TM":
        r = (eps2 * k1z - eps1 * k2z) / (eps2 * k1z + eps1 * k2z)
        t = 1.0 + r
        h0 = n1 / ETA0  # |E_incident| = 1

        def wave(k, amp, eps):
            H0 = amp * _YHAT
            E0 = -np.cross(k, H0) / (OMEGA * EPS0 * eps)
            return k, E0, H0

        waves1 = [wave(ki, h0, eps1), wave(kr, h0 * r * r_scale, eps1)]
        waves2 = [wave(kt, h0 * t, eps2)]
    else:  # pragma: no cover - defensive
        raise ValueError(pol)

    to_t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.complex128)  # noqa: E731
    conv = lambda ws: [(to_t(k), to_t(E0), to_t(H0)) for k, E0, H0 in ws]  # noqa: E731
    return conv(waves1), conv(waves2), complex(r), complex(t)


def eval_waves(waves, coords: torch.Tensor):
    """Sum of plane waves at real ``coords`` (N, 3); supports complex k."""
    E = torch.zeros(coords.shape[0], 3, dtype=torch.complex128, device=coords.device)
    H = torch.zeros_like(E)
    for k, E0, H0 in waves:
        phase_re = coords @ k.real.to(coords.dtype)
        phase_im = coords @ k.imag.to(coords.dtype)
        env = torch.exp(-phase_im)
        f = torch.complex(env * torch.cos(phase_re), env * torch.sin(phase_re))
        E = E + E0.unsqueeze(0) * f.unsqueeze(1)
        H = H + H0.unsqueeze(0) * f.unsqueeze(1)
    return E, H


class PlaneWaveSumNetwork(nn.Module):
    """Exact analytic field of one medium as an ``(N, 6, 2)`` float64 network."""

    def __init__(self, waves):
        super().__init__()
        self.waves = waves

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return join_complex(*eval_waves(self.waves, coords))


class TwoSidedNetwork(nn.Module):
    """Single network switching between the two media's exact fields at z = 0."""

    def __init__(self, waves1, waves2):
        super().__init__()
        self.net1 = PlaneWaveSumNetwork(waves1)
        self.net2 = PlaneWaveSumNetwork(waves2)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        side2 = (coords[:, 2] > 0).view(-1, 1, 1)
        return torch.where(side2, self.net2(coords), self.net1(coords))


def interface_points(n: int = 48, seed: int = 0) -> torch.Tensor:
    """Points on the z = 0 interface spanning about one wavelength in x, y."""
    g = torch.Generator().manual_seed(seed)
    xy = (torch.rand(n, 2, generator=g, dtype=torch.float64) - 0.5) * LAM0
    return torch.cat([xy, torch.zeros(n, 1, dtype=torch.float64)], dim=1)


def side_coords(sign: float, n: int = 32, seed: int = 1) -> torch.Tensor:
    """Random points strictly inside one half-space, |z| in [0.05, 0.45] lam0."""
    g = torch.Generator().manual_seed(seed)
    pts = (torch.rand(n, 3, generator=g, dtype=torch.float64) - 0.5) * LAM0
    pts[:, 2] = sign * (0.05 * LAM0 + pts[:, 2].abs() * 0.8)
    return pts.requires_grad_(True)


def eps_tensor(eps, n_pts: int) -> torch.Tensor:
    eye = torch.eye(3, dtype=torch.complex128)
    return (eye * complex(eps)).unsqueeze(0).expand(n_pts, -1, -1)


def _max_abs(residual: torch.Tensor) -> float:
    return float(residual.abs().max())


def _interface_fields(eps2, theta_deg, pol, r_scale=1.0, n=48):
    waves1, waves2, r, t = fresnel_waves(EPS1, eps2, np.deg2rad(theta_deg), pol, r_scale)
    pts = interface_points(n)
    E1, H1 = eval_waves(waves1, pts)
    E2, H2 = eval_waves(waves2, pts)
    return pts, (E1, H1), (E2, H2), r, t


# --------------------------------------------------------------------------
# 0. Zero-parameter cross-checks of the Fresnel construction itself
# --------------------------------------------------------------------------
class TestFresnelSelfChecks:
    def test_brewster_angle_rp_vanishes(self):
        """TM reflection vanishes at th_B = atan(n2/n1) -- exact zero of r_p."""
        n1, n2 = np.sqrt(EPS1), np.sqrt(EPS2)
        theta_b = np.arctan(n2 / n1)
        _, _, r, _ = fresnel_waves(EPS1, EPS2, theta_b, "TM")
        assert abs(r) < 1e-12

    @pytest.mark.parametrize("theta", ANGLES_DEG)
    @pytest.mark.parametrize("pol", POLS)
    def test_energy_conservation_lossless(self, pol, theta):
        """R + T = 1 for a lossless interface (power, not amplitude, ratios)."""
        th = np.deg2rad(theta)
        _, _, r, t = fresnel_waves(EPS1, EPS2, th, pol)
        kx = np.sqrt(EPS1) * K0 * np.sin(th)
        k1z, k2z = _kz(EPS1, kx), _kz(EPS2, kx)
        if pol == "TE":
            T = (k2z / k1z).real * abs(t) ** 2
        else:  # H-amplitude convention: T = Re(k2z/eps2)/(k1z/eps1) |t|^2
            T = (k2z / EPS2).real / (k1z / EPS1).real * abs(t) ** 2
        assert abs(r) ** 2 + T == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("eps2", [EPS2, EPS2_LOSSY], ids=["lossless", "lossy"])
    @pytest.mark.parametrize("pol", POLS)
    def test_lossy_wave_decays_into_medium_2(self, pol, eps2):
        (waves1, waves2, _, _) = fresnel_waves(EPS1, eps2, np.deg2rad(30.0), pol)
        k_t = waves2[0][0]
        assert float(k_t[2].imag) >= 0.0
        if eps2 is EPS2_LOSSY:
            assert float(k_t[2].imag) > 0.0


# --------------------------------------------------------------------------
# 1 & 2 & 7. BoundaryConditions residuals for the exact fields at z = 0
# --------------------------------------------------------------------------
class TestBoundaryConditionsFresnel:
    bc = BoundaryConditions(interface_normal=(0.0, 0.0, 1.0))

    @pytest.mark.parametrize("eps2", [EPS2, EPS2_LOSSY], ids=["lossless", "lossy"])
    @pytest.mark.parametrize("theta", ANGLES_DEG)
    @pytest.mark.parametrize("pol", POLS)
    def test_exact_fields_satisfy_all_continuity(self, pol, theta, eps2):
        pts, (E1, H1), (E2, H2), _, _ = _interface_fields(eps2, theta, pol)
        n_pts = pts.shape[0]
        assert _max_abs(self.bc.tangential_E_continuity(E1, E2)) / ESCALE < 1e-10
        assert _max_abs(self.bc.tangential_H_continuity(H1, H2)) / HSCALE < 1e-10
        res_d = self.bc.normal_D_continuity(
            E1, E2, eps_tensor(EPS1, n_pts), eps_tensor(eps2, n_pts), relative=True
        )
        assert _max_abs(res_d) / ESCALE < 1e-10
        assert _max_abs(self.bc.normal_B_continuity(H1, H2, relative=True)) / HSCALE < 1e-10

    @pytest.mark.parametrize("eps2", [EPS2, EPS2_LOSSY], ids=["lossless", "lossy"])
    @pytest.mark.parametrize("theta", [30.0, 55.0])
    @pytest.mark.parametrize("pol", POLS)
    def test_perturbed_r_breaks_tangential_continuity(self, pol, theta, eps2):
        """r -> 1.05 r (t kept exact) must show up in both tangential residuals."""
        _, (E1, H1), (E2, H2), r, _ = _interface_fields(eps2, theta, pol, r_scale=1.05)
        floor = 0.01 * abs(r)
        assert _max_abs(self.bc.tangential_E_continuity(E1, E2)) / ESCALE > floor
        assert _max_abs(self.bc.tangential_H_continuity(H1, H2)) / HSCALE > floor

    @pytest.mark.parametrize("theta", [30.0, 55.0])
    def test_tm_oblique_normal_E_jumps_but_D_continuous(self, theta):
        """The discriminating case: eps E_z continuous while E_z itself is not."""
        pts, (E1, H1), (E2, H2), _, _ = _interface_fields(EPS2, theta, "TM")
        n_pts = pts.shape[0]
        res_d = self.bc.normal_D_continuity(
            E1, E2, eps_tensor(EPS1, n_pts), eps_tensor(EPS2, n_pts), relative=True
        )
        assert _max_abs(res_d) / ESCALE < 1e-10
        # normal E is discontinuous: eps1 E1z = eps2 E2z => E1z = (eps2/eps1) E2z
        jump = _max_abs(E2[:, 2] - E1[:, 2]) / ESCALE
        assert jump > 0.1
        expected = abs(EPS2 / EPS1 - 1.0) * _max_abs(E2[:, 2])
        assert jump == pytest.approx(expected, rel=1e-10)

    @pytest.mark.parametrize("theta", [30.0, 55.0])
    def test_tm_oblique_perturbed_r_breaks_normal_D(self, theta):
        pts, (E1, _), (E2, _), r, _ = _interface_fields(EPS2, theta, "TM", r_scale=1.05)
        n_pts = pts.shape[0]
        res_d = self.bc.normal_D_continuity(
            E1, E2, eps_tensor(EPS1, n_pts), eps_tensor(EPS2, n_pts), relative=True
        )
        floor = 0.005 * abs(r) * np.sin(np.deg2rad(theta))
        assert _max_abs(res_d) / ESCALE > floor


# --------------------------------------------------------------------------
# 3. The combined spp_boundary_conditions stack
# --------------------------------------------------------------------------
class TestSPPBoundaryStack:
    """``spp_boundary_conditions(E1, H1, E2, H2, eps1_tensor, eps2_scalar)``
    is generic enough to express the dielectric-dielectric problem: medium 1
    plays the "metamaterial" (eps1 * I tensor), medium 2 the dielectric."""

    bc = BoundaryConditions(interface_normal=(0.0, 0.0, 1.0))

    @pytest.mark.parametrize("eps2", [EPS2, EPS2_LOSSY], ids=["lossless", "lossy"])
    @pytest.mark.parametrize("theta", ANGLES_DEG)
    @pytest.mark.parametrize("pol", POLS)
    def test_stack_vanishes_for_exact_fields(self, pol, theta, eps2):
        pts, (E1, H1), (E2, H2), _, _ = _interface_fields(eps2, theta, pol)
        res = self.bc.spp_boundary_conditions(
            E1, H1, E2, H2, eps_tensor(EPS1, pts.shape[0]), eps_dielectric=eps2, relative=True
        )
        assert res.shape == (pts.shape[0], 16)
        assert _max_abs(res) / ESCALE < 1e-10

    @pytest.mark.parametrize("pol", POLS)
    def test_stack_nonzero_for_perturbed_r(self, pol):
        pts, (E1, H1), (E2, H2), r, _ = _interface_fields(EPS2, 30.0, pol, r_scale=1.05)
        res = self.bc.spp_boundary_conditions(
            E1, H1, E2, H2, eps_tensor(EPS1, pts.shape[0]), eps_dielectric=EPS2, relative=True
        )
        assert _max_abs(res) / ESCALE > 0.01 * abs(r) * HSCALE


# --------------------------------------------------------------------------
# 4. InterfaceBoundaryLoss with per-side networks
# --------------------------------------------------------------------------
class TestInterfaceBoundaryLossFresnel:
    CASES = [
        ("TE", 30.0, EPS2),
        ("TM", 30.0, EPS2),
        ("TM", 55.0, EPS2),
        ("TM", 30.0, EPS2_LOSSY),
    ]

    @staticmethod
    def _loss(eps2, offset=1e-9):
        return InterfaceBoundaryLoss(
            BoundaryConditions((0.0, 0.0, 1.0)),
            eps_medium_1=EPS1,
            eps_medium_2=eps2,
            offset=offset,
        )

    @pytest.mark.parametrize("pol,theta,eps2", CASES)
    def test_two_networks_exact_solution(self, pol, theta, eps2):
        waves1, waves2, _, _ = fresnel_waves(EPS1, eps2, np.deg2rad(theta), pol)
        loss = self._loss(eps2).compute(
            PlaneWaveSumNetwork(waves1),
            interface_points(),
            network_2=PlaneWaveSumNetwork(waves2),
        )
        assert float(loss) < 1e-12

    @pytest.mark.parametrize("pol,theta,eps2", CASES)
    def test_two_networks_perturbed_r(self, pol, theta, eps2):
        waves1, waves2, r, _ = fresnel_waves(EPS1, eps2, np.deg2rad(theta), pol, r_scale=1.05)
        loss = self._loss(eps2).compute(
            PlaneWaveSumNetwork(waves1),
            interface_points(),
            network_2=PlaneWaveSumNetwork(waves2),
        )
        assert float(loss) > (0.001 * abs(r)) ** 2

    @pytest.mark.parametrize("pol", POLS)
    def test_single_conditional_network(self, pol):
        """One network holding both media's fields (switch at z = 0)."""
        waves1, waves2, _, _ = fresnel_waves(EPS1, EPS2, np.deg2rad(30.0), pol)
        loss = self._loss(EPS2).compute(TwoSidedNetwork(waves1, waves2), interface_points())
        assert float(loss) < 1e-12


# --------------------------------------------------------------------------
# 5. TangentialContinuityLoss with a single conditional network
# --------------------------------------------------------------------------
class TestTangentialContinuityLossFresnel:
    @staticmethod
    def _setup(pol, r_scale=1.0):
        waves1, waves2, r, _ = fresnel_waves(EPS1, EPS2, np.deg2rad(30.0), pol, r_scale)
        pts = interface_points()
        normals = torch.zeros_like(pts)
        normals[:, 2] = 1.0
        return TwoSidedNetwork(waves1, waves2), pts, normals, r

    @pytest.mark.parametrize("pol", POLS)
    def test_exact_solution_default_offset(self, pol):
        """Default offset 1e-6 m << LAM0 = 1 m: residual ~ (2 k delta)^2."""
        net, pts, normals, _ = self._setup(pol)
        loss = TangentialContinuityLoss().compute(net, pts, normals)
        assert float(loss) < 1e-8

    @pytest.mark.parametrize("pol", POLS)
    def test_exact_solution_small_offset(self, pol):
        net, pts, normals, _ = self._setup(pol)
        loss = TangentialContinuityLoss(offset=1e-9).compute(net, pts, normals)
        assert float(loss) < 1e-14

    @pytest.mark.parametrize("pol", POLS)
    def test_perturbed_r_detected(self, pol):
        net, pts, normals, r = self._setup(pol, r_scale=1.05)
        loss = TangentialContinuityLoss(offset=1e-9).compute(net, pts, normals)
        assert float(loss) > (0.001 * abs(r)) ** 2


# --------------------------------------------------------------------------
# Maxwell residual per side: each medium's field solves Maxwell in its medium
# --------------------------------------------------------------------------
class TestMaxwellResidualPerSide:
    @pytest.mark.parametrize("eps2", [EPS2, EPS2_LOSSY], ids=["lossless", "lossy"])
    @pytest.mark.parametrize("pol", POLS)
    def test_curl_residual_each_side(self, pol, eps2):
        waves1, waves2, _, _ = fresnel_waves(EPS1, eps2, np.deg2rad(30.0), pol)
        curl_loss = MaxwellCurlLoss(frequency=OMEGA)
        loss1 = curl_loss.compute(
            PlaneWaveSumNetwork(waves1), side_coords(-1.0), epsilon=EPS1
        )
        loss2 = curl_loss.compute(
            PlaneWaveSumNetwork(waves2), side_coords(+1.0), epsilon=eps2
        )
        # residual scale is k0 |E| ~ 2 pi; squared machine noise is ~1e-28
        assert float(loss1.detach()) < 1e-22
        assert float(loss2.detach()) < 1e-22

    @pytest.mark.parametrize("pol", POLS)
    def test_wrong_epsilon_gives_large_residual(self, pol):
        """Sanity: the residual check has teeth (medium-2 field vs eps1)."""
        _, waves2, _, _ = fresnel_waves(EPS1, EPS2, np.deg2rad(30.0), pol)
        curl_loss = MaxwellCurlLoss(frequency=OMEGA)
        loss = curl_loss.compute(PlaneWaveSumNetwork(waves2), side_coords(+1.0), epsilon=EPS1)
        # residual_H picks up omega eps0 (eps2 - eps1) |E| ~ 2e-2 per component
        assert float(loss.detach()) > 1e-5
