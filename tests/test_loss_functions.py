"""Tests for every loss in ``src.models.loss_functions`` using analytic plane waves."""

import numpy as np
import pytest
import torch

from src.constants import EPS0, ETA0, MU0
from src.models.loss_functions import (
    EM_CompositeLoss,
    InterfaceBoundaryLoss,
    MaxwellCurlLoss,
    MaxwellDivergenceLoss,
    MetamaterialConstitutiveLoss,
    PowerFlowLoss,
    RadiationLoss,
    SPPBoundaryLoss,
    TangentialContinuityLoss,
    WaveguideLoss,
    _resolve_permittivity,
)
from src.physics.boundary_conditions import BoundaryConditions
from src.physics.metamaterial import MetamaterialProperties
from tests.conftest import K0, OMEGA, make_plane_wave, sample_coords

# Scale of the Ampère residual for a unit-amplitude plane wave, (k0 |H|)^2 = (k0 / eta0)^2.
# This is the smaller of the two curl residual scales, so normalising by it is conservative.
CURL_SCALE = (K0 / ETA0) ** 2


def _normalised_curl_loss(loss_value: torch.Tensor) -> float:
    return float(loss_value) / CURL_SCALE


# --------------------------------------------------------------------------- permittivity spec
class TestResolvePermittivity:
    @pytest.fixture
    def ref(self):
        return torch.zeros(1, dtype=torch.complex128)

    def test_none_is_identity(self, ref):
        t = _resolve_permittivity(None, torch.zeros(4, 3), ref)
        assert t.shape == (4, 3, 3)
        assert torch.allclose(t, torch.eye(3, dtype=torch.complex128).expand(4, -1, -1))

    def test_scalar_diag_full_and_batched(self, ref):
        c = torch.zeros(2, 3)
        assert torch.allclose(_resolve_permittivity(2.0 + 1j, c, ref)[0], (2.0 + 1j) * torch.eye(3, dtype=torch.complex128))
        d = _resolve_permittivity(torch.tensor([1.0, 2.0, 3.0]), c, ref)
        assert torch.allclose(torch.diagonal(d[1]).real, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
        full = torch.ones(3, 3)
        assert _resolve_permittivity(full, c, ref).shape == (2, 3, 3)
        batched = torch.ones(2, 3, 3)
        assert _resolve_permittivity(batched, c, ref).shape == (2, 3, 3)

    def test_metamaterial(self, ref):
        m = MetamaterialProperties(-2.0, 3.0, optical_axis="z")
        t = _resolve_permittivity(m, torch.zeros(2, 3), ref)
        assert complex(t[0, 2, 2]) == -2.0 and complex(t[0, 0, 0]) == 3.0

    def test_bad_shape(self, ref):
        with pytest.raises(ValueError):
            _resolve_permittivity(torch.ones(4, 4), torch.zeros(2, 3), ref)


# --------------------------------------------------------------------------- curl
class TestMaxwellCurlLoss:
    @pytest.fixture
    def loss_fn(self):
        return MaxwellCurlLoss(frequency=OMEGA)

    def test_initialization(self, loss_fn):
        assert loss_fn.omega == OMEGA
        assert loss_fn.mu0 == MU0 and loss_fn.eps0 == EPS0
        assert loss_fn.maxwell_solver.k0 == pytest.approx(K0)

    @pytest.mark.parametrize(
        "epsilon",
        [
            None,
            1.0,
            torch.tensor([1.0, 1.0, 1.0]),
            torch.eye(3),
            MetamaterialProperties(1.0, 1.0, optical_axis="z"),
        ],
        ids=["none", "scalar", "diag3", "full3x3", "metamaterial"],
    )
    def test_vacuum_plane_wave_gives_zero_loss(self, loss_fn, plane_wave_net, coords64, epsilon):
        loss = loss_fn.compute(network=plane_wave_net, coords=coords64, epsilon=epsilon)
        assert loss.dtype == torch.float64
        assert _normalised_curl_loss(loss) < 1e-20

    def test_float32_plane_wave(self, loss_fn):
        net = make_plane_wave(dtype=torch.float32)
        coords = sample_coords(32, dtype=torch.float32)
        loss = loss_fn.compute(network=net, coords=coords)
        assert _normalised_curl_loss(loss) < 1e-5

    def test_dielectric_plane_wave(self, loss_fn):
        """A wave with |k| = sqrt(eps) k0 satisfies Maxwell in a dielectric with that eps."""
        eps = 2.25
        net = make_plane_wave(k_scale=np.sqrt(eps))
        coords = sample_coords(32)
        assert _normalised_curl_loss(loss_fn.compute(network=net, coords=coords, epsilon=eps)) < 1e-20
        # ...but not in vacuum
        assert _normalised_curl_loss(loss_fn.compute(network=net, coords=coords)) > 1e-3

    def test_wrong_wavenumber_gives_positive_loss(self, loss_fn, coords64):
        net = make_plane_wave(k_scale=0.5)
        loss = loss_fn.compute(network=net, coords=coords64)
        # Only Ampère fails: |res_H|^2 = (ωε0 (1 - k²/k0²))² = (0.75 k0/η0)², averaged over 3 components
        expected = (0.75 * K0 / ETA0) ** 2 / 3
        assert float(loss) == pytest.approx(expected, rel=1e-6)

    def test_legacy_material_props(self, loss_fn, plane_wave_net, coords64):
        n = coords64.shape[0]
        props = torch.zeros(n, 2, 2, dtype=torch.float64)
        props[:, 0, 0] = 1.0  # mu_r = 1
        props[:, 1, 0] = 1.0  # eps_r = 1
        loss = loss_fn.compute(network=plane_wave_net, coords=coords64, material_props=props)
        assert _normalised_curl_loss(loss) < 1e-20
        props[:, 1, 0] = 4.0  # wrong eps
        assert _normalised_curl_loss(loss_fn.compute(network=plane_wave_net, coords=coords64, material_props=props)) > 1e-3

    def test_mu_r_tensor(self, loss_fn, plane_wave_net, coords64):
        mu = torch.ones(coords64.shape[0], dtype=torch.complex128)
        loss = loss_fn.compute(network=plane_wave_net, coords=coords64, mu_r=mu)
        assert _normalised_curl_loss(loss) < 1e-20

    def test_weight_applied_via_call(self, coords64):
        net = make_plane_wave(k_scale=0.5)
        a = MaxwellCurlLoss(OMEGA, weight=1.0)(network=net, coords=coords64)
        b = MaxwellCurlLoss(OMEGA, weight=3.0)(network=net, coords=coords64)
        assert float(b) == pytest.approx(3 * float(a))

    def test_backpropagates_to_network(self):
        torch.manual_seed(0)
        from src.models.pinn_network import ComplexPINN

        net = ComplexPINN(spatial_dim=3, frequency=OMEGA, hidden_dims=[8, 8], fourier_modes=8)
        loss = MaxwellCurlLoss(OMEGA)(network=net, coords=sample_coords(8, dtype=torch.float32))
        loss.backward()
        assert all(p.grad is not None for p in net.parameters())


# --------------------------------------------------------------------------- divergence
class TestMaxwellDivergenceLossConstructor:
    def test_weight_only(self):
        assert MaxwellDivergenceLoss(weight=2.5).weight == 2.5

    def test_frequency_is_rejected_with_an_explanatory_message(self):
        """The curl loss takes a frequency and this one does not; the error has
        to say which class and why, not just 'unexpected keyword argument'."""
        with pytest.raises(TypeError) as excinfo:
            MaxwellDivergenceLoss(frequency=2e15)
        message = str(excinfo.value)
        assert "MaxwellDivergenceLoss" in message
        assert "frequency" in message
        assert "MaxwellCurlLoss" in message


class TestMaxwellDivergenceLoss:
    def test_plane_wave_zero(self, plane_wave_net, coords64):
        loss = MaxwellDivergenceLoss().compute(network=plane_wave_net, coords=coords64)
        assert float(loss) / K0**2 < 1e-20

    def test_anisotropic_epsilon_still_zero_for_transverse_wave(self, plane_wave_net, coords64):
        # k along x, E along y: div(eps E) = i eps_yy k_x... no, k·E = 0 so still zero.
        loss = MaxwellDivergenceLoss().compute(network=plane_wave_net, coords=coords64, epsilon=torch.tensor([1.0, 3.0, 2.0]))
        assert float(loss) / K0**2 < 1e-20

    def test_longitudinal_wave_nonzero(self, coords64):
        net = make_plane_wave(direction=(1, 0, 0), pol=(1, 0, 0))  # E parallel to k
        loss = MaxwellDivergenceLoss().compute(network=net, coords=coords64)
        # div E = i k·E = i k0 -> mean |div|^2 = k0^2 (scalar residual, no component averaging)
        assert float(loss) == pytest.approx(K0**2, rel=1e-8)

    def test_charge_density_subtracted(self, coords64):
        net = make_plane_wave(direction=(1, 0, 0), pol=(1, 0, 0))
        E, _ = net.fields(coords64)
        rho = 1j * K0 * E[:, 0]
        loss = MaxwellDivergenceLoss().compute(network=net, coords=coords64, charge_density=rho.detach())
        assert float(loss) / K0**2 < 1e-20


# --------------------------------------------------------------------------- power flow
class TestPowerFlowLoss:
    def test_plane_wave_zero(self, plane_wave_net, coords64):
        loss = PowerFlowLoss().compute(network=plane_wave_net, coords=coords64)
        S_scale = (K0 / (2 * ETA0)) ** 2  # (k0 |S|)^2
        assert float(loss) / S_scale < 1e-18

    def test_nonzero_for_field_with_diverging_power(self, coords64):
        class Radial(torch.nn.Module):
            def forward(self, c):
                E = c  # E = r
                H = torch.stack([-c[:, 1], c[:, 0], torch.zeros_like(c[:, 0])], 1)
                return torch.stack([torch.cat([E, H], 1), torch.zeros(c.shape[0], 6, dtype=c.dtype)], -1)

        loss = PowerFlowLoss().compute(network=Radial(), coords=sample_coords(16, extent=1.0))
        assert float(loss) > 0


# --------------------------------------------------------------------------- interfaces
class TestTangentialContinuityLoss:
    def test_zero_for_continuous_network(self, plane_wave_net):
        pts = sample_coords(32).detach()
        pts[:, 2] = 0.0
        normals = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64).expand(32, -1)
        loss = TangentialContinuityLoss(offset=1e-9).compute(plane_wave_net, pts, normals)
        assert float(loss) < 1e-24

    def test_nonzero_for_discontinuous_network(self):
        class Step(torch.nn.Module):
            def forward(self, c):
                jump = (c[:, 2] > 0).to(c.dtype).unsqueeze(1)
                out = torch.zeros(c.shape[0], 6, 2, dtype=c.dtype)
                out[:, 0, 0] = jump[:, 0]  # Ex jumps across z = 0
                return out

        pts = torch.zeros(8, 3, dtype=torch.float64)
        normals = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64).expand(8, -1)
        loss = TangentialContinuityLoss(offset=1e-9).compute(Step(), pts, normals)
        # |n x ΔE|^2 = 1 in one of three components
        assert float(loss) == pytest.approx(1.0 / 3.0)


class TestInterfaceBoundaryLoss:
    def test_zero_when_same_network_and_eps(self, plane_wave_net):
        pts = sample_coords(16).detach()
        pts[:, 2] = 0.0
        loss_fn = InterfaceBoundaryLoss(BoundaryConditions((0, 0, 1)), eps_medium_1=2.0, eps_medium_2=2.0, offset=1e-9)
        loss = loss_fn.compute(plane_wave_net, pts, network_2=plane_wave_net)
        assert float(loss) < 1e-24

    def test_uses_constructor_coords(self, plane_wave_net):
        pts = torch.zeros(4, 3, dtype=torch.float64)
        loss_fn = InterfaceBoundaryLoss(interface_coords=pts)
        assert float(loss_fn.compute(plane_wave_net)) < 1e-24

    def test_missing_coords_raises(self, plane_wave_net):
        with pytest.raises(ValueError):
            InterfaceBoundaryLoss().compute(plane_wave_net)

    def test_nonzero_for_different_eps(self, plane_wave_net):
        """Same E on both sides with different eps violates normal-D continuity when Ez != 0."""
        net = make_plane_wave(direction=(1, 0, 0), pol=(0, 0, 1))  # E along the normal z
        pts = torch.zeros(4, 3, dtype=torch.float64)
        loss_fn = InterfaceBoundaryLoss(eps_medium_1=1.0, eps_medium_2=3.0)
        # residual (relative units) = (3 - 1) Ez over 16 entries: mean = 4/16 = 0.25
        assert float(loss_fn.compute(net, pts)) == pytest.approx(0.25, rel=1e-6)

    def test_nonzero_for_different_networks(self, plane_wave_net):
        other = make_plane_wave(pol=(0, 0.5, 0))
        pts = torch.zeros(4, 3, dtype=torch.float64)
        assert float(InterfaceBoundaryLoss().compute(plane_wave_net, pts, network_2=other)) > 0


# --------------------------------------------------------------------------- SPP / waveguide / radiation
class TestSPPBoundaryLoss:
    def test_zero_for_exact_envelope(self):
        L = 1e-7

        class Envelope(torch.nn.Module):
            def forward(self, c):
                out = torch.zeros(c.shape[0], 6, 2, dtype=c.dtype)
                out[:, 0, 0] = torch.exp(-c[:, 2].abs() / L)
                return out

        coords = sample_coords(32, extent=3e-7).detach()
        coords[0, 2] = 0.0  # ensure max is attained at the interface
        assert float(SPPBoundaryLoss(1e7, decay_length=L).compute(Envelope(), coords)) < 1e-24

    def test_nonzero_for_plane_wave(self, plane_wave_net):
        coords = sample_coords(32, extent=3e-7).detach()
        assert float(SPPBoundaryLoss(1e7, decay_length=1e-7).compute(plane_wave_net, coords)) > 1e-3


class TestWaveguideLoss:
    def test_zero_for_matching_beta(self, coords64):
        net = make_plane_wave(direction=(1, 0, 0), pol=(1, 0, 0))
        loss = WaveguideLoss(propagation_direction=0).compute(net, coords64, beta=K0)
        assert float(loss) / K0**2 < 1e-18

    def test_nonzero_for_wrong_beta(self, coords64):
        net = make_plane_wave(direction=(1, 0, 0), pol=(1, 0, 0))
        loss = WaveguideLoss().compute(net, coords64, beta=0.5 * K0)
        assert float(loss) == pytest.approx((0.5 * K0) ** 2, rel=1e-6)


class TestRadiationLoss:
    """The radial direction is ``coords / (|r| + 1e-8)``: exact only when |r| >> 1e-8 m."""

    @staticmethod
    def _x_axis_points(r_min, r_max, n=8):
        r = torch.linspace(r_min, r_max, n, dtype=torch.float64)
        return torch.stack([r, torch.zeros_like(r), torch.zeros_like(r)], 1).requires_grad_(True)

    def test_outgoing_wave_along_radius(self):
        """A plane wave along +x evaluated on the +x axis satisfies the Sommerfeld condition."""
        net = make_plane_wave(direction=(1, 0, 0))
        loss = RadiationLoss().compute(net, self._x_axis_points(1.0, 2.0), k0=K0)
        assert float(loss) / K0**2 < 1e-12

    def test_incoming_wave_penalised(self):
        net = make_plane_wave(direction=(-1, 0, 0))
        loss = RadiationLoss().compute(net, self._x_axis_points(1.0, 2.0), k0=K0)
        # residual = -2 i k0 E in one of three components -> mean|res|^2 = 4 k0^2 / 3
        assert float(loss) == pytest.approx(4 * K0**2 / 3, rel=1e-6)
    def test_outgoing_wave_at_micron_scale(self):
        net = make_plane_wave(direction=(1, 0, 0))
        loss = RadiationLoss().compute(net, self._x_axis_points(1e-6, 2e-6), k0=K0)
        assert float(loss) / K0**2 < 1e-12


# --------------------------------------------------------------------------- constitutive placeholder
def test_metamaterial_constitutive_loss_not_implemented():
    loss = MetamaterialConstitutiveLoss(MetamaterialProperties(1.0, 1.0))
    with pytest.raises(NotImplementedError):
        loss.compute()
    with pytest.raises(NotImplementedError):
        loss(network=None, coords=None)


# --------------------------------------------------------------------------- composite
class TestEMCompositeLoss:
    def test_aggregates_components(self, plane_wave_net, coords64):
        comp = EM_CompositeLoss(
            {"maxwell_curl": MaxwellCurlLoss(OMEGA, weight=2.0), "divergence": MaxwellDivergenceLoss(weight=1.0)},
            adaptive_weights=False,
        )
        total, parts = comp.compute(network=make_plane_wave(k_scale=0.5), coords=coords64)
        assert set(parts) == {"maxwell_curl", "divergence"}
        assert float(total) == pytest.approx(sum(float(v) for v in parts.values()))
        assert float(parts["maxwell_curl"]) == pytest.approx(2.0 * (0.75 * K0 / ETA0) ** 2 / 3, rel=1e-6)
        assert comp.step_count == 1
        assert comp.get_physics_residuals()["divergence"] == float(parts["divergence"])

    def test_forwards_only_accepted_kwargs(self, coords64):
        comp = EM_CompositeLoss(
            {"curl": MaxwellCurlLoss(OMEGA), "wg": WaveguideLoss(), "rad": RadiationLoss()},
            adaptive_weights=False,
        )
        # Transverse wave propagating along (1, 0, 1)/sqrt(2): Ex != 0 with phase velocity beta = k0/sqrt(2)
        net = make_plane_wave(direction=(1, 0, 1), pol=(1, 0, -1))
        r = torch.linspace(1.0, 2.0, 4, dtype=torch.float64) / np.sqrt(2)
        pts = torch.stack([r, torch.zeros_like(r), r], 1)  # radial direction parallel to k
        total, parts = comp.compute(
            network=net, coords=coords64, beta=K0 / np.sqrt(2), boundary_coords=pts, k0=K0, epsilon=None
        )
        assert set(parts) == {"curl", "wg", "rad"}
        assert float(total) / K0**2 < 1e-12

    def test_missing_required_kwarg_raises_type_error(self, plane_wave_net, coords64):
        comp = EM_CompositeLoss({"curl": MaxwellCurlLoss(OMEGA), "wg": WaveguideLoss()}, adaptive_weights=False)
        with pytest.raises(TypeError, match=r"Loss 'wg' requires keyword argument\(s\) \['beta'\]"):
            comp.compute(network=plane_wave_net, coords=coords64)

    def test_adaptive_weights_rebalance(self, coords64):
        curl = MaxwellCurlLoss(OMEGA)
        div = MaxwellDivergenceLoss()
        comp = EM_CompositeLoss({"maxwell_curl": curl, "div": div}, adaptive_weights=True, update_interval=1)
        net = make_plane_wave(k_scale=0.5)  # large curl loss, zero divergence loss
        comp.compute(network=net, coords=coords64)
        # Weights were rescaled toward the mean; the dominant curl term is down-weighted
        assert curl.weight != 1.0
        assert curl.weight < div.weight
        assert comp.running_means["maxwell_curl"] > comp.running_means["div"]
