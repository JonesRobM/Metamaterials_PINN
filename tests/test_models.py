"""Tests for the network architectures in ``src.models``."""

import numpy as np
import pytest
import torch

from src.analytical import analytical_potential
from src.constants import C0, EPS0, ETA0
from src.models.electrostatics_pinn import ElectrostaticsPINN, boundary_loss, laplace_residual
from src.models.field_format import join_complex, split_complex, to_complex
from src.models.pinn_network import (
    ComplexLinear,
    ComplexPINN,
    ElectromagneticActivation,
    ElectromagneticPINN,
    FourierEMFeatures,
    MetamaterialDeepONet,
    MultiFrequencyPINN,
    NondimensionalPINN,
    SPPNetwork,
)

OMEGA = 2 * np.pi * 1e15
SMALL = dict(hidden_dims=[16, 16], fourier_modes=8)


def _coords(n=12, dim=3, extent=1e-6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(n, dim, generator=g) * 2 - 1) * extent


def _assert_grads_populated(model: torch.nn.Module, loss: torch.Tensor):
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads and all(g is not None for g in grads)
    assert any(torch.any(g != 0) for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)


# --------------------------------------------------------------------------- building blocks
class TestBuildingBlocks:
    def test_complex_linear_matches_complex_matmul(self):
        torch.manual_seed(0)
        layer = ComplexLinear(4, 3)
        x = torch.randn(5, 4, 2)
        out = layer(x)
        assert out.shape == (5, 3, 2)
        W = torch.complex(layer.weight_real, layer.weight_imag)
        b = torch.complex(layer.bias_real, layer.bias_imag)
        expected = torch.complex(x[..., 0], x[..., 1]) @ W.T + b
        assert torch.allclose(torch.complex(out[..., 0], out[..., 1]), expected, atol=1e-6)

    def test_complex_linear_no_bias(self):
        layer = ComplexLinear(4, 3, bias=False)
        assert layer.bias_real is None and layer.bias_imag is None
        assert layer(torch.zeros(2, 4, 2)).abs().sum() == 0

    @pytest.mark.parametrize("kind", ["complex_tanh", "modulus", "split", "unknown"])
    def test_activation_shapes(self, kind):
        act = ElectromagneticActivation(kind)
        x = torch.randn(7, 5, 2)
        y = act(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()

    def test_modulus_activation_preserves_magnitude(self):
        act = ElectromagneticActivation("modulus")
        x = torch.randn(7, 5, 2)
        y = act(x)
        assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5)


class TestFourierEMFeatures:
    @pytest.mark.parametrize("encoding_size,expected", [(8, 8), (7, 6), (1, 0), (128, 128)])
    def test_output_dim_even_and_odd(self, encoding_size, expected):
        f = FourierEMFeatures(3, encoding_size, include_dc=True)
        assert f.encoding_size == expected
        assert f.output_dim == expected + 3
        out = f(torch.randn(4, 3))
        assert out.shape == (4, f.output_dim)

    def test_output_dim_without_dc(self):
        f = FourierEMFeatures(2, 10, include_dc=False)
        assert f.output_dim == 10
        assert f(torch.randn(3, 2)).shape == (3, 10)

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_k_vector_magnitudes_in_range(self, dim):
        f = FourierEMFeatures(dim, 20, frequency_range=(0.5, 5.0))
        norms = f.k_vectors.norm(dim=1)
        assert f.k_vectors.shape == (10, dim)
        assert torch.all(norms >= 0.5 - 1e-5) and torch.all(norms <= 5.0 + 1e-5)

    def test_features_bounded(self):
        f = FourierEMFeatures(3, 16, include_dc=False)
        out = f(torch.randn(10, 3) * 100)
        assert torch.all(out.abs() <= 1.0 + 1e-6)


# --------------------------------------------------------------------------- PINNs
class TestFourierScaling:
    def _coords_over_one_wavelength(self):
        lam = 2 * np.pi * C0 / OMEGA
        x = torch.linspace(0, lam, 16)
        return lam, torch.stack([x, torch.zeros_like(x), torch.zeros_like(x)], 1)

    def test_raw_si_coordinates_need_explicit_k_range(self):
        # Default band (0.1, 20) rad per input unit is far below k0 ~ 2e7 rad/m
        lam, coords = self._coords_over_one_wavelength()
        net = ElectromagneticPINN(spatial_dim=3, frequency=OMEGA, use_fourier=True, **SMALL)
        feats = net.fourier_encoder(coords)[:, 3:]
        assert float((feats.max(dim=0).values - feats.min(dim=0).values).max()) < 1e-3
        k0 = OMEGA / C0
        net = ElectromagneticPINN(spatial_dim=3, frequency=OMEGA, use_fourier=True,
                                  fourier_k_range=(0.1 * k0, 20 * k0), **SMALL)
        feats = net.fourier_encoder(coords)[:, 3:]
        assert float((feats.max(dim=0).values - feats.min(dim=0).values).max()) > 0.1

    def test_nondimensional_wrapper_resolves_a_wavelength(self):
        lam, coords = self._coords_over_one_wavelength()
        core = ElectromagneticPINN(spatial_dim=3, frequency=OMEGA, use_fourier=True, **SMALL)
        net = NondimensionalPINN(core, lam, NondimensionalPINN.em_field_scale(1.0))
        feats = core.fourier_encoder(coords / net.length_scale)[:, 3:]
        assert float((feats.max(dim=0).values - feats.min(dim=0).values).max()) > 0.1
        out = net(coords)
        assert out.shape == (16, 6, 2)
        E, H = net.get_fields(coords)
        assert E.shape == (16, 3, 2) and H.shape == (16, 3, 2)
        # H channel is scaled down by the wave impedance
        assert torch.allclose(net.field_scale[0, 3:, 0] * ETA0, torch.ones(3), rtol=1e-6)


class TestElectromagneticPINN:
    @pytest.mark.parametrize("complex_valued", [True, False])
    @pytest.mark.parametrize("use_fourier", [True, False])
    def test_forward_shape(self, complex_valued, use_fourier):
        torch.manual_seed(0)
        net = ElectromagneticPINN(spatial_dim=3, field_components=6, complex_valued=complex_valued,
                                  frequency=OMEGA, use_fourier=use_fourier, **SMALL)
        out = net(_coords())
        assert out.shape == (12, 6, 2)
        assert torch.isfinite(out).all()

    def test_time_dependent_layout_takes_extra_column(self):
        net = ElectromagneticPINN(spatial_dim=2, frequency=None, **SMALL)
        assert net(torch.randn(5, 3)).shape == (5, 6, 2)

    def test_e_only_network(self):
        net = ElectromagneticPINN(spatial_dim=3, field_components=3, frequency=OMEGA, **SMALL)
        E, H = net.get_fields(_coords())
        assert E.shape == (12, 3, 2)
        assert torch.all(H == 0)

    def test_get_fields_splits_components(self):
        net = ElectromagneticPINN(frequency=OMEGA, **SMALL)
        c = _coords()
        out = net(c)
        E, H = net.get_fields(c)
        assert torch.equal(E, out[:, :3]) and torch.equal(H, out[:, 3:])

    def test_unsupported_field_components_raise(self):
        net = ElectromagneticPINN(field_components=4, frequency=OMEGA, **SMALL)
        with pytest.raises(ValueError):
            net.get_fields(_coords())

    @pytest.mark.parametrize("complex_valued", [True, False])
    def test_gradient_flow(self, complex_valued):
        torch.manual_seed(0)
        net = ElectromagneticPINN(frequency=OMEGA, complex_valued=complex_valued, **SMALL)
        _assert_grads_populated(net, net(_coords()).pow(2).mean())

    def test_output_not_constant(self):
        torch.manual_seed(0)
        net = ElectromagneticPINN(frequency=OMEGA, **SMALL)
        out = net(_coords(64))
        assert float(out.std(dim=0).max()) > 0


class TestComplexPINN:
    def test_defaults_and_shape(self):
        net = ComplexPINN(spatial_dim=3, frequency=OMEGA, **SMALL)
        assert net.complex_valued and net.field_components == 6
        assert net(_coords()).shape == (12, 6, 2)

    def test_gradient_flow(self):
        torch.manual_seed(0)
        net = ComplexPINN(spatial_dim=3, frequency=OMEGA, **SMALL)
        _assert_grads_populated(net, net(_coords()).abs().mean())

    def test_compute_em_derivatives_matches_autograd(self):
        torch.manual_seed(0)
        net = ComplexPINN(spatial_dim=3, frequency=OMEGA, use_fourier=False, hidden_dims=[16, 16])
        c = _coords(6, extent=1.0)
        d = net.compute_em_derivatives(c, field_component=1, spatial_derivative=0)
        assert d.shape == (6, 2)
        # Finite-difference cross-check on the real part of Ey w.r.t. x
        h = 1e-3
        cp = c.detach().clone()
        cp[:, 0] += h
        cm = c.detach().clone()
        cm[:, 0] -= h
        with torch.no_grad():
            fd = (net(cp)[:, 1, 0] - net(cm)[:, 1, 0]) / (2 * h)
        assert torch.allclose(d[:, 0], fd, rtol=1e-2, atol=1e-4)


class TestSPPNetwork:
    @pytest.fixture
    def spp(self):
        torch.manual_seed(0)
        return SPPNetwork(interface_position=0.0, metal_permittivity=-20 + 1j,
                          dielectric_permittivity=2.25, frequency=OMEGA, decay_length=1e-7, **SMALL)

    def test_forward_shape(self, spp):
        assert spp(_coords()).shape == (12, 6, 2)

    def test_k_spp_has_positive_imaginary_part(self, spp):
        assert spp.k_spp.imag.item() > 0
        assert spp.k_spp.real.item() > spp.k0  # bound mode: k_spp > k0

    def test_k_spp_matches_closed_form(self, spp):
        eps_m, eps_d = -20 + 1j, 2.25
        expected = spp.k0 * np.sqrt(eps_m * eps_d / (eps_m + eps_d))
        if expected.imag < 0:
            expected = -expected
        assert complex(spp.k_spp) == pytest.approx(expected, rel=1e-5)

    def test_envelope_decays_away_from_interface(self, spp):
        L = spp.decay_length
        xy = torch.zeros(5, 2)
        z = torch.tensor([0.0, 0.5, 1.0, 3.0, 8.0]) * L
        c = torch.cat([xy, z.unsqueeze(1)], dim=1)
        with torch.no_grad():
            out = spp(c)
            base = ElectromagneticPINN.forward(spp, c)
        ratio = out.norm(dim=(1, 2)) / base.norm(dim=(1, 2))
        # Envelope is exp(-|z|/L) * (1 + 0.1 * tanh(.)) so lies within [0.9, 1.1] * exp(-|z|/L)
        expected = torch.exp(-z / L)
        assert torch.all(ratio <= 1.1 * expected + 1e-7)
        assert torch.all(ratio >= 0.9 * expected - 1e-7)
        assert torch.all(ratio[1:] < ratio[:-1])

    def test_field_magnitude_decreases_with_distance(self, spp):
        L = spp.decay_length
        g = torch.Generator().manual_seed(1)
        xy = (torch.rand(50, 2, generator=g) * 2 - 1) * 1e-6
        near = torch.cat([xy, torch.zeros(50, 1)], 1)
        far = torch.cat([xy, torch.full((50, 1), 10 * L)], 1)
        with torch.no_grad():
            m_near = spp(near).norm(dim=(1, 2)).mean()
            m_far = spp(far).norm(dim=(1, 2)).mean()
        assert m_far < 1e-3 * m_near

    def test_symmetric_envelope(self, spp):
        c_plus = torch.tensor([[0.0, 0.0, 2e-7]])
        c_minus = torch.tensor([[0.0, 0.0, -2e-7]])
        with torch.no_grad():
            r_plus = spp(c_plus).norm() / ElectromagneticPINN.forward(spp, c_plus).norm()
            r_minus = spp(c_minus).norm() / ElectromagneticPINN.forward(spp, c_minus).norm()
        assert abs(float(r_plus) - float(r_minus)) < 0.2 * np.exp(-2)

    def test_gradient_flow(self, spp):
        _assert_grads_populated(spp, spp(_coords()).pow(2).mean())


class TestMultiFrequencyPINN:
    @pytest.fixture
    def net(self):
        torch.manual_seed(0)
        return MultiFrequencyPINN(frequency_range=(1e14, 1e16), num_frequency_modes=3,
                                  spatial_dim=3, **SMALL)

    def test_forward_shape(self, net):
        freq = torch.full((12, 1), 1e15)
        out = net(_coords(), freq)
        assert out.shape == (12, 6, 2)
        assert torch.isfinite(out).all()

    def test_interpolation_weights_sum_to_one(self, net):
        freq = torch.logspace(14, 16, 5).unsqueeze(1)
        freq_norm = (torch.log10(freq) - 14) / 2
        w = net.freq_interpolator(freq_norm)
        assert w.shape == (5, 3)
        assert torch.allclose(w.sum(dim=1), torch.ones(5), atol=1e-6)

    def test_gradient_flow(self, net):
        freq = torch.full((12, 1), 3e15)
        _assert_grads_populated(net, net(_coords(), freq).pow(2).mean())


class TestMetamaterialDeepONet:
    @pytest.fixture
    def net(self):
        torch.manual_seed(0)
        return MetamaterialDeepONet(material_param_dim=9, spatial_dim=3, field_components=6,
                                    branch_hidden=[16], trunk_hidden=[16], latent_dim=8)

    def test_forward_shape(self, net):
        n = 10
        out = net(torch.randn(n, 9), _coords(n), torch.full((n, 1), 1e15))
        assert out.shape == (n, 6)
        assert torch.isfinite(out).all()

    def test_gradient_flow(self, net):
        n = 10
        out = net(torch.randn(n, 9), _coords(n), torch.full((n, 1), 1e15))
        _assert_grads_populated(net, out.pow(2).mean())

    def test_depends_on_material_parameters(self, net):
        n = 10
        c, f = _coords(n), torch.full((n, 1), 1e15)
        with torch.no_grad():
            a = net(torch.ones(n, 9), c, f)
            b = net(-torch.ones(n, 9), c, f)
        assert not torch.allclose(a, b)


# --------------------------------------------------------------------------- electrostatics
class _AnalyticPotential(torch.nn.Module):
    """2-D (line-charge) potential ``-(q / (2 pi eps0)) ln r``, matching src.analytical."""

    def __init__(self, q=1e-9, q_pos=(0.0, 0.0)):
        super().__init__()
        self.q, self.q_pos = q, q_pos

    def forward(self, xy):
        k = 1.0 / (2.0 * np.pi * EPS0)
        r = torch.sqrt((xy[:, 0] - self.q_pos[0]) ** 2 + (xy[:, 1] - self.q_pos[1]) ** 2)
        return (-k * self.q * torch.log(r)).unsqueeze(1)


class _Harmonic2D(torch.nn.Module):
    """``V = x^2 - y^2 + log r`` is harmonic in 2-D away from the origin."""

    def forward(self, xy):
        r2 = xy[:, 0] ** 2 + xy[:, 1] ** 2
        return (xy[:, 0] ** 2 - xy[:, 1] ** 2 + 0.5 * torch.log(r2)).unsqueeze(1)


class TestElectrostaticsPINN:
    def test_forward_shape(self):
        net = ElectrostaticsPINN(num_layers=3, hidden_dim=8)
        assert net(torch.randn(7, 2)).shape == (7, 1)

    def test_laplace_residual_shape_and_gradient_flow(self):
        torch.manual_seed(0)
        net = ElectrostaticsPINN(num_layers=3, hidden_dim=8)
        res = laplace_residual(net, torch.randn(9, 2))
        assert res.shape == (9,)
        res.pow(2).mean().backward()
        # The output bias cannot influence a Laplacian, so it legitimately has no grad.
        weights = [p for name, p in net.named_parameters() if name.endswith("weight")]
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in weights)
        assert any(torch.any(p.grad != 0) for p in weights)
        assert net.net[-1].bias.grad is None

    def test_boundary_loss_zero_for_exact_values(self):
        net = ElectrostaticsPINN(num_layers=3, hidden_dim=8)
        c = torch.randn(5, 2)
        with torch.no_grad():
            v = net(c)
        assert boundary_loss(net, c, v).item() == 0.0

    def test_laplace_residual_vanishes_for_harmonic_function(self):
        coords = torch.tensor([[1.0, 0.5], [-0.7, 1.2], [2.0, -1.0], [0.3, -0.4]], dtype=torch.float64)
        res = laplace_residual(_Harmonic2D(), coords)
        assert torch.allclose(res, torch.zeros_like(res), atol=1e-10)

    def test_laplace_residual_vanishes_for_analytical_point_charge(self):
        q, q_pos = 1e-9, (0.0, 0.0)
        coords = torch.tensor([[1.0, 0.5], [-0.7, 1.2], [2.0, -1.0]], dtype=torch.float64)
        net = _AnalyticPotential(q, q_pos)
        # Sanity: the wrapper reproduces src.analytical.analytical_potential
        ref = analytical_potential(coords[:, 0].numpy(), coords[:, 1].numpy(), q, q_pos)
        assert np.allclose(net(coords).squeeze(1).detach().numpy(), ref, rtol=1e-6)
        res = laplace_residual(net, coords)
        r = coords.norm(dim=1)
        scale = (net(coords).squeeze(1) / r**2).abs()
        assert torch.all(res.abs() < 1e-6 * scale)


# --------------------------------------------------------------------------- field format
class TestFieldFormat:
    def test_round_trip_split_to_complex(self):
        g = torch.Generator().manual_seed(0)
        z = torch.complex(torch.randn(5, 6, generator=g), torch.randn(5, 6, generator=g))
        fmt = split_complex(z)
        assert fmt.shape == (5, 6, 2)
        E, H = to_complex(fmt)
        assert torch.equal(torch.cat([E, H], dim=1), z)
        assert torch.equal(join_complex(E, H), fmt)

    def test_e_only_tensor(self):
        fmt = torch.randn(4, 3, 2)
        E, H = to_complex(fmt)
        assert E.shape == (4, 3) and torch.all(H == 0)

    def test_preserves_autograd(self):
        fmt = torch.randn(4, 6, 2, requires_grad=True)
        E, H = to_complex(fmt)
        (E.abs().sum() + H.abs().sum()).backward()
        assert fmt.grad is not None

    def test_invalid_inputs(self):
        with pytest.raises(TypeError):
            split_complex(torch.randn(3, 6))
        with pytest.raises(ValueError):
            to_complex(torch.randn(3, 6))
        with pytest.raises(ValueError):
            to_complex(torch.randn(3, 4, 2))
        with pytest.raises(TypeError):
            to_complex(torch.complex(torch.randn(3, 6, 2), torch.randn(3, 6, 2)))
