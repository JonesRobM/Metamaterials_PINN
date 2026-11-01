"""Tests for examples/validate_spp_dispersion.py."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from examples import validate_spp_dispersion as vd
from src.analytical import analytical_spp_fields
from src.models import MaxwellCurlLoss

DEVICE = torch.device("cpu")


def test_band_and_domain_constants():
    assert vd.OMEGA0 == pytest.approx(2 * np.pi * vd.C0 / 633e-9)
    assert vd.OMEGA_MIN == pytest.approx(0.85 * vd.OMEGA0)
    assert vd.OMEGA_MAX == pytest.approx(1.15 * vd.OMEGA0)
    # Non-dispersive uniaxial case
    assert vd.EPS_T == -4 + 0.2j and vd.EPS_N == 3 + 0.05j and vd.EPS_D == 1.0
    # Worst case over the band is ω_min: domain sized there
    k, kd, km = vd.mode_constants(vd.OMEGA_MIN)
    assert vd.X_MAX == pytest.approx(2 * 2 * np.pi / k.real)
    assert vd.Z_MIN == pytest.approx(-3.5 / km.real)
    assert vd.Z_MAX == pytest.approx(1.2 / kd.real)
    # Benchmarked worst-case scales (λ_spp 693 nm, δ_m 75.5 nm, δ_d 302 nm)
    assert vd.X_MAX == pytest.approx(1386.8e-9, rel=1e-3)
    assert vd.Z_MIN == pytest.approx(-264.2e-9, rel=1e-2)
    assert vd.Z_MAX == pytest.approx(362.9e-9, rel=1e-2)
    # All dimensionless wavenumbers fit the Fourier band (0.1, 40) at BOTH edges
    for omega in (vd.OMEGA_MIN, vd.OMEGA_MAX):
        for c in vd.mode_constants(omega):
            assert 0.1 < c.real * vd.LAMBDA0 < 40.0
    # ε is non-dispersive, so k_spp scales exactly linearly with ω
    k_hi = vd.mode_constants(vd.OMEGA_MAX)[0]
    assert k_hi.real / k.real == pytest.approx(1.15 / 0.85, rel=1e-12)


def test_omega_hat_normalisation():
    """The frequency feature maps the band onto [-1, 1] with ω₀ at 0."""
    assert vd.omega_hat(vd.OMEGA_MIN) == pytest.approx(-1.0)
    assert vd.omega_hat(vd.OMEGA0) == pytest.approx(0.0)
    assert vd.omega_hat(vd.OMEGA_MAX) == pytest.approx(1.0)
    for f in (0.85, 0.93, 1.0, 1.07, 1.15):
        omega = f * vd.OMEGA0
        assert vd.omega_from_hat(vd.omega_hat(omega)) == pytest.approx(omega)
        # The ratio recovered inside the loss folding is ω/ω₀
        w_col = torch.tensor([[vd.omega_hat(omega)]], dtype=torch.float64)
        _, mu = vd.interior_material_args(w_col, metal=False)
        assert float(mu[0]) == pytest.approx(f, rel=1e-12)


def test_validation_and_lbfgs_frequency_sets():
    """9 validation ω incl. both ends; the odd grid points held out from L-BFGS."""
    ratios = [f / vd.OMEGA0 for f in vd.VALIDATION_OMEGAS]
    assert len(ratios) == 9
    assert ratios[0] == pytest.approx(0.85) and ratios[-1] == pytest.approx(1.15)
    lbfgs_ratios = {round(f / vd.OMEGA0, 6) for f in vd.LBFGS_OMEGAS}
    held_out = [r for r in ratios if round(r, 6) not in lbfgs_ratios]
    assert len(held_out) == 4  # 0.8875, 0.9625, 1.0375, 1.1125


@pytest.mark.parametrize("omega_factor", [0.85, 1.15])
def test_anchor_target_matches_analytical_at_band_edges(omega_factor):
    """The scaled anchor equals analytical_spp_fields at the band-edge ω's."""
    torch.manual_seed(0)
    omega = omega_factor * vd.OMEGA0
    boundary = vd.sample_boundary_points(60, device=DEVICE)
    target_hat = vd.analytical_fields_hat(boundary / vd.LAMBDA0, omega)
    round_trip = target_hat * vd.FIELD_SCALE
    E, H = analytical_spp_fields(
        boundary, omega, vd.EPS_T, vd.EPS_N, eps_dielectric=vd.EPS_D, H0=vd.H0
    )
    direct = torch.cat(
        [torch.stack([E.real, E.imag], -1), torch.stack([H.real, H.imag], -1)], dim=1
    ).to(torch.float32)
    scale = float(direct.abs().max())
    assert torch.allclose(round_trip, direct, rtol=1e-3, atol=1e-3 * scale)
    # The scaled anchor stays O(1) across the band (k/ω is ω-independent here)
    assert 0.1 < float(target_hat.abs().max()) < 3.0


class _AnalyticalHatMode(nn.Module):
    """Exact mode at a fixed ω in the dimensionless frame (3-column coords)."""

    def __init__(self, omega: float):
        super().__init__()
        self.omega = float(omega)

    def forward(self, coords_hat: torch.Tensor) -> torch.Tensor:
        return vd.analytical_fields_hat(coords_hat, self.omega)


@pytest.mark.parametrize("omega_factor", [0.85, 1.12])
def test_per_omega_loss_frequency_scaling(omega_factor):
    """The folded material args make the curl loss's frequency scale with ω.

    With epsilon = (ω/ω₀)·ε and mu_r = ω/ω₀ the exact mode at ω is a zero of
    the frequency-2π loss; with the unscaled args (r = 1) it is not.
    """
    torch.manual_seed(0)
    omega = omega_factor * vd.OMEGA0
    net = _AnalyticalHatMode(omega)
    loss = MaxwellCurlLoss(frequency=vd.OMEGA_HAT_REF, mu0=1.0, eps0=1.0)
    w = vd.omega_hat(omega)
    for metal in (False, True):
        n = 48
        coords = torch.rand(n, 3, dtype=torch.float64) * 0.5
        coords[:, 2] = (coords[:, 2] * 0.25 + 0.02) * (-1.0 if metal else 1.0)
        w_col = torch.full((n, 1), w, dtype=torch.float64)
        eps_rows, mu = vd.interior_material_args(w_col, metal=metal)
        c1 = coords.clone().requires_grad_(True)
        res_scaled = loss.compute(network=net, coords=c1, epsilon=eps_rows, mu_r=mu)
        assert float(res_scaled.detach()) < 1e-12
        if omega_factor != 1.0:
            eps_1, mu_1 = vd.interior_material_args(torch.zeros_like(w_col), metal=metal)
            c2 = coords.clone().requires_grad_(True)
            res_unscaled = loss.compute(network=net, coords=c2, epsilon=eps_1, mu_r=mu_1)
            assert float(res_unscaled.detach()) > 1e-3


def test_sampling_uses_per_omega_scales_and_guard():
    torch.manual_seed(0)
    n = 4000
    for omega in (vd.OMEGA_MIN, vd.OMEGA_MAX):
        coords = vd.sample_collocation_points(n, omega, device=DEVICE)
        assert coords.shape == (n, 3) and coords.requires_grad
        c = coords.detach()
        assert torch.all(c[:, 0] >= 0) and torch.all(c[:, 0] <= vd.X_MAX)
        assert torch.all(c[:, 2] >= vd.Z_MIN) and torch.all(c[:, 2] <= vd.Z_MAX)
        assert torch.all(c[:, 2].abs() >= vd.GUARD * (1 - 1e-6))
        assert int((c[:, 2] < 0).sum()) == int(round(vd.METAL_FRACTION * n))
    # Metal strata follow the sampled ω's own penetration depth: the ω_max mode
    # is shallower, so its |z| quartile sits closer to the interface.
    torch.manual_seed(1)
    z_lo = vd.sample_collocation_points(n, vd.OMEGA_MIN, device=DEVICE).detach()[:, 2]
    torch.manual_seed(1)
    z_hi = vd.sample_collocation_points(n, vd.OMEGA_MAX, device=DEVICE).detach()[:, 2]
    assert float(z_hi[z_hi < 0].abs().median()) < float(z_lo[z_lo < 0].abs().median())


def test_training_batch_layout():
    torch.manual_seed(0)
    omegas = [0.9 * vd.OMEGA0, 1.1 * vd.OMEGA0]
    batch = vd.sample_training_batch(64, 24, 8, omegas, device=DEVICE)
    n_air = batch["coords_air"].shape[0]
    n_metal = batch["coords_metal"].shape[0]
    assert n_air + n_metal == 64
    assert batch["coords_air"].shape[1] == 3  # 3-col: the ω̂ ride along separately
    assert batch["w_air"].shape == (n_air, 1)
    assert batch["eps_air"].shape == (n_air, 3, 3)
    assert batch["mu_air"].shape == (n_air,)
    # ω̂ columns take only the block values
    w_vals = {round(float(v), 6) for v in torch.cat([batch["w_air"], batch["w_metal"]])}
    assert w_vals == {round(vd.omega_hat(o), 6) for o in omegas}
    # Metal ε rows are (ω/ω₀)·diag(ε_t, ε_t, ε_n)
    r = 1 + vd.BAND_HALF_WIDTH * batch["w_metal"][0, 0].double()
    expected = torch.diag(torch.tensor([vd.EPS_T, vd.EPS_T, vd.EPS_N], dtype=torch.complex128))
    assert torch.allclose(batch["eps_metal"][0], r * expected)
    # Anchor targets align with their blocks' ω
    assert batch["boundary"].shape[0] == batch["target"].shape[0] == batch["w_bc"].shape[0]


def test_network_shapes_and_ez_jump():
    torch.manual_seed(0)
    network = vd.create_network(hidden_dims=(32, 32), fourier_modes=16, device=DEVICE)
    coords = torch.rand(10, 3) * torch.tensor([vd.X_MAX, vd.Y_MAX, vd.Z_MAX])
    out = network(coords, vd.OMEGA0)
    assert out.shape == (10, 6, 2) and out.dtype == torch.float32
    # at_omega gives the 3-column SI view used by the validation pipeline
    net3 = network.at_omega(1.1 * vd.OMEGA0)
    assert net3(coords).shape == (10, 6, 2)
    # The DisplacementAdapter enforces the D_z jump: E_z just below/above the
    # interface differ by ε_n/ε_d for the continuous MLP output
    eps = 1e-12
    pts = torch.tensor([[100e-9, 50e-9, -eps], [100e-9, 50e-9, +eps]], dtype=torch.float32)
    E, _ = vd.to_complex(network(pts, vd.OMEGA0))
    ratio = E[1, 2] / E[0, 2]
    assert torch.allclose(
        ratio, torch.tensor(vd.EPS_N / complex(vd.EPS_D), dtype=ratio.dtype), rtol=1e-3
    )


def test_validate_on_analytical_network_all_probe_frequencies():
    """Pipeline self-check: the ω-consuming analytical mode is near-perfect at
    every probe frequency (frequency-feature plumbing + conventions)."""
    torch.manual_seed(0)
    model = vd.AnalyticalDispersionSPP()
    per_freq = vd.validate_band(model, vd.VALIDATION_OMEGAS, n_points=500, device=DEVICE)
    assert len(per_freq) == 9
    for m in per_freq.values():
        assert all(np.isfinite(v) for v in m.values())
        assert m["rel_l2_E"] < 1e-6 and m["rel_l2_H"] < 1e-6
        assert m["k_spp_rel_error"] < 1e-6
        assert m["kappa_d_fit_rel_error"] < 1e-6
        assert m["kappa_m_fit_rel_error"] < 1e-6
        assert m["decay_sign_correct_air"] == 1.0
        assert m["decay_sign_correct_metal"] == 1.0
        for side in ("air", "metal"):
            assert m[f"curl_E_residual_rel_{side}"] < 1e-5
            assert m[f"curl_H_residual_rel_{side}"] < 1e-5
    summary = vd.summarise(per_freq)
    assert summary["success_tier"] == "stretch"
    assert summary["bound_mode_everywhere"] == 1.0


def test_validate_on_untrained_network_finite():
    torch.manual_seed(0)
    network = vd.create_network(hidden_dims=(32, 32), fourier_modes=16, device=DEVICE)
    m = vd.validate_at_omega(network, vd.OMEGA0, n_points=300, device=DEVICE)
    assert all(np.isfinite(v) for v in m.values())


def test_success_tier_classification():
    base = {
        "worst_rel_l2": 0.01, "worst_k_spp_rel_error": 1e-3, "bound_mode_everywhere": 1.0,
    }
    assert vd.success_tier(base) == "stretch"
    assert vd.success_tier({**base, "worst_rel_l2": 0.05}) == "target"
    assert vd.success_tier({**base, "worst_rel_l2": 0.3}) == "minimum"
    assert vd.success_tier({**base, "worst_rel_l2": 0.05, "worst_k_spp_rel_error": 0.05}) == (
        "minimum"
    )
    assert vd.success_tier({**base, "worst_rel_l2": 0.9}) == "not met"
    assert vd.success_tier({**base, "bound_mode_everywhere": 0.0}) == "not met"


def test_parse_args_defaults():
    args = vd.parse_args([])
    assert args.epochs == vd.N_EPOCHS and args.lbfgs_dtype == "float64"
    assert args.lbfgs_steps == vd.LBFGS_STEPS
    args = vd.parse_args(["--quick", "--lbfgs-dtype", "float32"])
    assert args.quick and args.lbfgs_dtype == "float32"


@pytest.mark.slow
def test_short_training_reduces_loss():
    """A --quick-style short run must decrease the loss (multi-ω batching path)."""
    torch.manual_seed(0)
    np.random.seed(0)
    network = vd.create_network(hidden_dims=(32, 32), fourier_modes=16, device=DEVICE)
    network, history = vd.train(
        network, n_epochs=100, n_points=192, device=DEVICE, log_every=1000, lbfgs_steps=1
    )
    assert set(history) == {"epoch", "total", "curl", "div", "continuity", "boundary", "lr"}
    assert len(history["total"]) == 101  # 100 Adam epochs + 1 L-BFGS step
    assert all(np.isfinite(history["total"]))
    assert min(history["total"][-10:]) < history["total"][0]
    # The float64 L-BFGS phase must hand back a float32 network
    assert all(p.dtype == torch.float32 for p in network.parameters())
