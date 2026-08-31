"""Tests for examples/validate_spp.py."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from examples import validate_spp as vs
from src.analytical import analytical_spp_fields

DEVICE = torch.device("cpu")


@pytest.fixture
def uniaxial_case():
    """Switch the module to the uniaxial case, restoring silver afterwards."""
    vs.configure_case("uniaxial")
    yield vs
    vs.configure_case("silver")


def test_constants():
    assert vs.OMEGA == pytest.approx(2 * np.pi * vs.C0 / 633e-9)
    assert vs.K0 == pytest.approx(vs.OMEGA / vs.C0)
    # Default case is silver (isotropic)
    assert vs.CASE == "silver"
    assert vs.EPS_METAL_T == vs.EPS_METAL_N == -18.3 + 0.55j
    # Benchmarked SPP quantities for silver/air at 633 nm
    assert vs.LAMBDA_SPP == pytest.approx(615.5e-9, rel=1e-3)
    assert vs.DELTA_D == pytest.approx(419e-9, rel=1e-2)
    assert vs.DELTA_M == pytest.approx(22.9e-9, rel=1e-2)
    # Domain derives from the mode scales
    assert vs.Z_MIN == pytest.approx(-4.4 * vs.DELTA_M)
    assert vs.Z_MAX == pytest.approx(1.2 * vs.DELTA_D)
    # Dimensionless frame: k0 * lambda0 = 2 pi = the loss's frequency
    assert vs.K0 * vs.LAMBDA0 == pytest.approx(vs.OMEGA_HAT)
    # Metal-side decay wavenumber must sit inside the widened Fourier band
    assert vs.KAPPA_M.real * vs.LAMBDA0 < 40.0
    # Output paths are per-case
    assert vs.MODEL_PATH.name == "spp_validation.pth"
    assert vs.FIGURE_PREFIX == ""


def test_constants_uniaxial(uniaxial_case):
    """The uniaxial case rederives every scale from the anisotropic dispersion."""
    assert vs.CASE == "uniaxial"
    assert vs.EPS_METAL_T == -4 + 0.2j  # in-plane (xx, yy)
    assert vs.EPS_METAL_N == 3 + 0.05j  # normal (zz)
    # k_spp^2 = k0^2 eps_d eps_n (eps_t - eps_d) / (eps_t eps_n - eps_d^2)
    eps_t, eps_n, eps_d = vs.EPS_METAL_T, vs.EPS_METAL_N, complex(vs.EPS_DIEL)
    k_expected = np.sqrt(
        vs.K0**2 * eps_d * eps_n * (eps_t - eps_d) / (eps_t * eps_n - eps_d**2)
    )
    assert vs.K_SPP == pytest.approx(k_expected, rel=1e-12)
    # Benchmarked scales for eps_t = -4+0.2j, eps_n = 3+0.05j, eps_d = 1
    assert vs.LAMBDA_SPP == pytest.approx(589.4e-9, rel=1e-3)
    assert vs.DELTA_D == pytest.approx(257e-9, rel=1e-2)
    assert vs.DELTA_M == pytest.approx(64.2e-9, rel=1e-2)
    # Bound-mode branch on both sides
    assert vs.KAPPA_D.real > 0 and vs.KAPPA_M.real > 0
    # All dimensionless wavenumbers fit inside the Fourier band (0.1, 40)
    for k in (vs.K_SPP.real, vs.KAPPA_D.real, vs.KAPPA_M.real):
        assert 0.1 < k * vs.LAMBDA0 < 40.0
    # Domain follows the new scales
    assert vs.Z_MIN == pytest.approx(-4.4 * vs.DELTA_M)
    assert vs.Z_MAX == pytest.approx(1.2 * vs.DELTA_D)
    # Metal preconditioning uses the largest |eps component| = |eps_t| = 4.005
    stiff = max(abs(eps_t), abs(eps_n))
    assert vs.METAL_CURL_WEIGHT_ADAM == pytest.approx(stiff**-vs.METAL_CURL_EXPONENT_ADAM)
    assert vs.METAL_CURL_WEIGHT_LBFGS == pytest.approx(stiff**-vs.METAL_CURL_EXPONENT_LBFGS)
    assert vs.METAL_DIV_WEIGHT == pytest.approx(stiff**-vs.METAL_DIV_EXPONENT)
    # Output paths are per-case
    assert vs.MODEL_PATH.name == "spp_validation_uniaxial.pth"
    assert vs.FIGURE_PREFIX == "uniaxial_"


def test_configure_case_rejects_unknown():
    with pytest.raises(ValueError):
        vs.configure_case("gold")
    assert vs.CASE == "silver"  # unchanged


def test_sampling_respects_strata_and_guard_band():
    torch.manual_seed(0)
    n = 4000
    coords = vs.sample_collocation_points(n, device=DEVICE)
    assert coords.shape == (n, 3)
    assert coords.requires_grad
    coords = coords.detach()
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    assert torch.all(x >= 0) and torch.all(x <= vs.X_MAX)
    assert torch.all(y >= 0) and torch.all(y <= vs.Y_MAX)
    assert torch.all(z >= vs.Z_MIN) and torch.all(z <= vs.Z_MAX)
    # Guard band around the interface is excluded
    assert torch.all(z.abs() >= vs.GUARD * (1 - 1e-6))
    # Stratification: exactly the requested metal fraction
    n_metal = int((z < 0).sum())
    assert n_metal == int(round(vs.METAL_FRACTION * n))
    # Metal-side points biased toward the interface (median << uniform median 50 nm)
    z_metal = z[z < 0].abs()
    assert float(z_metal.median()) < 25e-9
    # Air-side points biased toward the interface but with a uniform floor
    z_air = z[z > 0]
    assert float(z_air.mean()) < 0.47 * vs.Z_MAX  # uniform mean would be ~0.5 Z_MAX
    assert float(z_air.max()) > 0.8 * vs.Z_MAX  # floor still covers the far region
    # Custom guard width is honoured
    coords_v = vs.sample_collocation_points(500, guard=vs.VAL_GUARD, device=DEVICE)
    assert torch.all(coords_v[:, 2].abs() >= vs.VAL_GUARD * (1 - 1e-6))


def test_boundary_points_on_faces_and_outside_guard():
    torch.manual_seed(0)
    pts = vs.sample_boundary_points(60, device=DEVICE)
    assert pts.shape == (60, 3)
    on_face = (
        (pts[:, 0] == 0.0) | (pts[:, 0] == vs.X_MAX)
        | (pts[:, 1] == 0.0) | (pts[:, 1] == vs.Y_MAX)
        | (pts[:, 2] == vs.Z_MIN) | (pts[:, 2] == vs.Z_MAX)
    )
    assert torch.all(on_face)
    assert torch.all(pts[:, 2].abs() >= vs.GUARD * (1 - 1e-6))


def test_epsilon_tensor_correct_on_both_sides():
    coords = torch.tensor(
        [[0.0, 0.0, -10e-9], [100e-9, 10e-9, 10e-9], [0.0, 0.0, -1e-9]],
        dtype=torch.float32,
    )
    eps = vs.epsilon_tensor(coords)
    assert eps.shape == (3, 3, 3)
    assert eps.dtype == torch.complex64
    eye = torch.eye(3, dtype=torch.complex64)
    assert torch.allclose(eps[0], eye * vs.EPS_METAL_T, rtol=1e-6)
    assert torch.allclose(eps[2], eye * vs.EPS_METAL_T, rtol=1e-6)
    assert torch.allclose(eps[1], eye * complex(vs.EPS_DIEL), rtol=1e-6)
    # Off-diagonals are zero
    off = eps * (1 - eye.abs())
    assert torch.all(off.abs() == 0)
    # Same result in the dimensionless frame (only the sign of z matters)
    eps_hat = vs.epsilon_tensor(coords / vs.LAMBDA0)
    assert torch.allclose(eps, eps_hat)


def test_epsilon_tensor_uniaxial(uniaxial_case):
    """Below the interface the tensor is diag(eps_t, eps_t, eps_n)."""
    coords = torch.tensor([[0.0, 0.0, -10e-9], [0.0, 0.0, 10e-9]], dtype=torch.float32)
    eps = vs.epsilon_tensor(coords)
    expected_metal = torch.diag(
        torch.tensor(
            [vs.EPS_METAL_T, vs.EPS_METAL_T, vs.EPS_METAL_N], dtype=torch.complex64
        )
    )
    assert torch.allclose(eps[0], expected_metal, rtol=1e-6)
    assert torch.allclose(eps[1], torch.eye(3, dtype=torch.complex64), rtol=1e-6)
    # The (3,) diagonal spec fed to the interior losses matches the tensor
    diag = vs.metal_eps_diag()
    assert torch.allclose(torch.diag_embed(diag).to(torch.complex64), eps[0], rtol=1e-6)


class _ConstantMLP(nn.Module):
    """Stub MLP returning a fixed [N, 6, 2] pattern (all channels 1 + 0j)."""

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(coords.shape[0], 6, 2, dtype=coords.dtype)
        out[:, :, 0] = 1.0
        return out


@pytest.mark.parametrize("case", ["silver", "uniaxial"])
def test_displacement_adapter_divides_dz_by_normal_permittivity(case):
    """Channel 2 (D_z) is divided by the zz permittivity: eps_n below, eps_d above."""
    vs.configure_case(case)
    try:
        adapter = vs.DisplacementAdapter(
            _ConstantMLP(), eps_below=vs.EPS_METAL_N, eps_above=vs.EPS_DIEL
        )
        coords = torch.tensor([[0.0, 0.0, -0.1], [0.0, 0.0, 0.1]], dtype=torch.float32)
        out = adapter(coords)
        fields = torch.complex(out[..., 0], out[..., 1])
        # E_z below = D_z / eps_n (NOT eps_t); above = D_z / eps_d
        assert torch.allclose(
            fields[0, 2], torch.tensor(1.0 / vs.EPS_METAL_N, dtype=torch.complex64)
        )
        assert torch.allclose(
            fields[1, 2], torch.tensor(1.0 / complex(vs.EPS_DIEL), dtype=torch.complex64)
        )
        # All other channels pass through untouched
        untouched = torch.cat([fields[:, :2], fields[:, 3:]], dim=1)
        assert torch.allclose(untouched, torch.ones_like(untouched))
    finally:
        vs.configure_case("silver")


def test_displacement_adapter_float64():
    """The adapter has no dtype-frozen buffers: float64 coords give float64 out."""
    adapter = vs.DisplacementAdapter(
        _ConstantMLP(), eps_below=vs.EPS_METAL_N, eps_above=vs.EPS_DIEL
    )
    coords = torch.tensor([[0.0, 0.0, -0.1]], dtype=torch.float64)
    out = adapter(coords)
    assert out.dtype == torch.float64


def test_anchor_target_matches_analytical_through_scaling_round_trip():
    torch.manual_seed(0)
    boundary = vs.sample_boundary_points(60, device=DEVICE)
    target_hat = vs.analytical_fields_hat(boundary / vs.LAMBDA0)
    round_trip = target_hat * vs.FIELD_SCALE
    direct = vs.analytical_fields_si(boundary)
    # float32 coordinate rounding through the oscillatory phase limits the
    # round-trip to ~1e-4 absolute on O(10) fields; a wrong scale would be O(1).
    scale = float(direct.abs().max())
    assert torch.allclose(round_trip, direct, rtol=1e-3, atol=1e-3 * scale)
    # The scaled anchor is O(1): the network can represent it
    assert 0.1 < float(target_hat.abs().max()) < 3.0


def test_anchor_matches_analytical_uniaxial(uniaxial_case):
    """The uniaxial anchor equals analytical_spp_fields with the eps_t/eps_n split."""
    torch.manual_seed(0)
    boundary = vs.sample_boundary_points(60, device=DEVICE)
    anchor = vs.analytical_fields_si(boundary)
    E, H = analytical_spp_fields(
        boundary, vs.OMEGA, -4 + 0.2j, 3 + 0.05j, eps_dielectric=1.0, H0=vs.H0
    )
    direct = torch.cat(
        [torch.stack([E.real, E.imag], -1), torch.stack([H.real, H.imag], -1)], dim=1
    ).to(torch.float32)
    assert torch.allclose(anchor, direct, rtol=1e-6, atol=1e-6 * float(direct.abs().max()))
    # Scaled anchor is O(1) for the uniaxial mode too
    target_hat = vs.analytical_fields_hat(boundary / vs.LAMBDA0)
    assert 0.1 < float(target_hat.abs().max()) < 3.0


def test_analytical_fields_follow_input_dtype():
    """float64 coords (the L-BFGS phase) give float64 anchors."""
    coords = torch.tensor([[0.0, 0.0, 50e-9]], dtype=torch.float64)
    assert vs.analytical_fields_si(coords).dtype == torch.float64
    assert vs.analytical_fields_hat(coords / vs.LAMBDA0).dtype == torch.float64
    assert vs.analytical_fields_si(coords.to(torch.float32)).dtype == torch.float32


def test_create_network_output_shape():
    torch.manual_seed(0)
    network = vs.create_network(hidden_dims=(32, 32), fourier_modes=16, device=DEVICE)
    assert isinstance(network, nn.Module)
    coords = vs.sample_collocation_points(10, device=DEVICE)
    out = network(coords)
    assert out.shape == (10, 6, 2)
    assert out.dtype == torch.float32
    assert network.core(coords / vs.LAMBDA0).shape == (10, 6, 2)


@pytest.mark.parametrize("case", ["silver", "uniaxial"])
def test_validate_on_analytical_mode_is_near_perfect(case):
    """Pipeline self-check: the exact mode through the full validation path."""
    vs.configure_case(case)
    try:
        torch.manual_seed(0)
        metrics = vs.validate(vs.AnalyticalSPP(), n_points=1500, device=DEVICE)
        assert all(isinstance(v, float) and np.isfinite(v) for v in metrics.values())
        assert metrics["rel_l2_E"] < 1e-6
        assert metrics["rel_l2_H"] < 1e-6
        for side in ("air", "metal"):
            assert metrics[f"curl_E_residual_rel_{side}"] < 1e-5
            assert metrics[f"curl_H_residual_rel_{side}"] < 1e-5
        assert metrics["k_spp_rel_error"] < 1e-6
        assert metrics["kappa_d_fit_rel_error"] < 1e-6
        assert metrics["kappa_m_fit_rel_error"] < 1e-6
        assert metrics["decay_sign_correct_air"] == 1.0
        assert metrics["decay_sign_correct_metal"] == 1.0
        # The continuity residual at +-2 nm is NOT ~0 even for the exact mode: the
        # physical envelope differs across the 4 nm gap (exp(-kappa_m * 2 nm) ~ 0.92
        # for silver), so ~8% on H is the physical floor of this metric, not an error.
        assert metrics["continuity_E_rel"] < 0.15
        assert metrics["continuity_H_rel"] < 0.15
        assert vs.success_tier(metrics) == "stretch"
    finally:
        vs.configure_case("silver")


def test_validate_on_untrained_network_returns_finite_metrics():
    torch.manual_seed(0)
    network = vs.create_network(hidden_dims=(32, 32), fourier_modes=16, device=DEVICE)
    metrics = vs.validate(network, n_points=400, device=DEVICE)
    assert all(np.isfinite(v) for v in metrics.values())


def test_success_tier_classification():
    base = {
        "rel_l2_E": 1e-3, "rel_l2_H": 1e-3,
        "k_spp_rel_error": 1e-3,
        "kappa_d_fit_rel_error": 1e-2, "kappa_m_fit_rel_error": 1e-2,
        "decay_sign_correct_air": 1.0, "decay_sign_correct_metal": 1.0,
    }
    assert vs.success_tier(base) == "stretch"
    assert vs.success_tier({**base, "rel_l2_E": 0.02, "kappa_m_fit_rel_error": 0.05}) == "target"
    assert vs.success_tier({**base, "rel_l2_E": 0.3, "k_spp_rel_error": 0.05}) == "minimum"
    assert vs.success_tier({**base, "rel_l2_E": 0.9}) == "not met"
    assert vs.success_tier({**base, "rel_l2_E": 0.3, "decay_sign_correct_metal": 0.0}) == "not met"


def test_write_metrics_json_merges_cases(tmp_path):
    """Per-case entries coexist; a legacy flat file is migrated to 'silver'."""
    path = tmp_path / "metrics.json"
    # Legacy layout: a single flat object from the pre-case script
    path.write_text('{"metrics": {"rel_l2_E": 0.05}, "figures": {}}')
    vs.write_metrics_json(path, "uniaxial", {"rel_l2_E": 0.03}, {"rel_l2_E": 0.0}, {})
    import json

    data = json.loads(path.read_text())
    assert set(data) == {"silver", "uniaxial"}
    assert data["silver"]["metrics"]["rel_l2_E"] == 0.05
    assert data["uniaxial"]["metrics"]["rel_l2_E"] == 0.03
    # Overwriting one case keeps the other
    vs.write_metrics_json(path, "silver", {"rel_l2_E": 0.01}, {}, {})
    data = json.loads(path.read_text())
    assert data["silver"]["metrics"]["rel_l2_E"] == 0.01
    assert data["uniaxial"]["metrics"]["rel_l2_E"] == 0.03


def test_parse_args_case_selection():
    args = vs.parse_args([])
    assert args.case == "silver" and args.lbfgs_dtype == "float64"
    args = vs.parse_args(["--case", "uniaxial", "--lbfgs-dtype", "float32"])
    assert args.case == "uniaxial" and args.lbfgs_dtype == "float32"
    with pytest.raises(SystemExit):
        vs.parse_args(["--case", "gold"])


@pytest.mark.slow
def test_short_training_reduces_loss():
    """A --quick-style ~100-epoch run must decrease the loss."""
    torch.manual_seed(0)
    np.random.seed(0)
    network = vs.create_network(hidden_dims=(32, 32), fourier_modes=16, device=DEVICE)
    network, history = vs.train(
        network, n_epochs=100, n_points=192, device=DEVICE, log_every=1000, lbfgs_steps=1
    )
    assert set(history) == {"epoch", "total", "curl", "div", "continuity", "boundary", "lr"}
    assert len(history["total"]) == 101  # 100 Adam epochs + 1 L-BFGS step
    assert all(np.isfinite(history["total"]))
    assert min(history["total"][-10:]) < history["total"][0]
    # The float64 L-BFGS phase must hand back a float32 network
    assert all(p.dtype == torch.float32 for p in network.parameters())
