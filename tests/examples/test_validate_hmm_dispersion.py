"""Tests for examples/validate_hmm_dispersion.py (dispersive-ε SPP dispersion)."""

import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from examples import validate_hmm_dispersion as vh
from src.analytical import analytical_spp_fields
from src.constants import C0
from src.effective_medium import hmm_permittivities
from src.models import MaxwellCurlLoss, MaxwellDivergenceLoss

DEVICE = torch.device("cpu")
EDGES = (vh.OMEGA_MIN, vh.OMEGA_MAX)


# --------------------------------------------------------------------- band / material
def test_band_read_from_hmm_summary_not_hardcoded():
    """The band and design come from figures/hyperbolic/hmm_summary.json."""
    with open(vh.HMM_SUMMARY_PATH) as fh:
        summary = json.load(fh)
    band = summary["recommended_band"]
    assert vh.OMEGA_MIN == pytest.approx(band["omega"][0])
    assert vh.OMEGA_MAX == pytest.approx(band["omega"][1])
    assert vh.FILL_FRACTION == pytest.approx(summary["design"]["fill_fraction"])
    assert vh.EPS_D2 == pytest.approx(summary["design"]["eps_dielectric_layer"])
    assert vh.EPS_D == pytest.approx(summary["design"]["eps_superstrate"])
    # Sanity: the documented 450-885 nm window
    lam_nm = [2 * np.pi * C0 / w * 1e9 for w in (vh.OMEGA_MAX, vh.OMEGA_MIN)]
    assert lam_nm[0] == pytest.approx(450.0, rel=1e-3)
    assert lam_nm[1] == pytest.approx(885.4, rel=1e-3)


def test_permittivity_is_dispersive_and_matches_effective_medium():
    """ε(ω) is the Ag/silica stack, and it genuinely varies across the band."""
    for omega in EDGES:
        eps_t, eps_n = vh.hmm_eps(omega)
        ref_t, ref_n = hmm_permittivities(omega, vh.FILL_FRACTION, vh.EPS_D2, **vh.DRUDE)
        assert eps_t == pytest.approx(complex(ref_t))
        assert eps_n == pytest.approx(complex(ref_n))
        assert eps_t.imag > 0 and eps_n.imag > 0  # passive, exp(-iωt) convention
    eps_t_lo, eps_n_lo = vh.hmm_eps(vh.OMEGA_MIN)
    eps_t_hi, eps_n_hi = vh.hmm_eps(vh.OMEGA_MAX)
    # This is the whole experiment: the material is NOT the same at the two edges.
    assert abs(eps_t_lo - eps_t_hi) > 5.0
    assert eps_t_lo.real < -5.0 and -1.0 < eps_t_hi.real < 0.0  # type-II throughout
    assert eps_n_lo.real > 0 and eps_n_hi.real > 0


def test_hmm_eps_torch_matches_numpy_version():
    """The torch mirror used inside the adapter agrees to machine precision."""
    omegas = np.linspace(vh.OMEGA_MIN, vh.OMEGA_MAX, 17)
    eps_t, eps_n = vh.hmm_eps_torch(torch.tensor(omegas))
    ref_t, ref_n = hmm_permittivities(omegas, vh.FILL_FRACTION, vh.EPS_D2, **vh.DRUDE)
    assert np.allclose(eps_t.numpy(), ref_t, rtol=1e-14, atol=0)
    assert np.allclose(eps_n.numpy(), ref_n, rtol=1e-14, atol=0)


def test_dispersion_is_nonlinear():
    """Re k_spp(ω) departs materially from a straight line — the point of the run."""
    om = np.linspace(vh.OMEGA_MIN, vh.OMEGA_MAX, 201)
    k = np.array([vh.mode_constants(float(w))[0].real for w in om])
    residual = k - np.polyval(np.polyfit(om, k, 1), om)
    span = k.max() - k.min()
    assert 100 * np.abs(residual).max() / span > 10.0
    assert k.max() / k.min() > 2.0  # 2.59x


# --------------------------------------------------------------------- domain sizing
@pytest.mark.parametrize("omega", EDGES)
def test_per_omega_domain_matches_that_omegas_analytic_scales(omega):
    """The box is sized from the mode constants of *that* ω, in SI and scaled units."""
    k, kappa_d, kappa_m = vh.mode_constants(omega)
    x_max, y_max, z_min, z_max = vh.domain_si(omega)
    lambda_spp = 2 * np.pi / k.real
    assert x_max == pytest.approx(vh.X_PERIODS * lambda_spp)
    assert z_min == pytest.approx(-vh.Z_METAL_DEPTHS / kappa_m.real)
    assert z_max == pytest.approx(vh.Z_AIR_DEPTHS / kappa_d.real)
    assert y_max == pytest.approx(vh.Y_WAVELENGTHS * 2 * np.pi * C0 / omega)
    # the scaled box is the SI box times k0(ω)
    hat = vh.domain_hat(omega)
    assert hat == pytest.approx(tuple(v * vh.k0_of(omega) for v in (x_max, y_max, z_min, z_max)))


def test_domain_actually_differs_between_band_edges():
    """A single fixed box would be wrong: the two edges differ several-fold."""
    lo, hi = vh.domain_si(vh.OMEGA_MIN), vh.domain_si(vh.OMEGA_MAX)
    assert lo[0] / hi[0] > 2.0  # x extent (2 λ_spp) shrinks 2.6x
    assert lo[3] / hi[3] > 5.0  # air-side depth (1.2/κ_d) shrinks ~7x
    assert hi[2] / lo[2] > 2.0  # metal-side depth (3.5/κ_m) grows 2.4x


# --------------------------------------------------------------------- input scaling
def test_k0_input_scaling_keeps_coordinates_and_wavenumbers_order_one():
    """Scaling by the LOCAL k₀ makes the problem nearly frequency-invariant."""
    for omega in EDGES:
        x_max, y_max, z_min, z_max = vh.domain_hat(omega)
        # every scaled extent stays within a small O(1)-to-10 window at BOTH edges
        for v in (x_max, y_max, -z_min, z_max):
            assert 0.5 < v < 15.0
        k, kappa_d, kappa_m = vh.mode_constants(omega)
        k0 = vh.k0_of(omega)
        for c in (k.real / k0, kappa_d.real / k0, kappa_m.real / k0):
            # all scaled wavenumbers sit inside the Fourier band with headroom
            assert vh.FOURIER_K_RANGE[0] < c < vh.FOURIER_K_RANGE[1] / 2
    # The scaled effective index varies far less than k_spp itself: this is why
    # scaling by k₀ is honest bookkeeping while scaling by k_spp would leak.
    k_lo = vh.mode_constants(vh.OMEGA_MIN)[0].real
    k_hi = vh.mode_constants(vh.OMEGA_MAX)[0].real
    n_lo = k_lo / vh.k0_of(vh.OMEGA_MIN)
    n_hi = k_hi / vh.k0_of(vh.OMEGA_MAX)
    assert k_hi / k_lo > 2.5
    assert n_hi / n_lo < 1.4


def test_omega_hat_normalisation():
    assert vh.omega_hat(vh.OMEGA_MIN) == pytest.approx(-1.0)
    assert vh.omega_hat(vh.OMEGA_MID) == pytest.approx(0.0)
    assert vh.omega_hat(vh.OMEGA_MAX) == pytest.approx(1.0)
    for w in np.linspace(vh.OMEGA_MIN, vh.OMEGA_MAX, 5):
        assert vh.omega_from_hat(vh.omega_hat(w)) == pytest.approx(w)
        rt = vh.omega_from_hat_torch(torch.tensor([vh.omega_hat(w)]))
        assert float(rt[0]) == pytest.approx(w)


def test_validation_and_lbfgs_frequency_sets():
    """9 validation ω incl. both ends; the odd grid points held out from L-BFGS."""
    assert len(vh.VALIDATION_OMEGAS) == 9
    assert vh.VALIDATION_OMEGAS[0] == pytest.approx(vh.OMEGA_MIN)
    assert vh.VALIDATION_OMEGAS[-1] == pytest.approx(vh.OMEGA_MAX)
    lbfgs = {round(float(w), 3) for w in vh.LBFGS_OMEGAS}
    held_out = [w for w in vh.VALIDATION_OMEGAS if round(float(w), 3) not in lbfgs]
    assert len(vh.LBFGS_OMEGAS) == 13 and len(held_out) == 4
    # The property that matters is the worst-case distance from an arbitrary ω to
    # the nearest refinement node: the 5-node set left it at Δ, and the error
    # bulged to 9e-2 midway between the two bluest nodes. Adding the half-way
    # points halves it to Δ/2. The gaps are NOT uniform — each held-out
    # validation frequency deliberately sits alone at the centre of a Δ-wide gap.
    nodes = np.asarray(vh.LBFGS_OMEGAS, dtype=float)
    delta = (vh.OMEGA_MAX - vh.OMEGA_MIN) / 8.0
    gaps = np.diff(nodes)
    assert gaps.max() == pytest.approx(delta)
    probe = np.linspace(vh.OMEGA_MIN, vh.OMEGA_MAX, 501)
    worst = np.abs(probe[:, None] - nodes[None, :]).min(axis=1).max()
    assert worst <= delta / 2 * (1 + 1e-9)
    # ...and each held-out frequency is exactly mid-gap, so none is ever refined.
    for w in held_out:
        assert np.abs(nodes - float(w)).min() == pytest.approx(delta / 2)


def test_band_fraction_narrowing_and_restore():
    """--band-fraction shrinks the band about its midpoint (the documented fallback)."""
    full_lo, full_hi = vh.OMEGA_MIN, vh.OMEGA_MAX
    try:
        vh.set_band_fraction(0.6)
        assert vh.BAND_FRACTION == pytest.approx(0.6)
        assert vh.OMEGA_MAX - vh.OMEGA_MIN == pytest.approx(0.6 * (full_hi - full_lo))
        assert 0.5 * (vh.OMEGA_MIN + vh.OMEGA_MAX) == pytest.approx(0.5 * (full_lo + full_hi))
        assert vh.VALIDATION_OMEGAS[0] == pytest.approx(vh.OMEGA_MIN)
        assert vh.omega_hat(vh.OMEGA_MIN) == pytest.approx(-1.0)
    finally:
        vh.set_band_fraction(1.0)
    assert vh.OMEGA_MIN == pytest.approx(full_lo) and vh.OMEGA_MAX == pytest.approx(full_hi)


# --------------------------------------------------------------------- permittivity rows
def test_eps_rows_use_the_row_frequency():
    """The interior ε tensor differs between two rows at different ω."""
    om = torch.tensor([vh.OMEGA_MIN, vh.OMEGA_MAX], dtype=torch.float64)
    eps_metal = vh.eps_tensor_rows(om, metal=True)
    eps_air = vh.eps_tensor_rows(om, metal=False)
    assert eps_metal.shape == (2, 3, 3) and eps_metal.dtype == torch.complex128
    # diagonal, and (ε_t, ε_t, ε_n) of that row's own frequency
    for i, omega in enumerate((vh.OMEGA_MIN, vh.OMEGA_MAX)):
        eps_t, eps_n = vh.hmm_eps(float(omega))
        expected = torch.diag(torch.tensor([eps_t, eps_t, eps_n], dtype=torch.complex128))
        assert torch.allclose(eps_metal[i], expected)
    # THE point of the experiment: two frequencies carry different material
    assert not torch.allclose(eps_metal[0], eps_metal[1])
    assert abs(complex(eps_metal[0, 0, 0]) - complex(eps_metal[1, 0, 0])) > 5.0
    # air rows are ε_d at every frequency
    assert torch.allclose(eps_air[0], eps_air[1])
    assert torch.allclose(eps_air[0], torch.eye(3, dtype=torch.complex128) * complex(vh.EPS_D))


def test_training_batch_rows_carry_their_own_material():
    torch.manual_seed(0)
    omegas = [vh.OMEGA_MIN, vh.OMEGA_MAX]
    batch = vh.sample_training_batch(64, 24, 8, omegas, device=DEVICE)
    n_air, n_metal = batch["coords_air"].shape[0], batch["coords_metal"].shape[0]
    assert n_air + n_metal == 64
    assert batch["coords_air"].shape[1] == 3  # ω̂ rides along separately
    assert batch["eps_metal"].shape == (n_metal, 3, 3)
    assert batch["eps_scale_metal"].shape == (n_metal,)
    # each metal row's ε matches ε(ω of that row)
    for i in (0, n_metal - 1):
        omega = float(batch["omega_metal"][i])
        eps_t, eps_n = vh.hmm_eps(omega)
        expected = torch.diag(torch.tensor([eps_t, eps_t, eps_n], dtype=torch.complex128))
        assert torch.allclose(batch["eps_metal"][i], expected)
        assert float(batch["eps_scale_metal"][i]) == pytest.approx(
            max(abs(eps_t), abs(eps_n))
        )
    # both frequencies present, and their ε differ
    w_vals = {round(float(v), 6) for v in torch.cat([batch["w_air"], batch["w_metal"]])}
    assert w_vals == {round(vh.omega_hat(o), 6) for o in omegas}
    assert batch["boundary"].shape[0] == batch["target"].shape[0] == batch["w_bc"].shape[0]


# --------------------------------------------------------------------- network / adapter
def test_network_shapes_and_omega_dependent_ez_jump():
    """The adapter divisor is ε_n(ω) — different at the two band edges."""
    torch.manual_seed(0)
    network = vh.create_network(hidden_dims=(32, 32), fourier_modes=16, device=DEVICE)
    coords = torch.rand(10, 3) * 1e-7
    out = network(coords, vh.OMEGA_MID)
    assert out.shape == (10, 6, 2) and out.dtype == torch.float32
    assert network.at_omega(vh.OMEGA_MAX)(coords).shape == (10, 6, 2)

    eps = 1e-13
    pts = torch.tensor([[5e-8, 2e-8, -eps], [5e-8, 2e-8, +eps]], dtype=torch.float32)
    ratios = []
    for omega in EDGES:
        with torch.no_grad():
            E, _ = vh.to_complex(network(pts, omega))
        ratio = E[1, 2] / E[0, 2]  # E_z(above) / E_z(below) = ε_n(ω) / ε_d
        expected = vh.hmm_eps(omega)[1] / complex(vh.EPS_D)
        assert torch.allclose(
            ratio, torch.tensor(expected, dtype=ratio.dtype), rtol=2e-3
        )
        ratios.append(complex(ratio))
    # ω-dependent divisor: the jump is NOT the same at the two edges
    assert abs(ratios[0] - ratios[1]) > 0.3


class _AnalyticalHatCore(nn.Module):
    """4-column core returning the exact mode at a fixed ω (ignores the ω̂ column)."""

    def __init__(self, omega: float):
        super().__init__()
        self.omega = float(omega)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return vh.analytical_fields_hat(coords[:, :3], self.omega)


# --------------------------------------------------------------------- interior losses
@pytest.mark.parametrize("omega", EDGES)
@pytest.mark.parametrize("metal", [False, True])
def test_scaled_maxwell_is_frequency_free_on_the_exact_mode(omega, metal):
    """
    In the k₀-scaled frame the exact mode is a zero of the residual at EVERY ω
    with no frequency prefactor: ω enters only through ε(ω).
    """
    torch.manual_seed(0)
    core = _AnalyticalHatCore(omega)
    n = 64
    coords = torch.rand(n, 3, dtype=torch.float64) * 0.5
    coords[:, 2] = (coords[:, 2] * 0.25 + 0.05) * (-1.0 if metal else 1.0)
    w_col = torch.full((n, 1), vh.omega_hat(omega), dtype=torch.float64)
    om_col = torch.full((n,), float(omega), dtype=torch.float64)
    eps = vh.eps_tensor_rows(om_col, metal=metal)

    c1 = coords.clone().requires_grad_(True)
    assert float(vh.curl_loss_weighted(core, c1, w_col, eps).detach()) < 1e-12
    c2 = coords.clone().requires_grad_(True)
    assert float(vh.divergence_loss_weighted(core, c2, w_col, eps).detach()) < 1e-12

    if metal:
        # The wrong ε (the other band edge's) is NOT a zero — ε(ω) is load-bearing.
        other = vh.OMEGA_MAX if omega == vh.OMEGA_MIN else vh.OMEGA_MIN
        eps_wrong = vh.eps_tensor_rows(torch.full_like(om_col, float(other)), metal=True)
        c3 = coords.clone().requires_grad_(True)
        assert float(vh.curl_loss_weighted(core, c3, w_col, eps_wrong).detach()) > 1e-4


@pytest.mark.parametrize("metal", [False, True])
def test_weighted_losses_match_library_losses_for_uniform_weights(metal):
    """``curl_loss_weighted``/``divergence_loss_weighted`` reduce to the library losses."""
    torch.manual_seed(1)
    omega = vh.OMEGA_MID
    core = _AnalyticalHatCore(omega)
    n = 48
    coords = torch.rand(n, 3, dtype=torch.float64) * 0.4 + 0.05
    coords[:, 2] = coords[:, 2] * (-1.0 if metal else 1.0)
    w_col = torch.full((n, 1), vh.omega_hat(omega), dtype=torch.float64)
    eps = vh.eps_tensor_rows(torch.full((n,), float(omega), dtype=torch.float64), metal=metal)
    net3 = vh.OmegaColumnNet(core, w_col)

    c1 = coords.clone().requires_grad_(True)
    mine = float(vh.curl_loss_weighted(core, c1, w_col, eps).detach())
    c2 = coords.clone().requires_grad_(True)
    ref = float(
        MaxwellCurlLoss(frequency=1.0, mu0=1.0, eps0=1.0)
        .compute(network=net3, coords=c2, epsilon=eps, mu_r=1.0)
        .detach()
    )
    assert mine == pytest.approx(ref, rel=1e-12, abs=1e-30)

    c3 = coords.clone().requires_grad_(True)
    mine_d = float(vh.divergence_loss_weighted(core, c3, w_col, eps).detach())
    c4 = coords.clone().requires_grad_(True)
    ref_d = float(
        MaxwellDivergenceLoss().compute(network=net3, coords=c4, epsilon=eps).detach()
    )
    assert mine_d == pytest.approx(ref_d, rel=1e-12, abs=1e-30)

    # A non-uniform weight actually changes the value.
    weights = torch.linspace(0.1, 2.0, n, dtype=torch.float64)
    c5 = coords.clone().requires_grad_(True)
    assert float(vh.curl_loss_weighted(core, c5, w_col, eps, weights).detach()) != mine


# --------------------------------------------------------------------- sampling / anchor
def test_sampling_respects_the_per_omega_box_and_guard():
    torch.manual_seed(0)
    n = 4000
    for omega in EDGES:
        x_max, y_max, z_min, z_max = vh.domain_hat(omega)
        c = vh.sample_collocation_hat(n, omega, device=DEVICE)
        assert c.shape == (n, 3)
        assert torch.all(c[:, 0] >= 0) and torch.all(c[:, 0] <= x_max)
        assert torch.all(c[:, 1] >= 0) and torch.all(c[:, 1] <= y_max)
        assert torch.all(c[:, 2] >= z_min) and torch.all(c[:, 2] <= z_max)
        assert torch.all(c[:, 2].abs() >= vh.GUARD_HAT * (1 - 1e-6))
        assert int((c[:, 2] < 0).sum()) == int(round(vh.METAL_FRACTION * n))
        si = vh.sample_collocation_si(n, omega, device=DEVICE)
        assert si.requires_grad
        assert float(si.detach()[:, 0].max()) <= vh.domain_si(omega)[0]
    # Air strata follow each ω's own decay length: the blue edge is shallower
    # in SI (δ_d = 78 nm vs 541 nm).
    torch.manual_seed(1)
    z_lo = vh.sample_collocation_si(n, vh.OMEGA_MIN, device=DEVICE).detach()[:, 2]
    torch.manual_seed(1)
    z_hi = vh.sample_collocation_si(n, vh.OMEGA_MAX, device=DEVICE).detach()[:, 2]
    assert float(z_hi[z_hi > 0].median()) < float(z_lo[z_lo > 0].median())


@pytest.mark.parametrize("omega", EDGES)
def test_anchor_matches_analytical_at_band_edges(omega):
    """The scaled anchor round-trips to analytical_spp_fields with ε(ω)."""
    torch.manual_seed(0)
    boundary_hat = vh.sample_boundary_hat(60, omega, device=DEVICE)
    target_hat = vh.analytical_fields_hat(boundary_hat, omega)
    round_trip = target_hat * vh.FIELD_SCALE

    eps_t, eps_n = vh.hmm_eps(omega)
    E, H = analytical_spp_fields(
        boundary_hat / vh.k0_of(omega), omega, eps_t, eps_n,
        eps_dielectric=vh.EPS_D, H0=vh.H0,
    )
    direct = torch.cat(
        [torch.stack([E.real, E.imag], -1), torch.stack([H.real, H.imag], -1)], dim=1
    ).to(torch.float32)
    scale = float(direct.abs().max())
    assert torch.allclose(round_trip, direct, rtol=1e-3, atol=1e-3 * scale)
    # k₀ scaling keeps the anchor O(1) at both edges (that is why it works)
    assert 0.3 < float(target_hat.abs().max()) < 3.0


# --------------------------------------------------------------------- validation pipeline
def test_validate_on_analytical_network_is_near_perfect():
    """Pipeline self-check at every probe frequency: measurement, not model, error."""
    torch.manual_seed(0)
    per_freq = vh.validate_band(
        vh.AnalyticalHMMSPP(), vh.VALIDATION_OMEGAS, n_points=500, device=DEVICE
    )
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
    summary = vh.summarise(per_freq)
    assert summary["success_tier"] == "stretch"
    assert summary["bound_mode_everywhere"] == 1.0
    # ...and the nonlinearity bookkeeping is self-consistent on the exact mode
    assert summary["rms_residual_about_analytical_curve_per_m"] < 1.0
    assert summary["rms_residual_about_straight_line_per_m"] > 1e5
    assert summary["curvature_capture_ratio"] > 1e3
    assert summary["nonlinearity_percent_origin_line"] > 20.0
    assert summary["nonlinearity_percent_lsq_line"] > 10.0


def test_validate_on_untrained_network_finite():
    torch.manual_seed(0)
    network = vh.create_network(hidden_dims=(32, 32), fourier_modes=16, device=DEVICE)
    m = vh.validate_at_omega(network, vh.OMEGA_MID, n_points=300, device=DEVICE)
    assert all(np.isfinite(v) for v in m.values())


def test_success_tier_classification():
    base = {"worst_rel_l2": 0.01, "worst_k_spp_rel_error": 1e-3, "bound_mode_everywhere": 1.0}
    assert vh.success_tier(base) == "stretch"
    assert vh.success_tier({**base, "worst_rel_l2": 0.05}) == "target"
    assert vh.success_tier({**base, "worst_rel_l2": 0.3}) == "minimum"
    assert vh.success_tier(
        {**base, "worst_rel_l2": 0.05, "worst_k_spp_rel_error": 0.05}
    ) == "minimum"
    assert vh.success_tier({**base, "worst_rel_l2": 0.9}) == "not met"
    assert vh.success_tier({**base, "bound_mode_everywhere": 0.0}) == "not met"


def test_parse_args_defaults():
    args = vh.parse_args([])
    assert args.epochs == vh.N_EPOCHS and args.lbfgs_dtype == "float64"
    assert args.lbfgs_steps == vh.LBFGS_STEPS and args.band_fraction == 1.0
    args = vh.parse_args(["--quick", "--lbfgs-dtype", "float32", "--band-fraction", "0.65"])
    assert args.quick and args.lbfgs_dtype == "float32" and args.band_fraction == 0.65


@pytest.mark.slow
def test_short_training_reduces_loss():
    """A --quick-style short run must decrease the loss (multi-ω, per-ω-box path)."""
    torch.manual_seed(0)
    np.random.seed(0)
    network = vh.create_network(hidden_dims=(32, 32), fourier_modes=16, device=DEVICE)
    network, history = vh.train(
        network, n_epochs=100, n_points=192, device=DEVICE, log_every=1000, lbfgs_steps=1
    )
    assert set(history) == {"epoch", "total", "curl", "div", "continuity", "boundary", "lr"}
    assert len(history["total"]) == 101  # 100 Adam epochs + 1 L-BFGS step
    assert all(np.isfinite(history["total"]))
    assert min(history["total"][-10:]) < history["total"][0]
    # The float64 L-BFGS phase must hand back a float32 network
    assert all(p.dtype == torch.float32 for p in network.parameters())
