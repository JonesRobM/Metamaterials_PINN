"""Tests for examples/validate_multilayer.py — the PINN on the real layered stack."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from examples import validate_multilayer as vm
from src.transfer_matrix import layer_boundaries, mode_field_profile, permittivity_profile

DEVICE = torch.device("cpu")


@pytest.fixture(scope="module")
def struct() -> vm.Structure:
    return vm.STRUCT


# --------------------------------------------------------------------- geometry
def test_constants_and_frame():
    assert vm.LAMBDA0 == pytest.approx(633e-9)
    assert vm.K0 == pytest.approx(vm.OMEGA / vm.C0)
    # k₀-scaled frame: Maxwell's curl equations are frequency-free, ω̂ = 1.
    assert vm.OMEGA_HAT == 1.0
    assert vm.PERIOD == pytest.approx(30e-9)
    assert vm.N_PERIODS == 6


def test_structure_is_the_metal_terminated_emt_validity_stack(struct):
    """Same construction as ``examples/emt_validity.multilayer_stack('metal')``."""
    from examples.emt_validity import multilayer_stack

    eps_layers, thicknesses = multilayer_stack(
        struct.period, struct.n_periods, vm.FILL_FRACTION, vm.EPS_AG, vm.EPS_D2, "metal"
    )
    assert struct.eps_layers == eps_layers
    assert struct.thicknesses == thicknesses
    # 6 bilayers => 12 finite layers, 13 interfaces, plus 2 semi-infinite media.
    assert len(struct.eps_layers) == 14
    assert struct.boundaries.size == 13
    # Silver against air at the top, silver substrate at the bottom.
    assert struct.eps_layers[-1] == pytest.approx(vm.EPS_AIR)
    assert struct.eps_layers[-2] == vm.EPS_AG
    assert struct.eps_layers[0] == vm.EPS_AG
    # Topmost interface pinned at z = 0.
    assert struct.boundaries[-1] == pytest.approx(0.0, abs=1e-18)
    assert struct.z0 == pytest.approx(-struct.n_periods * struct.period)


def test_layer_thicknesses_follow_the_fill_fraction(struct):
    d = np.asarray(struct.thicknesses)
    assert np.isclose(d, vm.FILL_FRACTION * struct.period).sum() == struct.n_periods
    assert np.isclose(d, (1 - vm.FILL_FRACTION) * struct.period).sum() == struct.n_periods


def test_tmm_mode_is_bound_and_differs_from_emt(struct):
    assert struct.k_tmm.real > vm.K0  # beyond the air light line => bound
    assert struct.k_tmm.imag > 0  # lossy, exp(-iωt)
    assert struct.kappa_d.real > 0
    info = struct.summary()
    # The gap this experiment exists to measure (emt_validity: ~2.3% / ~25%).
    assert info["emt_error_re"] == pytest.approx(0.023, abs=0.005)
    # (normalised by the *truth*; emt_validity quotes 25% normalised by the EMT value)
    assert info["emt_error_im"] == pytest.approx(0.34, abs=0.05)


def test_domain_covers_the_stack_the_substrate_pad_and_the_air_tail(struct):
    assert struct.z_min == pytest.approx(struct.z0 - vm.SUBSTRATE_PAD)
    assert struct.z_max == pytest.approx(vm.Z_AIR_DEPTHS / struct.kappa_d.real)
    assert struct.x_max == pytest.approx(vm.X_PERIODS * 2 * np.pi / struct.k_tmm.real)


def test_material_group_labels_every_medium(struct):
    mids = 0.5 * (struct.boundaries[:-1] + struct.boundaries[1:])
    groups = struct.material_group(mids)
    assert set(np.unique(groups)) == {0, 1}  # alternating Ag / silica inside the stack
    assert struct.material_group(np.array([struct.z_min]))[0] == 0  # Ag substrate
    assert struct.material_group(np.array([100e-9]))[0] == 2  # air


# ------------------------------------------------------- piecewise permittivity
def test_adapter_epsilon_matches_the_stack_profile_layer_by_layer(struct):
    """The torch-side ε(z) lookup must agree with the numpy transfer-matrix one."""
    adapter = vm.create_network(struct, hidden_dims=(8,), device=DEVICE).core
    z = np.linspace(struct.z_min, struct.z_max, 4001)
    want = permittivity_profile(z, struct.eps_layers, struct.thicknesses, z0=struct.z0)
    got = adapter.eps_zz(
        torch.as_tensor(z * vm.K0, dtype=torch.float32), torch.complex128
    ).numpy()
    assert np.abs(got - want).max() < 1e-6 * np.abs(want).max()


def test_adapter_epsilon_exactly_on_an_interface_takes_the_medium_above(struct):
    """Right-continuous, matching ``src.transfer_matrix.layer_index_at``."""
    adapter = vm.create_network(struct, hidden_dims=(8,), device=DEVICE).core
    bounds_hat = torch.as_tensor(struct.boundaries_hat, dtype=torch.float64)
    on = adapter.eps_zz(bounds_hat, torch.complex128).numpy()
    above = adapter.eps_zz(bounds_hat + 1e-9, torch.complex128).numpy()
    below = adapter.eps_zz(bounds_hat - 1e-9, torch.complex128).numpy()
    assert np.array_equal(on, above)
    assert not np.any(on == below)
    assert complex(on[-1]) == complex(vm.EPS_AIR)  # the stack/air interface reads as air


def test_adapter_divides_dz_by_the_correct_layers_epsilon(struct):
    r"""
    The methodological claim, tested: the network's channel 2 is ``D̂_z`` and the
    returned ``Ê_z`` is that divided by **the ε of the layer the point is in**.
    """
    torch.manual_seed(0)
    network = vm.create_network(struct, hidden_dims=(16, 16), device=DEVICE)
    adapter = network.core
    mids = 0.5 * (struct.boundaries[:-1] + struct.boundaries[1:])
    z = np.concatenate([[struct.z_min * 0.9], mids, [200e-9]])
    coords = torch.tensor(
        np.stack([np.zeros_like(z), np.zeros_like(z), z * vm.K0], axis=1), dtype=torch.float32
    )
    with torch.no_grad():
        raw = adapter.mlp(coords)  # the continuous D̂_z lives on channel 2
        out = adapter(coords)
    d_z = torch.complex(raw[:, 2, 0], raw[:, 2, 1]).to(torch.complex128)
    e_z = torch.complex(out[:, 2, 0], out[:, 2, 1]).to(torch.complex128)
    eps = torch.as_tensor(
        permittivity_profile(z, struct.eps_layers, struct.thicknesses, z0=struct.z0)
    )
    assert torch.allclose(e_z * eps, d_z, rtol=1e-5, atol=1e-8)
    # Every other channel is passed straight through.
    for channel in (0, 1, 3, 4, 5):
        assert torch.allclose(out[:, channel], raw[:, channel])


def test_adapter_makes_ez_jump_by_the_permittivity_ratio(struct):
    """A *continuous* MLP put through the adapter already has the right jumps."""
    torch.manual_seed(0)
    adapter = vm.create_network(struct, hidden_dims=(16, 16), device=DEVICE).core
    bounds_hat = torch.as_tensor(struct.boundaries_hat, dtype=torch.float32)
    delta = 1e-6  # scaled units, ~0.1 pm
    zeros = torch.zeros_like(bounds_hat)
    with torch.no_grad():
        lo = adapter(torch.stack([zeros, zeros, bounds_hat - delta], dim=1))
        hi = adapter(torch.stack([zeros, zeros, bounds_hat + delta], dim=1))
    ez_lo = torch.complex(lo[:, 2, 0], lo[:, 2, 1]).to(torch.complex128)
    ez_hi = torch.complex(hi[:, 2, 0], hi[:, 2, 1]).to(torch.complex128)
    eps_lo = adapter.eps_zz(bounds_hat - delta, torch.complex128)
    eps_hi = adapter.eps_zz(bounds_hat + delta, torch.complex128)
    assert torch.allclose(ez_hi / ez_lo, eps_lo / eps_hi, rtol=1e-3)


# ------------------------------------------------------------- Fourier features
def test_fourier_band_reaches_past_the_layer_wavenumber(struct):
    r"""The band along ``z`` must cover ``2π/a`` with harmonics to spare."""
    layer_k = 2 * np.pi / (struct.period * vm.K0)
    assert struct.kz_band_hat == pytest.approx(vm.FOURIER_KZ_HARMONICS * layer_k)
    assert struct.kz_band_hat > layer_k

    features = vm.LayeredFourierFeatures(struct.kz_band_hat)
    kz = features.k_vectors[:, 2].abs()
    assert float(kz.max()) == pytest.approx(struct.kz_band_hat, rel=1e-5)
    # ... and it is genuinely populated around 2π/a, not just at the far end.
    assert int(((kz > 0.8 * layer_k) & (kz < 1.25 * layer_k)).sum()) >= 3


def test_fourier_band_is_anisotropic(struct):
    """High wavenumbers along z only; x and y stay near the mode's own k_spp."""
    features = vm.LayeredFourierFeatures(struct.kz_band_hat)
    k = features.k_vectors
    assert float(k[:, 0].abs().max()) <= vm.FOURIER_KX_MAX + 1e-5
    assert float(k[:, 1].abs().max()) <= vm.FOURIER_KX_MAX + 1e-5
    assert float(k[:, 2].abs().max()) > 5 * vm.FOURIER_KX_MAX
    # The propagation wavenumber the field actually has is inside the x band.
    assert struct.k_tmm.real / vm.K0 < vm.FOURIER_KX_MAX


def test_fourier_output_dim_and_shape(struct):
    features = vm.LayeredFourierFeatures(struct.kz_band_hat)
    n = vm.FOURIER_Z_MODES + vm.FOURIER_X_MODES + vm.FOURIER_MIX_MODES
    assert features.num_modes == n
    assert features.output_dim == 2 * n + 3
    out = features(torch.zeros(5, 3))
    assert out.shape == (5, features.output_dim)


def test_fourier_modes_are_reproducible_from_a_generator(struct):
    a = vm.LayeredFourierFeatures(struct.kz_band_hat, generator=torch.Generator().manual_seed(3))
    b = vm.LayeredFourierFeatures(struct.kz_band_hat, generator=torch.Generator().manual_seed(3))
    assert torch.equal(a.k_vectors, b.k_vectors)


# ------------------------------------------------------------------- sampling
def test_collocation_stays_in_the_domain_and_clears_every_interface(struct):
    torch.manual_seed(0)
    coords = vm.sample_collocation_hat(4000, struct, device=DEVICE)
    x_max, y_max, z_min, z_max = struct.domain_hat
    assert coords[:, 0].min() >= 0 and coords[:, 0].max() <= x_max
    assert coords[:, 1].min() >= 0 and coords[:, 1].max() <= y_max
    assert coords[:, 2].min() >= z_min - 1e-6
    assert coords[:, 2].max() <= z_max + 1e-6
    guard_hat = vm.GUARD * vm.K0
    distance = (coords[:, 2].unsqueeze(1) - torch.as_tensor(
        struct.boundaries_hat, dtype=torch.float32
    ).unsqueeze(0)).abs().min(dim=1).values
    assert float(distance.min()) >= guard_hat * (1 - 1e-5)


def test_sampling_resolves_individual_layers(struct):
    """The resolution claim in the module docstring, as a measurement."""
    torch.manual_seed(0)
    report = vm.points_per_layer(2048, struct, trials=4)
    assert len(report["per_layer"]) == struct.boundaries.size - 1
    # Several points per layer, even in the thinnest and deepest one.
    assert report["min_per_layer"] > 10
    assert report["mean_per_metal_layer"] > 20
    assert report["mean_per_dielectric_layer"] > report["mean_per_metal_layer"]


def test_boundary_points_lie_on_the_six_faces(struct):
    torch.manual_seed(0)
    pts = vm.sample_boundary_hat(600, struct, device=DEVICE)
    x_max, y_max, z_min, z_max = struct.domain_hat
    on_face = (
        torch.isclose(pts[:, 0], torch.tensor(0.0))
        | torch.isclose(pts[:, 0], torch.tensor(x_max))
        | torch.isclose(pts[:, 1], torch.tensor(0.0))
        | torch.isclose(pts[:, 1], torch.tensor(y_max))
        | torch.isclose(pts[:, 2], torch.tensor(z_min))
        | torch.isclose(pts[:, 2], torch.tensor(z_max))
    )
    assert bool(on_face.all())


def test_interface_sampling_covers_all_interfaces(struct):
    torch.manual_seed(0)
    coords, normals = vm.sample_interface_hat(4000, struct, device=DEVICE)
    found = torch.unique(coords[:, 2])
    assert found.numel() == struct.boundaries.size
    assert torch.allclose(normals, torch.tensor([0.0, 0.0, 1.0]).expand_as(normals))
    # The continuity offset must stay inside the thinnest layer on both sides.
    assert vm.CONTINUITY_OFFSET < 0.5 * vm.FILL_FRACTION * struct.period


# --------------------------------------------------------------- anchor / truth
def test_tmm_fields_reproduce_the_transfer_matrix_profile(struct):
    """``tmm_fields_si`` is ``mode_field_profile`` × ``exp(i k_x x)``, nothing else."""
    z = np.linspace(struct.z_min, struct.z_max, 97)
    x = np.linspace(0.0, struct.x_max, 97)
    coords = torch.tensor(np.stack([x, np.zeros_like(x), z], axis=1), dtype=torch.float64)
    fields = vm.tmm_fields_si(coords, struct)
    prof = mode_field_profile(
        struct.k_tmm, vm.K0, struct.eps_layers, struct.thicknesses, z,
        z0=struct.z0, H0=vm.H0, h0_at=0.0, omega=vm.OMEGA,
    )
    phase = np.exp(1j * struct.k_tmm * x)
    got = torch.complex(fields[..., 0], fields[..., 1]).numpy()
    assert np.abs(got[:, 4] - prof.H_y * phase).max() < 1e-12 * np.abs(prof.H_y).max()
    assert np.abs(got[:, 0] - prof.E_x * phase).max() < 1e-12 * np.abs(prof.E_x).max()
    assert np.abs(got[:, 2] - prof.E_z * phase).max() < 1e-12 * np.abs(prof.E_z).max()
    # TM: only E_x, E_z, H_y are non-zero.
    assert np.abs(got[:, [1, 3, 5]]).max() == 0.0


def test_tmm_field_is_normalised_at_the_top_interface(struct):
    coords = torch.zeros(1, 3, dtype=torch.float64)
    fields = vm.tmm_fields_si(coords, struct)
    h_y = complex(fields[0, 4, 0], fields[0, 4, 1])
    assert h_y == pytest.approx(complex(vm.H0), abs=1e-12)


def test_anchor_target_matches_the_tmm_profile_through_the_scaling_round_trip(struct):
    """The soft-Dirichlet target is the TMM mode in the core's own scaled units."""
    torch.manual_seed(0)
    boundary = vm.sample_boundary_hat(240, struct, device=DEVICE)
    target = vm.tmm_fields_hat(boundary, struct)
    si = vm.tmm_fields_si(boundary / vm.K0, struct)
    scale = vm.FIELD_SCALE.to(dtype=si.dtype)
    assert torch.allclose(target * scale, si, rtol=1e-5, atol=1e-12)
    # Scaled fields are O(1) — the point of the non-dimensionalisation.
    assert 1e-4 < float(target.abs().max()) < 1e2


def test_emt_reference_is_the_homogenised_uniaxial_mode(struct):
    """The EMT baseline is the analytical uniaxial SPP with (ε_t, ε_n)."""
    from src.analytical import analytical_spp_fields

    z = np.linspace(-100e-9, 200e-9, 51)
    coords = torch.tensor(
        np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1), dtype=torch.float64
    )
    E, H = analytical_spp_fields(
        coords, vm.OMEGA, struct.eps_t, struct.eps_n, eps_dielectric=vm.EPS_AIR, H0=vm.H0
    )
    got = vm.emt_fields_si(coords, struct)
    assert torch.allclose(got[:, 4, 0], H[:, 1].real)
    assert torch.allclose(got[:, 2, 0], E[:, 2].real)


def test_emt_field_is_a_worse_fit_to_the_truth_than_a_perfect_solver(struct):
    """The baseline the field metric is measured against is genuinely nonzero."""
    z = np.linspace(struct.z0, struct.z_max, 500)
    coords = torch.tensor(
        np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1), dtype=torch.float64
    )
    ref = vm.tmm_fields_si(coords, struct)
    emt = vm.emt_fields_si(coords, struct)
    assert vm._relative_l2(emt, ref) > 0.1


# ------------------------------------------------------------------ validation
def test_validate_on_the_tmm_field_is_near_perfect(struct):
    """The pipeline self-check: feeding it the truth must return the truth."""
    torch.manual_seed(0)
    metrics = vm.validate(vm.TMMFieldModule(struct), struct, n_points=3000, device=DEVICE)
    assert metrics["rel_l2_total"] < 1e-12
    assert metrics["rel_l2_E"] < 1e-12
    assert metrics["rel_l2_H"] < 1e-12
    assert metrics["k_spp_re_rel_error"] < 1e-6
    assert metrics["k_spp_im_rel_error"] < 1e-4
    assert metrics["kappa_d_rel_error"] < 1e-6
    assert metrics["kappa_stack_rel_error"] < 1e-6
    assert metrics["ez_layer_contrast_rel_error"] < 1e-6
    assert metrics["bound_in_air"] > 0 and metrics["bound_in_stack"] > 0
    # Not zero: the offset formulation compares the field at z ± 0.5 nm, and the
    # exact mode varies by ~2 κ δ across that gap (see CONTINUITY_OFFSET).
    assert metrics["continuity_E_rel"] < 0.1
    assert metrics["continuity_H_rel"] < 0.1
    assert metrics["curl_E_residual_rel_air"] < 1e-6
    assert metrics["success_tier"] == "stretch"
    # The EMT baseline recorded alongside it is the number to beat.
    assert metrics["rel_l2_total_emt_vs_tmm"] > 0.1


def test_validate_reports_the_three_way_k_comparison(struct):
    metrics = vm.validate(vm.TMMFieldModule(struct), struct, n_points=1500, device=DEVICE)
    assert metrics["k_spp_re_tmm"] == pytest.approx(struct.k_tmm.real, rel=1e-12)
    assert metrics["k_spp_im_tmm"] == pytest.approx(struct.k_tmm.imag, rel=1e-12)
    assert metrics["k_spp_re_emt"] == pytest.approx(struct.k_emt.real, rel=1e-12)
    assert metrics["k_spp_im_emt"] == pytest.approx(struct.k_emt.imag, rel=1e-12)
    # An exact solver beats EMT on both components by construction.
    assert metrics["beats_emt_re"] > 0 and metrics["beats_emt_im"] > 0
    assert metrics["re_error_ratio_emt_over_pinn"] > 1
    assert metrics["im_error_ratio_emt_over_pinn"] > 1


def test_validate_on_the_emt_field_fails_the_field_tiers(struct):
    """Pushing the *homogenised* mode through the pipeline: it is not the truth."""
    metrics = vm.validate(vm.EMTFieldModule(struct), struct, n_points=1500, device=DEVICE)
    assert metrics["rel_l2_total"] > 0.1
    assert metrics["k_spp_re_rel_error"] == pytest.approx(metrics["emt_re_rel_error"], rel=1e-3)
    assert metrics["success_tier"] in {"minimum", "not met"}


def test_validate_on_an_untrained_network_returns_finite_metrics(struct):
    torch.manual_seed(0)
    network = vm.create_network(struct, hidden_dims=(16, 16), device=DEVICE)
    metrics = vm.validate(network, struct, n_points=800, device=DEVICE)
    assert all(
        np.isfinite(v) for v in metrics.values() if isinstance(v, float)
    )


def test_success_tier_classification():
    base = {
        "rel_l2_E": 0.02, "rel_l2_H": 0.02, "bound_in_air": 1.0, "bound_in_stack": 1.0,
        "ez_layer_contrast_rel_error": 0.1, "beats_emt_re": 1.0, "beats_emt_im": 1.0,
        "k_spp_re_rel_error": 0.005, "k_spp_im_rel_error": 0.005,
    }
    assert vm.success_tier(base) == "stretch"
    assert vm.success_tier({**base, "k_spp_im_rel_error": 0.05}) == "target"
    assert vm.success_tier({**base, "rel_l2_E": 0.3, "beats_emt_im": 0.0}) == "minimum"
    assert vm.success_tier({**base, "rel_l2_E": 0.9}) == "not met"
    assert vm.success_tier({**base, "bound_in_air": 0.0}) == "not met"
    # No layer structure => the minimum tier is not met however small the error.
    assert vm.success_tier({**base, "ez_layer_contrast_rel_error": 2.0}) == "not met"


# ---------------------------------------------------------------------- wiring
def test_create_network_output_shape(struct):
    network = vm.create_network(struct, hidden_dims=(16, 16), device=DEVICE)
    coords = torch.rand(7, 3) * 1e-7
    assert network(coords).shape == (7, 6, 2)


def test_configure_structure_rebuilds_everything():
    try:
        s = vm.configure_structure(20e-9, 4)
        assert s.period == pytest.approx(20e-9)
        assert s.n_periods == 4
        assert s.boundaries.size == 9
        assert vm.STRUCT is s
        assert s.kz_band_hat == pytest.approx(
            vm.FOURIER_KZ_HARMONICS * 2 * np.pi / (20e-9 * vm.K0)
        )
        assert layer_boundaries(s.thicknesses, s.z0)[-1] == pytest.approx(0.0, abs=1e-18)
    finally:
        vm.configure_structure(30e-9, 6)
    assert vm.STRUCT.n_periods == 6


def test_parse_args_defaults_and_overrides():
    args = vm.parse_args([])
    assert args.period_nm == pytest.approx(30.0) and args.n_periods == 6
    args = vm.parse_args(["--period-nm", "10", "--n-periods", "12", "--probe-only"])
    assert args.period_nm == 10.0 and args.n_periods == 12 and args.probe_only


@pytest.mark.slow
def test_probe_reduces_the_supervised_loss(struct):
    """The tractability probe must at least be fitting something."""
    out = vm.probe_representability(
        struct, n_epochs=60, n_points=512, device=DEVICE, seed=0, log_every=1000
    )
    history = out["probe_history"]
    assert min(history[-10:]) < history[0]
    assert 0.0 < out["probe_rel_l2"] < 2.0


@pytest.mark.slow
def test_short_training_reduces_loss(struct):
    """A ``--quick``-style run must decrease the loss and survive the L-BFGS phase."""
    torch.manual_seed(0)
    np.random.seed(0)
    network = vm.create_network(struct, hidden_dims=(32, 32), device=DEVICE)
    network, history = vm.train(
        network, struct, n_epochs=60, n_points=192, device=DEVICE, log_every=1000,
        lbfgs_steps=1,
    )
    assert set(history) == {"epoch", "total", "curl", "div", "continuity", "boundary", "lr"}
    assert len(history["total"]) == 61  # 60 Adam epochs + 1 L-BFGS step
    assert all(np.isfinite(history["total"]))
    assert min(history["total"][-10:]) < history["total"][0]
    # The float64 L-BFGS phase must hand back a float32 network.
    assert all(p.dtype == torch.float32 for p in network.parameters())


def test_layer_permittivities_survive_the_lbfgs_dtype_round_trip(struct):
    """
    Regression: ``Module.to(float64)`` used to silently strip ``Im ε_Ag``.

    The training loop promotes the core to float64 for L-BFGS and restores
    float32 afterwards. ``nn.Module.to(dtype)`` converts *complex* buffers with
    that same dtype, so a registered complex ε table would come back real —
    a lossless metal, and no ``Im k_spp`` to measure.
    """
    network = vm.create_network(struct, hidden_dims=(8,), device=DEVICE)
    before = network.core.eps_values.clone()
    network.core.to(torch.float64)
    network.to(torch.float32)
    assert torch.equal(network.core.eps_values, before)
    assert network.core.eps_values.dtype == torch.complex128
    assert float(before.imag.abs().max()) > 0.0
    # The same guard on the TMM reference module.
    tmm = vm.TMMFieldModule(struct)
    amps = tmm.amplitudes.clone()
    tmm.to(torch.float64)
    assert torch.equal(tmm.amplitudes, amps)


def test_tmm_reference_module_is_differentiable(struct):
    """
    The self-check has to run the *identical* pipeline, curl residuals included.

    A numpy round-trip would detach the graph and report ``∇×E = 0`` — a curl
    residual of order 1 — for a field that is exact.
    """
    z = torch.linspace(-150e-9, 200e-9, 33, dtype=torch.float64)
    coords = torch.stack([torch.zeros_like(z), torch.zeros_like(z), z], dim=1)
    coords.requires_grad_(True)
    fields = vm.tmm_fields_si(coords, struct)
    assert fields.requires_grad
    grad = torch.autograd.grad(fields[:, 4, 0].sum(), coords)[0]
    # ∂|H_y|/∂z is the mode's own decay, not zero.
    assert float(grad[:, 2].abs().max()) > 0.0
    assert torch.isfinite(grad).all()


def test_k_estimate_reports_a_spread_across_probe_lines(struct):
    """``Im k_spp`` is a ~1% amplitude change over the box, so its spread matters."""
    metrics = vm.estimate_k_spp(vm.TMMFieldModule(struct), struct, device=DEVICE)
    # The exact field has no scatter, so the spread metric is measuring the
    # network and nothing else.
    assert metrics["k_spp_im_spread_over_tmm"] < 1e-5
    assert metrics["k_spp_im_worst_rel_error"] < 1e-5
    assert metrics["k_spp_re_worst_rel_error"] < 1e-8
    assert metrics["k_spp_n_probe_lines"] == len(vm.K_PROBE_HEIGHTS)
    assert len(vm.K_PROBE_HEIGHTS) >= 10
    assert all(0.0 < h < 1.0 for h in vm.K_PROBE_HEIGHTS)


def test_k_estimate_weights_lines_by_signal_strength(struct):
    """
    The weighted mean must sit inside the range of the individual line fits and
    lean toward the strong-signal (low-z) lines, where the log-fit is best
    conditioned.
    """
    net = vm.TMMFieldModule(struct)
    low = vm.estimate_k_spp(net, struct, heights=(0.05,), device=DEVICE)["k_spp_im_pinn"]
    high = vm.estimate_k_spp(net, struct, heights=(0.95,), device=DEVICE)["k_spp_im_pinn"]
    both = vm.estimate_k_spp(net, struct, heights=(0.05, 0.95), device=DEVICE)["k_spp_im_pinn"]
    assert min(low, high) - 1e-6 <= both <= max(low, high) + 1e-6


def test_parse_args_eval_only():
    assert vm.parse_args(["--eval-only"]).eval_only
    assert not vm.parse_args([]).eval_only
