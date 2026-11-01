"""Tests for examples/validate_hmm_surrogate.py (the (ω, f) material-conditioned surrogate)."""

import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from examples import validate_hmm_surrogate as vs
from src.analytical import analytical_spp_fields
from src.constants import C0
from src.effective_medium import hmm_permittivities
from src.models import MaxwellCurlLoss, MaxwellDivergenceLoss
from src.physics.metamaterial import MetamaterialProperties

DEVICE = torch.device("cpu")
CORNERS = (
    (vs.OMEGA_MIN, vs.F_MIN),
    (vs.OMEGA_MIN, vs.F_MAX),
    (vs.OMEGA_MAX, vs.F_MIN),
    (vs.OMEGA_MAX, vs.F_MAX),
)


# --------------------------------------------------------------- design space
def test_design_rectangle_read_from_hmm_summary_not_hardcoded():
    """The material and the per-f qualifying bands come from hmm_summary.json."""
    with open(vs.HMM_SUMMARY_PATH) as fh:
        summary = json.load(fh)
    assert vs.EPS_D2 == pytest.approx(summary["design"]["eps_dielectric_layer"])
    assert vs.EPS_D == pytest.approx(summary["design"]["eps_superstrate"])
    scanned = {
        e["fill_fraction"]: e["band"]
        for e in summary["fill_fraction_scan"]
        if e.get("band") is not None
    }
    # f = 0.15 and 0.40 both have a qualifying band (they bound the design range);
    # the intersection is their inner edges, inset.
    assert 0.15 in scanned and 0.40 in scanned
    lo = max(scanned[f]["omega_over_omega_ref"][0] for f in scanned if 0.15 <= f <= 0.40)
    hi = min(scanned[f]["omega_over_omega_ref"][1] for f in scanned if 0.15 <= f <= 0.40)
    span = (hi - lo) * vs.OMEGA_REF
    assert vs.OMEGA_MIN == pytest.approx(lo * vs.OMEGA_REF + vs.DESIGN_INSET * span)
    assert vs.OMEGA_MAX == pytest.approx(hi * vs.OMEGA_REF - vs.DESIGN_INSET * span)
    # Sanity: the documented ~593-767 nm window.
    lam_nm = [2 * np.pi * C0 / w * 1e9 for w in (vs.OMEGA_MAX, vs.OMEGA_MIN)]
    assert lam_nm[0] == pytest.approx(593.3, rel=2e-3)
    assert lam_nm[1] == pytest.approx(766.4, rel=2e-3)


def test_design_space_is_bound_everywhere_independent_check():
    """
    Every point of the rectangle supports a bound, non-radiative SPP.

    Checked here *independently* of :func:`verify_design_space`: this walks a
    dense grid through ``MetamaterialProperties`` directly, applying both gates
    (``is_spp_supported`` and ``Re k_spp > sqrt(eps_d) k0``) by hand.
    """
    n_unbound = 0
    n_eff_min = np.inf
    for w in np.linspace(vs.OMEGA_MIN, vs.OMEGA_MAX, 25):
        for f in np.linspace(vs.F_MIN, vs.F_MAX, 17):
            eps_t, eps_n = hmm_permittivities(float(w), float(f), vs.EPS_D2, **vs.DRUDE)
            material = MetamaterialProperties(complex(eps_n), complex(eps_t), "z")
            k = material.spp_wavevector(float(w), vs.EPS_D, "x")
            n_eff = k.real / (float(w) / C0)
            if not (material.is_spp_supported(vs.EPS_D, "x") and n_eff > np.sqrt(vs.EPS_D)):
                n_unbound += 1
            n_eff_min = min(n_eff_min, n_eff)
    assert n_unbound == 0
    assert n_eff_min > 1.03  # bound with a real margin, not marginally


def test_verify_design_space_reports_a_fully_bound_rectangle_and_real_f_sensitivity():
    """The runtime verifier agrees, and f genuinely moves k_spp."""
    report = vs.verify_design_space(n_omega=21, n_fill=15)
    assert report["bound_fraction"] == 1.0
    assert report["n_eff_min"] > 1.0
    assert report["worst_margin_over_light_line"] > 0.02
    # Sweeping f at fixed ω must change k_spp by much more than the target
    # tolerance (1 %) -- otherwise the second conditioning axis is decoration.
    assert abs(1.0 - report["k_spp_ratio_along_f_at_mid_omega"]) > 0.05
    assert abs(1.0 - report["k_spp_ratio_along_f_at_blue_edge"]) > 0.15


def test_is_bound_agrees_with_metamaterial_properties():
    for omega, fill in CORNERS:
        bound, n_eff = vs.is_bound(omega, fill)
        eps_t, eps_n = vs.hmm_eps(omega, fill)
        material = MetamaterialProperties(eps_n, eps_t, "z")
        assert bound is material.is_spp_supported(vs.EPS_D, "x")
        assert n_eff == pytest.approx(
            material.spp_wavevector(omega, vs.EPS_D, "x").real / vs.k0_of(omega)
        )


# --------------------------------------------------------------- material
def test_permittivity_depends_on_both_omega_and_fill():
    """The whole point: at fixed ω, two fill fractions are different materials."""
    omega = vs.OMEGA_MID
    eps_t_lo, eps_n_lo = vs.hmm_eps(omega, vs.F_MIN)
    eps_t_hi, eps_n_hi = vs.hmm_eps(omega, vs.F_MAX)
    assert abs(eps_t_lo - eps_t_hi) > 3.0
    assert abs(eps_n_lo - eps_n_hi) > 1.0
    # ...and at fixed f, two frequencies are too.
    eps_t_red, _ = vs.hmm_eps(vs.OMEGA_MIN, vs.F_MID)
    eps_t_blue, _ = vs.hmm_eps(vs.OMEGA_MAX, vs.F_MID)
    assert abs(eps_t_red - eps_t_blue) > 1.0
    # Type-II throughout (Re ε_t < 0 < Re ε_n) and passive.
    for omega, fill in CORNERS:
        eps_t, eps_n = vs.hmm_eps(omega, fill)
        assert eps_t.real < 0.0 < eps_n.real
        assert eps_t.imag > 0 and eps_n.imag > 0


def test_hmm_eps_torch_matches_numpy_version_over_the_rectangle():
    """The torch mirror used inside the adapter agrees to machine precision."""
    w = np.linspace(vs.OMEGA_MIN, vs.OMEGA_MAX, 7)
    f = np.linspace(vs.F_MIN, vs.F_MAX, 7)
    W, F = np.meshgrid(w, f, indexing="ij")
    eps_t, eps_n = vs.hmm_eps_torch(torch.tensor(W.ravel()), torch.tensor(F.ravel()))
    ref_t, ref_n = hmm_permittivities(W.ravel(), F.ravel(), vs.EPS_D2, **vs.DRUDE)
    assert np.allclose(eps_t.numpy(), ref_t, rtol=1e-14, atol=0)
    assert np.allclose(eps_n.numpy(), ref_n, rtol=1e-14, atol=0)


def test_hmm_eps_torch_is_differentiable_in_fill():
    """The adapter's divisor must carry a gradient w.r.t. f for inverse design."""
    f = torch.tensor(vs.F_MID, dtype=torch.float64, requires_grad=True)
    w = torch.tensor(vs.OMEGA_MID, dtype=torch.float64)
    _, eps_n = vs.hmm_eps_torch(w, f)
    eps_n.real.backward()
    assert f.grad is not None and torch.isfinite(f.grad) and f.grad.abs() > 0


def test_eps_rows_use_the_rows_own_omega_and_fill():
    """Two rows differing only in f must carry different interior ε -- and vice versa."""
    omegas = torch.tensor([vs.OMEGA_MID, vs.OMEGA_MID, vs.OMEGA_MIN], dtype=torch.float64)
    fills = torch.tensor([vs.F_MIN, vs.F_MAX, vs.F_MIN], dtype=torch.float64)
    rows = vs.eps_tensor_rows(omegas, fills, metal=True)
    assert rows.shape == (3, 3, 3)
    for i, (w, f) in enumerate(zip(omegas.tolist(), fills.tolist(), strict=True)):
        eps_t, eps_n = vs.hmm_eps(w, f)
        assert rows[i, 0, 0].item() == pytest.approx(eps_t)
        assert rows[i, 1, 1].item() == pytest.approx(eps_t)
        assert rows[i, 2, 2].item() == pytest.approx(eps_n)
    # same ω, different f -> different material
    assert abs(rows[0, 0, 0] - rows[1, 0, 0]).item() > 3.0
    assert abs(rows[0, 2, 2] - rows[1, 2, 2]).item() > 1.0
    # same f, different ω -> different material
    assert abs(rows[0, 0, 0] - rows[2, 0, 0]).item() > 0.5
    # air rows are ε_d regardless of the design point
    air = vs.eps_tensor_rows(omegas, fills, metal=False)
    assert torch.allclose(air, torch.eye(3, dtype=torch.complex128) * complex(vs.EPS_D))


# --------------------------------------------------------------- normalisation
def test_omega_hat_and_fill_hat_normalisation():
    """Both conditioning features are linear maps of their range onto [-1, 1]."""
    assert vs.omega_hat(vs.OMEGA_MIN) == pytest.approx(-1.0)
    assert vs.omega_hat(vs.OMEGA_MAX) == pytest.approx(1.0)
    assert vs.omega_hat(vs.OMEGA_MID) == pytest.approx(0.0)
    assert vs.fill_hat(vs.F_MIN) == pytest.approx(-1.0)
    assert vs.fill_hat(vs.F_MAX) == pytest.approx(1.0)
    assert vs.fill_hat(vs.F_MID) == pytest.approx(0.0)
    for f in np.linspace(vs.F_MIN, vs.F_MAX, 9):
        assert vs.fill_from_hat(vs.fill_hat(float(f))) == pytest.approx(float(f))
        # linear, not merely monotone
        assert vs.fill_hat(float(f)) == pytest.approx(
            (float(f) - vs.F_MID) / (0.5 * (vs.F_MAX - vs.F_MIN))
        )
    hats = torch.tensor([-1.0, -0.25, 0.5, 1.0], dtype=torch.float64)
    assert torch.allclose(
        vs.fill_from_hat_torch(hats),
        torch.tensor([vs.fill_from_hat(float(h)) for h in hats], dtype=torch.float64),
    )


def test_validation_points_are_strictly_held_out_from_the_refinement_nodes():
    assert len(vs.LBFGS_POINTS) == vs.N_NODE_OMEGA * vs.N_NODE_FILL
    assert len(vs.VALIDATION_POINTS) >= 16
    node_hats = {(round(vs.omega_hat(w), 9), round(vs.fill_hat(f), 9)) for w, f in vs.LBFGS_POINTS}
    for w, f in vs.VALIDATION_POINTS:
        assert (round(vs.omega_hat(w), 9), round(vs.fill_hat(f), 9)) not in node_hats
        # each held-out point sits at a cell centre: half a node spacing away
        d_w = min(abs(vs.omega_hat(w) - vs.omega_hat(nw)) for nw, _ in vs.LBFGS_POINTS)
        d_f = min(abs(vs.fill_hat(f) - vs.fill_hat(nf)) for _, nf in vs.LBFGS_POINTS)
        assert d_w == pytest.approx(1.0 / (vs.N_NODE_OMEGA - 1))
        assert d_f == pytest.approx(1.0 / (vs.N_NODE_FILL - 1))
        assert vs.OMEGA_MIN < w < vs.OMEGA_MAX and vs.F_MIN < f < vs.F_MAX


def test_stratified_design_points_cover_every_cell():
    """One draw puts exactly one point in each stratum, unlike a uniform draw."""
    g = torch.Generator().manual_seed(3)
    points = vs.stratified_design_points(generator=g)
    assert len(points) == vs.N_BLOCKS
    cells = set()
    for w, f in points:
        assert vs.OMEGA_MIN <= w <= vs.OMEGA_MAX and vs.F_MIN <= f <= vs.F_MAX
        i = min(vs.STRATA_OMEGA - 1, int((vs.omega_hat(w) + 1.0) / 2.0 * vs.STRATA_OMEGA))
        j = min(vs.STRATA_FILL - 1, int((vs.fill_hat(f) + 1.0) / 2.0 * vs.STRATA_FILL))
        cells.add((i, j))
    assert len(cells) == vs.N_BLOCKS


def test_set_design_space_narrowing_and_restore():
    """Shrinking the f range moves the ω intersection and refreshes the node sets."""
    try:
        vs.set_design_space(0.20, 0.35)
        assert (vs.F_MIN, vs.F_MAX) == (0.20, 0.35)
        assert vs.OMEGA_MIN > 0 and vs.OMEGA_MAX > vs.OMEGA_MIN
        assert all(0.20 <= f <= 0.35 for _, f in vs.LBFGS_POINTS)
        assert all(0.20 < f < 0.35 for _, f in vs.VALIDATION_POINTS)
        report = vs.verify_design_space(n_omega=13, n_fill=9)
        assert report["bound_fraction"] == 1.0
    finally:
        vs.set_design_space()
    assert (vs.F_MIN, vs.F_MAX) == (0.15, 0.40)


# --------------------------------------------------------------- geometry
@pytest.mark.parametrize(("omega", "fill"), CORNERS)
def test_per_point_domain_matches_that_points_analytic_scales(omega, fill):
    k, kappa_d, kappa_m = vs.mode_constants(omega, fill)
    x_max, y_max, z_min, z_max = vs.domain_si(omega, fill)
    assert x_max == pytest.approx(vs.X_PERIODS * 2 * np.pi / k.real)
    assert z_min == pytest.approx(-vs.Z_METAL_DEPTHS / kappa_m.real)
    assert z_max == pytest.approx(vs.Z_AIR_DEPTHS / kappa_d.real)
    assert y_max == pytest.approx(vs.Y_WAVELENGTHS * 2 * np.pi / vs.k0_of(omega))


def test_domain_differs_between_fill_fractions_at_fixed_omega():
    """The box is a function of f too, not of ω alone."""
    lo = vs.domain_hat(vs.OMEGA_MID, vs.F_MIN)
    hi = vs.domain_hat(vs.OMEGA_MID, vs.F_MAX)
    assert abs(lo[2] / hi[2]) > 1.5  # metal depth: |ε_t| grows with f, mode is shallower
    assert abs(lo[3] / hi[3]) < 0.8  # air depth: n_eff drops with f, mode spreads out


def test_k0_scaling_keeps_coordinates_and_wavenumbers_order_one():
    """The k₀(ω) input scaling compresses the rectangle's 9.7× SI spread to O(1)."""
    scaled, si = [], []
    for omega, fill in CORNERS:
        k, kd, km = vs.mode_constants(omega, fill)
        k0 = vs.k0_of(omega)
        scaled += [k.real / k0, kd.real / k0, km.real / k0]
        si += [kd.real, km.real]
    assert 0.2 < min(scaled) and max(scaled) < 3.0
    assert max(si) / min(si) > 5.0  # ... which the SI numbers certainly are not
    for omega, fill in CORNERS:
        x_max, _, z_min, z_max = vs.domain_hat(omega, fill)
        assert 5.0 < x_max < 20.0
        assert -12.0 < z_min < 0.0 < z_max < 6.0


# --------------------------------------------------------------- network
def test_network_shapes_and_fill_dependent_ez_jump():
    """
    The adapter's E_z jump is ε_n(ω, f) — so it changes with f at fixed ω.

    Evaluated just above and just below the interface at one (ω, f) and then at
    a second f with the same ω: the ratio of the two Ê_z must track ε_n each
    time, and the two ε_n must differ.
    """
    torch.manual_seed(0)
    net = vs.create_network(device=DEVICE)
    ratios = []
    for fill in (vs.F_MIN, vs.F_MAX):
        omega = vs.OMEGA_MID
        eps = 1e-4
        base = torch.tensor([[0.3, 0.05, 0.0]], dtype=torch.float32)
        above = base.clone()
        above[0, 2] = eps
        below = base.clone()
        below[0, 2] = -eps
        cond = torch.tensor(
            [[vs.omega_hat(omega), vs.fill_hat(fill)]], dtype=torch.float32
        )
        out_a = net.core(torch.cat([above, cond], dim=1))
        out_b = net.core(torch.cat([below, cond], dim=1))
        assert out_a.shape == (1, 6, 2)
        ez_a = complex(out_a[0, 2, 0].item(), out_a[0, 2, 1].item())
        ez_b = complex(out_b[0, 2, 0].item(), out_b[0, 2, 1].item())
        _, eps_n = vs.hmm_eps(omega, fill)
        # D_z is continuous by construction, so E_z(above)/E_z(below) = ε_n/ε_d.
        assert (ez_a / ez_b) == pytest.approx(eps_n / vs.EPS_D, rel=2e-2)
        ratios.append(ez_a / ez_b)
    assert abs(ratios[0] - ratios[1]) > 1.0  # the divisor really depends on f


def test_forward_is_sensitive_to_the_fill_input():
    """Changing only f̂ changes the field: the network is genuinely conditioned."""
    torch.manual_seed(1)
    net = vs.create_network(device=DEVICE)
    coords = torch.rand(64, 3) * 1e-7
    a = net(coords, vs.OMEGA_MID, vs.F_MIN)
    b = net(coords, vs.OMEGA_MID, vs.F_MAX)
    assert torch.linalg.vector_norm(a - b) > 1e-3 * torch.linalg.vector_norm(a)


@pytest.mark.parametrize(("omega", "fill"), CORNERS)
def test_scaled_maxwell_is_frequency_free_on_the_exact_mode(omega, fill):
    """
    The k₀-scaling claim: ``∇̂×Ê = iĤ``, ``∇̂×Ĥ = −iεÊ`` at every design point.

    Feeding the analytical mode in scaled coordinates through the scaled
    operator (frequency = 1, μ₀ = ε₀ = 1) must leave a residual at round-off,
    with ε the design point's own tensor and no ω prefactor anywhere.
    """
    k0 = vs.k0_of(omega)
    coords_hat = vs.sample_collocation_hat(256, omega, fill, device=DEVICE).to(torch.float64)
    for metal in (False, True):
        z = coords_hat[:, 2]
        sel = (z < 0) if metal else (z > 0)
        pts = coords_hat[sel].clone().requires_grad_(True)
        eps_rows = vs.eps_tensor_rows(
            torch.full((pts.shape[0],), omega, dtype=torch.float64),
            torch.full((pts.shape[0],), fill, dtype=torch.float64),
            metal=metal,
        )
        fields = vs.analytical_fields_si(pts.detach() / k0, omega, fill).to(torch.float64)
        fields = fields / vs.FIELD_SCALE.to(torch.float64)
        E = torch.complex(fields[:, :3, 0], fields[:, :3, 1])
        H = torch.complex(fields[:, 3:, 0], fields[:, 3:, 1])
        # Re-evaluate with graph so the operator can differentiate.
        analytic = vs.analytical_fields_si(pts / k0, omega, fill).to(torch.float64)
        del E, H
        analytic = analytic / vs.FIELD_SCALE.to(torch.float64)
        E = torch.complex(analytic[:, :3, 0], analytic[:, :3, 1])
        H = torch.complex(analytic[:, 3:, 0], analytic[:, 3:, 1])
        curl_E = vs._SCALED_MAXWELL.curl_operator(E, pts)
        curl_H = vs._SCALED_MAXWELL.curl_operator(H, pts)
        eps_E = torch.einsum("nij,nj->ni", eps_rows.to(E.dtype), E)
        scale = E.abs().max()
        assert (curl_E - 1j * H).abs().max() < 1e-8 * scale.clamp_min(1e-30) * 10
        assert (curl_H + 1j * eps_E).abs().max() < 1e-6 * scale.clamp_min(1e-30) * 100


@pytest.mark.parametrize("metal", [False, True])
def test_weighted_losses_match_library_losses_for_uniform_weights(metal):
    """With row weight 1 the per-row losses reduce to the library ones."""
    torch.manual_seed(2)
    net = vs.create_network(device=DEVICE)
    omega, fill = vs.OMEGA_MID, vs.F_MID
    coords = (
        vs.sample_collocation_hat(96, omega, fill, device=DEVICE).detach().requires_grad_(True)
    )
    cond = torch.tensor([[vs.omega_hat(omega), vs.fill_hat(fill)]]).expand(96, 2)
    eps_rows = vs.eps_tensor_rows(
        torch.full((96,), omega, dtype=torch.float64),
        torch.full((96,), fill, dtype=torch.float64),
        metal=metal,
    )
    net3 = vs.ConditionColumnNet(net.core, cond)
    c1 = coords.detach().clone().requires_grad_(True)
    ours = float(vs.curl_loss_weighted(net.core, c1, cond, eps_rows).detach())
    c2 = coords.detach().clone().requires_grad_(True)
    theirs = float(
        MaxwellCurlLoss(frequency=1.0, mu0=1.0, eps0=1.0)
        .compute(network=net3, coords=c2, epsilon=eps_rows, mu_r=1.0)
        .detach()
    )
    assert ours == pytest.approx(theirs, rel=1e-6)
    c3 = coords.detach().clone().requires_grad_(True)
    ours_d = float(vs.divergence_loss_weighted(net.core, c3, cond, eps_rows).detach())
    c4 = coords.detach().clone().requires_grad_(True)
    theirs_d = float(
        MaxwellDivergenceLoss().compute(network=net3, coords=c4, epsilon=eps_rows).detach()
    )
    assert ours_d == pytest.approx(theirs_d, rel=1e-6)
    # A non-uniform per-row weight actually changes the value.
    c5 = coords.detach().clone().requires_grad_(True)
    weights = torch.linspace(0.1, 2.0, 96)
    assert float(vs.curl_loss_weighted(net.core, c5, cond, eps_rows, weights).detach()) != ours


# --------------------------------------------------------------- anchor / sampling
@pytest.mark.parametrize(("omega", "fill"), CORNERS)
def test_anchor_matches_analytical_at_design_space_corners(omega, fill):
    """
    The boundary anchor is exactly ``analytical_spp_fields`` at that block's own
    (ω, f), expressed in the scaled field units the core outputs.
    """
    batch = vs.sample_training_batch(64, 48, 16, [(omega, fill)], device=DEVICE)
    coords_hat = batch["boundary"]
    coords_si = coords_hat.to(torch.float64) / vs.k0_of(omega)
    eps_t, eps_n = vs.hmm_eps(omega, fill)
    E, H = analytical_spp_fields(
        coords_si, omega, eps_t, eps_n, eps_dielectric=vs.EPS_D, H0=vs.H0
    )
    expect = torch.cat(
        [
            torch.stack([E.real, E.imag], dim=-1) / vs.E_SCALE,
            torch.stack([H.real, H.imag], dim=-1) / vs.H_SCALE,
        ],
        dim=1,
    ).to(torch.float32)
    assert torch.allclose(batch["target"], expect, atol=1e-5, rtol=1e-4)
    # ... and the condition columns really carry this point's (ω̂, f̂)
    assert torch.allclose(
        batch["cond_bc"][0],
        torch.tensor([vs.omega_hat(omega), vs.fill_hat(fill)], dtype=torch.float32),
    )


def test_sampling_respects_the_per_point_box_and_guard():
    for omega, fill in CORNERS:
        x_max, y_max, z_min, z_max = vs.domain_hat(omega, fill)
        pts = vs.sample_collocation_hat(600, omega, fill, device=DEVICE)
        assert pts[:, 0].min() >= 0.0 and pts[:, 0].max() <= x_max
        assert pts[:, 1].min() >= 0.0 and pts[:, 1].max() <= y_max
        assert pts[:, 2].min() >= z_min - 1e-6 and pts[:, 2].max() <= z_max + 1e-6
        assert pts[:, 2].abs().min() >= vs.GUARD_HAT - 1e-9
        iface, normals = vs.sample_interface_hat(50, omega, fill, device=DEVICE)
        assert torch.allclose(iface[:, 2], torch.zeros(50))
        assert torch.allclose(normals[:, 2], torch.ones(50))


def test_training_batch_blocks_carry_their_own_material():
    """A two-block batch at one ω and two f's must carry two different ε sets."""
    points = [(vs.OMEGA_MID, vs.F_MIN), (vs.OMEGA_MID, vs.F_MAX)]
    batch = vs.sample_training_batch(400, 96, 48, points, device=DEVICE)
    eps = batch["eps_metal"]
    unique = torch.unique(torch.round(eps[:, 0, 0].real * 1e6) / 1e6)
    assert unique.numel() == 2
    lo, hi = sorted(complex(v, 0).real for v in unique.tolist())
    assert hi - lo > 3.0
    scales = torch.unique(torch.round(batch["eps_scale_metal"] * 1e6) / 1e6)
    assert scales.numel() == 2


# --------------------------------------------------------------- differentiable estimator
def test_differentiable_k_spp_estimator_has_finite_nonzero_gradient_in_fill():
    """
    ``k_spp_from_network`` must be differentiable in f̂ — the demo depends on it.

    Uses the *analytical* mode wrapped as a core, so the value is also checked
    against the exact k_spp; the gradient is what matters.
    """
    torch.manual_seed(4)
    net = vs.create_network(device=DEVICE)
    f_hat = torch.tensor(0.2, requires_grad=True)
    k = vs.k_spp_from_network(net.core, vs.OMEGA_MID, f_hat, n_line=128, device=DEVICE)
    assert torch.isfinite(k)
    k.backward()
    assert f_hat.grad is not None
    assert torch.isfinite(f_hat.grad)
    assert f_hat.grad.abs() > 0.0


def test_differentiable_estimator_recovers_the_exact_k_spp_on_an_analytical_core():
    """On a core that *is* the analytical mode the estimator returns Re k_spp."""

    class AnalyticalCore(nn.Module):
        """5-column scaled-frame core returning the exact mode (no parameters used)."""

        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))

        def forward(self, coords: torch.Tensor) -> torch.Tensor:
            omega = float(vs.omega_from_hat(float(coords[0, 3])))
            fill = float(vs.fill_from_hat(float(coords[0, 4])))
            out = vs.analytical_fields_hat(coords[:, :3].detach(), omega, fill)
            return out.to(coords.dtype) + 0.0 * self.dummy

    core = AnalyticalCore()
    for fill in (vs.F_MIN, vs.F_MID, vs.F_MAX):
        f_hat = torch.tensor(vs.fill_hat(fill))
        k = vs.k_spp_from_network(core, vs.OMEGA_MID, f_hat, n_line=512, device=DEVICE)
        assert float(k.detach()) == pytest.approx(
            vs.mode_constants(vs.OMEGA_MID, fill)[0].real, rel=1e-3
        )


def test_closed_form_fill_for_index_inverts_the_analytics():
    """The cross-check root find recovers the fill fraction it was given."""
    for f_true in (0.18, 0.25, 0.33):
        omega = vs.OMEGA_MID
        _, n_target = vs.is_bound(omega, f_true)
        assert vs.closed_form_fill_for_index(omega, n_target) == pytest.approx(f_true, abs=1e-8)
    with pytest.raises(ValueError):
        vs.closed_form_fill_for_index(vs.OMEGA_MID, 5.0)


def test_default_inverse_targets_are_interior_and_consistent():
    targets = vs.default_inverse_targets()
    assert len(targets) == 3
    for omega, n_target in targets:
        assert vs.OMEGA_MIN <= omega <= vs.OMEGA_MAX
        f = vs.closed_form_fill_for_index(omega, n_target)
        assert vs.F_MIN + 0.02 < f < vs.F_MAX - 0.02


def test_inverse_design_demo_on_an_analytical_surrogate_finds_the_right_fill():
    """
    End-to-end demo against a surrogate that is exactly right.

    With the analytical mode standing in for the trained network, descending
    through :func:`k_spp_from_network` must land on the closed-form fill
    fraction — which isolates the optimisation loop from the network's error.
    """

    class AnalyticalSurrogate(nn.Module):
        def __init__(self):
            super().__init__()
            self.core = _AnalyticalCoreDifferentiableInF()

    model = AnalyticalSurrogate()
    omega = vs.OMEGA_MID
    _, n_target = vs.is_bound(omega, 0.26)
    records = vs.inverse_design_demo(
        model, [(omega, n_target)], n_steps=250, lr=0.05, device=DEVICE
    )
    assert len(records) == 1
    rec = records[0]
    assert rec["fill_fraction_closed_form"] == pytest.approx(0.26, abs=1e-6)
    assert rec["fill_fraction_pinn"] == pytest.approx(0.26, abs=5e-3)


class _AnalyticalCoreDifferentiableInF(nn.Module):
    """
    Analytical mode as a core whose output depends *smoothly* on the f̂ column.

    ``analytical_spp_fields`` takes Python complex scalars, so it cannot carry a
    gradient in f. This stand-in interpolates it: it evaluates the exact mode on
    a fine f grid once and returns a differentiable linear blend of the two
    bracketing samples, which is enough for the optimiser to have a real
    gradient while the values stay the exact ones at the grid points.
    """

    def __init__(self, n_grid: int = 601):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.n_grid = n_grid

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        omega = float(vs.omega_from_hat(float(coords[0, 3].detach())))
        f_hat = coords[:, 4:5]
        f_grid = torch.linspace(-1.0, 1.0, self.n_grid, dtype=coords.dtype)
        pos = (f_hat[0, 0].detach() + 1.0) / 2.0 * (self.n_grid - 1)
        i0 = int(torch.clamp(torch.floor(pos), 0, self.n_grid - 2))
        lo, hi = float(f_grid[i0]), float(f_grid[i0 + 1])
        out_lo = vs.analytical_fields_hat(
            coords[:, :3].detach(), omega, vs.fill_from_hat(lo)
        ).to(coords.dtype)
        out_hi = vs.analytical_fields_hat(
            coords[:, :3].detach(), omega, vs.fill_from_hat(hi)
        ).to(coords.dtype)
        w = ((f_hat[:, 0] - lo) / (hi - lo)).view(-1, 1, 1)
        return (1.0 - w) * out_lo + w * out_hi + 0.0 * self.dummy


# --------------------------------------------------------------- validation pipeline
def test_validate_on_analytical_network_is_near_perfect():
    """The pipeline's conventions are self-consistent: exact mode in, ~0 error out."""
    model = vs.AnalyticalHMMSPP()
    for omega, fill in ((vs.OMEGA_MIN, vs.F_MIN), (vs.OMEGA_MID, vs.F_MID),
                        (vs.OMEGA_MAX, vs.F_MAX)):
        m = vs.validate_at_point(model, omega, fill, n_points=1500, device=DEVICE)
        assert m["rel_l2_E"] < 1e-6
        assert m["rel_l2_H"] < 1e-6
        assert m["k_spp_rel_error"] < 1e-6
        assert m["kappa_d_fit_rel_error"] < 1e-4
        assert m["kappa_m_fit_rel_error"] < 1e-4
        # The continuity probe compares z = ±0.02/k₀, so even the exact mode
        # shows a residual of order κ̂·0.02 ≈ 1-5 % from its own decay across
        # that offset; it is a measurement floor, not an error.
        assert m["continuity_E_rel"] < 5e-2
        assert m["continuity_H_rel"] < 5e-2
        assert m["decay_sign_correct_air"] > 0 and m["decay_sign_correct_metal"] > 0
        assert all(np.isfinite(v) for v in m.values())
        for side in ("air", "metal"):
            assert m[f"curl_E_residual_rel_{side}"] < 1e-5
            assert m[f"curl_H_residual_rel_{side}"] < 1e-5
    summary = vs.summarise(
        vs.validate_grid(model, vs.VALIDATION_POINTS[:4], n_points=1200, device=DEVICE)
    )
    assert summary["success_tier"] == "stretch"
    assert summary["bound_mode_everywhere"] == 1.0


def test_validate_on_untrained_network_is_finite():
    torch.manual_seed(5)
    net = vs.create_network(device=DEVICE)
    m = vs.validate_at_point(net, vs.OMEGA_MID, vs.F_MID, n_points=600, device=DEVICE)
    assert np.isfinite(m["rel_l2_E"]) and np.isfinite(m["k_spp_fit"])


def test_design_space_maps_shapes_and_keys():
    model = vs.AnalyticalHMMSPP()
    maps = vs.design_space_maps(model, n_omega=3, n_fill=3, n_points=300, device=DEVICE)
    assert maps["k_pinn"].shape == (3, 3)
    assert np.allclose(maps["k_pinn"], maps["k_exact"], rtol=1e-5)
    assert maps["rel_l2"].max() < 1e-6


def test_success_tier_classification():
    base = {"bound_mode_everywhere": 1.0}
    assert vs.success_tier({**base, "worst_rel_l2": 0.02, "worst_k_spp_rel_error": 0.001}) == "stretch"
    assert vs.success_tier({**base, "worst_rel_l2": 0.05, "worst_k_spp_rel_error": 0.008}) == "target"
    assert vs.success_tier({**base, "worst_rel_l2": 0.3, "worst_k_spp_rel_error": 0.2}) == "minimum"
    assert vs.success_tier({**base, "worst_rel_l2": 0.8, "worst_k_spp_rel_error": 0.2}) == "not met"
    assert vs.success_tier(
        {"bound_mode_everywhere": 0.0, "worst_rel_l2": 0.001, "worst_k_spp_rel_error": 0.0}
    ) == "not met"


def test_parse_args_defaults():
    args = vs.parse_args([])
    assert args.epochs == vs.N_EPOCHS
    assert args.n_points == vs.BATCH_SIZE
    assert args.lbfgs_steps == vs.LBFGS_STEPS
    assert args.f_min == 0.15 and args.f_max == 0.40
    assert args.lbfgs_dtype == "float64"
    quick = vs.parse_args(["--quick", "--f-min", "0.2", "--f-max", "0.35"])
    assert quick.quick and quick.f_min == 0.2 and quick.f_max == 0.35


# --------------------------------------------------------------- training smoke
def test_short_training_reduces_loss():
    """A --quick-sized run must actually descend."""
    torch.manual_seed(0)
    net = vs.create_network(device=DEVICE)
    _, history = vs.train(
        net, n_epochs=40, n_points=256, device=DEVICE, log_every=1000, lbfgs_steps=0
    )
    assert len(history["total"]) == 40
    assert np.all(np.isfinite(history["total"]))
    assert np.mean(history["total"][-10:]) < np.mean(history["total"][:10])


def test_resume_continues_the_history_and_never_checkpoints_a_worse_iterate(tmp_path):
    """
    Chunked training must produce one continuous curve, not several.

    The run is split across processes by a wall-clock limit, so the history and
    the best-so-far bar have to survive in the checkpoint; without the bar, a
    resumed chunk's first post-ramp epoch (at the restarted, full learning rate)
    would overwrite a better iterate.
    """
    torch.manual_seed(7)
    net = vs.create_network(device=DEVICE)
    ck = tmp_path / "m.partial.pth"
    _, h1 = vs.train(
        net, n_epochs=12, n_points=128, device=DEVICE, log_every=5, lbfgs_steps=0,
        checkpoint_path=ck,
    )
    stored = vs.load_history(ck)
    assert stored is not None and len(stored["epoch"]) == 12
    assert stored["epoch"] == list(range(12))

    net2 = vs.create_network(device=DEVICE)
    best = vs.load_checkpoint_into(net2, ck)
    _, h2 = vs.train(
        net2, n_epochs=8, n_points=128, device=DEVICE, log_every=5, lbfgs_steps=0,
        checkpoint_path=ck, initial_history=stored, initial_best_loss=best,
    )
    assert h2["epoch"] == list(range(20))
    assert h2["total"][:12] == h1["total"]
    assert h2["wall_s"] == sorted(h2["wall_s"])  # cumulative across chunks
    assert h2["wall_s"][12] > h2["wall_s"][11]
    after = vs.load_checkpoint_into(vs.create_network(device=DEVICE), ck)
    assert after <= best + 1e-12


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(6)
    net = vs.create_network(device=DEVICE)
    path = tmp_path / "ckpt.pth"
    vs._write_checkpoint(path, net.core.state_dict(), 1.25, "adam:0")
    other = vs.create_network(device=DEVICE)
    assert vs.load_checkpoint_into(other, path) == pytest.approx(1.25)
    coords = torch.rand(16, 3) * 1e-7
    assert torch.allclose(
        net(coords, vs.OMEGA_MID, vs.F_MID), other(coords, vs.OMEGA_MID, vs.F_MID)
    )
