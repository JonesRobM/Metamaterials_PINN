"""
Unit tests for the shared experiment machinery in :mod:`src.experiments`.

These pieces were previously reachable only through the five
``examples/validate_*.py`` experiments, so their edge cases — an interface point
that lands exactly on a boundary, a ramp of length zero, a checkpoint written
while the network is in float64, a success tier one ULP either side of its
threshold — were exercised only incidentally, if at all. Here they are tested
directly.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.experiments import (
    ColumnConditionedNet,
    DisplacementAdapter,
    LayeredAdapter,
    LinearFeature,
    TrainingConfig,
    TwoMediumAdapter,
    add_common_args,
    add_core_args,
    add_output_args,
    banded_success_tier,
    continuity_residuals,
    fit_decay_constants,
    k0_of,
    load_checkpoint_history,
    load_core_checkpoint,
    plot_two_phase_history,
    ramp_at,
    relative_l2,
    run_training,
    weighted_curl_loss,
    weighted_divergence_loss,
    write_checkpoint,
    write_json_report,
)
from src.experiments.training import HISTORY_KEYS

DEVICE = torch.device("cpu")


# ===================================================================== helpers
class ConstantMLP(nn.Module):
    """Returns 1 + 0j on all six channels, so the adapter's effect is visible."""

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(coords.shape[0], 6, 2, dtype=coords.dtype,
                          device=coords.device)
        out[:, :, 0] = 1.0
        return out


class TinyCore(nn.Module):
    """A minimal differentiable ``coords -> [N, 6, 2]`` network."""

    def __init__(self, in_features: int = 3):
        super().__init__()
        self.linear = nn.Linear(in_features, 12)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.linear(coords).reshape(-1, 6, 2)


class TinyPINN(nn.Module):
    """The ``network.core`` shape :func:`run_training` expects."""

    def __init__(self, in_features: int = 3):
        super().__init__()
        self.core = TinyCore(in_features)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.core(coords)


# ===================================================== displacement adapters
class TestDisplacementAdapter:
    def test_base_class_requires_an_epsilon_profile(self):
        """The base adapter is abstract: it has no ε of its own to divide by."""
        adapter = DisplacementAdapter(ConstantMLP())
        with pytest.raises(NotImplementedError):
            adapter(torch.zeros(2, 3))

    def test_wrapped_network_is_stored_as_mlp(self):
        """Serialisation contract: keys are the wrapped net's under ``mlp.``.

        Every shipped ``artifacts/models/*.pth`` was written against this
        prefix; renaming the attribute would make them all unloadable.
        """
        adapter = TwoMediumAdapter(TinyCore(), eps_below=-4 + 0.2j, eps_above=1.0)
        assert all(k.startswith("mlp.") for k in adapter.state_dict())

    @pytest.mark.parametrize(
        "eps_below", [-18.3 + 0.55j, 3 + 0.05j, -4 + 0.2j],
        ids=["silver", "uniaxial-normal", "type-II"],
    )
    def test_two_medium_divides_dz_by_the_local_normal_permittivity(self, eps_below):
        eps_above = 1.0
        adapter = TwoMediumAdapter(ConstantMLP(), eps_below, eps_above)
        coords = torch.tensor([[0.0, 0.0, -0.1], [0.0, 0.0, 0.1]])
        fields = torch.view_as_complex(adapter(coords).contiguous())
        assert fields[0, 2] == pytest.approx(1.0 / eps_below)
        assert fields[1, 2] == pytest.approx(1.0 / eps_above)

    def test_two_medium_leaves_every_other_channel_untouched(self):
        adapter = TwoMediumAdapter(ConstantMLP(), -18.3 + 0.55j, 1.0)
        coords = torch.tensor([[0.0, 0.0, -0.1], [0.0, 0.0, 0.1]])
        fields = torch.view_as_complex(adapter(coords).contiguous())
        others = torch.cat([fields[:, :2], fields[:, 3:]], dim=1)
        assert torch.allclose(others, torch.ones_like(others))

    def test_the_ez_jump_is_exactly_the_permittivity_ratio(self):
        """The point of the whole construction: an *exact* discontinuity.

        ``D_z`` is continuous, so ``E_z`` must jump by exactly ``ε_below/ε_above``
        — not approximately, and not only in the limit of a converged network.
        """
        eps_below, eps_above = -18.3 + 0.55j, 2.25
        adapter = TwoMediumAdapter(ConstantMLP(), eps_below, eps_above)
        delta = 1e-9
        below = torch.view_as_complex(
            adapter(torch.tensor([[0.0, 0.0, -delta]])).contiguous()
        )[0, 2]
        above = torch.view_as_complex(
            adapter(torch.tensor([[0.0, 0.0, delta]])).contiguous()
        )[0, 2]
        assert (above / below) == pytest.approx(eps_below / eps_above)

    def test_exactly_on_the_interface_takes_the_medium_above(self):
        """``z < 0`` is strict, so z = 0 is the upper medium — as everywhere else."""
        adapter = TwoMediumAdapter(ConstantMLP(), -18.3 + 0.55j, 1.0)
        fields = torch.view_as_complex(
            adapter(torch.tensor([[0.0, 0.0, 0.0]])).contiguous()
        )
        assert fields[0, 2] == pytest.approx(1.0 + 0j)

    def test_no_dtype_frozen_state_so_float64_coords_give_float64_out(self):
        """The float64 L-BFGS phase promotes the module; ε must follow, not resist."""
        adapter = TwoMediumAdapter(ConstantMLP(), -4 + 0.2j, 1.0)
        out = adapter(torch.tensor([[0.0, 0.0, -0.1]], dtype=torch.float64))
        assert out.dtype == torch.float64


class TestLayeredAdapter:
    """A three-layer stack: ε = 2 below −1, 5 in [−1, 1), 1 above."""

    BOUNDS = np.array([-1.0, 1.0])
    EPS = (2.0 + 0.0j, 5.0 + 0.5j, 1.0 + 0.0j)

    def adapter(self):
        return LayeredAdapter(ConstantMLP(), self.BOUNDS, self.EPS)

    def test_epsilon_matches_the_profile_layer_by_layer(self):
        z = torch.tensor([-3.0, -1.5, -0.5, 0.5, 1.5, 3.0], dtype=torch.float64)
        got = self.adapter().eps_zz(z, torch.complex128).numpy()
        want = np.array([self.EPS[0], self.EPS[0], self.EPS[1],
                         self.EPS[1], self.EPS[2], self.EPS[2]])
        assert np.allclose(got, want)

    def test_exactly_on_a_boundary_takes_the_medium_above(self):
        """``bucketize(right=True)`` matches ``src.transfer_matrix.layer_index_at``."""
        a = self.adapter()
        bounds = torch.as_tensor(self.BOUNDS, dtype=torch.float64)
        on = a.eps_zz(bounds, torch.complex128).numpy()
        above = a.eps_zz(bounds + 1e-9, torch.complex128).numpy()
        below = a.eps_zz(bounds - 1e-9, torch.complex128).numpy()
        assert np.allclose(on, above)
        assert not np.allclose(on, below)

    def test_ez_jumps_by_the_ratio_at_every_interface(self):
        """One construction fixes *all* the jumps, which is the N-layer claim."""
        a = self.adapter()
        delta = 1e-7
        for i, z0 in enumerate(self.BOUNDS):
            lo = torch.view_as_complex(
                a(torch.tensor([[0.0, 0.0, z0 - delta]], dtype=torch.float64))
                .contiguous()
            )[0, 2]
            hi = torch.view_as_complex(
                a(torch.tensor([[0.0, 0.0, z0 + delta]], dtype=torch.float64))
                .contiguous()
            )[0, 2]
            assert complex(hi / lo) == pytest.approx(self.EPS[i] / self.EPS[i + 1])

    def test_complex_epsilon_survives_a_float64_round_trip(self):
        """Regression: ``Module.to(float64)`` used to silently strip ``Im ε``.

        The table is held as a plain tensor rather than a registered buffer for
        exactly this reason — a lossy metal that came back lossless would take
        ``Im k_spp`` with it.
        """
        a = self.adapter()
        before = a.eps_values.clone()
        a.to(torch.float64)
        a.to(torch.float32)
        assert torch.equal(a.eps_values, before)
        assert a.eps_values.imag.abs().max() > 0

    def test_a_single_boundary_reduces_to_the_two_medium_case(self):
        eps_below, eps_above = -18.3 + 0.55j, 1.0
        layered = LayeredAdapter(ConstantMLP(), np.array([0.0]), (eps_below, eps_above))
        two = TwoMediumAdapter(ConstantMLP(), eps_below, eps_above)
        coords = torch.tensor([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]])
        assert torch.allclose(layered(coords), two(coords))


# =============================================================== scaling
class TestLinearFeature:
    def test_endpoints_map_to_minus_one_and_one(self):
        f = LinearFeature(2.0, 6.0)
        assert f.to_hat(2.0) == pytest.approx(-1.0)
        assert f.to_hat(6.0) == pytest.approx(1.0)
        assert f.to_hat(4.0) == pytest.approx(0.0)

    def test_round_trip_is_the_identity(self):
        f = LinearFeature(0.15, 0.40)
        for v in (0.15, 0.2, 0.31, 0.40):
            assert f.from_hat(f.to_hat(v)) == pytest.approx(v)

    def test_extrapolates_outside_the_range_rather_than_clamping(self):
        """Off-band ω has to stay distinguishable: a clamp would alias it."""
        f = LinearFeature(-1.0, 1.0)
        assert f.to_hat(3.0) == pytest.approx(3.0)

    def test_centred_holds_mid_and_half_span_exactly(self):
        """``centred`` must not re-derive the midpoint from the endpoints.

        The round trip through ``mid ± half_span`` moves the feature by an ULP,
        which is invisible physically but would break a bit-for-bit
        reproducibility claim about a published run.
        """
        mid, half = 2.9757891e15, 0.15 * 2.9757891e15
        f = LinearFeature.centred(mid, half)
        assert f.mid == mid
        assert f.half_span == half
        for x in (0.85, 1.0, 1.15):
            assert f.to_hat(x * mid) == (x * mid - mid) / half

    def test_torch_form_agrees_with_the_python_form(self):
        f = LinearFeature(1.0, 3.0)
        hats = torch.tensor([-1.0, -0.25, 0.0, 0.5, 1.0], dtype=torch.float32)
        got = f.from_hat_torch(hats)
        assert got.dtype == torch.float64  # ε(ω) needs the precision
        for h, g in zip(hats.tolist(), got.tolist(), strict=True):
            assert g == pytest.approx(f.from_hat(h))

    def test_torch_form_is_differentiable(self):
        """The adapter backpropagates through ε(ω̂), so this cannot be detached."""
        f = LinearFeature(1.0, 3.0)
        hat = torch.tensor([0.5], requires_grad=True, dtype=torch.float64)
        f.from_hat_torch(hat).sum().backward()
        assert hat.grad is not None
        assert hat.grad.item() == pytest.approx(f.half_span)

    @pytest.mark.parametrize("lo,hi", [(1.0, 1.0), (0.0, 0.0)])
    def test_a_degenerate_range_is_rejected(self, lo, hi):
        with pytest.raises(ValueError, match="non-degenerate"):
            LinearFeature(lo, hi)

    def test_centred_rejects_a_zero_half_span(self):
        with pytest.raises(ValueError, match="non-zero"):
            LinearFeature.centred(1.0, 0.0)

    def test_k0_is_omega_over_c(self):
        from src.constants import C0
        assert k0_of(3.0e15) == pytest.approx(3.0e15 / C0)
        # λ = 633 nm -> k₀ = 2π/λ
        omega = 2 * np.pi * C0 / 633e-9
        assert k0_of(omega) == pytest.approx(2 * np.pi / 633e-9, rel=1e-12)


# =============================================================== conditioning
class TestColumnConditionedNet:
    def test_condition_columns_are_appended_to_the_coordinates(self):
        core = TinyCore(in_features=5)
        coords = torch.randn(7, 3)
        cond = torch.randn(7, 2)
        wrapped = ColumnConditionedNet(core, cond)
        assert torch.allclose(wrapped(coords), core(torch.cat([coords, cond], dim=1)))

    def test_only_spatial_derivatives_can_be_formed(self):
        """``∂/∂ω`` appears nowhere in Maxwell's equations, so it must not be taken.

        The differentiated tensor is the 3-column ``coords``; the condition block
        enters inside the forward, after the operators have their handle on it.
        So the gradient the curl and divergence operators see is ``(N, 3)`` and
        an ``∂/∂ω`` term is not merely unused but unrepresentable.
        """
        core = TinyCore(in_features=4)
        coords = torch.randn(5, 3, requires_grad=True)
        cond = torch.randn(5, 1)  # as the experiments pass it: no grad
        out = ColumnConditionedNet(core, cond)(coords)
        (grad,) = torch.autograd.grad(out.sum(), coords)
        assert grad.shape == (5, 3)
        assert grad.abs().sum() > 0
        assert cond.grad is None

    def test_the_wrapper_contributes_no_parameters_of_its_own(self):
        """It must never widen a ``state_dict`` — checkpoints depend on that."""
        core = TinyCore(in_features=4)
        wrapped = ColumnConditionedNet(core, torch.zeros(3, 1))
        assert set(wrapped.state_dict()) == {f"core.{k}" for k in core.state_dict()}

    def test_the_condition_block_follows_the_coordinate_dtype(self):
        core = TinyCore(in_features=4).to(torch.float64)
        coords = torch.randn(4, 3, dtype=torch.float64)
        out = ColumnConditionedNet(core, torch.zeros(4, 1, dtype=torch.float32))(coords)
        assert out.dtype == torch.float64


class TestWeightedLosses:
    """With uniform weights the losses must reduce to their unweighted selves."""

    def batch(self, n=12, seed=0):
        torch.manual_seed(seed)
        core = TinyCore(in_features=4).to(torch.float64)
        coords = torch.randn(n, 3, dtype=torch.float64, requires_grad=True)
        cond = torch.randn(n, 1, dtype=torch.float64)
        eps = torch.eye(3, dtype=torch.complex128).expand(n, 3, 3).clone()
        return ColumnConditionedNet(core, cond), coords, eps

    @pytest.mark.parametrize("loss_fn", [weighted_curl_loss, weighted_divergence_loss])
    def test_unit_row_weights_reproduce_the_unweighted_loss(self, loss_fn):
        net, coords, eps = self.batch()
        plain = loss_fn(net, coords, eps).detach()
        weighted = loss_fn(net, coords, eps, torch.ones(coords.shape[0],
                                                        dtype=torch.float64))
        assert float(weighted.detach()) == pytest.approx(float(plain), rel=1e-12)

    @pytest.mark.parametrize("loss_fn", [weighted_curl_loss, weighted_divergence_loss])
    def test_a_uniform_row_weight_scales_the_loss_by_that_factor(self, loss_fn):
        net, coords, eps = self.batch()
        plain = loss_fn(net, coords, eps).detach()
        scaled = loss_fn(net, coords, eps,
                         torch.full((coords.shape[0],), 3.0, dtype=torch.float64))
        assert float(scaled.detach()) == pytest.approx(3.0 * float(plain), rel=1e-12)

    @pytest.mark.parametrize("loss_fn", [weighted_curl_loss, weighted_divergence_loss])
    def test_non_uniform_weights_change_the_loss(self, loss_fn):
        net, coords, eps = self.batch()
        w = torch.linspace(0.1, 5.0, coords.shape[0], dtype=torch.float64)
        plain_loss = loss_fn(net, coords, eps).detach()
        assert float(loss_fn(net, coords, eps, w).detach()) != float(plain_loss)

    @pytest.mark.parametrize("loss_fn", [weighted_curl_loss, weighted_divergence_loss])
    def test_the_losses_are_non_negative_and_differentiable(self, loss_fn):
        net, coords, eps = self.batch()
        loss = loss_fn(net, coords, eps)
        assert float(loss.detach()) >= 0.0
        loss.backward()
        assert any(p.grad is not None for p in net.parameters())


# =============================================================== ramp schedule
class TestRampSchedule:
    def test_the_first_epoch_already_carries_some_physics(self):
        """A zero-weight first step would waste an epoch fitting only the anchor."""
        assert ramp_at(0, 100, 0.25) == pytest.approx(1 / 25)

    def test_the_ramp_is_linear_up_to_the_fraction(self):
        for epoch in range(25):
            assert ramp_at(epoch, 100, 0.25) == pytest.approx((epoch + 1) / 25)

    def test_the_ramp_reaches_one_exactly_at_the_boundary_epoch(self):
        assert ramp_at(24, 100, 0.25) == pytest.approx(1.0)

    def test_the_ramp_is_clamped_at_one_afterwards(self):
        for epoch in (25, 50, 99, 10_000):
            assert ramp_at(epoch, 100, 0.25) == 1.0

    @pytest.mark.parametrize("frac", [0.0, 1e-9])
    def test_a_vanishing_fraction_means_full_physics_immediately(self, frac):
        """``max(1, ...)`` must guard the division; a zero fraction is 'no ramp'."""
        assert ramp_at(0, 100, frac) == 1.0

    def test_a_full_fraction_ramps_across_the_whole_run(self):
        assert ramp_at(0, 10, 1.0) == pytest.approx(0.1)
        assert ramp_at(9, 10, 1.0) == pytest.approx(1.0)

    def test_the_ramp_never_exceeds_one(self):
        for n in (1, 7, 100):
            for frac in (0.0, 0.25, 0.5, 1.0):
                assert all(ramp_at(e, n, frac) <= 1.0 for e in range(n))


# ================================================================ checkpoints
class TestCheckpointRoundTrip:
    def test_write_then_load_recovers_the_weights_and_the_loss(self, tmp_path):
        torch.manual_seed(0)
        net = TinyPINN()
        path = tmp_path / "m.partial.pth"
        write_checkpoint(path, net.core.state_dict(), 1.25, "adam:0")

        other = TinyPINN()
        assert load_core_checkpoint(other, path) == pytest.approx(1.25)
        coords = torch.randn(8, 3)
        assert torch.allclose(net(coords), other(coords))

    def test_the_write_is_atomic_and_leaves_no_temporary_behind(self, tmp_path):
        """An hour-long CPU run must not be destroyed by a crash mid-``torch.save``."""
        path = tmp_path / "m.partial.pth"
        write_checkpoint(path, TinyPINN().core.state_dict(), 0.5, "adam:0")
        assert path.exists()
        assert list(tmp_path.iterdir()) == [path]

    def test_missing_parent_directories_are_created(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "m.pth"
        write_checkpoint(path, TinyPINN().core.state_dict(), 0.5, "adam:0")
        assert path.exists()

    def test_a_second_write_replaces_the_first(self, tmp_path):
        path = tmp_path / "m.pth"
        net = TinyPINN()
        write_checkpoint(path, net.core.state_dict(), 9.0, "adam:0")
        write_checkpoint(path, net.core.state_dict(), 0.1, "lbfgs:3")
        blob = torch.load(path, map_location="cpu", weights_only=False)
        assert blob["best_loss"] == pytest.approx(0.1)
        assert blob["phase"] == "lbfgs:3"

    def test_history_is_omitted_from_the_blob_when_not_supplied(self, tmp_path):
        """The non-chunked experiments' checkpoints keep their original shape."""
        path = tmp_path / "m.pth"
        write_checkpoint(path, TinyPINN().core.state_dict(), 1.0, "adam:0")
        blob = torch.load(path, map_location="cpu", weights_only=False)
        assert "history" not in blob
        assert load_checkpoint_history(path) is None

    def test_history_round_trips_when_supplied(self, tmp_path):
        path = tmp_path / "m.pth"
        history = {"epoch": [0, 1, 2], "total": [3.0, 2.0, 1.0]}
        write_checkpoint(path, TinyPINN().core.state_dict(), 1.0, "final", history)
        assert load_checkpoint_history(path) == history

    def test_a_float64_state_dict_round_trips(self, tmp_path):
        """The best iterate is often captured mid-L-BFGS, in double precision."""
        path = tmp_path / "m.pth"
        net = TinyPINN().to(torch.float64)
        write_checkpoint(path, net.core.state_dict(), 1.0, "lbfgs:0")
        blob = torch.load(path, map_location="cpu", weights_only=False)
        assert all(v.dtype == torch.float64 for v in blob["state_dict"].values())

    def test_a_checkpoint_without_a_recorded_loss_reports_nan(self, tmp_path):
        path = tmp_path / "m.pth"
        torch.save({"state_dict": TinyPINN().core.state_dict()}, path)
        assert math.isnan(load_core_checkpoint(TinyPINN(), path))


# ============================================================== training loop
def _make_problem(in_features=3):
    """A trivial least-squares problem in the shape ``run_training`` expects."""
    network = TinyPINN(in_features)
    target = torch.zeros(16, 6, 2)

    def sample_batch(n_int, n_bc, n_if, dtype):
        return {
            "coords": torch.randn(16, in_features, dtype=dtype),
            "sizes": (n_int, n_bc, n_if),
            "dtype": dtype,
        }

    def compute_losses(batch, ramp=1.0, scale=1.0):
        pred = network.core(batch["coords"])
        mse = torch.mean((pred - target.to(pred.dtype)) ** 2)
        parts = [mse * f for f in (1.0, 0.5, 0.25, 0.125)]
        return ramp * scale * mse, *parts

    return network, sample_batch, compute_losses


class TestRunTraining:
    def config(self, **kw):
        base = dict(n_epochs=6, n_points=8, n_boundary=4, n_interface=2,
                    learning_rate=1e-2, physics_ramp_frac=0.5, log_every=10 ** 9)
        base.update(kw)
        return TrainingConfig(**base)

    def test_history_carries_exactly_the_seven_standard_keys(self, caplog):
        import logging
        net, sample, losses = _make_problem()
        _, history = run_training(net, self.config(), sample, losses,
                                  logging.getLogger("t"))
        assert set(history) == set(HISTORY_KEYS)
        assert len(history["total"]) == 6
        assert history["epoch"] == list(range(6))

    def test_wall_clock_is_recorded_only_when_asked_for(self):
        import logging
        net, sample, losses = _make_problem()
        _, h = run_training(net, self.config(), sample, losses,
                            logging.getLogger("t"), track_wall=True)
        assert "wall_s" in h
        assert h["wall_s"] == sorted(h["wall_s"])

    def test_the_loss_descends(self):
        import logging
        torch.manual_seed(0)
        net, sample, losses = _make_problem()
        # No ramp: with one, the *recorded* total rises while the ramp does,
        # which says nothing about whether the optimiser is working.
        _, h = run_training(net, self.config(n_epochs=40, physics_ramp_frac=0.0),
                            sample, losses, logging.getLogger("t"))
        assert np.mean(h["total"][-10:]) < np.mean(h["total"][:10])

    def test_the_ramp_is_applied_to_the_objective(self):
        """Early epochs are scaled down; the logged components are not."""
        import logging
        torch.manual_seed(0)
        net, sample, losses = _make_problem()
        _, h = run_training(net, self.config(n_epochs=4, physics_ramp_frac=1.0),
                            sample, losses, logging.getLogger("t"))
        # ``curl`` is the unramped mse and ``total`` is ramp * mse.
        for epoch, (total, curl) in enumerate(zip(h["total"], h["curl"], strict=True)):
            assert total == pytest.approx(ramp_at(epoch, 4, 1.0) * curl)

    def test_the_lbfgs_phase_appends_its_steps_and_restores_float32(self):
        import logging
        torch.manual_seed(0)
        net, sample, losses = _make_problem()
        _, h = run_training(net, self.config(lbfgs_steps=3), sample, losses,
                            logging.getLogger("t"),
                            lbfgs_loss_kwargs={"scale": 0.5})
        assert len(h["total"]) == 9  # 6 Adam + 3 L-BFGS
        assert h["epoch"] == list(range(9))
        assert all(math.isnan(x) for x in h["lr"][-3:])  # no LR during L-BFGS
        assert all(p.dtype == torch.float32 for p in net.parameters())

    def test_a_separate_lbfgs_sampler_is_used_for_phase_two_only(self):
        """The band sweeps pin L-BFGS to fixed frequencies; Adam resamples."""
        import logging
        net, sample, losses = _make_problem()
        seen = []

        def lbfgs_sample(n_int, n_bc, n_if, dtype):
            seen.append((n_int, n_bc, n_if, dtype))
            return sample(n_int, n_bc, n_if, dtype)

        cfg = self.config(lbfgs_steps=1, lbfgs_points_factor=2)
        run_training(net, cfg, sample, losses, logging.getLogger("t"),
                     lbfgs_sample_batch=lbfgs_sample)
        assert seen == [(16, 8, 4, torch.float64)]

    def test_the_best_iterate_is_restored_not_the_last(self):
        """A diverging tail must not be what gets saved."""
        import logging
        net = TinyPINN()
        calls = {"n": 0}
        losses_seen = [5.0, 1.0, 7.0, 9.0]

        def sample_batch(n_int, n_bc, n_if, dtype):
            return None

        def compute_losses(batch, ramp=1.0, **kw):
            value = losses_seen[calls["n"] % len(losses_seen)]
            calls["n"] += 1
            # Depends on the parameters so ``backward`` has something to do.
            loss = value + 0.0 * sum(p.sum() for p in net.parameters())
            return loss, loss, loss, loss, loss

        cfg = self.config(n_epochs=4, physics_ramp_frac=0.0)
        _, h = run_training(net, cfg, sample_batch, compute_losses,
                            logging.getLogger("t"))
        assert h["total"] == pytest.approx(losses_seen)

    def test_a_partially_ramped_epoch_never_becomes_the_best_iterate(self, tmp_path):
        """Its total is not comparable to the full objective, so it must not win."""
        import logging
        net = TinyPINN()
        # 4 epochs at frac 0.5 -> the ramp completes at epoch 1. Epoch 0's total
        # is the smallest of the four, but it is the only partially ramped one,
        # so the best iterate must be epoch 3's 0.8 instead.
        seq = [0.1, 0.9, 0.85, 0.8]
        assert ramp_at(0, 4, 0.5) < 1.0 and ramp_at(1, 4, 0.5) == 1.0

        def compute_losses(batch, ramp=1.0, **kw):
            loss = seq[min(len(seq) - 1, compute_losses.i)] + 0.0 * sum(
                p.sum() for p in net.parameters()
            )
            compute_losses.i += 1
            return loss, loss, loss, loss, loss
        compute_losses.i = 0

        ck = tmp_path / "m.partial.pth"
        cfg = self.config(n_epochs=4, physics_ramp_frac=0.5, log_every=1)
        run_training(net, cfg, lambda *a: None, compute_losses,
                     logging.getLogger("t"), checkpoint_path=ck,
                     final_checkpoint=True)
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        assert blob["best_loss"] == pytest.approx(0.8)  # not the ramped 0.1

    def test_no_checkpoint_is_written_when_no_path_is_given(self, tmp_path):
        import logging
        net, sample, losses = _make_problem()
        run_training(net, self.config(log_every=1), sample, losses,
                     logging.getLogger("t"))
        assert list(tmp_path.iterdir()) == []

    def test_resume_continues_the_epoch_numbering_and_the_curve(self, tmp_path):
        import logging
        torch.manual_seed(3)
        net, sample, losses = _make_problem()
        log = logging.getLogger("t")
        ck = tmp_path / "m.partial.pth"
        _, h1 = run_training(net, self.config(n_epochs=5, log_every=1), sample,
                             losses, log, checkpoint_path=ck, save_history=True,
                             track_wall=True, final_checkpoint=True)
        stored = load_checkpoint_history(ck)
        assert stored is not None and stored["epoch"] == list(range(5))

        net2, sample2, losses2 = _make_problem()
        best = load_core_checkpoint(net2, ck)
        _, h2 = run_training(net2, self.config(n_epochs=4, log_every=1), sample2,
                             losses2, log, checkpoint_path=ck, save_history=True,
                             track_wall=True, initial_history=stored,
                             initial_best_loss=best)
        assert h2["epoch"] == list(range(9))
        assert h2["total"][:5] == h1["total"]
        assert h2["wall_s"] == sorted(h2["wall_s"])

    def test_a_resumed_chunk_cannot_checkpoint_a_worse_iterate(self, tmp_path):
        """Its cosine schedule restarts at the full LR, so its early epochs are worse."""
        import logging
        net = TinyPINN()

        def compute_losses(batch, ramp=1.0, **kw):
            loss = 42.0 + 0.0 * sum(p.sum() for p in net.parameters())
            return loss, loss, loss, loss, loss

        ck = tmp_path / "m.partial.pth"
        cfg = self.config(n_epochs=3, physics_ramp_frac=0.0, log_every=1)
        run_training(net, cfg, lambda *a: None, compute_losses,
                     logging.getLogger("t"), checkpoint_path=ck,
                     initial_best_loss=0.001, final_checkpoint=True)
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        assert blob["best_loss"] == pytest.approx(0.001)

    def test_the_final_checkpoint_is_written_even_without_an_improvement(self, tmp_path):
        """Otherwise a resumed chunk's slice of the history would be lost."""
        import logging
        net, sample, losses = _make_problem()
        ck = tmp_path / "m.partial.pth"
        run_training(net, self.config(), sample, losses, logging.getLogger("t"),
                     checkpoint_path=ck, save_history=True, final_checkpoint=True)
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        assert blob["phase"] == "final"
        assert len(blob["history"]["epoch"]) == 6

    def test_a_non_finite_lbfgs_loss_stops_the_refinement(self):
        import logging
        net = TinyPINN()

        def compute_losses(batch, ramp=1.0, **kw):
            loss = torch.tensor(float("nan")) + 0.0 * sum(
                p.sum() for p in net.parameters()
            )
            return loss, loss, loss, loss, loss

        cfg = self.config(n_epochs=0, lbfgs_steps=5, physics_ramp_frac=0.0)
        _, h = run_training(net, cfg, lambda *a: None, compute_losses,
                            logging.getLogger("t"))
        assert len(h["total"]) == 1  # stopped after the first bad step


# ================================================================== reporting
class TestSuccessTiers:
    BASE = {"worst_rel_l2": 0.0, "worst_k_spp_rel_error": 0.0,
            "bound_mode_everywhere": 1.0}
    TIERS = dict(stretch=(0.02, 0.002), target=(0.1, 0.01))

    def tier(self, **over):
        return banded_success_tier({**self.BASE, **over}, **self.TIERS)

    def test_a_perfect_sweep_is_stretch(self):
        assert self.tier() == "stretch"

    @pytest.mark.parametrize("rel,k,want", [
        (0.019, 0.0019, "stretch"),
        (0.02, 0.0019, "target"),    # rel exactly on the stretch bound
        (0.019, 0.002, "target"),    # k exactly on the stretch bound
        (0.099, 0.0099, "target"),
        (0.1, 0.0099, "minimum"),    # rel exactly on the target bound
        (0.099, 0.01, "minimum"),    # k exactly on the target bound
        (0.499, 1.0, "minimum"),     # the minimum tier ignores k entirely
        (0.5, 0.0, "not met"),       # rel exactly on the minimum bound
        (0.6, 0.0, "not met"),
    ])
    def test_the_thresholds_are_strict_inequalities(self, rel, k, want):
        assert self.tier(worst_rel_l2=rel, worst_k_spp_rel_error=k) == want

    @pytest.mark.parametrize("rel,k", [(0.0, 0.0), (0.01, 0.001), (0.4, 0.05)])
    def test_losing_the_mode_anywhere_fails_every_tier(self, rel, k):
        """A sweep that lost the mode at one corner has not validated the band."""
        assert self.tier(worst_rel_l2=rel, worst_k_spp_rel_error=k,
                         bound_mode_everywhere=0.0) == "not met"

    def test_a_different_experiments_thresholds_are_honoured(self):
        """The surrogate grades itself more leniently than the dispersion runs."""
        summary = {**self.BASE, "worst_rel_l2": 0.025,
                   "worst_k_spp_rel_error": 0.004}
        assert banded_success_tier(summary, stretch=(0.02, 0.002),
                                   target=(0.1, 0.01)) == "target"
        assert banded_success_tier(summary, stretch=(0.03, 0.005),
                                   target=(0.1, 0.01)) == "stretch"

    def test_a_custom_minimum_ceiling_is_honoured(self):
        assert self.tier(worst_rel_l2=0.3, worst_k_spp_rel_error=1.0) == "minimum"
        assert banded_success_tier(
            {**self.BASE, "worst_rel_l2": 0.3, "worst_k_spp_rel_error": 1.0},
            **self.TIERS, minimum_rel=0.2,
        ) == "not met"

    def test_a_missing_summary_key_is_an_error_not_a_silent_pass(self):
        with pytest.raises(KeyError):
            banded_success_tier({"worst_rel_l2": 0.0}, **self.TIERS)


class TestRelativeL2:
    def test_an_exact_match_is_zero(self):
        x = torch.randn(20, 3)
        assert relative_l2(x, x) == pytest.approx(0.0)

    def test_it_is_the_norm_ratio(self):
        ref = torch.ones(4)
        pred = torch.ones(4) * 1.5
        assert relative_l2(pred, ref) == pytest.approx(0.5)

    def test_a_zero_reference_gives_a_finite_number_not_nan(self):
        """A nan here would poison a ``max`` over a whole sweep, silently."""
        got = relative_l2(torch.ones(3), torch.zeros(3))
        assert math.isfinite(got) and got > 0

    def test_it_works_on_complex_tensors(self):
        ref = torch.tensor([1 + 1j, 2 - 1j])
        assert relative_l2(ref, ref) == pytest.approx(0.0)


class TestJsonReport:
    def test_sections_round_trip_in_order(self, tmp_path):
        path = tmp_path / "metrics.json"
        sections = {"summary": {"a": 1}, "figures": {"f": "x.png"},
                    "run_info": {"seed": 0}}
        write_json_report(path, sections)
        with open(path) as fh:
            assert json.load(fh) == sections
        assert list(json.loads(path.read_text())) == list(sections)

    def test_missing_parent_directories_are_created(self, tmp_path):
        path = tmp_path / "figs" / "metrics.json"
        write_json_report(path, {"a": 1})
        assert path.exists()

    def test_the_file_is_indented_so_runs_can_be_diffed(self, tmp_path):
        path = tmp_path / "m.json"
        write_json_report(path, {"a": {"b": 1}})
        assert "\n  " in path.read_text()

    def test_a_rewrite_replaces_rather_than_appends(self, tmp_path):
        path = tmp_path / "m.json"
        write_json_report(path, {"a": 1, "b": 2})
        write_json_report(path, {"a": 3})
        assert json.loads(path.read_text()) == {"a": 3}


class TestMeasurement:
    """A network that *is* an exact decaying SPP-like mode, measured."""

    KAPPA_D, KAPPA_M = 2.0e7, 5.0e7  # 1/m

    class ExactMode(nn.Module):
        """``H_y = e^{-κ_d z}`` above, ``e^{+κ_m z}`` below; everything else 0.

        Continuous at z = 0 and decaying on both sides — so the continuity
        residual is ~0 and the decay fits must recover κ exactly.
        """

        def __init__(self, kappa_d, kappa_m):
            super().__init__()
            self.kappa_d, self.kappa_m = kappa_d, kappa_m

        def forward(self, coords):
            z = coords[:, 2]
            hy = torch.where(z >= 0, torch.exp(-self.kappa_d * z),
                             torch.exp(self.kappa_m * z))
            out = torch.zeros(coords.shape[0], 6, 2, dtype=coords.dtype)
            out[:, 4, 0] = hy  # H_y, real part
            out[:, 0, 0] = hy  # E_x, so E has a scale too
            return out

    def mode(self):
        return self.ExactMode(self.KAPPA_D, self.KAPPA_M)

    def test_the_decay_fits_recover_kappa_on_both_sides(self):
        got = fit_decay_constants(
            self.mode(), self.KAPPA_D, self.KAPPA_M,
            x=0.0, y=0.0, z_min=-1e-7, z_max=2e-7, guard=1e-9,
        )
        assert got["kappa_d_fit"] == pytest.approx(self.KAPPA_D, rel=1e-6)
        assert got["kappa_m_fit"] == pytest.approx(self.KAPPA_M, rel=1e-6)
        assert got["kappa_d_fit_rel_error"] < 1e-6
        assert got["kappa_m_fit_rel_error"] < 1e-6

    def test_the_analytical_reference_is_echoed_back(self):
        got = fit_decay_constants(
            self.mode(), self.KAPPA_D, self.KAPPA_M,
            x=0.0, y=0.0, z_min=-1e-7, z_max=2e-7, guard=1e-9,
        )
        assert got["kappa_d_analytical"] == pytest.approx(self.KAPPA_D)
        assert got["kappa_m_analytical"] == pytest.approx(self.KAPPA_M)

    def test_a_bound_mode_has_both_decay_signs_correct(self):
        got = fit_decay_constants(
            self.mode(), self.KAPPA_D, self.KAPPA_M,
            x=0.0, y=0.0, z_min=-1e-7, z_max=2e-7, guard=1e-9,
        )
        assert got["decay_sign_correct_air"] == 1.0
        assert got["decay_sign_correct_metal"] == 1.0

    def test_a_growing_field_is_flagged_as_the_wrong_branch(self):
        """The sign gate is the whole point: a radiative solution must not pass.

        A small relative error alone would not reveal it, so ``decay_sign_correct``
        is reported separately and the success tiers gate on it.
        """
        unbound = self.ExactMode(-self.KAPPA_D, self.KAPPA_M)  # grows into the air
        got = fit_decay_constants(
            unbound, self.KAPPA_D, self.KAPPA_M,
            x=0.0, y=0.0, z_min=-1e-7, z_max=2e-7, guard=1e-9,
        )
        assert got["decay_sign_correct_air"] == 0.0
        assert got["decay_sign_correct_metal"] == 1.0

    def test_a_continuous_field_has_a_near_zero_continuity_residual(self):
        n = 64
        coords = torch.stack([torch.linspace(0, 1e-6, n), torch.zeros(n),
                              torch.zeros(n)], dim=1)
        normals = torch.zeros(n, 3)
        normals[:, 2] = 1.0
        got = continuity_residuals(self.mode(), coords, normals, 1e-12)
        assert got["continuity_E_rel"] < 1e-4
        assert got["continuity_H_rel"] < 1e-4

    def test_a_discontinuous_tangential_field_is_caught(self):
        class Jump(nn.Module):
            def forward(self, coords):
                out = torch.zeros(coords.shape[0], 6, 2, dtype=coords.dtype)
                out[:, 0, 0] = torch.where(coords[:, 2] >= 0, 1.0, -1.0)  # E_x flips
                out[:, 4, 0] = 1.0
                return out

        n = 32
        coords = torch.stack([torch.linspace(0, 1e-6, n), torch.zeros(n),
                              torch.zeros(n)], dim=1)
        normals = torch.zeros(n, 3)
        normals[:, 2] = 1.0
        got = continuity_residuals(Jump(), coords, normals, 1e-9)
        assert got["continuity_E_rel"] == pytest.approx(2.0, rel=1e-6)
        assert got["continuity_H_rel"] < 1e-9

    def test_a_zero_field_gives_a_finite_residual_not_a_division_by_zero(self):
        class Zero(nn.Module):
            def forward(self, coords):
                return torch.zeros(coords.shape[0], 6, 2, dtype=coords.dtype)

        n = 8
        coords = torch.zeros(n, 3)
        normals = torch.zeros(n, 3)
        normals[:, 2] = 1.0
        got = continuity_residuals(Zero(), coords, normals, 1e-9)
        assert all(math.isfinite(v) for v in got.values())

    def test_the_measurements_take_no_gradients(self):
        """Both run under ``no_grad``: they are diagnostics, not part of a loss."""
        net = TinyPINN()
        n = 16
        coords = torch.zeros(n, 3, requires_grad=True)
        normals = torch.zeros(n, 3)
        normals[:, 2] = 1.0
        continuity_residuals(net, coords, normals, 1e-9)
        assert all(p.grad is None for p in net.parameters())


class TestTwoPhaseHistoryPlot:
    def history(self, n_adam=20, n_lbfgs=4):
        h = {"epoch": list(range(n_adam + n_lbfgs)),
             "lr": [1e-3] * n_adam + [float("nan")] * n_lbfgs}
        for k in ("total", "curl", "div", "continuity", "boundary"):
            h[k] = [10.0 ** (-i / 10) for i in range(n_adam + n_lbfgs)]
        return h

    def test_a_two_phase_run_gets_two_panels(self, tmp_path):
        import matplotlib.pyplot as plt
        path = plot_two_phase_history(self.history(), tmp_path, "t")
        assert Path(path).exists()
        assert plt.get_fignums() == []  # the figure is closed, not leaked

    @pytest.mark.parametrize("n_adam,n_lbfgs", [(20, 0), (0, 5)],
                             ids=["adam-only", "lbfgs-only"])
    def test_a_single_phase_run_still_produces_a_figure(self, tmp_path, n_adam,
                                                        n_lbfgs):
        """A ``--quick`` smoke run has no L-BFGS phase and must not crash."""
        path = plot_two_phase_history(self.history(n_adam, n_lbfgs), tmp_path, "t")
        assert Path(path).exists() and Path(path).stat().st_size > 0

    def test_missing_parent_directories_are_created(self, tmp_path):
        path = plot_two_phase_history(self.history(), tmp_path / "deep" / "figs", "t")
        assert Path(path).exists()

    def test_the_filename_is_configurable(self, tmp_path):
        path = plot_two_phase_history(self.history(), tmp_path, "t",
                                      filename="curve.png")
        assert Path(path).name == "curve.png"

    def test_marking_restarts_changes_the_figure(self, tmp_path):
        """Only when there are restarts to mark — otherwise it is a no-op."""
        import hashlib
        h = self.history(20, 4)
        h["lr"] = [1e-3 * (1.0 - (i % 10) / 10) + 1e-6 for i in range(20)] + \
                  [float("nan")] * 4

        def sha(p):
            return hashlib.sha256(Path(p).read_bytes()).hexdigest()

        plain = sha(plot_two_phase_history(h, tmp_path / "a", "t"))
        marked = sha(plot_two_phase_history(h, tmp_path / "b", "t",
                                            mark_restarts=True))
        assert plain != marked
        # A monotonically decreasing LR has no restarts, so the flag does nothing.
        flat = self.history(20, 4)
        assert sha(plot_two_phase_history(flat, tmp_path / "c", "t")) == \
            sha(plot_two_phase_history(flat, tmp_path / "d", "t", mark_restarts=True))


class TestCommonArgs:
    def parser(self, **kw):
        p = argparse.ArgumentParser()
        base = dict(epochs=4000, n_points=2048, lr=1e-3, device="cpu",
                    lbfgs_steps=50, quick_epochs=200, figures_dir="figs",
                    model_out="m.pth")
        base.update(kw)
        add_common_args(p, **base)
        return p

    def test_the_defaults_come_from_the_experiments_constants(self):
        args = self.parser().parse_args([])
        assert (args.epochs, args.n_points, args.lr) == (4000, 2048, 1e-3)
        assert args.lbfgs_steps == 50
        assert args.seed == 0
        assert args.lbfgs_dtype == "float64"
        assert not args.quick and not args.resume

    def test_every_option_can_be_overridden(self):
        args = self.parser().parse_args(
            ["--epochs", "10", "--n-points", "32", "--lr", "0.5", "--seed", "7",
             "--device", "cuda", "--lbfgs-steps", "0", "--lbfgs-dtype", "float32",
             "--resume", "--quick"]
        )
        assert (args.epochs, args.n_points, args.lr, args.seed) == (10, 32, 0.5, 7)
        assert args.device == "cuda" and args.lbfgs_dtype == "float32"
        assert args.lbfgs_steps == 0 and args.resume and args.quick

    def test_an_unknown_lbfgs_dtype_is_rejected(self):
        with pytest.raises(SystemExit):
            self.parser().parse_args(["--lbfgs-dtype", "float16"])

    def test_resume_can_be_withheld_for_the_short_experiment(self):
        p = argparse.ArgumentParser()
        add_core_args(p, epochs=1, n_points=1, lr=1.0, device="cpu", lbfgs_steps=0)
        add_output_args(p, quick_epochs=10, figures_dir="f", model_out=None,
                        resume=False)
        assert not hasattr(p.parse_args([]), "resume")
        with pytest.raises(SystemExit):
            p.parse_args(["--resume"])

    def test_paths_are_parsed_as_paths(self, tmp_path):
        from pathlib import Path
        args = self.parser().parse_args(["--figures-dir", str(tmp_path),
                                         "--model-out", str(tmp_path / "x.pth")])
        assert isinstance(args.figures_dir, Path)
        assert isinstance(args.model_out, Path)

    def test_the_experiments_own_options_sit_between_the_shared_blocks(self):
        """Ordering contract, so ``--help`` reads the same across experiments."""
        p = argparse.ArgumentParser()
        add_core_args(p, epochs=1, n_points=1, lr=1.0, device="cpu", lbfgs_steps=0)
        p.add_argument("--band-fraction", type=float, default=1.0)
        add_output_args(p, quick_epochs=10, figures_dir="f", model_out="m.pth")
        options = [a.option_strings[0] for a in p._actions if a.option_strings]
        assert options.index("--lbfgs-dtype") < options.index("--band-fraction")
        assert options.index("--band-fraction") < options.index("--resume")

    def test_the_quick_help_names_the_experiments_epoch_count(self):
        p = self.parser(quick_epochs=137)
        quick = next(a for a in p._actions if a.dest == "quick")
        assert "137 epochs" in quick.help

    def test_extra_quick_work_can_be_named(self):
        p = argparse.ArgumentParser()
        add_output_args(p, quick_epochs=10, figures_dir="f", model_out="m.pth",
                        quick_extra=", tiny probe")
        quick = next(a for a in p._actions if a.dest == "quick")
        assert quick.help.endswith(", tiny probe")
