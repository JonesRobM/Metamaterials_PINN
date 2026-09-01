"""Tests for examples/ablation_study.py.

The point of an ablation study is that each arm differs from the control in
*exactly* the intended way, so most of what is worth testing here is the
configuration rather than the physics: that ``no_adapter`` really has no
adapter, that ``no_anchor``'s objective really contains no anchor term, that
``no_ramp``'s ramp really is 1 at epoch 0, and that ``uniform_weights`` really
weights the two half-spaces alike. Several of those are checked against the
*recorded loss history* rather than against the condition dataclass, because the
dataclass says what was intended and the history says what was optimised.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from examples import ablation_study as ab
from examples import validate_spp as vs
from src.experiments import ramp_at

DEVICE = torch.device("cpu")

#: Small enough to train in a test, same shape as the real thing.
TINY = {"hidden_dims": (16, 16), "fourier_modes": 8}

#: The four fields that define an arm; everything else is naming.
BEHAVIOURAL_FIELDS = (
    "use_adapter",
    "boundary_weight",
    "physics_ramp_frac",
    "per_medium_weighting",
)

#: Which field each ablation is allowed to move.
EXPECTED_CHANGE = {
    "no_adapter": "use_adapter",
    "no_anchor": "boundary_weight",
    "no_ramp": "physics_ramp_frac",
    "uniform_weights": "per_medium_weighting",
}


def _tiny_network(condition):
    torch.manual_seed(0)
    return ab.build_network(condition, device=DEVICE, **TINY)


# --------------------------------------------------------------- the registry
def test_registry_has_five_arms_with_control_first():
    assert list(ab.CONDITIONS) == [
        "control", "no_adapter", "no_anchor", "no_ramp", "uniform_weights"
    ]
    assert next(iter(ab.CONDITIONS.values())).key == "control"
    # Every arm has a human label and a one-line statement of what moved
    for c in ab.CONDITIONS.values():
        assert c.label and c.changed


def test_control_is_exactly_the_validate_spp_recipe():
    """The control must track validate_spp.py's constants, not a copy of them."""
    c = ab.CONDITIONS["control"]
    assert c.use_adapter is True
    assert c.boundary_weight == vs.BOUNDARY_WEIGHT
    assert c.physics_ramp_frac == vs.PHYSICS_RAMP_FRAC
    assert c.per_medium_weighting is True
    assert c.metal_curl_weights() == (vs.METAL_CURL_WEIGHT_ADAM, vs.METAL_CURL_WEIGHT_LBFGS)
    assert c.metal_div_weight() == vs.METAL_DIV_WEIGHT


@pytest.mark.parametrize("key", sorted(EXPECTED_CHANGE))
def test_each_arm_differs_from_the_control_in_exactly_one_field(key):
    control = dataclasses.asdict(ab.CONDITIONS["control"])
    arm = dataclasses.asdict(ab.CONDITIONS[key])
    differing = [f for f in BEHAVIOURAL_FIELDS if arm[f] != control[f]]
    assert differing == [EXPECTED_CHANGE[key]], (
        f"{key} should move only {EXPECTED_CHANGE[key]}, moved {differing}"
    )


def test_the_ablated_values_are_the_intended_ones():
    assert ab.CONDITIONS["no_adapter"].use_adapter is False
    assert ab.CONDITIONS["no_anchor"].boundary_weight == 0.0
    assert ab.CONDITIONS["no_ramp"].physics_ramp_frac == 0.0
    assert ab.CONDITIONS["uniform_weights"].per_medium_weighting is False


def test_no_ramp_means_full_physics_weight_from_epoch_zero():
    """``physics_ramp_frac = 0`` must actually disable the ramp, not shorten it."""
    n = ab.N_EPOCHS
    assert ramp_at(0, n, ab.CONDITIONS["no_ramp"].physics_ramp_frac) == 1.0
    # while the control still starts far below full weight
    assert ramp_at(0, n, ab.CONDITIONS["control"].physics_ramp_frac) < 0.01


def test_uniform_weights_drops_the_per_medium_preconditioner():
    uniform = ab.CONDITIONS["uniform_weights"]
    assert uniform.metal_curl_weights() == (1.0, 1.0)
    assert uniform.metal_div_weight() == 1.0
    # and the control's really is a preconditioner, i.e. not 1
    control = ab.CONDITIONS["control"]
    assert control.metal_curl_weights()[0] < 0.1
    assert control.metal_div_weight() < 0.1


def test_resolve_conditions():
    assert [c.key for c in ab.resolve_conditions("all")] == list(ab.CONDITIONS)
    assert [c.key for c in ab.resolve_conditions("control, no_ramp")] == [
        "control", "no_ramp"
    ]
    with pytest.raises(ValueError):
        ab.resolve_conditions("control,gold")


# ---------------------------------------------------------------- the network
def test_adapter_present_only_in_the_arms_that_should_have_it():
    for key, condition in ab.CONDITIONS.items():
        network = _tiny_network(condition)
        assert ab.has_adapter(network) is condition.use_adapter, key
        assert isinstance(network, vs.SPPPINN)


def test_no_adapter_arm_predicts_ez_directly():
    """With the adapter gone, channel 2 passes through; with it, it is divided by eps_zz."""
    coords = torch.tensor([[0.0, 0.0, -10e-9], [0.0, 0.0, 10e-9]], dtype=torch.float32)
    bare = _tiny_network(ab.CONDITIONS["no_adapter"])
    wrapped = _tiny_network(ab.CONDITIONS["control"])
    # Same seed, same MLP: the only difference is the adapter around it.
    mlp_out = bare.core(coords / vs.LAMBDA0)
    adapted = wrapped.core(coords / vs.LAMBDA0)
    mlp_ez = torch.complex(mlp_out[:, 2, 0], mlp_out[:, 2, 1])
    adapted_ez = torch.complex(adapted[:, 2, 0], adapted[:, 2, 1])
    eps = torch.tensor(
        [complex(vs.EPS_METAL_N), complex(vs.EPS_DIEL)], dtype=torch.complex64
    )
    assert torch.allclose(adapted_ez, mlp_ez / eps, rtol=1e-5, atol=1e-7)
    # Every other channel is untouched by the adapter
    other = [0, 1, 3, 4, 5]
    assert torch.allclose(adapted[:, other], mlp_out[:, other], rtol=1e-6, atol=1e-8)


def test_all_arms_start_from_identical_weights():
    """The adapter adds no parameters, so seeding gives every arm the same start."""
    ref = _tiny_network(ab.CONDITIONS["control"]).core.mlp.state_dict()
    for condition in ab.CONDITIONS.values():
        network = _tiny_network(condition)
        mlp = network.core.mlp if condition.use_adapter else network.core
        got = mlp.state_dict()
        assert set(got) == set(ref)
        for k in ref:
            assert torch.equal(got[k], ref[k]), f"{condition.key}/{k}"


# ------------------------------------------------- the objective, as optimised
def _one_epoch(condition, seed=0):
    """One Adam epoch; returns the recorded (total, curl, div, cont, bc)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    network = ab.build_network(condition, device=DEVICE, **TINY)
    torch.manual_seed(seed)
    np.random.seed(seed)
    _, history = ab.train_condition(
        network, condition, n_epochs=1, n_points=64, lbfgs_steps=0,
        device=DEVICE, log_every=1000, checkpoint_path=None,
    )
    return {k: history[k][0] for k in ("total", "curl", "div", "continuity", "boundary")}


@pytest.mark.parametrize("key", ["control", "no_anchor", "no_ramp"])
def test_recorded_total_decomposes_with_this_arms_weights(key):
    """
    The logged total must equal ``ramp·(curl + div + cont) + w_bc·bc`` for *this*
    arm's ramp and anchor weight — which is what says the ablation reached the
    objective rather than only the dataclass.
    """
    condition = ab.CONDITIONS[key]
    h = _one_epoch(condition)
    ramp = ramp_at(0, 1, condition.physics_ramp_frac)
    expected = (
        ramp * (h["curl"] + vs.DIVERGENCE_WEIGHT * h["div"]
                + vs.CONTINUITY_WEIGHT * h["continuity"])
        + condition.boundary_weight * h["boundary"]
    )
    assert h["total"] == pytest.approx(expected, rel=1e-5)


def test_no_anchor_total_contains_no_anchor_contribution():
    """The anchor is still *measured* at weight 0, and still excluded from the total."""
    h = ab.CONDITIONS["no_anchor"], _one_epoch(ab.CONDITIONS["no_anchor"])
    _, values = h
    assert values["boundary"] > 0  # the diagnostic is recorded
    ramp = ramp_at(0, 1, ab.CONDITIONS["no_anchor"].physics_ramp_frac)
    physics_only = ramp * (
        values["curl"] + vs.DIVERGENCE_WEIGHT * values["div"]
        + vs.CONTINUITY_WEIGHT * values["continuity"]
    )
    assert values["total"] == pytest.approx(physics_only, rel=1e-5)
    # ... and the control's total is dominated by the anchor it does carry
    control = _one_epoch(ab.CONDITIONS["control"])
    assert control["total"] > vs.BOUNDARY_WEIGHT * control["boundary"] * 0.99


def test_uniform_weighting_raises_the_recorded_curl_term():
    """
    Dropping the ``1/|eps_m|`` preconditioner multiplies the metal curl residual
    by ``|eps_m| ~ 18``, so the same weights at the same seed give a much larger
    curl loss. This is measured, not asserted from the weight value.
    """
    control = _one_epoch(ab.CONDITIONS["control"])
    uniform = _one_epoch(ab.CONDITIONS["uniform_weights"])
    assert uniform["curl"] > 3 * control["curl"]
    assert uniform["div"] > 3 * control["div"]
    # The anchor term is untouched by the change
    assert uniform["boundary"] == pytest.approx(control["boundary"], rel=1e-5)


# --------------------------------------------------------------- measurements
def test_interface_jump_on_the_exact_mode_recovers_eps_ratio():
    """
    The analytical mode's measured jump is ``eps_m/eps_d`` times the finite-offset
    factor ``exp((kappa_m - kappa_d)*offset)`` — about +8.6% at the +-2 nm offset.
    That is the metric's floor, not an error, which is why the study reports the
    exact mode's own value alongside every arm's.
    """
    torch.manual_seed(0)
    jump = ab.interface_jump(vs.AnalyticalSPP(), n_points=512, device=DEVICE)
    exact = abs(complex(vs.EPS_METAL_N) / complex(vs.EPS_DIEL))
    assert jump["ez_jump_ratio_exact_abs"] == pytest.approx(exact)
    expected = exact * np.exp(
        (vs.KAPPA_M.real - vs.KAPPA_D.real) * ab.JUMP_OFFSET
    )
    assert jump["ez_jump_ratio_abs"] == pytest.approx(expected, rel=0.02)
    assert 0.05 < jump["ez_jump_rel_error"] < 0.12


class _ContinuousEz(nn.Module):
    """A network whose E_z is smooth through z = 0 — an adapter-less limit."""

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(coords.shape[0], 6, 2, dtype=coords.dtype)
        out[:, :, 0] = 1.0
        return out


def test_interface_jump_flags_a_continuous_ez():
    """A field with no jump scores ~|1 - eps_m/eps_d| / |eps_m/eps_d| ~ 0.95."""
    jump = ab.interface_jump(_ContinuousEz(), n_points=64, device=DEVICE)
    assert jump["ez_jump_ratio_abs"] == pytest.approx(1.0, rel=1e-6)
    exact = complex(vs.EPS_METAL_N) / complex(vs.EPS_DIEL)
    assert jump["ez_jump_rel_error"] == pytest.approx(abs(1 - exact) / abs(exact), rel=1e-6)


def test_amplitude_ratio_is_one_for_the_exact_mode_and_zero_for_collapse():
    torch.manual_seed(0)
    exact = ab.amplitude_ratios(vs.AnalyticalSPP(), n_points=500, device=DEVICE)
    assert exact["amplitude_ratio_E"] == pytest.approx(1.0, rel=1e-6)
    assert exact["amplitude_ratio_H"] == pytest.approx(1.0, rel=1e-6)

    class _Zero(nn.Module):
        def forward(self, coords):
            return torch.zeros(coords.shape[0], 6, 2, dtype=coords.dtype)

    torch.manual_seed(0)
    collapsed = ab.amplitude_ratios(_Zero(), n_points=500, device=DEVICE)
    assert collapsed["amplitude_ratio_E"] == 0.0
    assert collapsed["amplitude_ratio_H"] == 0.0


def test_evaluate_returns_every_promised_metric_key():
    torch.manual_seed(0)
    network = ab.build_network(ab.CONDITIONS["control"], device=DEVICE, **TINY)
    metrics = ab.evaluate(network, n_points=400, device=DEVICE)
    missing = [k for k in ab.METRIC_KEYS if k not in metrics]
    assert missing == []
    assert all(np.isfinite(metrics[k]) for k in ab.METRIC_KEYS)


# ------------------------------------------------------------- checkpointing
def test_checkpoint_path_is_per_condition(tmp_path):
    paths = {
        ab.checkpoint_path_for(c, tmp_path).name for c in ab.CONDITIONS.values()
    }
    assert paths == {
        f"ablation_{k}.partial.pth" for k in ab.CONDITIONS
    }


def test_resume_from_absent_checkpoint_runs_the_whole_schedule(tmp_path):
    network = _tiny_network(ab.CONDITIONS["control"])
    history, best, left = ab.resume_from(network, tmp_path / "nope.pth", 500)
    assert history is None and np.isinf(best) and left == 500


def test_resume_subtracts_the_epochs_already_on_record(tmp_path):
    """A resumed arm finishes the declared schedule; it does not extend it."""
    condition = ab.CONDITIONS["control"]
    ckpt = ab.checkpoint_path_for(condition, tmp_path)
    network = _tiny_network(condition)
    ab.train_condition(
        network, condition, n_epochs=3, n_points=64, lbfgs_steps=0,
        device=DEVICE, log_every=1000, checkpoint_path=ckpt,
    )
    assert ckpt.exists()
    fresh = _tiny_network(condition)
    history, best, left = ab.resume_from(fresh, ckpt, 10)
    assert history is not None and len(history["epoch"]) == 3
    assert np.isfinite(best)
    assert left == 7
    # Never negative, even if the checkpoint is longer than the request
    assert ab.resume_from(fresh, ckpt, 2)[2] == 0


# -------------------------------------------------------- reporting & figures
def _fake_results():
    out = []
    for i, condition in enumerate(ab.CONDITIONS.values()):
        metrics = {k: 0.01 * (i + 1) for k in ab.METRIC_KEYS}
        metrics["amplitude_ratio_E"] = 1.0 - 0.2 * i
        metrics["final_loss"] = 1e-3 * (i + 1)
        out.append(
            ab.Result(
                condition=condition, metrics=metrics,
                history={"epoch": [0, 1], "total": [1.0, 0.5]},
            )
        )
    return out


def test_report_schema(tmp_path):
    results = _fake_results()
    reference = {k: 0.0 for k in ab.METRIC_KEYS}
    schedule = {"epochs": 10, "n_points": 32, "lbfgs_steps": 0}
    report = ab.build_report(results, reference, schedule, {"fig": "x.png"})
    assert set(report) == {
        "experiment", "case", "schedule", "analytical_reference", "conditions", "figures"
    }
    assert report["experiment"] == "spp_ablation" and report["case"] == "silver"
    assert set(report["conditions"]) == set(ab.CONDITIONS)
    for key, entry in report["conditions"].items():
        assert set(entry) == {"label", "changed", "config", "metrics"}
        assert set(entry["config"]) == {
            "use_adapter", "boundary_weight", "physics_ramp_frac",
            "per_medium_weighting", "metal_curl_weight_adam",
            "metal_curl_weight_lbfgs", "metal_div_weight",
        }
        assert entry["config"]["use_adapter"] is ab.CONDITIONS[key].use_adapter
        assert all(k in entry["metrics"] for k in ab.METRIC_KEYS)
    # Serialises
    json.loads(json.dumps(report))


def test_summary_table_has_a_row_per_arm_plus_the_exact_mode():
    table = ab.summary_table(_fake_results(), {k: 0.0 for k in ab.METRIC_KEYS})
    lines = table.splitlines()
    assert len(lines) == 2 + len(ab.CONDITIONS) + 1  # header, rule, arms, exact
    assert "condition" in lines[0] and "E_z jump err" in lines[0]
    assert "(exact mode)" in lines[-1]
    for condition in ab.CONDITIONS.values():
        assert any(condition.label in ln for ln in lines)


def test_every_arm_has_its_own_colour_and_short_label():
    assert set(ab.CONDITION_COLORS) == set(ab.CONDITIONS)
    assert set(ab.SHORT_LABELS) == set(ab.CONDITIONS)
    assert len(set(ab.CONDITION_COLORS.values())) == len(ab.CONDITIONS)


def test_plot_ablation_writes_both_figures(tmp_path):
    paths = ab.plot_ablation(
        _fake_results(), reference={k: 1e-4 for k in ab.METRIC_KEYS}, out_dir=tmp_path
    )
    assert set(paths) == {"ablation_metrics", "ablation_training_curves"}
    for p in paths.values():
        assert tmp_path.joinpath(p).exists() or __import__("pathlib").Path(p).exists()


def test_parse_args_defaults_and_overrides():
    args = ab.parse_args([])
    assert args.conditions == "all" and args.epochs == ab.N_EPOCHS
    assert args.n_points == ab.N_POINTS and args.lbfgs_steps == ab.LBFGS_STEPS
    assert args.resume is False and args.quick is False
    args = ab.parse_args(["--conditions", "no_anchor", "--quick", "--resume"])
    assert args.conditions == "no_anchor" and args.quick and args.resume


# -------------------------------------------------------------- end-to-end
@pytest.mark.slow
def test_quick_run_of_two_conditions_produces_the_schema(tmp_path):
    """A ``--quick`` two-arm run completes and fills in every promised metric."""
    report = ab.main(
        [
            "--quick", "--conditions", "control,no_adapter",
            "--figures-dir", str(tmp_path), "--model-dir", str(tmp_path / "models"),
            "--no-figure",
        ]
    )
    assert set(report["conditions"]) == {"control", "no_adapter"}
    assert report["schedule"]["quick"] is True
    assert report["schedule"]["epochs"] == ab.QUICK_EPOCHS
    for entry in report["conditions"].values():
        for key in ab.METRIC_KEYS:
            assert np.isfinite(entry["metrics"][key]), key
        assert np.isfinite(entry["metrics"]["final_loss"])
        assert entry["metrics"]["train_time_s"] > 0
        assert entry["metrics"]["success_tier"] in {
            "stretch", "target", "minimum", "not met"
        }
    # The exact-mode floor row is present and is (nearly) perfect
    assert report["analytical_reference"]["rel_l2_total"] < 1e-6
    # results.json landed and round-trips
    saved = json.loads((tmp_path / "results.json").read_text())
    assert set(saved["conditions"]) == {"control", "no_adapter"}
    # A --quick run must not litter artifacts/models
    assert not (tmp_path / "models").exists()


@pytest.mark.slow
def test_partial_run_merges_rather_than_clobbers_existing_arms(tmp_path):
    """Running one arm at a time must not delete the arms already on disk."""
    common = [
        "--quick", "--figures-dir", str(tmp_path), "--no-figure",
        "--model-dir", str(tmp_path / "models"),
    ]
    ab.main([*common, "--conditions", "control"])
    ab.main([*common, "--conditions", "no_ramp"])
    saved = json.loads((tmp_path / "results.json").read_text())
    assert set(saved["conditions"]) == {"control", "no_ramp"}
