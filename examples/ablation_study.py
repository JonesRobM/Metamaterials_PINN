r"""
Ablation study: which of the SPP recipe's design choices actually earn their place?

``examples/validate_spp.py`` trains a PINN to the silver/air surface-plasmon mode
at 633 nm and reaches the *stretch* tier. Four ingredients of that recipe have
until now been asserted rather than measured — the README calls the first of them
"the most reusable idea the project has produced" and
``docs/plans/2026-09-01-hmm-surrogate-results.md`` says of it, verbatim, "Not
ablated — asserted as a design choice and tested for coverage, not for its effect
on the final error." This script removes them one at a time and measures what
happens.

The five conditions
-------------------

=====================  =====================================================
``control``            the full recipe, exactly as ``validate_spp.py``
``no_adapter``         the MLP predicts ``E_z`` directly; the displacement
                       adapter is gone, so ``ε(z)`` never divides channel 2
                       and the interface jump must be *learned*
``no_anchor``          boundary weight 0: interior physics + tangential
                       continuity only. The Maxwell residual loss is minimised
                       exactly by ``E = H = 0``, so this is the condition that
                       tests the project's most-repeated claim
``no_ramp``            ``physics_ramp_frac = 0``: interior physics at full
                       weight from epoch 0, no warm-up behind the anchor
``uniform_weights``    the per-medium ``1/|ε_m|`` preconditioner is dropped;
                       both half-spaces' curl and divergence residuals are
                       weighted 1, in both the Adam and L-BFGS phases
=====================  =====================================================

Everything else — seed, network, sampling, schedule, learning rate, validation
points — is identical across conditions. Each condition differs from the control
in exactly one place, and that place is asserted in
``tests/examples/test_ablation_study.py`` rather than left to inspection.

A shortened schedule, deliberately
----------------------------------
The defaults here (1200 Adam epochs at 1024 points, 8 float64 L-BFGS steps) are a
fraction of ``validate_spp.py``'s production schedule (4000 epochs at 2048
points, 50 L-BFGS steps), so **no absolute number in this study should be quoted
as the method's accuracy** — ``validate_spp.py``'s own metrics.json is the place
for that. What the shortened schedule buys is five conditions inside a 90-minute
CPU budget, and since every condition gets the *same* shortened schedule, the
*differences* between them — which is the entire question — remain meaningful.

Metrics per condition
---------------------
Relative L2 against the analytical mode (overall and split air / metal), the
recovered ``Re k_spp``, the ``κ_d`` and ``κ_m`` decay fits, the ``E_z``
interface-jump ratio measured against the exact ``ε_m/ε_d``, the field amplitude
relative to the analytical mode (the collapse detector), and the final loss.
Because every metric has a measurement floor set by the finite evaluation offsets
and the guard band, the **analytical mode itself** is pushed through the identical
pipeline and reported alongside as the ``analytical`` row.

Usage::

    python examples/ablation_study.py [--conditions control,no_adapter]
                                      [--epochs 1200] [--n-points 1024]
                                      [--lbfgs-steps 8] [--seed 0] [--resume]
                                      [--quick]

Checkpoints go to ``artifacts/models/ablation_<condition>.partial.pth`` (hence
``--model-dir``, not ``--model-out``: there is one per condition), the figure and
``results.json`` to ``figures/ablation/``.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples import validate_spp as vs  # noqa: E402
from src.analytical import analytical_spp_fields  # noqa: E402
from src.experiments import (  # noqa: E402
    TrainingConfig,
    TwoMediumAdapter,
    add_core_args,
    load_checkpoint_history,
    load_core_checkpoint,
    run_training,
    write_json_report,
)
from src.models import (  # noqa: E402
    ElectromagneticPINN,
    MaxwellCurlLoss,
    MaxwellDivergenceLoss,
    TangentialContinuityLoss,
    to_complex,
)

logger = logging.getLogger("ablation_study")

DEVICE = torch.device("cpu")

# --------------------------------------------------------------------- schedule
# Shortened but identical across conditions — see the module docstring. Sized so
# one condition is ~7 minutes on a CPU: 1200 x 0.26 s/epoch + 8 x 9.5 s/L-BFGS.
N_EPOCHS = 1200
N_POINTS = 1024
LBFGS_STEPS = 8
LEARNING_RATE = 1e-3
VAL_POINTS = 8000  # validation collocation points per condition
QUICK_EPOCHS = 30
QUICK_POINTS = 256

#: Offset at which the ``E_z`` jump is measured, in metres. The same ±2 nm as the
#: validation guard band, so the measurement never straddles the ε discontinuity.
JUMP_OFFSET = vs.VAL_GUARD

FIGURES_DIR = REPO_ROOT / "figures" / "ablation"
MODEL_DIR = REPO_ROOT / "artifacts" / "models"


# ------------------------------------------------------------------- conditions
@dataclass(frozen=True)
class Condition:
    """
    One arm of the study: a name, and the single knob it turns.

    The defaults *are* the control, so each ablation is written as one keyword
    override and a reader can see at a glance that nothing else moved. The four
    behavioural fields map onto the four claims under test:

    ``use_adapter``
        Wrap the MLP in :class:`~src.experiments.adapter.TwoMediumAdapter`, so
        channel 2 carries the continuous ``D̂_z`` and the ``E_z`` jump is exact
        by construction. ``False`` lets the MLP emit ``E_z`` directly.
    ``boundary_weight``
        Weight of the soft-Dirichlet anchor on the six domain faces. ``0``
        removes it and leaves an objective whose global minimum is ``E = H = 0``.
    ``physics_ramp_frac``
        Fraction of the Adam schedule over which the interior physics weight
        climbs 0 → 1. ``0`` means full weight from epoch 0
        (:func:`src.experiments.ramp_at` returns 1 immediately).
    ``per_medium_weighting``
        Precondition the metal-side curl and divergence residuals by
        ``max(|ε_t|, |ε_n|)^{-p}``. ``False`` weights both half-spaces at 1.
    """

    key: str
    label: str
    changed: str
    use_adapter: bool = True
    boundary_weight: float = vs.BOUNDARY_WEIGHT
    physics_ramp_frac: float = vs.PHYSICS_RAMP_FRAC
    per_medium_weighting: bool = True

    def metal_curl_weights(self) -> Tuple[float, float]:
        """``(adam, lbfgs)`` metal-side curl weights for this condition."""
        if not self.per_medium_weighting:
            return 1.0, 1.0
        return vs.METAL_CURL_WEIGHT_ADAM, vs.METAL_CURL_WEIGHT_LBFGS

    def metal_div_weight(self) -> float:
        """Metal-side divergence weight for this condition."""
        return vs.METAL_DIV_WEIGHT if self.per_medium_weighting else 1.0


_CONTROL = Condition(
    key="control",
    label="full recipe",
    changed="nothing (validate_spp.py --case silver)",
)

#: The study, in reporting order. ``control`` must stay first: the figure and the
#: summary table both read it as the baseline every other row is compared to.
CONDITIONS: Dict[str, Condition] = {
    c.key: c
    for c in (
        _CONTROL,
        replace(
            _CONTROL,
            key="no_adapter",
            label="no displacement adapter",
            changed="network predicts E_z directly; eps(z) never applied",
            use_adapter=False,
        ),
        replace(
            _CONTROL,
            key="no_anchor",
            label="no boundary anchor",
            changed="boundary weight 100 -> 0 (physics + continuity only)",
            boundary_weight=0.0,
        ),
        replace(
            _CONTROL,
            key="no_ramp",
            label="no physics ramp",
            changed="physics_ramp_frac 0.25 -> 0 (full weight from epoch 0)",
            physics_ramp_frac=0.0,
        ),
        replace(
            _CONTROL,
            key="uniform_weights",
            label="uniform loss weighting",
            changed="metal curl/div weights 1/|eps_m| -> 1 in both phases",
            per_medium_weighting=False,
        ),
    )
}


def resolve_conditions(spec: str) -> List[Condition]:
    """Parse a ``--conditions`` string (``"all"`` or a comma-separated list)."""
    if spec.strip().lower() == "all":
        return list(CONDITIONS.values())
    keys = [k.strip() for k in spec.split(",") if k.strip()]
    unknown = [k for k in keys if k not in CONDITIONS]
    if unknown:
        raise ValueError(f"Unknown condition(s) {unknown}; choose from {list(CONDITIONS)}")
    return [CONDITIONS[k] for k in keys]


# --------------------------------------------------------------------- network
def build_network(
    condition: Condition,
    hidden_dims: Tuple[int, ...] = (128, 128, 128, 128),
    fourier_modes: int = 128,
    device: torch.device = DEVICE,
) -> vs.SPPPINN:
    """
    The control's network from :func:`validate_spp.create_network`, adapter optional.

    The MLP is built identically in every condition — same architecture, same
    Fourier band, and (given the same seed immediately before this call) the same
    initial weights, because the adapter contributes no parameters of its own.
    ``no_adapter`` simply hands the bare MLP to :class:`validate_spp.SPPPINN`, so
    channel 2 of its output is read as ``Ê_z`` rather than ``D̂_z``.
    """
    mlp = ElectromagneticPINN(
        spatial_dim=3,
        field_components=6,
        hidden_dims=list(hidden_dims),
        complex_valued=True,
        frequency=vs.OMEGA,
        use_fourier=True,
        fourier_modes=fourier_modes,
        fourier_k_range=(0.1, 40.0),
        activation_type="complex_tanh",
    )
    core: nn.Module = mlp
    if condition.use_adapter:
        core = TwoMediumAdapter(mlp, eps_below=vs.EPS_METAL_N, eps_above=vs.EPS_DIEL)
    return vs.SPPPINN(core).to(device)


def has_adapter(network: nn.Module) -> bool:
    """Whether ``network.core`` is a displacement adapter (test/reporting helper)."""
    return isinstance(network.core, TwoMediumAdapter)


# -------------------------------------------------------------------- training
def train_condition(
    network: vs.SPPPINN,
    condition: Condition,
    *,
    n_epochs: int = N_EPOCHS,
    n_points: int = N_POINTS,
    learning_rate: float = LEARNING_RATE,
    lbfgs_steps: int = LBFGS_STEPS,
    lbfgs_dtype: torch.dtype = torch.float64,
    device: torch.device = DEVICE,
    log_every: int = 200,
    checkpoint_path: Optional[Path] = None,
    initial_history: Optional[Dict[str, list]] = None,
    initial_best_loss: float = float("inf"),
) -> Tuple[vs.SPPPINN, Dict[str, list]]:
    """
    Train one arm. Same physics and same closures as :func:`validate_spp.train`,
    with the four ablated knobs read off ``condition``.

    This mirrors rather than calls ``validate_spp.train`` because that function
    hard-codes the three weights the study exists to vary; keeping the copy here
    means the production script stays untouched by the experiment measuring it.
    The optimiser, ramp, best-iterate tracking and checkpointing are still
    :func:`src.experiments.run_training` — only the objective is local.
    """
    core = network.core
    curl_loss = MaxwellCurlLoss(frequency=vs.OMEGA_HAT, mu0=1.0, eps0=1.0)
    div_loss = MaxwellDivergenceLoss()
    cont_loss = TangentialContinuityLoss(offset=vs.CONTINUITY_OFFSET / vs.LAMBDA0)
    eps_m_diag = vs.metal_eps_diag()
    w_curl_adam, w_curl_lbfgs = condition.metal_curl_weights()
    w_div_m = condition.metal_div_weight()

    def compute_losses(batch, ramp=1.0, w_curl_m=None):
        coords_air, coords_metal, iface_hat, normals, boundary_hat, target_hat = batch
        if w_curl_m is None:
            w_curl_m = w_curl_adam
        l_curl = curl_loss.compute(
            network=core, coords=coords_air, epsilon=complex(vs.EPS_DIEL)
        ) + w_curl_m * curl_loss.compute(
            network=core, coords=coords_metal, epsilon=eps_m_diag
        )
        l_div = div_loss.compute(
            network=core, coords=coords_air, epsilon=complex(vs.EPS_DIEL)
        ) + w_div_m * div_loss.compute(
            network=core, coords=coords_metal, epsilon=eps_m_diag
        )
        l_cont = cont_loss.compute(
            network=core, interface_coords=iface_hat, normal_vectors=normals
        )
        # Always computed, even at weight 0: the no-anchor arm's boundary MSE is
        # the diagnostic that says how far its field drifted from the true mode.
        l_bc = torch.mean((core(boundary_hat) - target_hat) ** 2)
        total = (
            ramp
            * (l_curl + vs.DIVERGENCE_WEIGHT * l_div + vs.CONTINUITY_WEIGHT * l_cont)
            + condition.boundary_weight * l_bc
        )
        return total, l_curl, l_div, l_cont, l_bc

    def sample_batch(n_int, n_bc, n_if, dtype=torch.float32):
        coords_hat = (
            (vs.sample_collocation_points(n_int, device=device) / vs.LAMBDA0)
            .detach()
            .to(dtype)
        )
        metal = coords_hat[:, 2] < 0
        coords_air = coords_hat[~metal].requires_grad_(True)
        coords_metal = coords_hat[metal].requires_grad_(True)
        iface, normals = vs.sample_interface_points(n_if, device=device)
        iface_hat = (iface / vs.LAMBDA0).to(dtype)
        normals = normals.to(dtype)
        boundary_hat = (vs.sample_boundary_points(n_bc, device=device) / vs.LAMBDA0).to(dtype)
        with torch.no_grad():
            target_hat = vs.analytical_fields_hat(boundary_hat)
        return coords_air, coords_metal, iface_hat, normals, boundary_hat, target_hat

    return run_training(
        network,
        TrainingConfig(
            n_epochs=n_epochs,
            n_points=n_points,
            n_boundary=max(6, n_points // 2),
            n_interface=max(1, n_points // 4),
            learning_rate=learning_rate,
            physics_ramp_frac=condition.physics_ramp_frac,
            lbfgs_steps=lbfgs_steps,
            lbfgs_dtype=lbfgs_dtype,
            lbfgs_points_factor=vs.LBFGS_POINTS_FACTOR,
            log_every=log_every,
        ),
        sample_batch,
        compute_losses,
        logger,
        lbfgs_loss_kwargs={"w_curl_m": w_curl_lbfgs},
        checkpoint_path=checkpoint_path,
        save_history=checkpoint_path is not None,
        final_checkpoint=checkpoint_path is not None,
        initial_history=initial_history,
        initial_best_loss=initial_best_loss,
        lbfgs_note=f" [{condition.key}]",
    )


def checkpoint_path_for(condition: Condition, model_dir: Path = MODEL_DIR) -> Path:
    """``artifacts/models/ablation_<key>.partial.pth`` — one checkpoint per arm."""
    return Path(model_dir) / f"ablation_{condition.key}.partial.pth"


def resume_from(
    network: nn.Module, path: Path, n_epochs: int
) -> Tuple[Optional[Dict[str, list]], float, int]:
    """
    Warm-start ``network`` from a ``.partial.pth``, if one is there.

    Returns ``(history, best_loss, epochs_still_to_run)``. Epochs already on
    record are subtracted so a resumed run finishes the *declared* schedule
    rather than extending it — a longer schedule for one arm would silently
    invalidate the comparison this whole script exists to make.
    """
    path = Path(path)
    if not path.exists():
        return None, float("inf"), n_epochs
    best = load_core_checkpoint(network, path)
    history = load_checkpoint_history(path)
    done = 0
    if history and history.get("epoch"):
        done = min(n_epochs, int(max(history["epoch"])) + 1)
    logger.info("resumed %s: %d epochs on record, best loss %.3e", path.name, done, best)
    return history, best, max(0, n_epochs - done)


# ----------------------------------------------------------------- measurement
def interface_jump(
    network: nn.Module,
    offset: float = JUMP_OFFSET,
    n_points: int = 2048,
    device: torch.device = DEVICE,
) -> Dict[str, float]:
    r"""
    The measured ``E_z`` jump across ``z = 0``, against the exact ``ε_m/ε_d``.

    ``D_z = ε₀ ε_zz E_z`` is continuous, so the exact mode satisfies
    ``E_z(0⁺) / E_z(0⁻) = ε_m / ε_d`` — a complex number of modulus ≈ 18.3 for
    silver at 633 nm. The ratio is estimated by complex least squares over the
    sampled interface points,

    .. math:: r = \frac{\sum \overline{E_z^-} E_z^+}{\sum |E_z^-|^2}

    rather than by averaging per-point quotients, which would be dominated by
    whichever points happen to sit near a node of ``E_z``.

    This is the metric with a *predictable* answer for the ``no_adapter`` arm: a
    continuous MLP evaluated 2 nm either side of the interface can only produce
    ``r ≈ 1``, i.e. a relative error approaching ``|1 − ε_m/ε_d| / |ε_m/ε_d| ≈
    0.95``, unless it has genuinely learned a near-discontinuity.

    Returns ``ez_jump_ratio_abs``, ``ez_jump_ratio_exact_abs`` and
    ``ez_jump_rel_error`` (the modulus of the complex difference, normalised).
    """
    coords, normals = vs.sample_interface_points(n_points, device=device)
    with torch.no_grad():
        E_air, _ = to_complex(network(coords + offset * normals))
        E_metal, _ = to_complex(network(coords - offset * normals))
        a = E_metal[:, 2].to(torch.complex128)
        b = E_air[:, 2].to(torch.complex128)
        denom = (a.conj() * a).real.sum().clamp_min(1e-60)
        ratio = complex(((a.conj() * b).sum() / denom).item())
    exact = complex(vs.EPS_METAL_N) / complex(vs.EPS_DIEL)
    return {
        "ez_jump_ratio_abs": abs(ratio),
        "ez_jump_ratio_exact_abs": abs(exact),
        "ez_jump_rel_error": abs(ratio - exact) / abs(exact),
    }


def amplitude_ratios(
    network: nn.Module, n_points: int = VAL_POINTS, device: torch.device = DEVICE
) -> Dict[str, float]:
    r"""
    ``‖E_pred‖ / ‖E_exact‖`` and the same for ``H``, over the validation volume.

    The collapse detector. The Maxwell residual losses are homogeneous of degree
    2 in the fields, so ``E = H = 0`` is an exact global minimiser of the
    ``no_anchor`` objective; a ratio near 0 says the arm found it. A ratio near 1
    with a large relative L2, by contrast, says the arm kept the amplitude and
    got the *shape* wrong — a different failure, and worth telling apart.
    """
    coords = vs.sample_collocation_points(n_points, guard=vs.VAL_GUARD, device=device)
    with torch.no_grad():
        E, H = to_complex(network(coords))
        E_ref, H_ref = analytical_spp_fields(
            coords.detach(), vs.OMEGA, vs.EPS_METAL_T, vs.EPS_METAL_N,
            eps_dielectric=vs.EPS_DIEL, H0=vs.H0,
        )
        out = {}
        for name, pred, ref in (("E", E, E_ref), ("H", H, H_ref)):
            num = torch.linalg.vector_norm(pred.to(torch.complex128))
            den = torch.linalg.vector_norm(ref.to(torch.complex128)).clamp_min(1e-30)
            out[f"amplitude_ratio_{name}"] = (num / den).item()
    return out


def evaluate(
    network: nn.Module, n_points: int = VAL_POINTS, device: torch.device = DEVICE
) -> Dict[str, float]:
    """
    Every metric for one arm: ``validate_spp.validate`` plus the two above.

    Reusing the production validator matters — the relative-L2, ``k_spp`` and κ
    numbers here are then directly comparable with ``validate_spp.py``'s own
    metrics.json, modulo the shortened schedule.
    """
    network.eval()
    metrics = dict(vs.validate(network, n_points=n_points, device=device))
    metrics.update(amplitude_ratios(network, n_points=n_points, device=device))
    metrics.update(interface_jump(network, device=device))
    return metrics


#: Metric keys the results JSON promises for every condition. Asserted in the
#: tests, so a metric silently disappearing from the schema fails the suite.
METRIC_KEYS: Tuple[str, ...] = (
    "rel_l2_total",
    "rel_l2_E",
    "rel_l2_H",
    "rel_l2_E_air",
    "rel_l2_E_metal",
    "rel_l2_H_air",
    "rel_l2_H_metal",
    "k_spp_fit",
    "k_spp_rel_error",
    "kappa_d_fit_rel_error",
    "kappa_m_fit_rel_error",
    "decay_sign_correct_air",
    "decay_sign_correct_metal",
    "continuity_E_rel",
    "continuity_H_rel",
    "amplitude_ratio_E",
    "amplitude_ratio_H",
    "ez_jump_ratio_abs",
    "ez_jump_rel_error",
)


# ------------------------------------------------------------------ one arm
@dataclass
class Result:
    """One arm's condition, metrics and loss history."""

    condition: Condition
    metrics: Dict[str, float]
    history: Dict[str, list] = field(default_factory=dict)

    def as_json(self) -> Dict[str, object]:
        c = self.condition
        return {
            "label": c.label,
            "changed": c.changed,
            "config": {
                "use_adapter": c.use_adapter,
                "boundary_weight": c.boundary_weight,
                "physics_ramp_frac": c.physics_ramp_frac,
                "per_medium_weighting": c.per_medium_weighting,
                "metal_curl_weight_adam": c.metal_curl_weights()[0],
                "metal_curl_weight_lbfgs": c.metal_curl_weights()[1],
                "metal_div_weight": c.metal_div_weight(),
            },
            "metrics": self.metrics,
        }


def run_condition(
    condition: Condition,
    *,
    n_epochs: int = N_EPOCHS,
    n_points: int = N_POINTS,
    learning_rate: float = LEARNING_RATE,
    lbfgs_steps: int = LBFGS_STEPS,
    lbfgs_dtype: torch.dtype = torch.float64,
    val_points: int = VAL_POINTS,
    seed: int = 0,
    device: torch.device = DEVICE,
    log_every: int = 200,
    model_dir: Optional[Path] = MODEL_DIR,
    resume: bool = False,
    hidden_dims: Tuple[int, ...] = (128, 128, 128, 128),
    fourier_modes: int = 128,
) -> Result:
    """
    Build, train and measure one arm, with the seed reset around each stage.

    Three separate seedings, on purpose. ``seed`` before construction gives every
    arm identical initial weights; ``seed`` again before training gives every arm
    the identical stream of collocation batches; ``seed + 1000`` before
    evaluation gives every arm the identical validation points. Without the last
    one the arms would be graded on different samples, and differences of a few
    percent would be sampling noise rather than results.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    network = build_network(
        condition, hidden_dims=hidden_dims, fourier_modes=fourier_modes, device=device
    )
    ckpt = checkpoint_path_for(condition, model_dir) if model_dir is not None else None

    history_in, best_in, epochs_left = (None, float("inf"), n_epochs)
    if resume and ckpt is not None:
        history_in, best_in, epochs_left = resume_from(network, ckpt, n_epochs)

    logger.info(
        "condition %-16s | adapter=%-5s bc_weight=%-6.1f ramp=%.2f per_medium=%-5s | "
        "epochs=%d n_points=%d lbfgs=%d",
        condition.key, condition.use_adapter, condition.boundary_weight,
        condition.physics_ramp_frac, condition.per_medium_weighting,
        epochs_left, n_points, lbfgs_steps,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.perf_counter()
    network, history = train_condition(
        network, condition,
        n_epochs=epochs_left, n_points=n_points, learning_rate=learning_rate,
        lbfgs_steps=lbfgs_steps, lbfgs_dtype=lbfgs_dtype, device=device,
        log_every=log_every, checkpoint_path=ckpt,
        initial_history=history_in, initial_best_loss=best_in,
    )
    train_time = time.perf_counter() - t0

    torch.manual_seed(seed + 1000)
    np.random.seed(seed + 1000)
    metrics = evaluate(network, n_points=val_points, device=device)
    totals = [t for t in history["total"] if math.isfinite(t)]
    metrics["final_loss"] = history["total"][-1] if history["total"] else float("nan")
    metrics["best_loss"] = min(totals) if totals else float("nan")
    metrics["train_time_s"] = train_time
    metrics["epochs"] = float(n_epochs)
    metrics["n_points"] = float(n_points)
    metrics["lbfgs_steps"] = float(lbfgs_steps)
    metrics["seed"] = float(seed)
    metrics["success_tier"] = vs.success_tier(metrics)
    return Result(condition=condition, metrics=metrics, history=history)


def analytical_baseline(
    val_points: int = VAL_POINTS, seed: int = 0, device: torch.device = DEVICE
) -> Dict[str, float]:
    """
    The exact mode through the identical measurement pipeline: every metric's floor.

    Without this row a reader cannot tell a real error from the finite-offset
    artefact of the metric itself — the ±2 nm continuity residual, for instance,
    is ~8% for the exact mode and 0% for nothing.
    """
    torch.manual_seed(seed + 1000)
    np.random.seed(seed + 1000)
    return evaluate(vs.AnalyticalSPP().to(device), n_points=val_points, device=device)


# ---------------------------------------------------------------------- figure
#: One fixed hue per condition, in the validated categorical order (blue, orange,
#: aqua, yellow, magenta). Colour follows the *condition*, not its rank, so a
#: partial run (``--conditions control,no_ramp``) paints the same arms the same
#: colours as the full study. Every bar is also direct-labelled and named on the
#: x axis, so identity never rests on colour alone.
CONDITION_COLORS: Dict[str, str] = {
    "control": "#2a78d6",
    "no_adapter": "#eb6834",
    "no_anchor": "#1baf7a",
    "no_ramp": "#eda100",
    "uniform_weights": "#e87ba4",
}
_INK = "#0b0b0b"
_INK_MUTED = "#52514e"
_FLOOR = "#8a8a85"

#: ``(metric key, panel title, log scale)`` — the eight panels, in reading order.
PANELS: Tuple[Tuple[str, str, bool], ...] = (
    ("rel_l2_total", "relative L2 (all fields)", True),
    ("rel_l2_E_air", "relative L2, E (air side)", True),
    ("rel_l2_E_metal", "relative L2, E (metal side)", True),
    ("amplitude_ratio_E", "field amplitude  ||E|| / ||E_exact||", False),
    ("k_spp_rel_error", "Re k_spp relative error", True),
    ("kappa_d_fit_rel_error", "kappa_d fit relative error", True),
    ("kappa_m_fit_rel_error", "kappa_m fit relative error", True),
    ("ez_jump_rel_error", "E_z interface jump vs eps_m/eps_d", True),
)

#: Short x-axis names — the full labels do not fit under a 4-across grid.
SHORT_LABELS: Dict[str, str] = {
    "control": "control",
    "no_adapter": "no\nadapter",
    "no_anchor": "no\nanchor",
    "no_ramp": "no\nramp",
    "uniform_weights": "uniform\nweights",
}


def _bar_label(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    if value == 0:
        return "0"
    if 1e-2 <= abs(value) < 1e3:
        return f"{value:.3g}"
    return f"{value:.1e}"


def plot_ablation(
    results: Sequence[Result],
    reference: Optional[Dict[str, float]] = None,
    out_dir: Path = FIGURES_DIR,
) -> Dict[str, str]:
    """
    Small multiples — one panel per metric, one bar per condition.

    Small multiples rather than one grouped chart because the metrics are on
    unrelated scales and mean unrelated things; grouping them would force either
    a shared log axis that flattens the differences or a second y-axis, which is
    never the answer. Each panel keeps a single axis, and the dashed grey rule is
    that metric's floor as measured on the exact analytical mode.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = [r.condition.key for r in results]
    colors = [CONDITION_COLORS.get(k, "#4a3aa7") for k in keys]
    names = [SHORT_LABELS.get(k, k) for k in keys]
    x = np.arange(len(results))

    ncol = min(4, max(1, len(PANELS)))
    nrow = int(math.ceil(len(PANELS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.5 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, (key, title, log) in zip(axes, PANELS, strict=False):
        values = [float(r.metrics.get(key, float("nan"))) for r in results]
        plotted = [v if math.isfinite(v) and (not log or v > 0) else np.nan for v in values]
        bars = ax.bar(x, plotted, width=0.62, color=colors, linewidth=0)
        if log:
            ax.set_yscale("log")
        ax.set_title(title, fontsize=10, color=_INK, pad=14)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8, color=_INK_MUTED)
        ax.tick_params(axis="y", labelsize=8, colors=_INK_MUTED, length=0)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(_FLOOR)

        # The exact mode's own value of this metric — drawn only when it is close
        # enough to the bars to be read off the same axis. Several metrics have a
        # floor near machine precision; stretching the axis down to 1e-9 would
        # squash every bar into a line to make a point the table already makes.
        floor = None if reference is None else float(reference.get(key, float("nan")))
        finite = [v for v in plotted if v == v]
        show_floor = (
            floor is not None
            and math.isfinite(floor)
            and (not log or floor > 0)
            and bool(finite)
            and (not log or floor >= min(finite) / 50)
        )
        if finite:
            span = [*finite, floor] if show_floor else finite
            if log:
                ax.set_ylim(min(span) / 3, max(span) * 4)
            else:
                ax.set_ylim(0, max(1.15, max(span) * 1.25))
        if log:
            # Minor tick labels ("2 x 10^0", "3 x 10^0", ...) appear whenever a log
            # axis spans under a decade, and they crowd out the majors.
            ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        # Direct labels: the panels are small and a reader should not have to
        # interpolate a log axis by eye.
        for rect, value in zip(bars, values, strict=True):
            height = rect.get_height()
            ax.annotate(
                _bar_label(value),
                (rect.get_x() + rect.get_width() / 2, height if height == height else 0),
                textcoords="offset points", xytext=(0, 3), ha="center",
                fontsize=7.5, color=_INK,
            )
        if show_floor:
            ax.axhline(floor, color=_FLOOR, linestyle="--", linewidth=1.0)
            ax.annotate(
                "exact mode", (len(results) - 0.45, floor), textcoords="offset points",
                xytext=(0, 3), ha="right", fontsize=7, color=_FLOOR,
            )
        elif floor is not None and math.isfinite(floor):
            ax.annotate(
                f"exact mode: {_bar_label(floor)}", (0.5, 0.94), xycoords="axes fraction",
                ha="center", fontsize=7, color=_FLOOR,
            )
        if key == "amplitude_ratio_E":
            ax.axhline(1.0, color=_FLOOR, linestyle="--", linewidth=1.0)

    for ax in axes[len(PANELS):]:
        ax.set_visible(False)

    fig.suptitle(
        "SPP PINN ablation — silver/air at 633 nm, identical shortened schedule per arm\n"
        "lower is better in every panel except the amplitude ratio, where 1.0 is correct",
        fontsize=11, color=_INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = out_dir / "ablation_metrics.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)

    paths = {"ablation_metrics": str(path)}

    # Training curves, one line per arm: how the no-anchor and no-ramp arms fail
    # is visible in the loss trace and nowhere in the table.
    if any(r.history.get("total") for r in results):
        fig, ax = plt.subplots(figsize=(8, 5))
        for r in results:
            if not r.history.get("total"):
                continue
            ax.semilogy(
                r.history["epoch"], r.history["total"],
                label=r.condition.label, linewidth=1.2,
                color=CONDITION_COLORS.get(r.condition.key, "#4a3aa7"),
            )
        ax.set_xlabel("epoch (Adam, then L-BFGS steps appended)", color=_INK_MUTED)
        ax.set_ylabel("training loss (this arm's own objective)", color=_INK_MUTED)
        ax.set_title(
            "Ablation training curves — note the objectives differ, so the\n"
            "curves are not comparable in level, only in shape",
            fontsize=10, color=_INK,
        )
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        p = out_dir / "ablation_training_curves.png"
        fig.savefig(p, dpi=150, facecolor="white")
        plt.close(fig)
        paths["ablation_training_curves"] = str(p)
    return paths


# ---------------------------------------------------------------------- report
#: Columns of the console/markdown summary table: ``(metric key, header)``.
TABLE_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("rel_l2_total", "rel L2"),
    ("rel_l2_E_air", "L2 air"),
    ("rel_l2_E_metal", "L2 metal"),
    ("amplitude_ratio_E", "|E|/|E_ex|"),
    ("k_spp_rel_error", "k_spp err"),
    ("kappa_d_fit_rel_error", "kappa_d err"),
    ("kappa_m_fit_rel_error", "kappa_m err"),
    ("ez_jump_rel_error", "E_z jump err"),
    ("final_loss", "final loss"),
)


def summary_table(
    results: Sequence[Result], reference: Optional[Dict[str, float]] = None
) -> str:
    """A pipe-delimited summary table, ready to paste into the results doc."""
    headers = ["condition", *(h for _, h in TABLE_COLUMNS)]
    rows = [
        [r.condition.label, *(_bar_label(float(r.metrics.get(k, float("nan"))))
                              for k, _ in TABLE_COLUMNS)]
        for r in results
    ]
    if reference is not None:
        rows.append(
            ["(exact mode)", *(_bar_label(float(reference.get(k, float("nan"))))
                               for k, _ in TABLE_COLUMNS)]
        )
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    def line(cells):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths, strict=True)) + " |"
    return "\n".join(
        [line(headers), "|" + "|".join("-" * (w + 2) for w in widths) + "|",
         *(line(r) for r in rows)]
    )


def build_report(
    results: Sequence[Result],
    reference: Dict[str, float],
    schedule: Dict[str, object],
    figures: Dict[str, str],
) -> Dict[str, object]:
    """The results.json payload: schedule, the exact-mode floor, one entry per arm."""
    return {
        "experiment": "spp_ablation",
        "case": "silver",
        "schedule": schedule,
        "analytical_reference": reference,
        "conditions": {r.condition.key: r.as_json() for r in results},
        "figures": figures,
    }


# ------------------------------------------------------------------------ main
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_core_args(
        p, epochs=N_EPOCHS, n_points=N_POINTS, lr=LEARNING_RATE,
        device=str(DEVICE), lbfgs_steps=LBFGS_STEPS,
    )
    p.add_argument(
        "--conditions", type=str, default="all",
        help=f"comma-separated subset of {list(CONDITIONS)}, or 'all'",
    )
    p.add_argument(
        "--val-points", type=int, default=VAL_POINTS,
        help="validation collocation points per condition",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="warm-start each arm from artifacts/models/ablation_<key>.partial.pth",
    )
    p.add_argument(
        "--quick", action="store_true",
        help=f"smoke run: {QUICK_EPOCHS} epochs, {QUICK_POINTS} points, no L-BFGS",
    )
    p.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    # A directory, not a file: there is one checkpoint per condition.
    p.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    p.add_argument(
        "--no-figure", action="store_true",
        help="skip the figures (used by the fast tests)",
    )
    return p.parse_args(argv)


def main(argv=None) -> Dict[str, object]:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    vs.configure_case("silver")
    device = torch.device(args.device)
    conditions = resolve_conditions(args.conditions)
    n_epochs, n_points, lbfgs_steps, val_points = (
        (QUICK_EPOCHS, QUICK_POINTS, 0, 800)
        if args.quick
        else (args.epochs, args.n_points, args.lbfgs_steps, args.val_points)
    )
    lbfgs_dtype = torch.float64 if args.lbfgs_dtype == "float64" else torch.float32
    schedule: Dict[str, object] = {
        "epochs": n_epochs, "n_points": n_points, "lbfgs_steps": lbfgs_steps,
        "lbfgs_dtype": args.lbfgs_dtype, "learning_rate": args.lr, "seed": args.seed,
        "val_points": val_points, "quick": bool(args.quick),
        "note": (
            "Shortened relative to validate_spp.py (4000 epochs / 2048 points / 50 "
            "L-BFGS steps) to fit five arms in one CPU budget. Identical across arms, "
            "so differences are meaningful and absolute values are not."
        ),
    }
    logger.info("schedule: %s", schedule)

    reference = analytical_baseline(val_points=val_points, seed=args.seed, device=device)
    logger.info(
        "exact-mode floor: rel L2 %.2e, |E|/|E_ex| %.4f, E_z jump %.4f (exact %.4f), "
        "jump err %.2e",
        reference["rel_l2_total"], reference["amplitude_ratio_E"],
        reference["ez_jump_ratio_abs"], reference["ez_jump_ratio_exact_abs"],
        reference["ez_jump_rel_error"],
    )

    results: List[Result] = []
    for condition in conditions:
        result = run_condition(
            condition,
            n_epochs=n_epochs, n_points=n_points, learning_rate=args.lr,
            lbfgs_steps=lbfgs_steps, lbfgs_dtype=lbfgs_dtype, val_points=val_points,
            seed=args.seed, device=device,
            model_dir=None if args.quick else args.model_dir, resume=args.resume,
        )
        results.append(result)
        logger.info(
            "%-16s rel L2 %.3e | |E|/|E_ex| %.3f | k_spp err %.2e | kappa_m err %.2e | "
            "E_z jump err %.3f | tier %s | %.0f s",
            condition.key, result.metrics["rel_l2_total"],
            result.metrics["amplitude_ratio_E"], result.metrics["k_spp_rel_error"],
            result.metrics["kappa_m_fit_rel_error"], result.metrics["ez_jump_rel_error"],
            result.metrics["success_tier"], result.metrics["train_time_s"],
        )

    figures = (
        {} if args.no_figure
        else plot_ablation(results, reference=reference, out_dir=args.figures_dir)
    )
    report = build_report(results, reference, schedule, figures)
    out = Path(args.figures_dir) / "results.json"
    merged = report
    if len(conditions) < len(CONDITIONS) and out.exists():
        # A partial run must not delete the arms already on disk: this study is
        # meant to be run one arm per process inside a wall-clock limit.
        import json

        try:
            with open(out) as fh:
                previous = json.load(fh)
            if isinstance(previous.get("conditions"), dict):
                merged = dict(report)
                merged["conditions"] = {**previous["conditions"], **report["conditions"]}
        except (OSError, ValueError):
            merged = report
    write_json_report(out, merged)
    print("\n" + summary_table(results, reference) + "\n")
    logger.info("wrote %s and %d figure(s)", out, len(figures))
    return merged


if __name__ == "__main__":
    main()
