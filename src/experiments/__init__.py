"""
Shared machinery for the ``examples/validate_*.py`` experiments.

The five SPP validation experiments differ in their *physics* — one interface
or twenty, fixed ε or dispersive, a single mode or a two-parameter surrogate —
but they run the same apparatus around it. That apparatus lives here so each
experiment file is the physics plus a thin wiring layer:

==================  =======================================================
:mod:`.adapter`      displacement adapters: the exact ``E_z`` interface jump
:mod:`.training`     the Adam → float64-L-BFGS loop, ramping, best-iterate
                     tracking and atomic checkpointing behind ``--resume``
:mod:`.scaling`      the ``k₀(ω)`` nondimensional frame and ``[-1, 1]`` features
:mod:`.conditioning` parameter-conditioned networks and per-row weighted losses
:mod:`.measurement`  continuity residuals and decay-constant fits
:mod:`.reporting`    metrics.json, success tiers, the common argparse options
:mod:`.plotting`     the two-phase training-history figure
==================  =======================================================

Nothing here knows about a particular material, geometry or mode: an
experiment passes in its own sampling and loss closures. That boundary is what
lets the machinery be shared without the physics leaking between experiments.
"""

from src.experiments.adapter import (
    DisplacementAdapter,
    LayeredAdapter,
    TwoMediumAdapter,
)
from src.experiments.conditioning import (
    SCALED_MAXWELL,
    ColumnConditionedNet,
    weighted_curl_loss,
    weighted_divergence_loss,
)
from src.experiments.measurement import (
    continuity_residuals,
    fit_decay_constants,
)
from src.experiments.plotting import plot_two_phase_history
from src.experiments.reporting import (
    add_common_args,
    add_core_args,
    add_output_args,
    banded_success_tier,
    relative_l2,
    write_json_report,
)
from src.experiments.scaling import LinearFeature, k0_of
from src.experiments.training import (
    TrainingConfig,
    load_checkpoint_history,
    load_core_checkpoint,
    ramp_at,
    run_training,
    write_checkpoint,
)

__all__ = [
    "DisplacementAdapter",
    "LayeredAdapter",
    "TwoMediumAdapter",
    "SCALED_MAXWELL",
    "ColumnConditionedNet",
    "weighted_curl_loss",
    "weighted_divergence_loss",
    "LinearFeature",
    "k0_of",
    "TrainingConfig",
    "load_checkpoint_history",
    "load_core_checkpoint",
    "ramp_at",
    "run_training",
    "write_checkpoint",
    "continuity_residuals",
    "fit_decay_constants",
    "plot_two_phase_history",
    "add_common_args",
    "add_core_args",
    "add_output_args",
    "banded_success_tier",
    "relative_l2",
    "write_json_report",
]
