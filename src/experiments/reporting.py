r"""
How an experiment reports itself: metrics.json, success tiers, the CLI.

Three small pieces of ceremony that every ``examples/validate_*.py`` performs
identically, and that are worth having in one place mostly so they *stay*
identical — a metrics file whose shape drifts between experiments is a metrics
file nobody can compare across experiments.

The success tiers are the interesting one. Each experiment declares, up front,
what "it worked" means at three levels of ambition — *minimum* (the mode is
qualitatively right), *target* (the number a paper would quote), *stretch* (as
good as the method gets) — and grades itself against them afterwards. Writing
the thresholds down before the run is what stops the bar moving to wherever the
result landed. :func:`banded_success_tier` implements the shape the band-sweep
experiments share; the ones with a genuinely different criterion (a layer
contrast, a comparison against effective-medium theory) keep their own ladder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import torch

__all__ = [
    "add_core_args",
    "add_output_args",
    "add_common_args",
    "banded_success_tier",
    "relative_l2",
    "write_json_report",
]


def relative_l2(pred: torch.Tensor, ref: torch.Tensor) -> float:
    """
    ``‖pred − ref‖ / ‖ref‖`` as a python float, the headline accuracy number.

    The denominator is clamped at 1e-30 rather than special-cased, so a
    zero reference gives a large finite number instead of a ``nan`` that would
    quietly poison a ``max`` over a whole sweep.
    """
    return (
        torch.linalg.vector_norm(pred - ref)
        / torch.linalg.vector_norm(ref).clamp_min(1e-30)
    ).item()


# ------------------------------------------------------------------- metrics
def write_json_report(path: Path, sections: Mapping[str, Any]) -> None:
    """
    Write ``sections`` to ``path`` as indented JSON.

    A thin wrapper, but it is the one place that decides these files are UTF-8,
    two-space-indented and written whole rather than appended to — which is what
    makes them diffable between runs.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(dict(sections), fh, indent=2)


# --------------------------------------------------------------------- tiers
def banded_success_tier(
    summary: Mapping[str, float],
    *,
    stretch: Tuple[float, float],
    target: Tuple[float, float],
    minimum_rel: float = 0.5,
    rel_key: str = "worst_rel_l2",
    k_key: str = "worst_k_spp_rel_error",
    bound_key: str = "bound_mode_everywhere",
) -> str:
    """
    Grade a band- or design-space sweep against its three declared criteria.

    Every tier requires a **bound mode everywhere** — a sweep that lost the mode
    at one corner has not validated the band, however good its average error is.
    Above that gate the ladder is worst-case relative L2 and worst-case ``k_spp``
    error, both taken over the whole sweep rather than averaged, for the same
    reason.

    Args:
        summary: The band summary; must carry ``rel_key``, ``k_key`` and
            ``bound_key``.
        stretch: ``(max rel L2, max k_spp rel error)`` for the stretch tier.
        target: The same pair for the target tier.
        minimum_rel: Relative-L2 ceiling for the minimum tier, which imposes no
            ``k_spp`` condition — recovering a bound mode of roughly the right
            shape is the whole of that claim.

    Returns:
        One of ``"stretch"``, ``"target"``, ``"minimum"``, ``"not met"``.
    """
    rel = summary[rel_key]
    k_err = summary[k_key]
    bound = summary[bound_key] > 0
    if bound and rel < stretch[0] and k_err < stretch[1]:
        return "stretch"
    if bound and rel < target[0] and k_err < target[1]:
        return "target"
    if bound and rel < minimum_rel:
        return "minimum"
    return "not met"


# ----------------------------------------------------------------------- CLI
def add_core_args(
    parser: argparse.ArgumentParser,
    *,
    epochs: int,
    n_points: int,
    lr: float,
    device: str,
    lbfgs_steps: int,
    n_points_help: str = "interior collocation points per epoch",
) -> argparse.ArgumentParser:
    """
    Add the optimiser/geometry-independent options shared by every experiment.

    Emitted in a fixed order — ``--epochs --n-points --lr --seed --device
    --lbfgs-steps --lbfgs-dtype`` — so ``--help`` reads the same everywhere.
    Call this first, add the experiment's own options next, then
    :func:`add_output_args`; that sequence puts each experiment's distinctive
    flags in the middle, where they are easy to spot.
    """
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--n-points", type=int, default=n_points, help=n_points_help)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=device)
    parser.add_argument(
        "--lbfgs-steps", type=int, default=lbfgs_steps,
        help="L-BFGS outer steps after Adam (0 disables)",
    )
    parser.add_argument(
        "--lbfgs-dtype", choices=("float64", "float32"), default="float64",
        help="precision of the L-BFGS refinement phase",
    )
    return parser


def add_output_args(
    parser: argparse.ArgumentParser,
    *,
    quick_epochs: int,
    figures_dir: Path,
    model_out: Optional[Path],
    resume: bool = True,
    quick_extra: str = "",
) -> argparse.ArgumentParser:
    """
    Add ``--resume``, ``--quick`` and the two output paths, in that order.

    Args:
        quick_epochs: Epoch count named in the ``--quick`` help text.
        figures_dir: Default figure directory.
        model_out: Default checkpoint path; ``None`` for the experiments that
            derive it from another flag (the per-case SPP run).
        resume: Whether this experiment supports warm-starting. Only the
            single-interface run does not — it is short enough not to need it.
        quick_extra: Appended to the ``--quick`` help, for extra work a smoke
            run also skips.
    """
    if resume:
        parser.add_argument(
            "--resume", action="store_true",
            help="warm-start from <model-out>.partial.pth if it exists",
        )
    parser.add_argument(
        "--quick", action="store_true",
        help=f"smoke run: {quick_epochs} epochs, 512 points, no L-BFGS{quick_extra}",
    )
    parser.add_argument("--figures-dir", type=Path, default=figures_dir)
    parser.add_argument(
        "--model-out", type=Path, default=model_out,
        **({"help": "checkpoint path (default: per-case under artifacts/models)"}
           if model_out is None else {}),
    )
    return parser


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    epochs: int,
    n_points: int,
    lr: float,
    device: str,
    lbfgs_steps: int,
    quick_epochs: int,
    figures_dir: Path,
    model_out: Optional[Path],
    n_points_help: str = "interior collocation points per epoch",
    resume: bool = True,
    quick_extra: str = "",
) -> argparse.ArgumentParser:
    """:func:`add_core_args` then :func:`add_output_args`, for experiments with
    no options of their own to slot between them."""
    add_core_args(
        parser, epochs=epochs, n_points=n_points, lr=lr, device=device,
        lbfgs_steps=lbfgs_steps, n_points_help=n_points_help,
    )
    return add_output_args(
        parser, quick_epochs=quick_epochs, figures_dir=figures_dir,
        model_out=model_out, resume=resume, quick_extra=quick_extra,
    )
