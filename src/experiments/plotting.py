r"""
The training-history figure, drawn the way the two-phase schedule needs.

Thousands of Adam epochs and ~10² L-BFGS steps share no useful x-axis. On one
axis the refinement — where most of the final accuracy is won — collapses into a
sliver at the right edge and the figure says nothing about it. So the phases are
drawn side by side, identified by the learning rate: the runner records ``NaN``
there for every L-BFGS row, which makes the split a property of the history
itself rather than something the caller has to remember to pass in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_two_phase_history"]

#: Loss components drawn on every panel, in legend order.
COMPONENT_KEYS = ("total", "curl", "div", "continuity", "boundary")


def plot_two_phase_history(
    history: Dict[str, list],
    out_dir: Path,
    title: str,
    *,
    ylabel: str = "loss (dimensionless, k₀-scaled frame)",
    mark_restarts: bool = False,
    filename: str = "training_history.png",
) -> str:
    """
    Draw the Adam and L-BFGS phases on their own axes and save the figure.

    A run with only one phase gets a single panel, so a ``--quick`` smoke run
    still produces a readable figure rather than an empty second axis.

    Args:
        history: The dict :func:`src.experiments.run_training` returns.
        out_dir: Directory to write into; created if absent.
        title: Figure suptitle (or the single panel's title when there is one
            phase only).
        ylabel: Left-hand axis label.
        mark_restarts: Draw a dotted vertical wherever the Adam learning rate
            jumps back up. Those are **warm restarts** — a run split into
            wall-clock-limited chunks resumes each one with a fresh cosine
            cycle, which is why its Adam curve has a sawtooth, and an unmarked
            sawtooth looks like an instability rather than a schedule.
        filename: Output file name.

    Returns:
        The path written, as a string.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    epoch = np.asarray(history["epoch"], dtype=float)
    lr = np.asarray(history["lr"], dtype=float)
    is_lbfgs = np.isnan(lr)
    n_adam = int((~is_lbfgs).sum())

    restarts = np.array([])
    if mark_restarts and n_adam > 1:
        adam_lr, adam_epoch = lr[~is_lbfgs], epoch[~is_lbfgs]
        restarts = adam_epoch[1:][adam_lr[1:] > adam_lr[:-1] * 1.5]

    if n_adam == 0 or int(is_lbfgs.sum()) == 0:
        fig, axis = plt.subplots(figsize=(8, 5))
        panels = [(axis, np.ones_like(epoch, dtype=bool), "epoch", "training", False)]
    else:
        fig, ax_pair = plt.subplots(
            1, 2, figsize=(12, 5), sharey=True,
            gridspec_kw={"width_ratios": [2.2, 1]},
        )
        panels = [
            (ax_pair[0], ~is_lbfgs, "Adam epoch", "Phase 1: Adam (cosine LR)", False),
            (ax_pair[1], is_lbfgs, "L-BFGS step",
             "Phase 2: float64 L-BFGS refinement", True),
        ]

    for ax, mask, xlabel, panel_title, renumber in panels:
        # The L-BFGS rows are appended after the Adam ones, so their stored
        # "epoch" continues the Adam count; on their own axis they read as steps.
        x = np.arange(int(mask.sum()), dtype=float) if renumber else epoch[mask]
        for key in COMPONENT_KEYS:
            ax.semilogy(x, np.asarray(history[key], dtype=float)[mask],
                        label=key, linewidth=1)
        if not renumber:
            for r in restarts:
                ax.axvline(r, color="0.4", ls=":", lw=1.0)
        ax.set_xlabel(xlabel)
        ax.set_title(panel_title, fontsize=10)
        ax.grid(alpha=0.3, which="both")
    panels[0][0].set_ylabel(ylabel)
    panels[0][0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = out_dir / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)
