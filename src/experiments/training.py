r"""
The two-phase optimiser the SPP experiments share, and its checkpointing.

Every ``examples/validate_*.py`` run is the same schedule:

1. **Adam**, cosine-annealed, on a freshly sampled collocation batch each
   epoch, with the interior physics terms multiplied by a ramp
   ``min(1, (epoch+1) / (physics_ramp_frac · n_epochs))``. The ramp exists
   because the stiff metal-side curl residual, applied at full weight from
   epoch 0, pins the network in the trivial ``E = H = 0`` basin before the
   boundary anchor has established the mode's amplitude.
2. **L-BFGS**, optionally in float64, on one *fixed* batch, with the metal-side
   curl preconditioner relaxed (``|ε|^{-1}`` → ``|ε|^{-1/2}``): float32 was the
   residual floor of the plane-wave experiment, and a fixed batch is what makes
   a quasi-Newton method meaningful at all.

Around that sit three things that are fiddly to get right and were previously
written out five times: **best-iterate tracking** (only once the ramp has
finished — a partially ramped total is not comparable to the full objective),
**atomic checkpointing** so an hour-long CPU run survives an interrupt, and
**resume**, which has to carry the loss history and the best-so-far bar across
processes or a resumed chunk's first epochs overwrite a better iterate.

The physics stays with the experiment. :func:`run_training` is given two
closures and knows nothing else:

``sample_batch(n_interior, n_boundary, n_interface, dtype) -> batch``
    Draws one training batch. The ``batch`` is opaque — a dict, a tuple,
    whatever the experiment's losses want. Pass ``lbfgs_sample_batch`` as well
    when phase 2 needs a *different* draw: the band-sweeping experiments
    resample random frequencies every Adam epoch but pin L-BFGS to a fixed set
    of them, because a quasi-Newton method optimising a batch that changes
    under it is not optimising anything.

``compute_losses(batch, ramp=1.0, **phase_kwargs) -> (total, curl, div, cont, bc)``
    The objective. ``total`` is the ramped training loss; the other four are the
    *unramped* components, which is what gets logged, so the history stays
    comparable across the ramp.

Phase 2 passes ``lbfgs_loss_kwargs`` instead of ``ramp`` (the ramp is complete
by then), which is how each experiment names its own relaxed metal weight —
``w_curl_m=...`` for the constant-ε runs, ``curl_power=...`` for the ones whose
preconditioner varies per row.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

__all__ = [
    "TrainingConfig",
    "load_checkpoint_history",
    "load_core_checkpoint",
    "ramp_at",
    "run_training",
    "write_checkpoint",
]

#: History fields every experiment records, in figure-plotting order.
HISTORY_KEYS: Tuple[str, ...] = (
    "epoch", "total", "curl", "div", "continuity", "boundary", "lr",
)

#: Extra field for experiments trained in wall-clock-limited chunks.
WALL_KEY = "wall_s"


# --------------------------------------------------------------------- ramp
def ramp_at(epoch: int, n_epochs: int, physics_ramp_frac: float) -> float:
    r"""
    The interior-physics multiplier at ``epoch`` (0-based), in ``(0, 1]``.

    Linear from the first epoch to ``physics_ramp_frac · n_epochs``, then held
    at 1. The first epoch is already non-zero: a zero-weight first step would
    waste an epoch fitting nothing but the anchor.

    ``physics_ramp_frac = 0`` (or any value that rounds the ramp to under one
    epoch) means "no ramp": full physics weight immediately.
    """
    ramp_epochs = max(1, int(physics_ramp_frac * n_epochs))
    return min(1.0, (epoch + 1) / ramp_epochs)


# --------------------------------------------------------------- checkpoints
def write_checkpoint(
    path: Path,
    state: dict,
    loss: float,
    phase: str,
    history: Optional[Dict[str, list]] = None,
) -> None:
    """
    Atomically save the best weights so far (``rename`` is atomic on POSIX).

    Training runs take about an hour on CPU; without this an interrupted run
    loses everything, and a *non*-atomic write loses everything to a crash
    landing mid-``torch.save``. Pass ``history`` for a run split into
    wall-clock-limited chunks: a training-history figure covering only the last
    chunk would be a figure of the last chunk, not of the training. It is
    omitted from the blob entirely when ``None``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob: Dict[str, Any] = {
        "state_dict": {k: v.cpu() for k, v in state.items()},
        "best_loss": loss,
        "phase": phase,
    }
    if history is not None:
        blob["history"] = history
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(blob, tmp)
    tmp.replace(path)


def load_core_checkpoint(network: nn.Module, path: Path) -> float:
    """Load core weights from a training checkpoint; returns its recorded loss."""
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    network.core.load_state_dict(blob["state_dict"])
    return float(blob.get("best_loss", float("nan")))


def load_checkpoint_history(path: Path) -> Optional[Dict[str, list]]:
    """The loss history stored in a checkpoint, if it has one."""
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    history = blob.get("history")
    return {k: list(v) for k, v in history.items()} if history else None


# ------------------------------------------------------------------- config
@dataclass
class TrainingConfig:
    """
    Schedule and batch sizes for :func:`run_training`.

    ``n_boundary`` and ``n_interface`` are absolute counts, not fractions: the
    band-sweeping experiments need a floor of several points *per sub-block*,
    which the experiment works out for itself. The L-BFGS phase multiplies all
    three counts by ``lbfgs_points_factor``, since a fixed batch can afford to
    be larger than a per-epoch one.
    """

    n_epochs: int
    n_points: int
    n_boundary: int
    n_interface: int
    learning_rate: float
    physics_ramp_frac: float
    lbfgs_steps: int = 0
    lbfgs_dtype: torch.dtype = torch.float64
    lbfgs_points_factor: int = 1
    log_every: int = 100


_ADAM_LOG = (
    "epoch %5d | total %.3e | curl %.3e | div %.3e | cont %.3e | bc %.3e | lr %.2e | %.0fs"
)
_LBFGS_LOG = "lbfgs %3d | total %.3e | curl %.3e | div %.3e | cont %.3e | bc %.3e | %.0fs"


def run_training(
    network: nn.Module,
    config: TrainingConfig,
    sample_batch: Callable[..., Any],
    compute_losses: Callable[..., Tuple[torch.Tensor, ...]],
    logger: Logger,
    *,
    lbfgs_sample_batch: Optional[Callable[..., Any]] = None,
    lbfgs_loss_kwargs: Optional[Dict[str, Any]] = None,
    checkpoint_path: Optional[Path] = None,
    save_history: bool = False,
    track_wall: bool = False,
    final_checkpoint: bool = False,
    initial_history: Optional[Dict[str, list]] = None,
    initial_best_loss: float = float("inf"),
    lbfgs_note: str = "",
) -> Tuple[nn.Module, Dict[str, list]]:
    r"""
    Run the Adam → L-BFGS schedule on ``network.core`` and restore its best iterate.

    Args:
        network: The SI-unit wrapper. Only ``network.core`` is optimised, and
            only ``network`` is converted back to float32 at the end.
        config: Schedule and batch sizes.
        sample_batch: ``(n_interior, n_boundary, n_interface, dtype) -> batch``.
        compute_losses: ``(batch, ramp=..., **phase) -> (total, curl, div, cont, bc)``.
        logger: The experiment's logger, so log lines carry its name.
        lbfgs_sample_batch: Draws the single fixed phase-2 batch, when that is
            not just a bigger draw from ``sample_batch``. Same signature.
        lbfgs_loss_kwargs: Phase-2 keyword arguments for ``compute_losses``
            (the relaxed metal curl weight). ``ramp`` is not passed: it is 1 by
            the time L-BFGS runs.
        checkpoint_path: Where to write the ``.partial.pth``; ``None`` disables
            checkpointing entirely.
        save_history: Store the loss history inside the checkpoint too, so a
            ``--resume`` run can continue the curve rather than restart it.
        track_wall: Record a cumulative ``wall_s`` column in the history.
        final_checkpoint: Write one last checkpoint after restoring the best
            iterate, even if this chunk never beat the inherited best —
            otherwise a resumed chunk's slice of the history would be lost.
        initial_history: History from a previous chunk. New epochs are numbered
            after the stored ones and the wall clock continues from its last value.
        initial_best_loss: The best-so-far bar inherited from a previous chunk.
            Without it, a resumed chunk's early epochs — at the restarted, full
            learning rate, so genuinely worse — would checkpoint over a better
            iterate.
        lbfgs_note: Appended to the phase-2 log line, e.g. how many fixed
            frequencies or design points the batch spans.

    Returns:
        ``(network, history)`` with the weights restored to the lowest-loss
        iterate and the network back in float32.
    """
    core = network.core
    lbfgs_loss_kwargs = dict(lbfgs_loss_kwargs or {})

    optimizer = torch.optim.Adam(core.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.n_epochs), eta_min=config.learning_rate * 1e-2
    )

    keys = HISTORY_KEYS + ((WALL_KEY,) if track_wall else ())
    history: Dict[str, list] = {k: list((initial_history or {}).get(k, [])) for k in keys}
    epoch0 = int(max(history["epoch"])) + 1 if history["epoch"] else 0
    # Cumulative wall clock across chunks, so the reported training time is the
    # whole run's and not just this process's.
    wall0 = float(history[WALL_KEY][-1]) if track_wall and history[WALL_KEY] else 0.0

    best_loss = float(initial_best_loss)
    best_state = {k: v.detach().clone() for k, v in core.state_dict().items()}

    def snapshot() -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in core.state_dict().items()}

    def checkpoint(phase: str) -> None:
        if checkpoint_path is None or not math.isfinite(best_loss):
            return
        write_checkpoint(
            checkpoint_path, best_state, best_loss, phase,
            history if save_history else None,
        )

    def record(epoch: int, total: float, parts: Tuple[float, ...], lr: float,
               elapsed: float) -> None:
        history["epoch"].append(epoch)
        history["total"].append(total)
        for key, value in zip(HISTORY_KEYS[2:6], parts, strict=True):
            history[key].append(value)
        history["lr"].append(lr)
        if track_wall:
            history[WALL_KEY].append(wall0 + elapsed)

    core.train()
    t0 = time.perf_counter()

    # ------------------------------------------------------------- phase 1
    for epoch in range(config.n_epochs):
        batch = sample_batch(
            config.n_points, config.n_boundary, config.n_interface, torch.float32
        )
        ramp = ramp_at(epoch, config.n_epochs, config.physics_ramp_frac)
        optimizer.zero_grad(set_to_none=True)
        loss, *components = compute_losses(batch, ramp=ramp)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        parts = tuple(c.item() for c in components)
        lr = optimizer.param_groups[0]["lr"]
        record(epoch0 + epoch, loss_val, parts, lr, time.perf_counter() - t0)

        # Totals from partially ramped epochs are not comparable to the full
        # objective, so track the best iterate only once the ramp is complete.
        if ramp >= 1.0 and loss_val < best_loss and math.isfinite(loss_val):
            best_loss = loss_val
            best_state = snapshot()
        # Checkpoint on a schedule, not only when a new best lands on a log
        # epoch. The final epoch is included explicitly: without it, a run whose
        # length is not a multiple of ``log_every`` never persists whatever it
        # learned after the last log, so an interrupt between the Adam and
        # L-BFGS phases discards up to ``log_every`` epochs of progress.
        if epoch % config.log_every == 0 or epoch == config.n_epochs - 1:
            checkpoint(f"adam:{epoch0 + epoch}")

        if epoch % config.log_every == 0 or epoch == config.n_epochs - 1:
            logger.info(_ADAM_LOG, epoch, loss_val, *parts, lr, time.perf_counter() - t0)

    # ------------------------------------------------------------- phase 2
    if config.lbfgs_steps > 0:
        core.load_state_dict(best_state)
        if config.lbfgs_dtype == torch.float64:
            core.to(torch.float64)
        logger.info("L-BFGS phase in %s%s", config.lbfgs_dtype, lbfgs_note)

        factor = config.lbfgs_points_factor
        batch = (lbfgs_sample_batch or sample_batch)(
            factor * config.n_points,
            factor * config.n_boundary,
            factor * config.n_interface,
            config.lbfgs_dtype,
        )
        lbfgs = torch.optim.LBFGS(
            core.parameters(), lr=1.0, max_iter=20, history_size=50,
            tolerance_grad=1e-12, tolerance_change=1e-14, line_search_fn="strong_wolfe",
        )
        parts_holder: Dict[str, Tuple[float, ...]] = {"parts": (0.0, 0.0, 0.0, 0.0)}

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            loss, *components = compute_losses(batch, **lbfgs_loss_kwargs)
            loss.backward()
            parts_holder["parts"] = tuple(c.item() for c in components)
            return loss

        for step in range(config.lbfgs_steps):
            loss_val = float(lbfgs.step(closure).detach())
            parts = parts_holder["parts"]
            record(
                epoch0 + config.n_epochs + step, loss_val, parts, float("nan"),
                time.perf_counter() - t0,
            )
            if loss_val < best_loss and math.isfinite(loss_val):
                best_loss = loss_val
                best_state = snapshot()
                # L-BFGS steps cost ~20 s each, so checkpoint every improvement:
                # an interrupted refinement then loses at most one step.
                checkpoint(f"lbfgs:{step}")
            logger.info(_LBFGS_LOG, step, loss_val, *parts, time.perf_counter() - t0)
            if not math.isfinite(loss_val):
                logger.warning("L-BFGS produced a non-finite loss; stopping refinement")
                break

    core.load_state_dict(best_state)
    network.to(torch.float32)  # evaluation/serialisation dtype; no-op for float32 phases
    if final_checkpoint:
        checkpoint("final")
    logger.info("restored best weights (loss %.3e)", best_loss)
    return network, history
