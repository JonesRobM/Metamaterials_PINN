"""
Inverse Design Through the Closed-Form SPP Dispersion

Gradient-based inverse design of a uniaxial metamaterial's effective-medium
parameters ``(ε_t, ε_n)`` by Adam descent *through* the analytical SPP
dispersion (:mod:`src.design`, the differentiable torch twin of
:class:`src.physics.metamaterial.MetamaterialProperties`). No neural networks
and no training runs — the physics itself is the differentiable model.

Setting: TM SPP at λ₀ = 633 nm on the interface between air (ε_d = 1) and a
uniaxial metamaterial ``diag(ε_t, ε_t, ε_n)``. The optimisation variables are
``(Re ε_t, Re ε_n)``; the imaginary parts are fixed at the passive values
``Im ε_t = 0.2``, ``Im ε_n = 0.05`` (the repo's uniaxial benchmark case). The
bound-mode (supported-region) gate enters as a soft penalty
(:func:`src.design.support_penalty_torch`, large loss outside), plus a soft
box ``|Re ε| ≤ 60`` keeping the search finite.

Problems (``--problem``):

1. ``wavevector`` — find ``(ε_t, ε_n)`` giving ``Re k_spp / k₀ = 1.8``.
   The target fixes one scalar, so the solution set is a 1-D contour in the
   ``(Re ε_t, Re ε_n)`` plane (degenerate problem): two different inits
   converge to two distinct designs, shown on a background map of
   ``Re k_spp/k₀`` with the target contour.
2. ``propagation`` — maximise ``L = 1/(2 Im k_spp)`` subject to the
   confinement constraint ``δ_d = 1/Re κ_d ≤ 300 nm`` (exterior penalty
   method; the objective is the dimensionless ``Im k_spp/k₀``). The
   constraint bound is swept over several values to trace the L-vs-δ_d
   Pareto front — the classic plasmonic trade-off: longer propagation costs
   confinement, and the constraint is active at every optimum.
3. ``enhancement`` — hit a target interface field-enhancement factor
   ``|E_z|/|E_x| = |k_spp|/|κ_d| = 2.5`` (achievable: it corresponds to a
   moderately bound mode with ``n_eff ≈ 1.09``).

Every found design is re-verified against the scalar reference
implementation (``MetamaterialProperties``); the achieved and reference
values agree to ~machine precision. Figures and ``design_results.json``
(found parameters + achieved vs target) go to ``figures/inverse_design/``.

Usage::

    python examples/inverse_design.py [--problem {wavevector,propagation,enhancement,all}]
                                      [--steps 2000] [--figures-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constants import C0  # noqa: E402
from src.design import (  # noqa: E402
    decay_constants_torch,
    field_enhancement_torch,
    is_spp_supported_torch,
    make_eps,
    penetration_depths_torch,
    propagation_length_torch,
    spp_wavevector_torch,
    support_penalty_torch,
)
from src.physics.metamaterial import MetamaterialProperties  # noqa: E402

logger = logging.getLogger("inverse_design")

# --------------------------------------------------------------------------- physics
LAMBDA0 = 633e-9  # m, free-space wavelength
OMEGA = 2 * math.pi * C0 / LAMBDA0  # rad/s
K0 = OMEGA / C0  # rad/m
EPS_D = 1.0  # air upper half-space
IM_T = 0.2  # fixed Im eps_t (passive, the uniaxial benchmark's value)
IM_N = 0.05  # fixed Im eps_n

# --------------------------------------------------------------------------- optimisation
DEFAULT_STEPS = 2000
LOG_EVERY = 50
LR = 0.03  # Adam step for the target problems
LR_PROPAGATION = 0.05  # problem 2 travels further in eps space
SUPPORT_WEIGHT = 1e3  # soft supported-region gate (large loss outside)
BOX = 60.0  # soft box |Re eps| <= BOX keeps the search finite
CONFINEMENT_WEIGHT = 30.0  # exterior penalty on delta_d > bound (problem 2)

# --------------------------------------------------------------------------- targets
TARGET_N_EFF = 1.8  # problem 1: Re k_spp / k0
WAVEVECTOR_INITS: Tuple[Tuple[float, float], ...] = ((-2.0, -2.0), (-0.3, 8.0))
DELTA_D_MAX_NM = 300.0  # problem 2 primary confinement bound
PARETO_BOUNDS_NM: Tuple[float, ...] = (150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 600.0)
PROPAGATION_INIT = (-10.0, -10.0)
TARGET_ENHANCEMENT = 2.5  # problem 3: |k_spp| / |kappa_d|
ENHANCEMENT_INIT = (-2.0, 3.0)

FIGURES_DIR = REPO_ROOT / "figures" / "inverse_design"

# A loss function maps (re_t, re_n, eps_t, eps_n) -> (scalar loss, achieved metric).
LossFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                  Tuple[torch.Tensor, float]]


# --------------------------------------------------------------------------- helpers
def reference_material(eps_t: complex, eps_n: complex) -> MetamaterialProperties:
    """Scalar reference for a design (optical axis z: ``eps_parallel`` = ε_n)."""
    return MetamaterialProperties(
        eps_parallel=eps_n, eps_perpendicular=eps_t, optical_axis="z", omega=OMEGA
    )


def regularisation(
    re_t: torch.Tensor, re_n: torch.Tensor, eps_t: torch.Tensor, eps_n: torch.Tensor
) -> torch.Tensor:
    """Soft supported-region gate plus the soft search box on ``|Re ε|``."""
    penalty = SUPPORT_WEIGHT * support_penalty_torch(eps_t, eps_n, EPS_D)
    penalty = penalty + torch.relu(re_t.abs() - BOX) ** 2 + torch.relu(re_n.abs() - BOX) ** 2
    return penalty


def optimise(
    loss_fn: LossFn,
    init: Tuple[float, float],
    steps: int,
    lr: float = LR,
    tag: str = "",
) -> Tuple[float, float, Dict[str, list]]:
    """
    Adam descent on ``(Re ε_t, Re ε_n)`` with the imaginary parts fixed.

    Returns the found real parts and a per-step trace
    (``step``, ``loss``, ``metric``, ``re_t``, ``re_n``).
    """
    re_t = torch.tensor(float(init[0]), dtype=torch.float64, requires_grad=True)
    re_n = torch.tensor(float(init[1]), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([re_t, re_n], lr=lr)
    trace: Dict[str, list] = {"step": [], "loss": [], "metric": [], "re_t": [], "re_n": []}
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        eps_t = make_eps(re_t, IM_T)
        eps_n = make_eps(re_n, IM_N)
        loss, metric = loss_fn(re_t, re_n, eps_t, eps_n)
        loss.backward()
        opt.step()
        trace["step"].append(step)
        trace["loss"].append(float(loss.detach()))
        trace["metric"].append(metric)
        trace["re_t"].append(float(re_t.detach()))
        trace["re_n"].append(float(re_n.detach()))
        if step % LOG_EVERY == 0 or step == steps - 1:
            logger.info(
                "%s step %5d | loss %.3e | metric %.6g | Re eps_t %+.4f | Re eps_n %+.4f",
                tag, step, trace["loss"][-1], metric, trace["re_t"][-1], trace["re_n"][-1],
            )
    return float(re_t.detach()), float(re_n.detach()), trace


def describe_design(re_t: float, re_n: float) -> Dict[str, object]:
    """
    Full description of a found design: torch metrics and their scalar-reference
    (:class:`MetamaterialProperties`) verification values.
    """
    eps_t = complex(re_t, IM_T)
    eps_n = complex(re_n, IM_N)
    et = torch.tensor(eps_t, dtype=torch.complex128)
    en = torch.tensor(eps_n, dtype=torch.complex128)
    with torch.no_grad():
        k, _, _ = decay_constants_torch(et, en, OMEGA, EPS_D)
        length = propagation_length_torch(et, en, OMEGA, EPS_D)
        delta_d, delta_m = penetration_depths_torch(et, en, OMEGA, EPS_D)
        enhancement = field_enhancement_torch(et, en, OMEGA, EPS_D)
        supported = bool(is_spp_supported_torch(et, en, EPS_D))
    m = reference_material(eps_t, eps_n)
    k_ref = m.spp_wavevector(eps_dielectric=EPS_D)
    return {
        "eps_t": [re_t, IM_T],
        "eps_n": [re_n, IM_N],
        "supported": supported,
        "reference_supported": bool(m.is_spp_supported(eps_dielectric=EPS_D)),
        "n_eff": float(k.real) / K0,
        "reference_n_eff": k_ref.real / K0,
        "L_um": float(length) * 1e6,
        "reference_L_um": m.propagation_length(eps_dielectric=EPS_D) * 1e6,
        "delta_d_nm": float(delta_d) * 1e9,
        "reference_delta_d_nm": m.penetration_depth_dielectric(eps_dielectric=EPS_D) * 1e9,
        "delta_m_nm": float(delta_m) * 1e9,
        "reference_delta_m_nm": m.penetration_depth_metamaterial(eps_dielectric=EPS_D) * 1e9,
        "enhancement": float(enhancement),
        "reference_enhancement": m.field_enhancement_factor(eps_dielectric=EPS_D),
    }


# --------------------------------------------------------------------------- figures
def _plot_convergence(
    traces: Sequence[Dict[str, list]],
    labels: Sequence[str],
    metric_label: str,
    target: float | None,
    title: str,
    path: Path,
) -> None:
    """Two-panel convergence figure: total loss (log) and achieved metric."""
    fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(11, 4.2))
    for trace, label in zip(traces, labels, strict=True):
        ax_loss.semilogy(trace["step"], trace["loss"], label=label, linewidth=1)
        ax_metric.plot(trace["step"], trace["metric"], label=label, linewidth=1)
    if target is not None:
        ax_metric.axhline(target, color="k", ls="--", lw=0.8, label="target")
    ax_loss.set_xlabel("Adam step")
    ax_loss.set_ylabel("loss")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(fontsize=8)
    ax_metric.set_xlabel("Adam step")
    ax_metric.set_ylabel(metric_label)
    ax_metric.grid(alpha=0.3)
    ax_metric.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_wavevector_map(
    traces: Sequence[Dict[str, list]], target: float, path: Path
) -> None:
    """Background map of ``Re k_spp/k₀`` with the target contour and trajectories."""
    re_t = np.linspace(-3.0, -0.02, 240)
    re_n = np.linspace(-3.0, 12.0, 300)
    T, N = np.meshgrid(re_t, re_n, indexing="xy")
    et = torch.tensor(T + 1j * IM_T, dtype=torch.complex128)
    en = torch.tensor(N + 1j * IM_N, dtype=torch.complex128)
    with torch.no_grad():
        n_eff = (spp_wavevector_torch(et, en, OMEGA, EPS_D).real / K0).numpy()
        supported = is_spp_supported_torch(et, en, EPS_D).numpy()
    n_eff = np.where(supported, n_eff, np.nan)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.pcolormesh(T, N, n_eff, cmap="viridis", vmin=1.0, vmax=3.0, shading="auto")
    fig.colorbar(im, ax=ax, label=r"Re $k_{spp}/k_0$")
    cs = ax.contour(T, N, n_eff, levels=[target], colors="w", linewidths=1.5)
    ax.clabel(cs, fmt={target: f"target {target}"}, fontsize=8)
    for i, trace in enumerate(traces):
        ax.plot(trace["re_t"], trace["re_n"], color=f"C{i + 1}", lw=1.2,
                label=f"trajectory (init {i + 1})")
        ax.plot(trace["re_t"][0], trace["re_n"][0], "o", color=f"C{i + 1}", ms=6, mfc="none")
        ax.plot(trace["re_t"][-1], trace["re_n"][-1], "*", color=f"C{i + 1}", ms=13)
    ax.set_xlabel(r"Re $\varepsilon_t$")
    ax.set_ylabel(r"Re $\varepsilon_n$")
    ax.set_title(
        rf"Degeneracy of the wavevector target (Im $\varepsilon$ fixed at {IM_T}/{IM_N}):"
        "\ntwo inits, two designs on the same solution contour"
    )
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_pareto(pareto: List[Dict[str, object]], path: Path) -> None:
    """L-vs-δ_d Pareto front from the constraint-bound sweep."""
    dd = [p["delta_d_nm"] for p in pareto]
    lengths = [p["L_um"] for p in pareto]
    bounds = [p["bound_nm"] for p in pareto]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.semilogy(dd, lengths, "o-", color="C0")
    for x, y, b in zip(dd, lengths, bounds, strict=True):
        ax.annotate(f"{b:.0f} nm", (x, y), textcoords="offset points", xytext=(6, -10),
                    fontsize=8)
    ax.set_xlabel(r"achieved confinement $\delta_d$ [nm]")
    ax.set_ylabel(r"propagation length $L$ [$\mu$m]")
    ax.set_title("Propagation-confinement Pareto front (labels: constraint bound)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- problem 1
def run_wavevector(
    steps: int = DEFAULT_STEPS,
    figures_dir: Path = FIGURES_DIR,
    target: float = TARGET_N_EFF,
    inits: Sequence[Tuple[float, float]] = WAVEVECTOR_INITS,
) -> Dict[str, object]:
    """
    Problem 1: hit ``Re k_spp/k₀ = target`` from several inits.

    The target under-determines ``(ε_t, ε_n)`` (one equation, two unknowns), so
    each init lands on a different point of the solution contour — the returned
    ``solution_separation`` quantifies the degeneracy.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    def loss_fn(re_t, re_n, eps_t, eps_n):
        k = spp_wavevector_torch(eps_t, eps_n, OMEGA, EPS_D)
        n_eff = k.real / K0
        loss = (n_eff - target) ** 2 + regularisation(re_t, re_n, eps_t, eps_n)
        return loss, float(n_eff.detach())

    solutions: List[Dict[str, object]] = []
    traces: List[Dict[str, list]] = []
    for i, init in enumerate(inits):
        re_t, re_n, trace = optimise(loss_fn, init, steps, lr=LR, tag=f"wavevector[{i}]")
        design = describe_design(re_t, re_n)
        design["init"] = list(init)
        design["target_n_eff"] = target
        design["final_loss"] = trace["loss"][-1]
        logger.info(
            "wavevector[%d]: eps_t = %.4f%+.2fj, eps_n = %.4f%+.2fj, "
            "achieved n_eff = %.6f (reference %.6f, target %.2f), supported=%s",
            i, re_t, IM_T, re_n, IM_N, design["n_eff"], design["reference_n_eff"],
            target, design["supported"],
        )
        solutions.append(design)
        traces.append(trace)

    separation = math.hypot(
        solutions[0]["eps_t"][0] - solutions[1]["eps_t"][0],
        solutions[0]["eps_n"][0] - solutions[1]["eps_n"][0],
    ) if len(solutions) >= 2 else 0.0

    conv_path = figures_dir / "wavevector_convergence.png"
    map_path = figures_dir / "wavevector_map.png"
    _plot_convergence(
        traces, [f"init {i + 1}: {init}" for i, init in enumerate(inits)],
        r"Re $k_{spp}/k_0$", target,
        f"Problem 1: target wavevector (two degenerate solutions, separation {separation:.2f})",
        conv_path,
    )
    _plot_wavevector_map(traces, target, map_path)

    return {
        "target_n_eff": target,
        "solutions": solutions,
        "solution_separation": separation,
        "figures": {"convergence": str(conv_path), "map": str(map_path)},
    }


# --------------------------------------------------------------------------- problem 2
def run_propagation(
    steps: int = DEFAULT_STEPS,
    figures_dir: Path = FIGURES_DIR,
    bounds_nm: Sequence[float] = PARETO_BOUNDS_NM,
    primary_nm: float = DELTA_D_MAX_NM,
) -> Dict[str, object]:
    """
    Problem 2: maximise ``L = 1/(2 Im k_spp)`` subject to ``δ_d ≤ bound``
    (exterior penalty), sweeping the bound to trace the Pareto front.

    The bounds are optimised in ascending order with warm starts; the entry at
    ``primary_nm`` is reported as *the* solution. The confinement constraint is
    active at every optimum — the L-vs-δ_d trade-off is monotone.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    bounds = sorted(float(b) for b in bounds_nm)
    if primary_nm not in bounds:
        raise ValueError(f"primary_nm={primary_nm} must be one of bounds_nm={bounds}")

    def make_loss(delta_max_m: float) -> LossFn:
        def loss_fn(re_t, re_n, eps_t, eps_n):
            k, kappa_d, _ = decay_constants_torch(eps_t, eps_n, OMEGA, EPS_D)
            # Dimensionless objective: minimise Im k/k0 = 1/(2 k0 L).
            im_hat = k.imag / K0
            # clamp_min keeps 1/Re kappa_d finite in the (heavily penalised)
            # unbound region; the support penalty supplies the restoring gradient.
            delta_d = 1.0 / kappa_d.real.clamp_min(1e-3 * K0)
            violation = torch.relu(delta_d / delta_max_m - 1.0)
            loss = (
                im_hat
                + CONFINEMENT_WEIGHT * violation**2
                + regularisation(re_t, re_n, eps_t, eps_n)
            )
            metric = float(0.5 / k.imag.detach()) * 1e6  # L in um
            return loss, metric
        return loss_fn

    pareto: List[Dict[str, object]] = []
    primary: Dict[str, object] = {}
    primary_trace: Dict[str, list] = {}
    warm = PROPAGATION_INIT
    for bound in bounds:
        tag = f"propagation[{bound:.0f}nm]"
        re_t, re_n, trace = optimise(
            make_loss(bound * 1e-9), warm, steps, lr=LR_PROPAGATION, tag=tag
        )
        warm = (re_t, re_n)
        design = describe_design(re_t, re_n)
        design["bound_nm"] = bound
        design["final_loss"] = trace["loss"][-1]
        logger.info(
            "%s: eps_t = %.3f%+.2fj, eps_n = %.3f%+.2fj, L = %.2f um "
            "(reference %.2f), delta_d = %.1f nm (bound %.0f), supported=%s",
            tag, re_t, IM_T, re_n, IM_N, design["L_um"], design["reference_L_um"],
            design["delta_d_nm"], bound, design["supported"],
        )
        pareto.append(design)
        if bound == primary_nm:
            primary = design
            primary_trace = trace

    conv_path = figures_dir / "propagation_convergence.png"
    pareto_path = figures_dir / "propagation_pareto.png"
    _plot_convergence(
        [primary_trace], [f"bound {primary_nm:.0f} nm"], r"$L$ [$\mu$m]", None,
        rf"Problem 2: maximise $L$ s.t. $\delta_d \leq$ {primary_nm:.0f} nm "
        rf"(found $L$ = {primary['L_um']:.1f} $\mu$m at $\delta_d$ = "
        rf"{primary['delta_d_nm']:.0f} nm)",
        conv_path,
    )
    _plot_pareto(pareto, pareto_path)

    return {
        "constraint_delta_d_nm": primary_nm,
        "solution": primary,
        "pareto": pareto,
        "figures": {"convergence": str(conv_path), "pareto": str(pareto_path)},
    }


# --------------------------------------------------------------------------- problem 3
def run_enhancement(
    steps: int = DEFAULT_STEPS,
    figures_dir: Path = FIGURES_DIR,
    target: float = TARGET_ENHANCEMENT,
    init: Tuple[float, float] = ENHANCEMENT_INIT,
) -> Dict[str, object]:
    """Problem 3: hit the target interface field-enhancement ``|k_spp|/|κ_d|``."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    def loss_fn(re_t, re_n, eps_t, eps_n):
        enhancement = field_enhancement_torch(eps_t, eps_n, OMEGA, EPS_D)
        loss = (enhancement - target) ** 2 + regularisation(re_t, re_n, eps_t, eps_n)
        return loss, float(enhancement.detach())

    re_t, re_n, trace = optimise(loss_fn, init, steps, lr=LR, tag="enhancement")
    design = describe_design(re_t, re_n)
    design["init"] = list(init)
    design["target_enhancement"] = target
    design["final_loss"] = trace["loss"][-1]
    logger.info(
        "enhancement: eps_t = %.4f%+.2fj, eps_n = %.4f%+.2fj, achieved %.6f "
        "(reference %.6f, target %.2f), supported=%s",
        re_t, IM_T, re_n, IM_N, design["enhancement"], design["reference_enhancement"],
        target, design["supported"],
    )

    conv_path = figures_dir / "enhancement_convergence.png"
    _plot_convergence(
        [trace], [f"init {init}"], r"$|k_{spp}|/|\kappa_d|$", target,
        f"Problem 3: target field enhancement {target}", conv_path,
    )

    return {
        "target_enhancement": target,
        "solution": design,
        "figures": {"convergence": str(conv_path)},
    }


# --------------------------------------------------------------------------- main
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--problem",
        choices=("wavevector", "propagation", "enhancement", "all"),
        default="all",
        help="which design problem to run",
    )
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Adam steps per run")
    p.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> Dict[str, object]:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    torch.manual_seed(args.seed)
    figures_dir: Path = args.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "inverse design at lambda0 = %.0f nm (eps_d = %s, Im eps fixed at %s/%s), "
        "problem=%s, steps=%d", LAMBDA0 * 1e9, EPS_D, IM_T, IM_N, args.problem, args.steps,
    )

    results: Dict[str, object] = {
        "lambda0_nm": LAMBDA0 * 1e9,
        "eps_d": EPS_D,
        "im_eps_t": IM_T,
        "im_eps_n": IM_N,
        "steps": args.steps,
    }
    if args.problem in ("wavevector", "all"):
        results["wavevector"] = run_wavevector(args.steps, figures_dir)
    if args.problem in ("propagation", "all"):
        results["propagation"] = run_propagation(args.steps, figures_dir)
    if args.problem in ("enhancement", "all"):
        results["enhancement"] = run_enhancement(args.steps, figures_dir)

    json_path = figures_dir / "design_results.json"
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("wrote %s", json_path)
    return results


if __name__ == "__main__":
    main()
