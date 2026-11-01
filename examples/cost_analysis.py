r"""
What Does the Surrogate Actually Cost? — an honest accounting

The design-space surrogate (``examples/validate_hmm_surrogate.py``) is sold as
amortisation: pay ~82 minutes of training once, then answer k_spp(ω, f)
queries from a forward pass. This script measures whether, and when, that deal
is worth taking, against the two references this project already has:

* the **closed-form** effective-medium dispersion (``mode_constants``), and
* the **transfer-matrix** solve of the real 6-period stack (``find_mode``),
  which is exact for planar multilayers.

The crossover query count is ``N* = T_train / (t_ref − t_query)``: how many
queries before training pays for itself. A negative or absurd ``N*`` is a
result, not a failure — the conclusion this script exists to make quantitative
is that **for planar stacks the surrogate never pays off**, because both
references are already fast and exact. The surrogate's case is geometries
where each reference query is a full-wave solve costing minutes; hypothetical
crossovers for that regime are reported alongside the measured ones.

Timings are medians over ``--repeats`` runs after a discarded warmup, on one
process (``torch`` threads left at default). They are indicative, not
benchmarks; run on an otherwise idle machine.

Usage::

    python examples/cost_analysis.py [--repeats 20] [--out-dir figures/cost_analysis]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.validate_hmm_surrogate import (  # noqa: E402
    OMEGA_REF,
    create_network,
    fill_hat,
    hmm_eps,
    k_spp_from_network,
    mode_constants,
)
from src.constants import C0  # noqa: E402
from src.effective_medium import drude_permittivity  # noqa: E402
from src.transfer_matrix import find_mode  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "figures" / "cost_analysis"
SURROGATE_CKPT = REPO_ROOT / "artifacts" / "models" / "hmm_surrogate.pth"
SURROGATE_METRICS = REPO_ROOT / "figures" / "hmm_surrogate" / "metrics.json"

# The design point every experiment shares: Ag/silica, f = 0.30, six 30 nm
# periods, evaluated at the reference frequency.
FILL = 0.30
PERIOD = 30e-9
N_PERIODS = 6
EPS_D2 = 2.25


def _median_time(fn: Callable[[], object], repeats: int) -> float:
    """Median wall-clock seconds over ``repeats`` calls, after one warmup."""
    fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(statistics.median(samples))


def _tmm_stack(omega: float):
    """(eps_layers, thicknesses) of the real metal-terminated stack."""
    eps_m = drude_permittivity(omega)
    eps, th = [1.0], []
    for _ in range(N_PERIODS):
        eps += [eps_m, EPS_D2]
        th += [FILL * PERIOD, (1.0 - FILL) * PERIOD]
    eps.append(eps_m)
    return eps, th


def measure(repeats: int = 20) -> Dict[str, float]:
    """Per-query costs (seconds) for each way of answering k_spp(ω, f)."""
    omega = OMEGA_REF
    k0 = omega / C0

    # 1) closed-form effective medium
    t_closed_form = _median_time(lambda: mode_constants(omega, FILL), repeats)

    # 2) transfer matrix on the real stack, seeded from the closed form
    eps_layers, thicknesses = _tmm_stack(omega)
    seed = mode_constants(omega, FILL)[0]
    t_tmm = _median_time(
        lambda: find_mode(seed, k0, eps_layers, thicknesses), repeats
    )

    # 3) the trained surrogate, queried the way the inverse-design loop does
    blob = torch.load(SURROGATE_CKPT, map_location="cpu", weights_only=False)
    network = create_network(device=torch.device("cpu"))
    network.load_state_dict(blob["state_dict"])
    network.eval()
    core = network.core  # the adapter-wrapped MLP, as inverse_design_demo uses it
    f_hat = torch.tensor(fill_hat(FILL), dtype=torch.float64)
    t_surrogate_k = _median_time(
        lambda: k_spp_from_network(core, omega, f_hat, device=torch.device("cpu")),
        repeats,
    )

    # 4) a raw surrogate forward pass (per batch of 512 field points), for scale
    coords = torch.randn(512, 5)
    with torch.no_grad():
        t_surrogate_forward = _median_time(lambda: core(coords), repeats)

    train_time_s = float(
        json.load(open(SURROGATE_METRICS))["summary"]["train_time_s"]
    )
    return {
        "t_closed_form_s": t_closed_form,
        "t_tmm_s": t_tmm,
        "t_surrogate_k_query_s": t_surrogate_k,
        "t_surrogate_forward_512pts_s": t_surrogate_forward,
        "surrogate_train_time_s": train_time_s,
        "repeats": float(repeats),
        "hmm_eps_at_ref": [  # provenance: which material the stack used
            [c.real, c.imag] for c in hmm_eps(omega, FILL)
        ],
    }


def crossovers(m: Dict[str, float]) -> Dict[str, float]:
    """Queries needed before training amortises, per reference. Negative means
    the surrogate is slower *per query* than the reference: it never pays."""
    train = m["surrogate_train_time_s"]
    q = m["t_surrogate_k_query_s"]
    out = {}
    for name, t_ref in (
        ("vs_closed_form", m["t_closed_form_s"]),
        ("vs_tmm", m["t_tmm_s"]),
        ("vs_hypothetical_60s_solver", 60.0),
        ("vs_hypothetical_600s_solver", 600.0),
    ):
        gain = t_ref - q
        out[f"crossover_queries_{name}"] = train / gain if gain > 0 else -1.0
    return out


def main(argv=None) -> Dict[str, float]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    m = measure(repeats=args.repeats)
    m.update(crossovers(m))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "cost_summary.json", "w") as fh:
        json.dump(m, fh, indent=2)

    us = 1e6
    print(f"per-query cost (median of {args.repeats}):")
    print(f"  closed-form EMT dispersion    {m['t_closed_form_s']*us:12.1f} us")
    print(f"  transfer matrix, real stack   {m['t_tmm_s']*us:12.1f} us")
    print(f"  surrogate k_spp query         {m['t_surrogate_k_query_s']*us:12.1f} us")
    print(f"  surrogate forward, 512 pts    {m['t_surrogate_forward_512pts_s']*us:12.1f} us")
    print(f"training cost: {m['surrogate_train_time_s']/60:.1f} min")
    print("amortisation crossover (queries; negative = never):")
    for k, v in m.items():
        if k.startswith("crossover"):
            print(f"  {k[len('crossover_queries_'):]:28s} {v:14.0f}")
    return m


if __name__ == "__main__":
    main()
