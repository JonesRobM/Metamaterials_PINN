# Contributing

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements-dev.txt
```

A plain `python -m venv` works too; `torch` is CPU-only here, so install it from
`https://download.pytorch.org/whl/cpu` to avoid pulling multi-gigabyte CUDA
wheels.

## Checks

```bash
.venv/bin/python -m pytest -q            # full suite
.venv/bin/python -m pytest -q -m "not slow"   # what CI runs
.venv/bin/ruff check .                   # lint
.venv/bin/python scripts/validate_physics.py
```

CI runs the non-slow suite on Python 3.10 and 3.12 with a coverage floor of
85%, plus a smoke run of the analytics-only examples.

## Conventions

**Sign convention.** Everything uses `e^{-iwt}`, so `curl E = i w mu0 H`,
`curl H = -i w eps0 eps E`, lossy media have `Im eps > 0`, and bound or decaying
solutions have `Im k > 0` and `Re kappa > 0`. Mixing conventions is the single
easiest way to introduce a silent sign error — the repo was internally
inconsistent once, and fixing it required flipping two curl signs rather than
negating every material parameter.

**Every physics routine needs an independent reference.** Not a comparison
against another routine in this repository — an exact solution, a literature
value, or a solver validated separately. Existing benchmarks live in
`tests/test_benchmark_*.py` (uniaxial extraordinary wave, Fresnel, silver SPP
against Johnson & Christy) and are the pattern to follow. Prefer a test that
fails loudly for a wrong answer over one that passes for a small number.

**Validate the measurement, not just the model.** Each experiment pushes its own
reference solution through the identical validation pipeline as a self-check, so
a reported error reflects the network rather than the metric.

**Long runs must checkpoint.** Training takes 30-80 minutes on CPU. Write the
best weights atomically at intervals and support `--resume`; a run that saves
only at the end will eventually be killed at 99% and lose everything (this has
happened here more than once).

**Report negative results.** If an experiment misses its target, say which
metric and why. `docs/plans/` records failures and diagnoses alongside
successes, and that is deliberate.

## Numerical practice

- Physics in `float64` wherever it is affordable; `float32` sets a residual
  floor around 1e-3 relative for these problems.
- Choose square-root branches explicitly. Principal branches silently pick the
  wrong sheet near resonances and branch cuts.
- Constants come from `src/constants.py` — never redefine them locally. `c` is
  exact by SI definition and `eps0` is derived, so cross-module comparisons are
  bit-exact.

## Git policy

Automated tooling must not run `git` or `gh` without per-command approval from
the repository owner. Stage the *content* of your work by editing files
normally, and leave version control to the owner.
