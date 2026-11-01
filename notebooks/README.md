# Notebooks

## `walkthrough.ipynb` — start here

The narrative tour of the project: the problem, the physics, the failure mode that
shaped the design, the idea that fixes it, the results, and the limits. Roughly
fifteen minutes to read top to bottom.

It is committed **with its outputs**, so it renders on GitHub without being run. If
you do run it, it executes end to end in about six seconds on a laptop CPU and
trains nothing. Everything in it is either computed live from `src/` — the Drude and
effective-medium permittivities, the surface-plasmon dispersion, an exact
transfer-matrix solve of the real 13-interface Ag/silica stack, and the
effective-medium error against layer period — or read from the `metrics.json` that
each training experiment writes into `figures/`.

```bash
uv pip install -r requirements-dev.txt jupyter        # jupyter is dev-only
.venv/bin/python -m jupyter lab notebooks/walkthrough.ipynb
```

The package itself does not depend on Jupyter; `requirements-dev.txt` does not
install it either. Install `jupyter` separately if you want to execute the notebook
rather than read it.

## Where to go next

| you want | look at |
| --- | --- |
| the equations, conventions and loss design | `docs/physics.md` |
| what each physics routine is benchmarked against | `docs/validation.md` |
| the experiments in detail | `docs/results.md` |
| the caveats | `docs/limitations.md` |
| per-experiment records — hyperparameters, what failed, what was fixed | `docs/plans/` |
| runnable studies (analytics in seconds, training in 30–80 min) | `examples/` |
