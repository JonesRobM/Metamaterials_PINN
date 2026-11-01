# CPU-only image for the Metamaterials PINN project.
#
# PyTorch's default wheels bundle CUDA (several GB) which is useless here — the
# experiments are CPU-bound by design — so we pull from the CPU wheel index.
#
#   docker build -t metamaterials-pinn .
#   docker run --rm metamaterials-pinn                    # physics validation
#   docker run --rm metamaterials-pinn pytest -q -m "not slow"
#   docker run --rm -v "$PWD/figures:/app/figures" metamaterials-pinn \
#       python examples/dispersion_analysis.py            # keep the figures

FROM python:3.12-slim

# Headless plotting, no .pyc clutter, unbuffered logs for `docker run` output.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# Dependencies first, so edits to source do not invalidate the (large) torch layer.
COPY pyproject.toml README.md LICENSE ./
RUN python -m pip install --upgrade pip \
 && pip install torch --index-url https://download.pytorch.org/whl/cpu

# Project sources. `pip install .` puts `src` and `config` on the path; the
# scripts/ and examples/ trees are not packaged, so they are copied and run
# from WORKDIR.
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY examples/ ./examples/
COPY tests/ ./tests/
COPY conftest.py ./
RUN pip install ".[dev]"

# Generate the design-study summary that two example test modules read at
# import time. It is a derived artefact (~1 s to produce), so the image builds
# its own rather than baking in a copy that could go stale; it doubles as a
# build-time smoke test of the analytics stack.
RUN python examples/hyperbolic_metamaterial.py > /dev/null

# Run as a non-root user; give it a writable home for matplotlib's cache.
RUN useradd --create-home --shell /bin/bash runner \
 && mkdir -p /tmp/matplotlib && chown -R runner:runner /app /tmp/matplotlib
USER runner

# Fast, meaningful default: validate the physics implementation (~2 s).
# Not a training run — those take 30-80 minutes.
CMD ["python", "scripts/validate_physics.py"]
