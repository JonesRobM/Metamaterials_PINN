# Thin wrappers around the commands in README.md / CONTRIBUTING.md.
PY := .venv/bin/python

.PHONY: help setup test test-fast lint figures site-assets docker-build docker-run clean

help:
	@echo "setup        Create .venv and install the project with dev extras"
	@echo "test         Full test suite"
	@echo "test-fast    Non-slow subset with coverage (what CI runs)"
	@echo "lint         ruff check"
	@echo "figures      Regenerate the analytics-only figures (~45 s)"
	@echo "site-assets  Refresh the figures the GitHub Pages site uses"
	@echo "docker-build Build the CPU image"
	@echo "docker-run   Run the image's default command (physics validation)"

setup:
	uv venv --python 3.12 .venv
	uv pip install -r requirements-dev.txt

test:
	$(PY) -m pytest -q

test-fast:
	$(PY) -m pytest -q -m "not slow" --cov=src --cov=config --cov-report=term-missing

lint:
	.venv/bin/ruff check .

# Analytics only; the PINN experiments take 30-80 minutes and are not included.
figures:
	$(PY) examples/dispersion_analysis.py
	$(PY) examples/hyperbolic_metamaterial.py
	$(PY) examples/inverse_design.py
	$(PY) examples/emt_validity.py

# The Pages site is served from docs/, which Jekyll cannot read outside of, so
# the figures it shows are mirrored here. Re-run after regenerating figures.
SITE_FIGURES := \
  figures/multilayer/field_profiles.png \
  figures/multilayer/k_spp_comparison.png \
  figures/hmm_surrogate/k_spp_surface.png \
  figures/hmm_dispersion/dispersion.png \
  figures/hyperbolic/hmm_spp_dispersion.png \
  figures/hyperbolic/hmm_permittivities.png \
  figures/emt_validity/emt_period_sweep.png \
  figures/dispersion/metamaterial_design_space.png \
  figures/spp_validation/field_maps.png \
  figures/inverse_design/wavevector_map.png \
  figures/ablation/ablation_training_curves.png

site-assets:
	@mkdir -p docs/assets
	@cp $(SITE_FIGURES) docs/assets/
	@echo "Mirrored $(words $(SITE_FIGURES)) figures into docs/assets/"

docker-build:
	docker build -t metamaterials-pinn .

docker-run:
	docker run --rm metamaterials-pinn

clean:
	find . -name '__pycache__' -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache coverage.xml
