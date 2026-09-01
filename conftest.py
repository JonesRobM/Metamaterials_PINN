"""Pytest configuration."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Generated artefact that two example test modules read at *import* time, so it
# has to exist before collection rather than inside a fixture.
HMM_SUMMARY = project_root / "figures" / "hyperbolic" / "hmm_summary.json"


def pytest_configure(config):
    """Regenerate the design-study summary if the checkout does not carry it.

    ``figures/hyperbolic/hmm_summary.json`` is produced by
    ``examples/hyperbolic_metamaterial.py`` and is an *input* to the surrogate
    and dispersion experiments, which read it at import time. A checkout
    without it — a CI runner, a fresh clone, a slim container image — would
    otherwise fail during collection with a bare ``FileNotFoundError`` instead
    of running. Rebuilding is deterministic and takes about two seconds, and
    only happens when the file is genuinely absent.
    """
    if HMM_SUMMARY.exists():
        return
    from examples.hyperbolic_metamaterial import main as build_hmm_study

    build_hmm_study([])
