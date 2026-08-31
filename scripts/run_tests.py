"""Run the full test suite under tests/ with verbose output.

Usage:
    python scripts/run_tests.py [extra pytest args...]
"""

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv=None):
    """Run all tests with detailed output."""
    argv = list(sys.argv[1:] if argv is None else argv)
    print("Running SPP Metamaterial PINN Tests")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print("\nRunning tests...")

    exit_code = pytest.main(
        [str(REPO_ROOT / "tests"), "-v", "--tb=short", "--durations=10", *argv]
    )

    if exit_code == 0:
        print("\nAll tests passed successfully!")
    else:
        print(f"\nTests failed with exit code {exit_code}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
