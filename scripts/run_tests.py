"""
Test runner script for physics module.

Run this script to execute all physics tests and verify implementation correctness.
"""

import sys
import pytest
import torch

def main():
    """Run all physics tests with detailed output."""
    print("Running SPP Metamaterial PINN Physics Tests")
    print("=" * 50)
    
    # Check PyTorch availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    print("\nRunning tests...")
    
    # Run tests with verbose output
    exit_code = pytest.main([
        "tests/test_physics.py",
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--durations=10",  # Show 10 slowest tests
        "-x"  # Stop on first failure
    ])
    
    if exit_code == 0:
        print("\n✓ All physics tests passed successfully!")
    else:
        print(f"\n✗ Tests failed with exit code {exit_code}")
        
    return exit_code


if __name__ == "__main__":
    sys.exit(main())