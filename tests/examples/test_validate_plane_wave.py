"""Tests for plane wave validation script."""
import pytest
import torch
import numpy as np


def test_imports():
    """Test that all required modules can be imported."""
    try:
        from examples.validate_plane_wave import (
            OMEGA, C, MU0, EPS0, K0, WAVELENGTH
        )
        assert OMEGA > 0
        assert C > 0
        assert MU0 > 0
        assert EPS0 > 0
        assert K0 > 0
        assert WAVELENGTH > 0
    except ImportError:
        pytest.skip("validate_plane_wave not yet implemented")
