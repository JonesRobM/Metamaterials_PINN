"""Shared fixtures and helpers for the test-suite."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # noqa: E402  (must run before pyplot is imported anywhere)

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.analytical import analytical_plane_wave
from src.constants import C0
from src.models.field_format import join_complex

OMEGA = 2 * np.pi * 1e15  # 1 PHz
K0 = OMEGA / C0


class PlaneWaveNetwork(nn.Module):
    """Wrap :func:`analytical_plane_wave` as a network returning ``(N, 6, 2)``.

    Args:
        k_vec: Real wavevector ``(3,)``.
        E0: Complex polarisation ``(3,)``.
        omega: Angular frequency.
    """

    def __init__(self, k_vec, E0, omega: float = OMEGA):
        super().__init__()
        self.register_buffer("k_vec", torch.as_tensor(k_vec))
        self.register_buffer("E0", torch.as_tensor(E0))
        self.omega = float(omega)

    def fields(self, coords: torch.Tensor):
        k = self.k_vec.to(coords.dtype)
        E0 = self.E0.to(torch.complex128 if coords.dtype == torch.float64 else torch.complex64)
        return analytical_plane_wave(coords, k, E0, self.omega)

    def get_fields(self, coords: torch.Tensor):
        """Complex ``(E, H)`` -- the interface used by ``src.utils.metrics``."""
        return self.fields(coords)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        E, H = self.fields(coords)
        return join_complex(E, H)


def make_plane_wave(k_scale: float = 1.0, direction=(1.0, 0.0, 0.0), pol=(0.0, 1.0, 0.0),
                    omega: float = OMEGA, dtype=torch.float64) -> PlaneWaveNetwork:
    """Plane wave with ``|k| = k_scale * k0`` (``k_scale = 1`` satisfies Maxwell in vacuum)."""
    k0 = omega / C0
    d = torch.tensor(direction, dtype=dtype)
    d = d / d.norm()
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    return PlaneWaveNetwork(k_scale * k0 * d, torch.tensor(pol, dtype=cdtype), omega)


def sample_coords(n: int = 64, extent: float = 1e-6, dtype=torch.float64, seed: int = 0,
                  dim: int = 3) -> torch.Tensor:
    """Random coordinates in ``[-extent, extent]^dim`` that require grad."""
    g = torch.Generator().manual_seed(seed)
    return ((torch.rand(n, dim, generator=g, dtype=dtype) * 2 - 1) * extent).requires_grad_(True)


@pytest.fixture
def plane_wave_net() -> PlaneWaveNetwork:
    return make_plane_wave()


@pytest.fixture
def coords64() -> torch.Tensor:
    return sample_coords()
