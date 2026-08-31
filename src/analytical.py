"""Closed-form reference solutions used by scripts and tests.

These are the analytical solutions the PINNs are compared against. They live
here so that training and visualisation scripts share a single definition.
"""

from typing import Sequence, Tuple, Union

import numpy as np
import torch

from src.constants import EPS0, MU0

ArrayLike = Union[np.ndarray, torch.Tensor, float]


def analytical_potential(
    x: ArrayLike, y: ArrayLike, q: float, q_pos: Sequence[float], r_ref: float = 1.0
) -> ArrayLike:
    """Electrostatic potential of a point charge in two dimensions.

    In 2-D a "point charge" is a line charge of density ``q`` per unit length,
    whose potential solves the 2-D Poisson equation:

        V(r) = -(q / (2 pi eps0)) ln(r / r_ref)

    This is the solution that satisfies ``∇²V = 0`` away from the charge in
    2-D (the 3-D Coulomb ``1/r`` does not). ``r_ref`` fixes the arbitrary
    additive constant (V = 0 at r = r_ref). A small offset avoids ``ln(0)``
    exactly at the charge location.

    Accepts NumPy arrays or torch tensors; with tensors the result stays on the
    autograd graph so ``E = -∇V`` can be obtained by differentiation.
    """
    xp = torch if isinstance(x, torch.Tensor) else np
    k = 1.0 / (2.0 * np.pi * EPS0)
    r = xp.sqrt((x - q_pos[0]) ** 2 + (y - q_pos[1]) ** 2)
    return -k * q * xp.log((r + 1e-9) / r_ref)


def analytical_point_charge_field(
    X: np.ndarray, Y: np.ndarray, q: float, q_pos: Sequence[float], r_min_sq: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """Electric field (Ex, Ey) of a 2-D point (line) charge on a grid.

    E = -∇V = (q / (2 pi eps0)) r_hat / r, consistent with
    :func:`analytical_potential`. Points closer than sqrt(r_min_sq) to the
    charge are left as zero.
    """
    k = 1.0 / (2.0 * np.pi * EPS0)
    rx = X - q_pos[0]
    ry = Y - q_pos[1]
    r_sq = rx**2 + ry**2
    mask = r_sq >= r_min_sq
    r2 = np.where(mask, r_sq, 1.0)
    Ex = np.where(mask, k * q * rx / r2, 0.0)
    Ey = np.where(mask, k * q * ry / r2, 0.0)
    return Ex, Ey


def analytical_plane_wave(
    coords: torch.Tensor,
    k_vec: torch.Tensor,
    E0_polarization: torch.Tensor,
    omega: float,
    mu0: float = MU0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Time-harmonic plane wave E = E0 exp(i k.r), H = (k x E) / (omega mu0).

    Args:
        coords: [N, 3] real coordinates.
        k_vec: [3] real wavevector.
        E0_polarization: [3] complex polarisation vector.
        omega: angular frequency [rad/s].
        mu0: permeability to use (defaults to vacuum).

    Returns:
        (E, H) as [N, 3] complex tensors.
    """
    phase = torch.sum(k_vec.unsqueeze(0) * coords, dim=1)
    exp_factor = torch.exp(1j * phase).unsqueeze(1)
    E = E0_polarization.unsqueeze(0) * exp_factor

    k_expanded = k_vec.unsqueeze(0).expand(coords.shape[0], -1).to(E.real.dtype)
    H_real = torch.cross(k_expanded, E.real, dim=1)
    H_imag = torch.cross(k_expanded, E.imag, dim=1)
    H = (H_real + 1j * H_imag) / (omega * mu0)
    return E, H


def complex_to_pinn_format(field: torch.Tensor) -> torch.Tensor:
    """Convert an [N, C] complex tensor to the PINN's [N, C, 2] (real, imag) layout."""
    return torch.stack([field.real, field.imag], dim=-1)


__all__ = [
    "analytical_potential",
    "analytical_point_charge_field",
    "analytical_plane_wave",
    "complex_to_pinn_format",
]
