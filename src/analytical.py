"""Closed-form reference solutions used by scripts and tests.

These are the analytical solutions the PINNs are compared against. They live
here so that training and visualisation scripts share a single definition.
"""

import cmath
import math
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


def analytical_spp_fields(
    coords: torch.Tensor,
    omega: float,
    eps_metal_t: complex,
    eps_metal_n: complex,
    eps_dielectric: complex = 1.0,
    H0: complex = 1.0,
    eps0: float = EPS0,
    mu0: float = MU0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact TM surface-plasmon-polariton mode at a planar interface (z = 0).

    Geometry and convention (e^{-iwt}): dielectric ``eps_dielectric`` fills
    z > 0, a uniaxial medium (optical axis z) with in-plane ``eps_metal_t`` and
    normal ``eps_metal_n`` fills z < 0; the mode propagates along +x with

        H = y_hat * H0 * exp(i k_spp x) * exp(-kappa_d z)   (z > 0)
        H = y_hat * H0 * exp(i k_spp x) * exp(+kappa_m z)   (z < 0)

    and E derived per medium from Ampere's law E = -(i w eps0 eps)^{-1} curl H,
    so that tangential E and H and normal D are continuous by construction
    (the matching condition kappa_d/eps_d + kappa_m/eps_t = 0 holds at k_spp).

        k_spp^2   = k0^2 eps_d eps_n (eps_t - eps_d) / (eps_t eps_n - eps_d^2)
        kappa_d^2 = k_spp^2 - eps_d k0^2
        kappa_m^2 = eps_t (k_spp^2 / eps_n - k0^2)

    with the Im(k_spp) >= 0 and Re(kappa) > 0 branches (identical to
    ``src.physics.metamaterial.MetamaterialProperties``; the isotropic limit is
    k_spp = k0 sqrt(eps_m eps_d / (eps_m + eps_d))).

    Args:
        coords: [N, 3] real coordinates (metres).
        omega: angular frequency (rad/s).
        eps_metal_t / eps_metal_n: in-plane / normal relative permittivity of
            the lower medium (equal values give the isotropic metal case).
        eps_dielectric: relative permittivity of the upper half-space.
        H0: complex H-field amplitude (A/m).

    Returns:
        (E, H) as [N, 3] complex tensors satisfying Maxwell in each half-space.
    """
    k0 = omega * math.sqrt(eps0 * mu0)  # = omega / c
    eps_t, eps_n, eps_d = complex(eps_metal_t), complex(eps_metal_n), complex(eps_dielectric)

    k_sq = k0**2 * eps_d * eps_n * (eps_t - eps_d) / (eps_t * eps_n - eps_d**2)
    k = cmath.sqrt(k_sq)
    if k.imag < 0 or (k.imag == 0 and k.real < 0):
        k = -k

    def decaying_root(z2: complex) -> complex:
        r = cmath.sqrt(z2)
        if r.real < 0 or (r.real == 0 and r.imag < 0):
            r = -r
        return r

    kappa_d = decaying_root(k**2 - eps_d * k0**2)
    kappa_m = decaying_root(eps_t * (k**2 / eps_n - k0**2))

    x = coords[:, 0].to(torch.float64)
    z = coords[:, 2].to(torch.float64)
    in_dielectric = z >= 0

    kt = torch.tensor(k, dtype=torch.complex128)
    kd = torch.tensor(kappa_d, dtype=torch.complex128)
    km = torch.tensor(kappa_m, dtype=torch.complex128)

    envelope = torch.where(in_dielectric, torch.exp(-kd * z), torch.exp(km * z))
    H_y = H0 * torch.exp(1j * kt * x) * envelope

    # E = -(i w eps0 eps)^{-1} curl H, evaluated analytically per medium:
    #   dielectric: curl H = ( kappa_d H_y, 0, i k H_y)
    #   metal:      curl H = (-kappa_m H_y, 0, i k H_y)
    pref = 1.0 / (omega * eps0)
    E_x = torch.where(
        in_dielectric,
        (1j * kd / eps_d) * pref * H_y,
        (-1j * km / eps_t) * pref * H_y,
    )
    E_z = torch.where(
        in_dielectric,
        (-kt / eps_d) * pref * H_y,
        (-kt / eps_n) * pref * H_y,
    )

    zeros = torch.zeros_like(H_y)
    E = torch.stack([E_x, zeros, E_z], dim=1)
    H = torch.stack([zeros, H_y, zeros], dim=1)
    return E, H


def complex_to_pinn_format(field: torch.Tensor) -> torch.Tensor:
    """Convert an [N, C] complex tensor to the PINN's [N, C, 2] (real, imag) layout."""
    return torch.stack([field.real, field.imag], dim=-1)


__all__ = [
    "analytical_potential",
    "analytical_point_charge_field",
    "analytical_plane_wave",
    "analytical_spp_fields",
    "complex_to_pinn_format",
]
