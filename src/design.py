"""
Differentiable uniaxial SPP dispersion for inverse design (torch).

Torch re-implementation of the closed-form surface-plasmon-polariton
dispersion of :class:`src.physics.metamaterial.MetamaterialProperties` (the
scalar reference implementation, validated to machine precision in
``tests/test_benchmark_spp.py``), in complex torch arithmetic with autograd
support. This lets the effective-medium parameters ``(ε_t, ε_n)`` be
*optimised* by gradient descent straight through the dispersion — the seed of
the inverse-design tool (see ``examples/inverse_design.py``). No neural
networks are involved.

Sign convention: time dependence ``exp(-iωt)`` — lossy media have
``Im ε > 0`` and a surface wave decaying along its propagation direction has
``Im k_spp > 0``.

Geometry: interface plane ``z = 0``; uniaxial metamaterial
``diag(ε_t, ε_t, ε_n)`` in ``z < 0`` against an isotropic dielectric ``ε_d``
in ``z > 0``; the TM SPP propagates along ``x``. The closed form is

    k_spp² = k₀² ε_d ε_n (ε_t − ε_d) / (ε_t ε_n − ε_d²)    branch Im ≥ 0
    κ_d²  = k_spp² − ε_d k₀²                                branch Re > 0
    κ_m²  = ε_t (k_spp² / ε_n − k₀²)                        branch Re > 0

with ``L = 1 / (2 Im k_spp)`` and penetration depths ``1 / Re κ``.

Branch selection
----------------
``torch.sqrt`` of a complex tensor returns the principal root (argument in
``(−π/2, π/2]``) and supports complex autograd. The physical branches are
selected *differentiably* with ``torch.where`` on that principal root: the
boolean flip mask carries no gradient, and the negation is smooth, so
gradients flow through whichever branch is active. The ``Im ≥ 0`` flip also
restores continuity across the principal branch cut (the negative real axis
of the squared argument): approaching ``k² = −a ± i0`` from either side gives
``k → ±i√a → +i√a`` after the flip, so the selected root — and its gradient —
is continuous across the seam (away from ``k² = 0`` itself, where the square
root is genuinely non-differentiable).
"""

from __future__ import annotations

import math
from typing import Tuple, Union

import torch
import torch.nn.functional as F

from .constants import C0

__all__ = [
    "make_eps",
    "constrain_im",
    "constrain_im_inverse",
    "spp_wavevector_torch",
    "decay_constants_torch",
    "propagation_length_torch",
    "penetration_depths_torch",
    "field_enhancement_torch",
    "is_spp_supported_torch",
    "support_penalty_torch",
]

Scalar = Union[float, complex, torch.Tensor]


# --------------------------------------------------------------------- coercion helpers
def _as_complex(x: Scalar) -> torch.Tensor:
    """Coerce to a complex tensor (numbers -> complex128; real tensors promoted)."""
    if isinstance(x, torch.Tensor):
        if x.is_complex():
            return x
        return x.to(torch.complex64 if x.dtype == torch.float32 else torch.complex128)
    return torch.as_tensor(complex(x), dtype=torch.complex128)


def _as_real(x: Union[float, torch.Tensor]) -> torch.Tensor:
    """Coerce to a real tensor (numbers -> float64)."""
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(float(x), dtype=torch.float64)


def make_eps(re: Union[float, torch.Tensor], im: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Complex permittivity from two real parts, differentiably.

    The optimisation parametrisation: keep ``re`` and ``im`` as real leaf
    tensors (``requires_grad=True``) and build ``ε = re + i·im`` per step;
    gradients flow back to both. Numbers are promoted to float64; mixed
    shapes broadcast.
    """
    if not isinstance(re, torch.Tensor):
        re = torch.as_tensor(float(re), dtype=torch.float64)
    if not isinstance(im, torch.Tensor):
        im = torch.as_tensor(float(im), dtype=torch.float64)
    dtype = torch.promote_types(re.dtype, im.dtype)
    re_b, im_b = torch.broadcast_tensors(re.to(dtype), im.to(dtype))
    return torch.complex(re_b, im_b)


def constrain_im(im_raw: Union[float, torch.Tensor], im_min: float = 1e-3) -> torch.Tensor:
    """
    Softplus map keeping ``Im ε ≥ im_min > 0`` (passivity under ``e^{-iωt}``).

    ``im = im_min + softplus(im_raw)`` — smooth, strictly above ``im_min``,
    with a well-behaved gradient everywhere. Use with :func:`make_eps` when
    the imaginary parts are free optimisation variables.
    """
    if im_min <= 0:
        raise ValueError("im_min must be > 0 (passive medium)")
    return im_min + F.softplus(_as_real(im_raw))


def constrain_im_inverse(im: Union[float, torch.Tensor], im_min: float = 1e-3) -> torch.Tensor:
    """Inverse of :func:`constrain_im`: the raw value whose softplus gives ``im``."""
    y = _as_real(im) - im_min
    if torch.any(y <= 0):
        raise ValueError("im must exceed im_min")
    return torch.log(torch.expm1(y))


# --------------------------------------------------------------------- branch selection
def _propagating_sqrt(z_sq: torch.Tensor) -> torch.Tensor:
    """
    Differentiable square root on the wavevector branch: ``Im ≥ 0`` (decay
    along propagation under ``e^{-iωt}``); for a real result choose ``Re > 0``.
    Mirrors ``metamaterial._propagating_root``.
    """
    r = torch.sqrt(z_sq)
    flip = (r.imag < 0) | ((r.imag == 0) & (r.real < 0))
    return torch.where(flip, -r, r)


def _decaying_sqrt(z_sq: torch.Tensor) -> torch.Tensor:
    """
    Differentiable square root on the bound-mode branch: ``Re > 0``
    (evanescent decay away from the interface). Mirrors
    ``metamaterial._decaying_root``.
    """
    r = torch.sqrt(z_sq)
    flip = (r.real < 0) | ((r.real == 0) & (r.imag < 0))
    return torch.where(flip, -r, r)


def _positive_reciprocal(x: torch.Tensor) -> torch.Tensor:
    """``1/x`` where ``x > 0`` else ``+inf``, with NaN-safe gradients.

    Both branches of a ``torch.where`` are differentiated, so a bare
    ``1/x`` would inject NaN gradients whenever any element has ``x ≤ 0``;
    the division is therefore computed on a safe denominator.
    """
    positive = x > 0
    safe = torch.where(positive, x, torch.ones_like(x))
    return torch.where(positive, 1.0 / safe, torch.full_like(x, math.inf))


# --------------------------------------------------------------------- dispersion
def spp_wavevector_torch(
    eps_t: Scalar, eps_n: Scalar, omega: Union[float, torch.Tensor], eps_d: Scalar = 1.0
) -> torch.Tensor:
    """
    Complex SPP wavevector ``k_spp`` (1/m), branch ``Im ≥ 0`` (else ``Re > 0``).

    Differentiable in ``eps_t``, ``eps_n`` and ``eps_d``; all arguments may be
    numbers or (batched) tensors and broadcast together.

    Args:
        eps_t: Metamaterial permittivity along the propagation direction (in-plane).
        eps_n: Metamaterial permittivity normal to the interface.
        omega: Angular frequency (rad/s).
        eps_d: Isotropic dielectric permittivity of the upper half-space.
    """
    eps_t, eps_n, eps_d = _as_complex(eps_t), _as_complex(eps_n), _as_complex(eps_d)
    k0 = _as_real(omega) / C0
    denom = eps_t * eps_n - eps_d**2
    k_sq = k0**2 * eps_d * eps_n * (eps_t - eps_d) / denom
    return _propagating_sqrt(k_sq)


def decay_constants_torch(
    eps_t: Scalar, eps_n: Scalar, omega: Union[float, torch.Tensor], eps_d: Scalar = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ``(k_spp, κ_d, κ_m)`` (1/m) with the bound-mode branch ``Re κ > 0``.

    ``κ_d² = k_spp² − ε_d k₀²`` (dielectric side) and
    ``κ_m² = ε_t (k_spp²/ε_n − k₀²)`` (metamaterial side). Differentiable;
    broadcasts like :func:`spp_wavevector_torch`.
    """
    eps_t, eps_n, eps_d = _as_complex(eps_t), _as_complex(eps_n), _as_complex(eps_d)
    k0 = _as_real(omega) / C0
    k = spp_wavevector_torch(eps_t, eps_n, omega, eps_d)
    kappa_d = _decaying_sqrt(k**2 - eps_d * k0**2)
    kappa_m = _decaying_sqrt(eps_t * (k**2 / eps_n - k0**2))
    return k, kappa_d, kappa_m


def propagation_length_torch(
    eps_t: Scalar, eps_n: Scalar, omega: Union[float, torch.Tensor], eps_d: Scalar = 1.0
) -> torch.Tensor:
    """Intensity propagation length ``L = 1/(2 Im k_spp)`` (m); ``+inf`` if lossless."""
    k = spp_wavevector_torch(eps_t, eps_n, omega, eps_d)
    return 0.5 * _positive_reciprocal(k.imag)


def penetration_depths_torch(
    eps_t: Scalar, eps_n: Scalar, omega: Union[float, torch.Tensor], eps_d: Scalar = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Field penetration depths ``(δ_d, δ_m) = (1/Re κ_d, 1/Re κ_m)`` (m);
    ``+inf`` where the mode is unbound on that side.
    """
    _, kappa_d, kappa_m = decay_constants_torch(eps_t, eps_n, omega, eps_d)
    return _positive_reciprocal(kappa_d.real), _positive_reciprocal(kappa_m.real)


def field_enhancement_torch(
    eps_t: Scalar, eps_n: Scalar, omega: Union[float, torch.Tensor], eps_d: Scalar = 1.0
) -> torch.Tensor:
    """
    Interface field-enhancement factor ``|E_z|/|E_x| = |k_spp|/|κ_d|`` on the
    dielectric side (a property of the mode; see
    ``MetamaterialProperties.field_enhancement_factor``). ``+inf`` if κ_d = 0.
    """
    k, kappa_d, _ = decay_constants_torch(eps_t, eps_n, omega, eps_d)
    mag_sq = kappa_d.real**2 + kappa_d.imag**2
    positive = mag_sq > 0
    safe = torch.sqrt(torch.where(positive, mag_sq, torch.ones_like(mag_sq)))
    return torch.where(positive, k.abs() / safe, torch.full_like(safe, math.inf))


# --------------------------------------------------------------------- supported-region gate
def is_spp_supported_torch(
    eps_t: Scalar,
    eps_n: Scalar,
    eps_d: Scalar = 1.0,
    rel_tol: float = 1e-6,
    bound_tol: float = 1e-3,
) -> torch.Tensor:
    """
    Boolean gate: does a bound TM surface mode exist? (Non-differentiable.)

    Batched, frequency-independent mirror of
    ``MetamaterialProperties.is_spp_supported``: with ``κ_d, κ_m`` on the
    ``Re > 0`` branch, both decay constants must be meaningfully bound
    (``Re κ > bound_tol·|κ|``) and the unsquared matching condition
    ``κ_d/ε_d + κ_m/ε_t = 0`` must hold to ``rel_tol`` (squaring the
    dispersion can introduce spurious roots). Returns a bool tensor.
    """
    with torch.no_grad():
        eps_t, eps_n, eps_d = _as_complex(eps_t), _as_complex(eps_n), _as_complex(eps_d)
        denom = eps_t * eps_n - eps_d**2
        denom_ok = denom.abs() > 0
        safe_denom = torch.where(denom_ok, denom, torch.ones_like(denom))
        n_sq = eps_d * eps_n * (eps_t - eps_d) / safe_denom  # (k_spp / k0)^2
        kappa_d = _decaying_sqrt(n_sq - eps_d)
        kappa_m = _decaying_sqrt(eps_t * (n_sq / eps_n - 1.0))
        bound = (kappa_d.real > bound_tol * kappa_d.abs()) & (
            kappa_m.real > bound_tol * kappa_m.abs()
        )
        matching = kappa_d / eps_d + kappa_m / eps_t
        scale = (kappa_d / eps_d).abs() + (kappa_m / eps_t).abs()
        matched = matching.abs() <= rel_tol * scale
        return denom_ok & bound & matched


def support_penalty_torch(eps_t: Scalar, eps_n: Scalar, eps_d: Scalar = 1.0) -> torch.Tensor:
    """
    Differentiable soft surrogate of :func:`is_spp_supported_torch` for use
    as an optimisation penalty (≈ 0 inside the supported region, growing
    outside).

    Three dimensionless terms (frequency scales out):

    * ``relu(−Re κ̂_d²)²`` — the mode must be evanescent in the dielectric
      (``κ̂_d² = (k/k₀)² − ε_d``);
    * ``relu(−Re κ̂_m²)²`` — and in the metamaterial;
    * ``|κ̂_d/ε_d + κ̂_m/ε_t|² / (|κ̂_d/ε_d|² + |κ̂_m/ε_t|²)`` — the unsquared
      matching residual, which rejects the spurious roots of the squared
      dispersion (≈ 2 for a spurious root, ≈ 0 for a genuine one).

    The matching term uses squared magnitudes throughout (no ``abs`` of a
    possibly-zero complex number), so its gradient is finite everywhere the
    square roots are (i.e. away from ``κ² = 0``).
    """
    eps_t, eps_n, eps_d = _as_complex(eps_t), _as_complex(eps_n), _as_complex(eps_d)
    denom = eps_t * eps_n - eps_d**2
    denom_ok = (denom.real**2 + denom.imag**2) > 0
    safe_denom = torch.where(denom_ok, denom, torch.ones_like(denom))
    n_sq = eps_d * eps_n * (eps_t - eps_d) / safe_denom
    kappa_d_sq = n_sq - eps_d
    kappa_m_sq = eps_t * (n_sq / eps_n - 1.0)

    penalty = F.relu(-kappa_d_sq.real) ** 2 + F.relu(-kappa_m_sq.real) ** 2

    kappa_d = _decaying_sqrt(kappa_d_sq)
    kappa_m = _decaying_sqrt(kappa_m_sq)
    a = kappa_d / eps_d
    b = kappa_m / eps_t
    m = a + b
    m_sq = m.real**2 + m.imag**2
    scale_sq = a.real**2 + a.imag**2 + b.real**2 + b.imag**2
    penalty = penalty + m_sq / scale_sq.clamp_min(1e-30)
    # A singular dispersion (ε_t ε_n = ε_d²) is maximally penalised.
    return torch.where(denom_ok, penalty, torch.full_like(penalty, 10.0))
