"""
Autograd-based differential operators shared by the physics and loss modules.

All operators act on *real* tensors and build a differentiable graph
(``create_graph=True``) so that the resulting residuals can themselves be
back-propagated during PINN training. Complex fields are handled by the
``*_complex`` wrappers, which apply the real operator to the real and
imaginary parts separately (differentiation is linear, so this is exact).

Conventions
-----------
* ``coords`` has shape ``(N, D)`` with ``D <= 3`` and column order ``(x, y, z)``.
  When ``D < 3`` the missing spatial derivatives are treated as zero, i.e. the
  fields are assumed invariant along the absent coordinates.
* ``coords`` must be a leaf that requires gradients (or be derived from one).
  If a field does not depend on ``coords`` at all (e.g. a constant), the
  derivative is returned as zeros rather than raising.
"""

from __future__ import annotations

import torch

__all__ = [
    "gradient",
    "jacobian",
    "curl",
    "divergence",
    "curl_complex",
    "divergence_complex",
]


def gradient(scalar: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """
    Gradient of a real scalar field with respect to spatial coordinates.

    Args:
        scalar: Real field values, shape ``(N,)`` or ``(N, 1)``.
        coords: Coordinates, shape ``(N, D)``.

    Returns:
        ``d(scalar)/d(coords)`` with shape ``(N, 3)``. Columns beyond ``D`` are zero.
    """
    if scalar.is_complex():
        raise TypeError("gradient() expects a real tensor; use the *_complex wrappers.")
    scalar = scalar.reshape(-1)
    n, d = coords.shape
    if d > 3:
        raise ValueError(f"coords must have at most 3 spatial columns, got {d}")

    out = torch.zeros(n, 3, device=coords.device, dtype=scalar.dtype)
    if not scalar.requires_grad or not coords.requires_grad:
        return out

    (grad,) = torch.autograd.grad(
        outputs=scalar,
        inputs=coords,
        grad_outputs=torch.ones_like(scalar),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    if grad is None:
        return out
    return torch.cat([grad, out[:, d:]], dim=1) if d < 3 else grad


def jacobian(field: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """
    Jacobian of a real vector field: ``J[n, i, j] = dF_i/dx_j``.

    Args:
        field: Real vector field, shape ``(N, 3)``.
        coords: Coordinates, shape ``(N, D)``.

    Returns:
        Tensor of shape ``(N, 3, 3)``. One autograd call per field component.
    """
    if field.dim() != 2 or field.shape[1] != 3:
        raise ValueError(f"field must have shape (N, 3), got {tuple(field.shape)}")
    return torch.stack([gradient(field[:, i], coords) for i in range(3)], dim=1)


def curl(field: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """
    Curl of a real vector field via autograd.

    Args:
        field: Real vector field ``[Fx, Fy, Fz]``, shape ``(N, 3)``.
        coords: Coordinates, shape ``(N, D)``.

    Returns:
        ``∇ × F`` with shape ``(N, 3)``.
    """
    J = jacobian(field, coords)  # J[:, i, j] = dF_i / dx_j
    curl_x = J[:, 2, 1] - J[:, 1, 2]
    curl_y = J[:, 0, 2] - J[:, 2, 0]
    curl_z = J[:, 1, 0] - J[:, 0, 1]
    return torch.stack([curl_x, curl_y, curl_z], dim=1)


def divergence(field: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """
    Divergence of a real vector field via autograd.

    Args:
        field: Real vector field ``[Fx, Fy, Fz]``, shape ``(N, 3)``.
        coords: Coordinates, shape ``(N, D)``.

    Returns:
        ``∇ · F`` with shape ``(N,)``.
    """
    if field.dim() != 2 or field.shape[1] != 3:
        raise ValueError(f"field must have shape (N, 3), got {tuple(field.shape)}")
    return sum(gradient(field[:, i], coords)[:, i] for i in range(3))


def curl_complex(field: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Curl of a complex vector field ``(N, 3)``; returns complex ``(N, 3)``."""
    if not field.is_complex():
        return curl(field, coords)
    return torch.complex(curl(field.real, coords), curl(field.imag, coords))


def divergence_complex(field: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Divergence of a complex vector field ``(N, 3)``; returns complex ``(N,)``."""
    if not field.is_complex():
        return divergence(field, coords)
    return torch.complex(divergence(field.real, coords), divergence(field.imag, coords))
