"""
Bridging helpers between the two field representations used in the project.

* ``src.physics`` works with **complex** tensors: ``E`` and ``H`` each of shape
  ``(N, 3)`` with dtype ``complex64``/``complex128``.
* ``src.models`` networks output **real** tensors of shape ``(N, 6, 2)`` with
  component order ``[Ex, Ey, Ez, Hx, Hy, Hz]`` along axis 1 and
  ``[real, imag]`` along axis 2.

These functions convert between the two without copying more than necessary
and preserve the autograd graph so that spatial derivatives of the complex
fields can be traced back to the network inputs.
"""

from __future__ import annotations

from typing import Tuple

import torch

__all__ = ["split_complex", "to_complex", "join_complex"]


def split_complex(fields_complex: torch.Tensor) -> torch.Tensor:
    """
    Convert a complex field tensor to the stacked real/imag network format.

    Args:
        fields_complex: Complex tensor of shape ``(N, K)`` (typically ``K = 6``).

    Returns:
        Real tensor of shape ``(N, K, 2)`` with ``[..., 0] = Re`` and ``[..., 1] = Im``.
    """
    if not fields_complex.is_complex():
        raise TypeError("split_complex expects a complex tensor")
    return torch.stack([fields_complex.real, fields_complex.imag], dim=-1)


def to_complex(fields: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert network output ``(N, 6, 2)`` into complex ``E`` and ``H`` fields.

    Args:
        fields: Real tensor of shape ``(N, 6, 2)`` ordered ``[Ex, Ey, Ez, Hx, Hy, Hz]``
            along axis 1 and ``[real, imag]`` along axis 2. A ``(N, 3, 2)``
            tensor is accepted for E-only networks, in which case ``H`` is zero.

    Returns:
        ``(E, H)`` complex tensors, each of shape ``(N, 3)``.
    """
    if fields.dim() != 3 or fields.shape[-1] != 2:
        raise ValueError(
            f"fields must have shape (N, 3 or 6, 2), got {tuple(fields.shape)}"
        )
    if fields.is_complex():
        raise TypeError("to_complex expects a real tensor with a trailing (re, im) axis")

    fields_c = torch.complex(fields[..., 0], fields[..., 1])  # (N, K)
    n_comp = fields_c.shape[1]
    if n_comp == 6:
        return fields_c[:, :3], fields_c[:, 3:]
    if n_comp == 3:
        return fields_c, torch.zeros_like(fields_c)
    raise ValueError(f"Expected 3 or 6 field components, got {n_comp}")


def join_complex(E: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """
    Inverse of :func:`to_complex`: pack complex ``E, H`` ``(N, 3)`` into ``(N, 6, 2)``.
    """
    return split_complex(torch.cat([E, H], dim=1))
