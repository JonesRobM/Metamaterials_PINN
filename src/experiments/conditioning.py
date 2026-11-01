r"""
Conditioning a spatial network on parameters, and the losses that go with it.

A surrogate over a band of frequencies — or over a whole design rectangle —
feeds the network extra input columns beyond ``(x, y, z)``: a normalised ω̂,
or ``(ω̂, f̂)``. That creates two problems the differential operators do not
solve on their own.

**The operators only accept spatial coordinates.** ``curl`` and ``divergence``
differentiate every input column they are given, and ``∂/∂ω`` appears nowhere
in Maxwell's equations. :class:`ColumnConditionedNet` therefore presents a
3-column *spatial view* of a wider core: the condition columns, aligned
row-for-row with the batch, are appended inside the forward, so gradients flow
through the spatial columns only. That is exactly the right semantics, and it
costs nothing — the wrapper holds no parameters of its own and never appears in
a ``state_dict``.

**The preconditioner is now per row.** A batch that mixes frequencies mixes
stiffnesses: the metal-side curl residual has to be weighted by |ε(row)|^-p,
and ``ε`` differs down the batch. :func:`weighted_curl_loss` and
:func:`weighted_divergence_loss` are the library losses
``MaxwellCurlLoss(frequency=1, mu0=1, eps0=1)`` and ``MaxwellDivergenceLoss``
in the scaled frame — where the curl equations are ``∇̂×Ê = i Ĥ`` and
``∇̂×Ĥ = −i ε Ê`` at *every* frequency, so no explicit ω survives — with each
row's squared residual scaled before the mean. With ``row_weight=None`` they
reduce to the library losses exactly, which the experiments' tests assert.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.models import to_complex
from src.physics import MaxwellEquations

__all__ = [
    "SCALED_MAXWELL",
    "ColumnConditionedNet",
    "weighted_curl_loss",
    "weighted_divergence_loss",
]

#: Maxwell's operators in the k₀(ω)-scaled frame, where the curl equations
#: carry no frequency of their own: ``∇̂×Ê = i Ĥ`` and ``∇̂×Ĥ = −i ε Ê``.
SCALED_MAXWELL = MaxwellEquations(1.0, mu0=1.0, eps0=1.0)


class ColumnConditionedNet(nn.Module):
    """
    A 3-column spatial view of a wider core, with fixed per-row condition columns.

    Args:
        core: The conditioned network, taking ``3 + k`` input columns.
        columns: ``(N, k)`` condition block, aligned row-for-row with the batch
            this will be called on. Held without grad: no ``∂/∂ω`` term enters
            Maxwell's equations.
    """

    def __init__(self, core: nn.Module, columns: torch.Tensor):
        super().__init__()
        self.core = core
        self.columns = columns

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.core(torch.cat([coords, self.columns.to(coords.dtype)], dim=1))


def weighted_curl_loss(
    net: nn.Module,
    coords: torch.Tensor,
    eps_rows: torch.Tensor,
    row_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""
    Scaled-frame curl residual ``‖∇̂×Ê − iĤ‖² + ‖∇̂×Ĥ + iεÊ‖²`` with per-row weights.

    Args:
        net: Returns ``[N, 6, 2]`` from ``coords``; usually a
            :class:`ColumnConditionedNet`.
        coords: ``(N, 3)`` scaled coordinates, requiring grad.
        eps_rows: ``(N, 3, 3)`` per-row permittivity tensors.
        row_weight: ``(N,)`` multiplier on each row's squared residual;
            ``None`` means an unweighted mean, identical to
            ``MaxwellCurlLoss(frequency=1, mu0=1, eps0=1).compute(...)``.
    """
    E, H = to_complex(net(coords))
    curl_E = SCALED_MAXWELL.curl_operator(E, coords)
    curl_H = SCALED_MAXWELL.curl_operator(H, coords)
    eps_E = torch.einsum("nij,nj->ni", eps_rows.to(E.dtype), E)
    res_E = curl_E - 1j * H
    res_H = curl_H + 1j * eps_E
    if row_weight is None:
        return torch.mean(res_E.abs() ** 2) + torch.mean(res_H.abs() ** 2)
    w = row_weight.reshape(-1, 1).to(res_E.real.dtype)
    return torch.mean(w * res_E.abs() ** 2) + torch.mean(w * res_H.abs() ** 2)


def weighted_divergence_loss(
    net: nn.Module,
    coords: torch.Tensor,
    eps_rows: torch.Tensor,
    row_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""``∇̂·(εÊ) = 0`` and ``∇̂·Ĥ = 0`` with per-row weights (see
    :func:`weighted_curl_loss`)."""
    E, H = to_complex(net(coords))
    div_D = SCALED_MAXWELL.divergence_operator(
        torch.einsum("nij,nj->ni", eps_rows.to(E.dtype), E), coords
    )
    div_H = SCALED_MAXWELL.divergence_operator(H, coords)
    if row_weight is None:
        return torch.mean(div_D.abs() ** 2) + torch.mean(div_H.abs() ** 2)
    w = row_weight.reshape(-1).to(div_D.real.dtype)
    return torch.mean(w * div_D.abs() ** 2) + torch.mean(w * div_H.abs() ** 2)
