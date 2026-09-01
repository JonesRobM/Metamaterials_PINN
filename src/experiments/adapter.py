r"""
Displacement adapters: the interface ``E_z`` discontinuity, built in exactly.

Every SPP experiment in ``examples/`` shares one structural idea. The physical
mode has ``E_z`` *jumping* across a material interface while the normal
displacement ``D_z = ε₀ ε_zz E_z`` is *continuous*. A continuous MLP cannot
represent that jump; its smoothed version fights the divergence and curl
residuals exactly where the collocation sampling is densest. So the wrapped
network is asked for the continuous quantity ``D̂_z`` on channel 2, and the
adapter divides it by the local ``ε_zz`` per point. The jump is then exact by
construction, at every interface, for free.

The only thing that differs between experiments is the ε profile:

===========================  =========================================
:class:`TwoMediumAdapter`    one interface at ``z = 0``, two constants
:class:`LayeredAdapter`      an N-layer stack, ``ε_zz`` from a table
subclass :class:`DisplacementAdapter`  anything else (ω- or design-dependent ε)
===========================  =========================================

so :class:`DisplacementAdapter` owns the shared forward pass and defers the
profile to :meth:`~DisplacementAdapter.eps_zz_at`.

``ε_zz`` — the *normal* (zz) tensor component — is the right divisor even for a
uniaxial medium, where it is ε_n and not ε_t, because ``D_z`` couples only to
the zz component.

Serialisation note: the wrapped network is always the submodule ``mlp``, so an
adapter's ``state_dict`` keys are the wrapped network's under an ``mlp.``
prefix, unchanged from before this module existed. Checkpoints keep loading.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

__all__ = ["DisplacementAdapter", "TwoMediumAdapter", "LayeredAdapter"]


class DisplacementAdapter(nn.Module):
    r"""
    Base adapter: divide channel 2 of ``mlp`` by ``ε_zz``, leave the rest alone.

    ``forward(coords)`` returns genuine ``(Ê, Ĥ)`` in the ``[N, 6, 2]``
    real/imaginary layout, with the exact ``E_z`` jump. All losses see this
    module, not the bare MLP.

    Subclasses supply the permittivity profile by overriding
    :meth:`eps_zz_at`. Nothing here is registered as a buffer or parameter, so
    ``.to(torch.float64)`` promotions for the L-BFGS phase (and the float32
    restore afterwards) cannot corrupt the ε values — see :class:`LayeredAdapter`
    for why that matters.

    Args:
        mlp: Wrapped network, ``coords -> [N, 6, 2]``, channel 2 carrying ``D̂_z``.
    """

    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.mlp = mlp

    def eps_zz_at(self, coords: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """``ε_zz`` at each row of ``coords``, as a complex tensor of ``dtype``."""
        raise NotImplementedError

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        out = self.mlp(coords)  # [N, 6, 2]; channel 2 carries D̂_z
        fields = torch.complex(out[..., 0], out[..., 1])  # [N, 6]
        eps = self.eps_zz_at(coords, fields.dtype)
        e_z = fields[:, 2] / eps
        fields = torch.cat([fields[:, :2], e_z.unsqueeze(1), fields[:, 3:]], dim=1)
        return torch.stack([fields.real, fields.imag], dim=-1)


class TwoMediumAdapter(DisplacementAdapter):
    r"""
    Adapter for a single interface at ``z = 0`` between two non-dispersive media.

    ``eps_below`` must be the **normal** (zz) component of the lower medium —
    ε_m for isotropic silver, ε_n (not ε_t) for a uniaxial one. Both values are
    kept as python ``complex`` and cast to the working dtype per forward, so
    ``.to(float64)`` conversions of the module are safe.
    """

    def __init__(self, mlp: nn.Module, eps_below: complex, eps_above: complex):
        super().__init__(mlp)
        self.eps_below = complex(eps_below)
        self.eps_above = complex(eps_above)

    def eps_zz_at(self, coords: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return torch.where(
            coords[:, 2] < 0,
            torch.tensor(self.eps_below, dtype=dtype, device=coords.device),
            torch.tensor(self.eps_above, dtype=dtype, device=coords.device),
        )


class LayeredAdapter(DisplacementAdapter):
    r"""
    Adapter for an N-layer stack: ``ε_zz`` looked up in the real layer profile.

    ``D_z`` is continuous across *every* interface, so the same construction
    that fixes one jump fixes all of them at once.

    Lookup convention: ``torch.bucketize(..., right=True)`` matches
    :func:`src.transfer_matrix.layer_index_at`, so a point exactly on an
    interface takes the medium **above**, as everywhere else in the project.

    The layer table is held as **plain float64/complex128 tensors, deliberately
    not registered buffers**. ``nn.Module.to(dtype)`` converts complex buffers
    with the same dtype it is given, so the float64 L-BFGS promotion (and the
    float32 restore afterwards) would silently rewrite ``ε_Ag = −17.88 + 0.20i``
    as the real number ``−17.88`` — losing the metal's loss, and with it the
    imaginary part of ``k_spp``. The two-medium adapters avoid this by accident,
    by storing python ``complex`` scalars; here the table is an array, so it is
    guarded on purpose. Only the device is tracked (lazily, in :meth:`eps_zz`).

    Args:
        mlp: Core network, ``coords_hat (N, 3) -> (N, 6, 2)``.
        boundaries_hat: Interface positions in ``k₀``-scaled units.
        eps_layers: Permittivities in the same increasing-``z`` order.
    """

    def __init__(
        self, mlp: nn.Module, boundaries_hat: np.ndarray, eps_layers: Sequence[complex]
    ):
        super().__init__(mlp)
        self.boundaries_hat = torch.as_tensor(
            np.asarray(boundaries_hat), dtype=torch.float64
        )
        self.eps_values = torch.as_tensor(
            np.asarray([complex(e) for e in eps_layers], dtype=complex),
            dtype=torch.complex128,
        )

    def eps_zz(self, z_hat: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """``ε_zz`` at each scaled ``z``, as a complex tensor of ``dtype``."""
        bounds = self.boundaries_hat.to(z_hat.device)
        idx = torch.bucketize(
            z_hat.detach().to(torch.float64).contiguous(), bounds, right=True
        )
        return self.eps_values.to(z_hat.device)[idx].to(dtype)

    def eps_zz_at(self, coords: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return self.eps_zz(coords[:, 2], dtype)
