r"""
The nondimensional frame the experiments train in.

Two independent scalings appear in every experiment, and they are easy to
confuse, so they are named apart here.

**Coordinate scaling** — :func:`k0_of`. The network never sees metres. Fixed-ω
experiments divide by λ₀ once; the band-sweeping ones divide by that frequency's
*own* ``k₀(ω) = ω/c``, so every ω sees a box of the same shape in scaled units
and the curl equations read ``∇̂×Ê = i Ĥ``, ``∇̂×Ĥ = −i ε Ê`` with no explicit
frequency at all. That is the whole reason for the k₀(ω) frame: it removes ω
from the operator and leaves it only in ε.

**Feature scaling** — :class:`LinearFeature`. A conditioning input (ω, or a
metal fill fraction) is mapped affinely onto ``[-1, 1]`` before it becomes a
network input column, because an MLP fed raw rad/s learns nothing. The map and
its inverse have to agree to machine precision — the inverse is applied
*inside* the displacement adapter to recover ε(ω, f) — so they are one object
rather than two functions that might drift apart.

Both are pure geometry: no material, no mode, no permittivity model.
"""

from __future__ import annotations

import torch

from src.constants import C0

__all__ = ["LinearFeature", "k0_of"]


def k0_of(omega: float) -> float:
    """Local free-space wavenumber ω/c — the coordinate-scaling factor."""
    return float(omega) / C0


class LinearFeature:
    r"""
    An affine map from a physical range onto the network feature range ``[-1, 1]``.

    ``LinearFeature(lo, hi)`` sends ``lo -> -1`` and ``hi -> +1``; equivalently
    :meth:`centred` names the same map by its midpoint. ``to_hat`` and
    ``from_hat`` are inverses, and :meth:`from_hat_torch` is the differentiable
    tensor form used inside the adapters, computed in float64 so the recovered
    ω matches the python one to machine precision.

    A band defined by its centre should be constructed with :meth:`centred`
    rather than by passing ``mid ± half_span`` here: recovering the midpoint
    from the endpoints is a different floating-point expression, and the
    round-trip moves the feature values by an ULP. Nothing physical depends on
    that, but a reproducibility claim does.

    Args:
        lo: Physical value mapped to ``-1``.
        hi: Physical value mapped to ``+1``.

    Raises:
        ValueError: if ``hi == lo`` (the map would not be invertible).
    """

    def __init__(self, lo: float, hi: float):
        lo, hi = float(lo), float(hi)
        if hi == lo:
            raise ValueError("LinearFeature needs a non-degenerate range (hi != lo)")
        self.lo = lo
        self.hi = hi
        self.mid = 0.5 * (lo + hi)
        self.half_span = 0.5 * (hi - lo)

    @classmethod
    def centred(cls, mid: float, half_span: float) -> "LinearFeature":
        """
        The same map named by its centre and half-width, held exactly.

        ``mid`` and ``half_span`` are stored as given, not re-derived from the
        endpoints, so ``to_hat`` is bit-for-bit ``(value - mid) / half_span``.
        """
        mid, half_span = float(mid), float(half_span)
        if half_span == 0.0:
            raise ValueError("LinearFeature needs a non-zero half_span")
        feature = cls.__new__(cls)
        feature.lo = mid - half_span
        feature.hi = mid + half_span
        feature.mid = mid
        feature.half_span = half_span
        return feature

    def to_hat(self, value: float) -> float:
        """Physical value -> normalised feature in ``[-1, 1]``."""
        return (float(value) - self.mid) / self.half_span

    def from_hat(self, hat: float) -> float:
        """Normalised feature -> physical value; the inverse of :meth:`to_hat`."""
        return self.mid + self.half_span * float(hat)

    def from_hat_torch(self, hat: torch.Tensor) -> torch.Tensor:
        """Tensor form of :meth:`from_hat`, differentiable, evaluated in float64."""
        return self.mid + self.half_span * hat.to(torch.float64)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"LinearFeature(lo={self.lo!r}, hi={self.hi!r})"
