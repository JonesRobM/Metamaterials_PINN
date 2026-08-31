"""
Frequency-domain Maxwell's equations for PINN-based SPP modelling.

Sign convention (used consistently across ``src.physics``, ``src.models`` and
the tests): time dependence ``exp(-iωt)``. Consequently

    ∇ × E =  iωμ₀μᵣ H          (Faraday)
    ∇ × H = -iωε₀εᵣ E          (Ampère, no free currents)
    ∇ · (εᵣ E) = 0             (Gauss, no free charges)
    ∇ · H = 0                  (no magnetic monopoles, μᵣ = 1)

With this convention a lossy medium has ``Im(ε) > 0``, a forward-propagating
wave is ``exp(+ik·r)`` and a decaying surface wave has ``Im(k_spp) > 0``.

Fields are complex tensors of shape ``(N, 3)``; residual methods return the
real and imaginary parts concatenated along the last axis so they can be fed
directly into a real-valued loss.
"""

from __future__ import annotations

import torch

from ..constants import EPS0, MU0
from .differential_ops import curl_complex, divergence_complex

__all__ = ["MaxwellEquations"]


class MaxwellEquations:
    """
    Frequency-domain Maxwell's equations (``exp(-iωt)`` convention).

    Args:
        omega: Angular frequency ω (rad/s).
        mu0: Vacuum permeability (H/m). Defaults to CODATA value.
        eps0: Vacuum permittivity (F/m). Defaults to CODATA value.
    """

    def __init__(self, omega: float, mu0: float = MU0, eps0: float = EPS0):
        if omega <= 0:
            raise ValueError("omega must be positive")
        self.omega = float(omega)
        self.mu0 = float(mu0)
        self.eps0 = float(eps0)
        self.c = 1.0 / (self.mu0 * self.eps0) ** 0.5  # equals C0 for default arguments
        self.k0 = self.omega / self.c

    # ------------------------------------------------------------------ operators
    @staticmethod
    def _require_grad(coords: torch.Tensor) -> torch.Tensor:
        if not coords.requires_grad:
            coords.requires_grad_(True)
        return coords

    def curl_operator(self, field: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        ``∇ × F`` for a complex vector field.

        Args:
            field: Complex vector field ``[Fx, Fy, Fz]``, shape ``(N, 3)``.
            coords: Coordinates ``[x, y, z]``, shape ``(N, 3)``.

        Returns:
            Complex curl, shape ``(N, 3)``.
        """
        return curl_complex(field, self._require_grad(coords))

    def divergence_operator(self, field: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """``∇ · F`` for a complex vector field ``(N, 3)``; returns complex ``(N,)``."""
        return divergence_complex(field, self._require_grad(coords))

    # ------------------------------------------------------------------ residuals
    @staticmethod
    def _split(z: torch.Tensor) -> torch.Tensor:
        """Concatenate real and imaginary parts along the last axis."""
        if z.dim() == 1:
            z = z.unsqueeze(1)
        return torch.cat([z.real, z.imag], dim=1)

    def curl_E_residual(
        self, E_field: torch.Tensor, H_field: torch.Tensor, coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Faraday residual ``∇ × E - iωμ₀H`` (μᵣ = 1).

        Returns:
            ``(N, 6)`` tensor ``[Re(res), Im(res)]``.
        """
        curl_E = self.curl_operator(E_field, coords)
        residual = curl_E - 1j * self.omega * self.mu0 * H_field
        return self._split(residual)

    def curl_H_residual(
        self,
        E_field: torch.Tensor,
        H_field: torch.Tensor,
        coords: torch.Tensor,
        epsilon_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Ampère residual ``∇ × H + iωε₀ εᵣ·E``.

        Args:
            epsilon_tensor: Relative permittivity tensor, shape ``(N, 3, 3)``.

        Returns:
            ``(N, 6)`` tensor ``[Re(res), Im(res)]``.
        """
        curl_H = self.curl_operator(H_field, coords)
        epsilon_E = torch.einsum("nij,nj->ni", epsilon_tensor.to(E_field.dtype), E_field)
        residual = curl_H + 1j * self.omega * self.eps0 * epsilon_E
        return self._split(residual)

    def divergence_E_residual(
        self, E_field: torch.Tensor, coords: torch.Tensor, epsilon_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Gauss residual ``∇ · (εᵣ E)`` (no free charges).

        Returns:
            ``(N, 2)`` tensor ``[Re, Im]``.
        """
        epsilon_E = torch.einsum("nij,nj->ni", epsilon_tensor.to(E_field.dtype), E_field)
        return self._split(self.divergence_operator(epsilon_E, coords))

    def divergence_B_residual(self, H_field: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Residual ``∇ · H`` (equivalent to ``∇ · B = 0`` for μᵣ = 1).

        Returns:
            ``(N, 2)`` tensor ``[Re, Im]``.
        """
        return self._split(self.divergence_operator(H_field, coords))

    def total_residual(
        self,
        E_field: torch.Tensor,
        H_field: torch.Tensor,
        coords: torch.Tensor,
        epsilon_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        All four residuals concatenated.

        Returns:
            ``(N, 16)``: ``[curl_E (6), curl_H (6), div_E (2), div_B (2)]``.
        """
        return torch.cat(
            [
                self.curl_E_residual(E_field, H_field, coords),
                self.curl_H_residual(E_field, H_field, coords, epsilon_tensor),
                self.divergence_E_residual(E_field, coords, epsilon_tensor),
                self.divergence_B_residual(H_field, coords),
            ],
            dim=1,
        )

    # ------------------------------------------------------------------ derived
    @staticmethod
    def poynting_vector(E_field: torch.Tensor, H_field: torch.Tensor) -> torch.Tensor:
        """
        Time-averaged Poynting vector ``S = ½ Re(E × H*)``.

        Args:
            E_field, H_field: Complex fields, shape ``(N, 3)``.

        Returns:
            Real tensor, shape ``(N, 3)``.
        """
        return 0.5 * torch.linalg.cross(E_field, H_field.conj(), dim=1).real
