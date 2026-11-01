"""
Boundary conditions for electromagnetic fields at metamaterial interfaces.

Continuity of tangential ``E`` and ``H`` and of normal ``D`` and ``B`` at a
source-free interface, plus perfect-conductor, surface-impedance and a
first-order radiation condition.

Sign convention: time dependence ``exp(-iωt)`` (see
:mod:`src.physics.maxwell_equations`); an outgoing wave is ``exp(+ikr)``.

All field inputs are complex tensors of shape ``(N, 3)``. Residual methods
return real tensors with the real and imaginary parts concatenated along the
last axis: ``(N, 6)`` for vector conditions and ``(N, 2)`` for scalar ones.
"""

from __future__ import annotations

import cmath
from typing import Tuple

import torch

from ..constants import EPS0, MU0

__all__ = ["BoundaryConditions"]


class BoundaryConditions:
    """
    Electromagnetic boundary conditions at a planar interface.

    1. Tangential E continuity: ``n × (E₂ − E₁) = 0``
    2. Tangential H continuity: ``n × (H₂ − H₁) = 0`` (no surface current)
    3. Normal D continuity:     ``n · (D₂ − D₁) = 0`` (no surface charge)
    4. Normal B continuity:     ``n · (B₂ − B₁) = 0``

    Args:
        interface_normal: Unit normal pointing from medium 1 into medium 2.
        eps0: Vacuum permittivity (F/m).
        mu0: Vacuum permeability (H/m).
    """

    def __init__(
        self,
        interface_normal: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        eps0: float = EPS0,
        mu0: float = MU0,
    ):
        normal = torch.tensor(interface_normal, dtype=torch.float32)
        norm = torch.norm(normal)
        if norm == 0:
            raise ValueError("interface_normal must be non-zero")
        self.interface_normal = normal / norm
        self.eps0 = eps0
        self.mu0 = mu0

    # ------------------------------------------------------------------ helpers
    def _normal_like(self, ref: torch.Tensor) -> torch.Tensor:
        """Normal vector broadcast to ``(N, 3)`` on ``ref``'s device and dtype."""
        n = self.interface_normal.to(device=ref.device)
        n = n.to(ref.dtype) if ref.is_complex() else n.to(ref.real.dtype)
        return n.unsqueeze(0).expand(ref.shape[0], -1)

    @staticmethod
    def _split(z: torch.Tensor) -> torch.Tensor:
        if not z.is_complex():
            z = torch.complex(z, torch.zeros_like(z))
        return torch.cat([z.real, z.imag], dim=1)

    @staticmethod
    def cross_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Batched cross product ``a × b`` supporting real/complex mixtures.

        Args:
            a, b: Tensors of shape ``(N, 3)`` or ``(3,)``.

        Returns:
            Tensor of shape ``(N, 3)``, complex if either input is complex.
        """
        if a.dim() == 1:
            a = a.unsqueeze(0)
        if b.dim() == 1:
            b = b.unsqueeze(0)
        if a.is_complex() or b.is_complex():
            dtype = a.dtype if a.is_complex() else b.dtype
            a, b = a.to(dtype), b.to(dtype)
        a, b = torch.broadcast_tensors(a, b)
        return torch.linalg.cross(a, b, dim=1)

    # ------------------------------------------------------------------ continuity
    def tangential_E_continuity(self, E1: torch.Tensor, E2: torch.Tensor) -> torch.Tensor:
        """
        ``n × (E₂ − E₁)``.

        Args:
            E1, E2: Complex E fields on each side, shape ``(N, 3)``.

        Returns:
            ``(N, 6)`` tensor ``[Re, Im]`` of the residual vector.
        """
        E_diff = E2 - E1
        return self._split(self.cross_product(self._normal_like(E_diff), E_diff))

    def tangential_H_continuity(self, H1: torch.Tensor, H2: torch.Tensor) -> torch.Tensor:
        """
        ``n × (H₂ − H₁)``.

        Returns:
            ``(N, 6)`` tensor ``[Re, Im]`` of the residual vector.
        """
        H_diff = H2 - H1
        return self._split(self.cross_product(self._normal_like(H_diff), H_diff))

    def normal_D_continuity(
        self,
        E1: torch.Tensor,
        E2: torch.Tensor,
        eps1_tensor: torch.Tensor,
        eps2_tensor: torch.Tensor,
        relative: bool = False,
    ) -> torch.Tensor:
        """
        ``n · (D₂ − D₁)`` with ``D = ε₀ εᵣ E``.

        Args:
            E1, E2: Complex E fields, shape ``(N, 3)``.
            eps1_tensor, eps2_tensor: Relative permittivity tensors, shape ``(N, 3, 3)``.
            relative: If True, omit the ``ε₀`` factor (residual in units of ``E``).

        Returns:
            ``(N, 2)`` tensor ``[Re, Im]`` of the scalar residual.
        """
        scale = 1.0 if relative else self.eps0
        D1 = scale * torch.einsum("nij,nj->ni", eps1_tensor.to(E1.dtype), E1)
        D2 = scale * torch.einsum("nij,nj->ni", eps2_tensor.to(E2.dtype), E2)
        D_diff = D2 - D1
        n = self._normal_like(D_diff)
        return self._split(torch.sum(n * D_diff, dim=1, keepdim=True))

    def normal_B_continuity(
        self, H1: torch.Tensor, H2: torch.Tensor, relative: bool = False
    ) -> torch.Tensor:
        """
        ``n · (B₂ − B₁)`` with ``B = μ₀ H`` (non-magnetic media).

        Args:
            relative: If True, omit the ``μ₀`` factor (residual in units of ``H``).

        Returns:
            ``(N, 2)`` tensor ``[Re, Im]`` of the scalar residual.
        """
        scale = 1.0 if relative else self.mu0
        B_diff = scale * (H2 - H1)
        n = self._normal_like(B_diff)
        return self._split(torch.sum(n * B_diff, dim=1, keepdim=True))

    def spp_boundary_conditions(
        self,
        E_metamaterial: torch.Tensor,
        H_metamaterial: torch.Tensor,
        E_dielectric: torch.Tensor,
        H_dielectric: torch.Tensor,
        eps_metamaterial: torch.Tensor,
        eps_dielectric: complex = 1.0,
        relative: bool = False,
    ) -> torch.Tensor:
        """
        All four continuity residuals at a metamaterial (medium 1) / dielectric
        (medium 2) interface.

        Args:
            E_metamaterial, H_metamaterial: Fields in the metamaterial, ``(N, 3)``.
            E_dielectric, H_dielectric: Fields in the dielectric, ``(N, 3)``.
            eps_metamaterial: Metamaterial permittivity tensor, ``(N, 3, 3)``.
            eps_dielectric: Scalar relative permittivity of the dielectric.
            relative: Use relative units for the D and B residuals.

        Returns:
            ``(N, 16)``: ``[tang E (6), tang H (6), norm D (2), norm B (2)]``.
        """
        n_pts = E_metamaterial.shape[0]
        eye = torch.eye(3, device=E_metamaterial.device, dtype=E_metamaterial.dtype)
        eps_diel_tensor = (eye * eps_dielectric).unsqueeze(0).expand(n_pts, -1, -1)

        return torch.cat(
            [
                self.tangential_E_continuity(E_metamaterial, E_dielectric),
                self.tangential_H_continuity(H_metamaterial, H_dielectric),
                self.normal_D_continuity(
                    E_metamaterial, E_dielectric, eps_metamaterial, eps_diel_tensor, relative
                ),
                self.normal_B_continuity(H_metamaterial, H_dielectric, relative),
            ],
            dim=1,
        )

    # ------------------------------------------------------------------ other conditions
    def perfect_conductor_boundary(self, E_field: torch.Tensor) -> torch.Tensor:
        """
        Perfect electric conductor: tangential E must vanish.

        Args:
            E_field: Complex E field at the boundary, shape ``(N, 3)``.

        Returns:
            ``(N, 6)`` tensor ``[Re, Im]`` of ``E − (n·E) n``.
        """
        n = self._normal_like(E_field)
        tangential_E = E_field - torch.sum(n * E_field, dim=1, keepdim=True) * n
        return self._split(tangential_E)

    def impedance_boundary_condition(
        self, E_tangential: torch.Tensor, H_tangential: torch.Tensor, surface_impedance: complex
    ) -> torch.Tensor:
        """
        Surface impedance condition ``n × E = Z_s (n × H) × n``.

        Args:
            E_tangential, H_tangential: Complex fields at the boundary, ``(N, 3)``.
            surface_impedance: Surface impedance ``Z_s`` (Ohm).

        Returns:
            ``(N, 6)`` tensor ``[Re, Im]`` of the residual vector.
        """
        n = self._normal_like(E_tangential)
        n_cross_E = self.cross_product(n, E_tangential)
        # (n × H) × n = H − (n·H) n for |n| = 1
        n_dot_H = torch.sum(n * H_tangential, dim=1, keepdim=True)
        H_tan = H_tangential - n_dot_H * n
        residual = n_cross_E - complex(surface_impedance) * H_tan
        return self._split(residual)

    def radiation_boundary_condition(
        self,
        E_field: torch.Tensor,
        H_field: torch.Tensor,
        k0: float,
        eps_background: complex = 1.0,
        normal_derivative_E: torch.Tensor | None = None,
        normal_derivative_H: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        First-order absorbing (Sommerfeld-type) condition ``(∂ₙ − ik) F = 0``
        with ``k = k₀ √ε_bg`` for outgoing waves ``exp(+ikr)``.

        Args:
            E_field, H_field: Complex fields at the boundary, ``(N, 3)``.
            k0: Free-space wavenumber (1/m).
            eps_background: Background relative permittivity.
            normal_derivative_E, normal_derivative_H: ``∂F/∂n`` at the boundary,
                ``(N, 3)``. If omitted they are taken as zero and the residual
                reduces to ``−ikF`` (a penalty on the field amplitude only).

        Returns:
            ``(N, 12)`` tensor ``[Re(E res), Re(H res), Im(E res), Im(H res)]``.
        """
        k = k0 * cmath.sqrt(complex(eps_background))
        dE = torch.zeros_like(E_field) if normal_derivative_E is None else normal_derivative_E
        dH = torch.zeros_like(H_field) if normal_derivative_H is None else normal_derivative_H
        residual = torch.cat([dE - 1j * k * E_field, dH - 1j * k * H_field], dim=1)
        return self._split(residual)

    def __repr__(self) -> str:
        return f"BoundaryConditions(normal={self.interface_normal.tolist()})"
