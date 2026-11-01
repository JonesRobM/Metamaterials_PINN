"""
Metamaterial constitutive relations for anisotropic SPP modelling.

Implements the permittivity tensor of a uniaxial (non-magnetic) metamaterial
and the analytical TM surface-plasmon-polariton dispersion at a planar
interface with an isotropic dielectric, for validation of PINN solutions.

Sign convention: time dependence ``exp(-iωt)`` (see
:mod:`src.physics.maxwell_equations`). Lossy media therefore have ``Im(ε) > 0``
and a surface wave propagating in ``+x`` and decaying with distance has
``Im(k_spp) > 0``.

Geometry used by the SPP methods
--------------------------------
* Interface plane ``z = 0``; metamaterial occupies ``z < 0``, isotropic
  dielectric ``ε_d`` occupies ``z > 0``.
* The SPP is TM polarised and propagates along ``propagation_direction``
  (``'x'`` by default), which must lie in the interface plane.

For a diagonal permittivity ``diag(ε_xx, ε_yy, ε_zz)`` with the field
``H = (0, H_y, 0) exp(i k x)`` the extraordinary-wave dispersion in the
anisotropic medium is ``k²/ε_zz + k_z²/ε_xx = k₀²``. Matching ``H_y`` and
``E_x`` at ``z = 0`` gives (see e.g. Elser, Podolskiy et al., *Appl. Phys.
Lett.* 89, 261102 (2006); Yermakov et al., *Phys. Rev. B* 91, 235423 (2015))

    k_spp² = k₀² ε_d ε_n (ε_t − ε_d) / (ε_t ε_n − ε_d²)

where ``ε_t`` is the metamaterial permittivity along the propagation direction
and ``ε_n`` the component normal to the interface. Setting ``ε_t = ε_n = ε_m``
recovers the isotropic result ``k_spp² = k₀² ε_d ε_m / (ε_d + ε_m)``. The
decay constants are

    κ_d² = k_spp² − ε_d k₀²          (dielectric side)
    κ_m² = ε_t (k_spp² / ε_n − k₀²)  (metamaterial side)

and the unsquared boundary condition is ``κ_d / ε_d + κ_m / ε_t = 0``.
"""

from __future__ import annotations

import cmath
import math
from typing import Optional, Tuple

import torch

from ..constants import C0, EPS0

__all__ = ["MetamaterialProperties"]

_AXES = ("x", "y", "z")


def _decaying_root(z2: complex) -> complex:
    """Square root with ``Re > 0`` (evanescent decay away from the interface)."""
    r = cmath.sqrt(z2)
    if r.real < 0 or (r.real == 0 and r.imag < 0):
        r = -r
    return r


def _propagating_root(z2: complex) -> complex:
    """
    Square root branch for a wavevector under the ``exp(-iωt)`` convention:
    ``Im > 0`` (decaying along propagation); for a real argument choose ``Re > 0``.
    """
    r = cmath.sqrt(z2)
    if r.imag < 0 or (r.imag == 0 and r.real < 0):
        r = -r
    return r


class MetamaterialProperties:
    """
    Uniaxial, non-magnetic metamaterial with permittivity ``diag(ε_⊥, ε_⊥, ε_∥)``
    (optical axis along ``optical_axis``).

    Args:
        eps_parallel: Relative permittivity parallel to the optical axis (ε_∥).
        eps_perpendicular: Relative permittivity perpendicular to the optical axis (ε_⊥).
        optical_axis: ``'x'``, ``'y'`` or ``'z'``.
        eps0: Vacuum permittivity (F/m).
        omega: Optional design angular frequency (rad/s). When given, ``k0`` and
            ``omega`` are available as attributes and may be omitted from method calls.
        wavelength: Alternative to ``omega``: free-space wavelength (m).
    """

    def __init__(
        self,
        eps_parallel: complex,
        eps_perpendicular: complex,
        optical_axis: str = "z",
        eps0: float = EPS0,
        omega: Optional[float] = None,
        wavelength: Optional[float] = None,
    ):
        self.eps_par = complex(eps_parallel)
        self.eps_perp = complex(eps_perpendicular)
        self.optical_axis = optical_axis.lower()
        self.eps0 = eps0

        if self.optical_axis not in _AXES:
            raise ValueError("Optical axis must be 'x', 'y', or 'z'")
        if omega is not None and wavelength is not None:
            raise ValueError("Specify either omega or wavelength, not both")

        if wavelength is not None:
            omega = 2.0 * math.pi * C0 / wavelength
        self.omega: Optional[float] = float(omega) if omega is not None else None

    # ------------------------------------------------------------------ frequency helpers
    @property
    def k0(self) -> Optional[float]:
        """Free-space wavenumber ω/c for the design frequency, or ``None``."""
        return None if self.omega is None else self.omega / C0

    def _resolve_k0(self, omega: Optional[float] = None, k0: Optional[float] = None) -> float:
        if k0 is not None:
            return float(k0)
        if omega is not None:
            return float(omega) / C0
        if self.k0 is None:
            raise ValueError(
                "No frequency available: pass omega (or k0) to the method, or set omega/"
                "wavelength in the constructor."
            )
        return self.k0

    # ------------------------------------------------------------------ tensors
    def eps_along(self, axis: str) -> complex:
        """Diagonal permittivity component along a Cartesian axis."""
        axis = axis.lower()
        if axis not in _AXES:
            raise ValueError("axis must be 'x', 'y', or 'z'")
        return self.eps_par if axis == self.optical_axis else self.eps_perp

    def permittivity_tensor(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Relative permittivity tensor at each coordinate (homogeneous medium).

        Args:
            coords: Coordinates, shape ``(N, D)``.

        Returns:
            Complex tensor of shape ``(N, 3, 3)``.
        """
        dtype = torch.complex128 if coords.dtype == torch.float64 else torch.complex64
        diag = torch.tensor(
            [self.eps_along(a) for a in _AXES], dtype=dtype, device=coords.device
        )
        return torch.diag_embed(diag).unsqueeze(0).expand(coords.shape[0], -1, -1)

    def effective_permittivity(
        self,
        kx: torch.Tensor,
        ky: torch.Tensor,
        k0: Optional[float] = None,
        omega: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Effective permittivity seen by a TM wave with in-plane wavevector ``(kx, ky)``.

        The extraordinary-wave dispersion is ``k_∥²/ε_n + k_z²/ε_t = k₀²`` where
        ``ε_t`` (``ε_n``) is the permittivity component transverse (normal) to the
        interface at ``z = 0``. Defining the effective permittivity through the
        isotropic-looking relation ``k_z² = ε_eff k₀² − k_∥²`` gives

            ε_eff = ε_t + (k_∥²/k₀²)(1 − ε_t/ε_n).

        With the optical axis normal to the interface (``'z'``) ``ε_t = ε_⊥``,
        ``ε_n = ε_∥``. With an in-plane optical axis ``ε_t`` depends on the
        direction of ``k_∥``: ``ε_t = (ε_xx kx² + ε_yy ky²) / k_∥²``.

        Args:
            kx, ky: In-plane wavevector components (same shape).
            k0: Free-space wavenumber. Falls back to ``omega`` or the constructor value.
            omega: Angular frequency, alternative to ``k0``.

        Returns:
            Complex tensor ``ε_eff`` broadcast to the shape of ``kx``.
        """
        k0_val = self._resolve_k0(omega, k0)
        k_par_sq = kx**2 + ky**2
        eps_n = self.eps_along("z")

        if self.optical_axis == "z":
            eps_t = self.eps_perp
        else:
            # Transverse component along the direction of k_∥.
            eps_x, eps_y = self.eps_along("x"), self.eps_along("y")
            zero = k_par_sq == 0
            safe = torch.where(zero, torch.ones_like(k_par_sq), k_par_sq)
            eps_t = (eps_x * kx**2 + eps_y * ky**2) / safe
            eps_t = torch.where(zero, torch.full_like(eps_t, eps_x), eps_t)

        return eps_t + (k_par_sq / k0_val**2) * (1.0 - eps_t / eps_n)

    # ------------------------------------------------------------------ SPP dispersion
    def _spp_components(self, propagation_direction: str) -> Tuple[complex, complex]:
        """Return ``(ε_t, ε_n)`` for an SPP propagating along ``propagation_direction``."""
        pd = propagation_direction.lower()
        if pd not in ("x", "y"):
            raise ValueError("propagation_direction must lie in the interface plane ('x' or 'y')")
        return self.eps_along(pd), self.eps_along("z")

    def spp_wavevector(
        self,
        omega: Optional[float] = None,
        eps_dielectric: complex = 1.0,
        propagation_direction: str = "x",
        k0: Optional[float] = None,
    ) -> complex:
        """
        Complex SPP wavevector ``k_spp`` (1/m) with ``Im(k_spp) >= 0``.

        See module docstring for the formula. The branch of the square root is
        chosen explicitly so the wave decays along its propagation direction
        under the ``exp(-iωt)`` convention.
        """
        k0_val = self._resolve_k0(omega, k0)
        eps_t, eps_n = self._spp_components(propagation_direction)
        eps_d = complex(eps_dielectric)
        denom = eps_t * eps_n - eps_d**2
        if denom == 0:
            raise ZeroDivisionError("SPP dispersion is singular: ε_t ε_n = ε_d²")
        k_sq = k0_val**2 * eps_d * eps_n * (eps_t - eps_d) / denom
        return _propagating_root(k_sq)

    def spp_dispersion_relation(
        self,
        omega: Optional[float] = None,
        eps_dielectric: complex = 1.0,
        propagation_direction: str = "x",
    ) -> Tuple[float, float]:
        """
        ``(Re k_spp, Im k_spp)`` at angular frequency ``omega``.

        Args:
            omega: Angular frequency (rad/s); defaults to the constructor value.
            eps_dielectric: Relative permittivity of the dielectric half-space.
            propagation_direction: In-plane propagation axis (``'x'`` or ``'y'``).
        """
        k = self.spp_wavevector(omega, eps_dielectric, propagation_direction)
        return k.real, k.imag

    def _decay_constants(
        self, omega: Optional[float], eps_dielectric: complex, propagation_direction: str
    ) -> Tuple[complex, complex, complex]:
        """Return ``(k_spp, κ_d, κ_m)`` with ``Re κ > 0`` (bound-mode branch)."""
        k0_val = self._resolve_k0(omega)
        eps_t, eps_n = self._spp_components(propagation_direction)
        eps_d = complex(eps_dielectric)
        k = self.spp_wavevector(omega, eps_d, propagation_direction)
        kappa_d = _decaying_root(k**2 - eps_d * k0_val**2)
        kappa_m = _decaying_root(eps_t * (k**2 / eps_n - k0_val**2))
        return k, kappa_d, kappa_m

    def decay_constants(
        self,
        omega: Optional[float] = None,
        eps_dielectric: complex = 1.0,
        propagation_direction: str = "x",
    ) -> Tuple[complex, complex, complex]:
        """Complex ``(k_spp, κ_d, κ_m)`` for the bound TM surface mode.

        Public counterpart to the penetration-depth helpers, which expose only
        ``1 / Re κ``. The full complex constants are what an analytical mode
        profile needs (see :func:`src.analytical.analytical_spp_fields`), so
        callers should not have to reach for the private method.

        Args:
            omega: Angular frequency (rad/s); defaults to the constructor value.
            eps_dielectric: Relative permittivity of the dielectric half-space.
            propagation_direction: In-plane propagation axis (``'x'`` or ``'y'``).

        Returns:
            ``(k_spp, κ_d, κ_m)`` on the ``Im k ≥ 0`` / ``Re κ > 0`` branches.
        """
        return self._decay_constants(omega, eps_dielectric, propagation_direction)

    def propagation_length(
        self,
        omega: Optional[float] = None,
        eps_dielectric: complex = 1.0,
        propagation_direction: str = "x",
    ) -> float:
        """SPP intensity propagation length ``L = 1 / (2 Im k_spp)`` (m)."""
        _, k_imag = self.spp_dispersion_relation(omega, eps_dielectric, propagation_direction)
        return 1.0 / (2.0 * k_imag) if k_imag > 0 else float("inf")

    def penetration_depth_metamaterial(
        self,
        omega: Optional[float] = None,
        eps_dielectric: complex = 1.0,
        propagation_direction: str = "x",
    ) -> float:
        """Field penetration depth ``1 / Re κ_m`` into the metamaterial (m)."""
        _, _, kappa_m = self._decay_constants(omega, eps_dielectric, propagation_direction)
        return 1.0 / kappa_m.real if kappa_m.real > 0 else float("inf")

    def penetration_depth_dielectric(
        self,
        omega: Optional[float] = None,
        eps_dielectric: complex = 1.0,
        propagation_direction: str = "x",
    ) -> float:
        """Field penetration depth ``1 / Re κ_d`` into the dielectric (m)."""
        _, kappa_d, _ = self._decay_constants(omega, eps_dielectric, propagation_direction)
        return 1.0 / kappa_d.real if kappa_d.real > 0 else float("inf")

    def field_enhancement_factor(
        self,
        omega: Optional[float] = None,
        eps_dielectric: complex = 1.0,
        propagation_direction: str = "x",
    ) -> float:
        """
        Ratio of normal to tangential electric-field amplitude at the interface
        on the dielectric side, ``|E_z| / |E_x| = |k_spp| / |κ_d|``.

        For an isotropic metal this is ``sqrt(|ε_m| / ε_d)`` (Maier, *Plasmonics*,
        Sec. 2.2), a measure of how strongly the SPP field is concentrated
        normal to the surface. It is a property of the mode alone and is not
        the excitation-dependent enhancement of a Kretschmann/grating coupler.
        """
        k, kappa_d, _ = self._decay_constants(omega, eps_dielectric, propagation_direction)
        if abs(kappa_d) == 0:
            return float("inf")
        return abs(k) / abs(kappa_d)

    def is_spp_supported(
        self,
        eps_dielectric: complex = 1.0,
        propagation_direction: str = "x",
        rel_tol: float = 1e-6,
        bound_tol: float = 1e-3,
    ) -> bool:
        """
        Whether a bound TM surface mode exists at the interface.

        The condition is checked in full (frequency independent, since ``k₀``
        scales out): with ``κ_d, κ_m`` on the ``Re > 0`` branch the unsquared
        matching condition ``κ_d/ε_d + κ_m/ε_t = 0`` must hold (squaring can
        introduce spurious roots) and both decay constants must have positive
        real part (``Re κ > bound_tol·|κ|``) so the mode is bound on both
        sides. For lossless media this
        reduces to ``ε_t < 0``, ``k_spp² > ε_d k₀²`` and ``κ_m² > 0``.
        """
        eps_t, eps_n = self._spp_components(propagation_direction)
        eps_d = complex(eps_dielectric)
        denom = eps_t * eps_n - eps_d**2
        if denom == 0:
            return False
        n_sq = eps_d * eps_n * (eps_t - eps_d) / denom  # (k_spp / k0)^2
        kappa_d = _decaying_root(n_sq - eps_d)
        kappa_m = _decaying_root(eps_t * (n_sq / eps_n - 1.0))
        # Both decay constants must be *meaningfully* real-positive. Near a
        # radiative (k_spp < sqrt(ε_d) k0) configuration a small loss term makes
        # Re κ tiny but positive; that is an oscillatory, not a bound, field.
        for kappa in (kappa_d, kappa_m):
            if kappa.real <= bound_tol * abs(kappa):
                return False
        matching = kappa_d / eps_d + kappa_m / eps_t
        scale = abs(kappa_d / eps_d) + abs(kappa_m / eps_t)
        return abs(matching) <= rel_tol * scale

    def __repr__(self) -> str:
        return (
            f"MetamaterialProperties(eps_∥={self.eps_par}, eps_⊥={self.eps_perp}, "
            f"optical_axis={self.optical_axis}, omega={self.omega})"
        )
