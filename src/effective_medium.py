r"""
Effective-medium theory for layered (metal/dielectric) uniaxial metamaterials.

A stack of alternating metal and dielectric layers with a period much smaller
than the wavelength behaves, to leading order, as a *homogeneous uniaxial*
medium — the canonical hyperbolic metamaterial (HMM). This module supplies the
three ingredients needed to drive
:class:`src.physics.metamaterial.MetamaterialProperties` with a physically
dispersive material: a Drude permittivity for the metal layers, the
effective-medium (Rytov) averages that turn ``(ε_m, ε_d2, f)`` into the
uniaxial pair ``(ε_t, ε_n)``, and the classification of the resulting
iso-frequency surface.

Everything is vectorised over NumPy arrays (and broadcasts between
arguments, so a fill-fraction column and a frequency row give a 2-D map) and
returns complex values.

Geometry and naming
-------------------
Layers are stacked along **z**, so the optical axis is ``z`` and the geometry
matches the SPP setup of :mod:`src.physics.metamaterial` (interface at
``z = 0``, metamaterial in ``z < 0``):

* ``ε_t`` — in-plane component (``ε_xx = ε_yy``), *transverse* to the interface
  normal, i.e. along the SPP propagation direction;
* ``ε_n`` — component along ``z``, *normal* to the interface, and along the
  optical axis.

Beware the constructor order of ``MetamaterialProperties``: its
``eps_parallel`` argument is the component *along the optical axis*, so for
``optical_axis='z'`` it takes ``ε_n``, and ``eps_perpendicular`` takes ``ε_t``::

    eps_t, eps_n = hmm_permittivities(omega, f, eps_d2)
    material = MetamaterialProperties(eps_n, eps_t, optical_axis="z", omega=omega)

Derivation of the two averages
------------------------------
Let the period be ``a = a_m + a_d2`` with metal fill fraction ``f = a_m / a``.
In the quasi-static (long-wavelength) limit the field inside each layer is
uniform, and the two polarisations are fixed by different continuity
conditions at the layer boundaries (normals along ``z``).

**In-plane (E parallel to the layers).** The tangential electric field is
continuous, so ``E_x`` takes the *same* value ``E`` in every layer. The
thickness-averaged displacement is then

    ⟨D_x⟩ = ε₀ [f ε_m + (1 − f) ε_d2] E,

and with ``ε_t ≡ ⟨D_x⟩ / (ε₀ ⟨E_x⟩)``

    **ε_t = f ε_m + (1 − f) ε_d2**            (arithmetic / parallel-capacitor mean)

**Normal (E along z).** The normal displacement is continuous across each
boundary (no free interface charge), so ``D_z`` takes the same value ``D`` in
every layer while ``E_z`` jumps. The averaged field is

    ⟨E_z⟩ = (D / ε₀) [f / ε_m + (1 − f) / ε_d2],

so ``ε_n ≡ ⟨D_z⟩ / (ε₀ ⟨E_z⟩)`` is the harmonic (series-capacitor) mean

    **ε_n = [f/ε_m + (1−f)/ε_d2]⁻¹ = ε_m ε_d2 / (f ε_d2 + (1 − f) ε_m)**

Both reduce to the constituent permittivity at ``f = 0`` and ``f = 1``
exactly. Writing ``D ≡ f ε_d2 + (1 − f) ε_m`` for the harmonic denominator and
splitting ``ε_n = ε_d2 ε_m \bar{D} / |D|²`` into real and imaginary parts gives
two exact identities used throughout this module:

    Re ε_n = ε_d2 [ (1 − f)|ε_m|² + f ε_d2 Re ε_m ] / |D|²
    Im ε_n = f ε_d2² Im ε_m / |D|²                                    (†)

so **the harmonic mean is passive wherever the metal is** (``Im ε_m > 0 ⇒
Im ε_n > 0`` for any ``0 < f < 1``), and its "pole" ``D → 0`` is a genuine
*loss resonance*, not a passivity violation: there ``Im ε_n`` grows like
``1/|D|²`` while ``Re ε_n`` passes smoothly through zero. Finite loss keeps
``|D| ≥ (1 − f) Im ε_m > 0``, so nothing actually diverges; the peak height is
``|ε_n| ≈ ε_d2 |ε_m| / ((1 − f) Im ε_m)``. This resonance is the physical
ENZ/epsilon-pole of the multilayer, and it is exactly where effective-medium
theory is least trustworthy (see below).

Validity limit (the honest caveat)
----------------------------------
These are the **leading term of a long-wavelength expansion**: they hold only
when the period is much smaller than every wavelength in the problem,

    a ≪ λ / |n|,  equivalently  a · k ≪ 1 for every k in the mode,

with corrections of order ``(a k)²`` (Rytov, *Sov. Phys. JETP* 2, 466 (1956);
Elser *et al.*, *Appl. Phys. Lett.* 90, 191109 (2007)). Two failure modes
matter here and neither is cured by making the formulas prettier:

1. **High-k modes.** A hyperbolic medium supports arbitrarily large ``k``; once
   ``k a ~ 1`` the effective medium stops describing them, the true dispersion
   is cut off by the finite period, and the EMT *under*-estimates loss.
   :func:`max_layer_period` turns a mode's largest wavenumber into the period
   the design must respect.
2. **Near the ENZ / pole frequencies**, where ``ε_n`` or ``ε_t`` is small, the
   local fields in the layers are enormous and non-local (spatial-dispersion)
   corrections become first-order. Numbers quoted within a few per cent of a
   crossing should be read as qualitative.

Additionally each layer must be thick enough (≳ 5 nm for noble metals) for a
bulk permittivity to mean anything at all.

Sign convention
---------------
Time dependence ``exp(-iωt)`` throughout the project, so **lossy media have
Im ε > 0**. The Drude form used here,

    ε(ω) = ε_∞ − ω_p² / (ω² + iγω)
         = [ε_∞ − ω_p²/(ω² + γ²)] + i [ω_p² γ / (ω (ω² + γ²))],

manifestly has ``Im ε > 0`` for ``ω, γ > 0`` — the imaginary part is a ratio of
positive quantities — which is verified in
``tests/test_effective_medium.py::TestDrude``.

Default metal
-------------
The Drude parameters default to the silver fit used by
``examples/dispersion_analysis.py`` (ε_∞ = 3.7, ħω_p = 9.1 eV, ħγ = 0.018 eV)
so the two studies describe the same metal. Its limitations are the ones
recorded there: no interband transitions, so it degrades rapidly below
~450 nm, and the free-electron damping under-represents the measured loss
(at 633 nm, Im ε ≈ 0.2 against the Johnson & Christy value 0.55).
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np

from .constants import C0

__all__ = [
    "EV",
    "HBAR",
    "HBAR_EVS",
    "EPS_INF_AG",
    "HBAR_OMEGA_P_AG_EV",
    "HBAR_GAMMA_AG_EV",
    "OMEGA_P_AG",
    "GAMMA_AG",
    "ANISOTROPY_CLASSES",
    "omega_from_wavelength",
    "wavelength_from_omega",
    "omega_from_photon_energy_ev",
    "photon_energy_ev",
    "drude_parameters_ev",
    "drude_permittivity",
    "layered_uniaxial",
    "hmm_permittivities",
    "classify_anisotropy",
    "transition_frequencies",
    "max_layer_period",
]

ArrayLike = Union[float, complex, np.ndarray]

# ------------------------------------------------------------------ constants
EV: float = 1.602176634e-19
"""Joules per electronvolt (exact, SI 2019)."""

HBAR: float = 1.054571817e-34
"""Reduced Planck constant [J s]."""

HBAR_EVS: float = HBAR / EV
"""Reduced Planck constant [eV s]: multiplies ω (rad/s) to give ħω (eV)."""

# Drude fit for silver, identical to examples/dispersion_analysis.py.
EPS_INF_AG: float = 3.7
HBAR_OMEGA_P_AG_EV: float = 9.1
HBAR_GAMMA_AG_EV: float = 0.018
OMEGA_P_AG: float = HBAR_OMEGA_P_AG_EV * EV / HBAR
"""Silver plasma frequency [rad/s]."""
GAMMA_AG: float = HBAR_GAMMA_AG_EV * EV / HBAR
"""Silver Drude damping rate [rad/s]."""

ANISOTROPY_CLASSES: Tuple[str, ...] = (
    "type-I",
    "type-II",
    "elliptic-dielectric",
    "elliptic-metallic",
)
"""The four classes returned by :func:`classify_anisotropy`."""


# ------------------------------------------------------------ unit conversions
def omega_from_wavelength(wavelength: ArrayLike) -> np.ndarray:
    """Angular frequency (rad/s) of a free-space wavelength (m)."""
    return 2.0 * np.pi * C0 / np.asarray(wavelength, dtype=float)


def wavelength_from_omega(omega: ArrayLike) -> np.ndarray:
    """Free-space wavelength (m) of an angular frequency (rad/s)."""
    return 2.0 * np.pi * C0 / np.asarray(omega, dtype=float)


def omega_from_photon_energy_ev(energy_ev: ArrayLike) -> np.ndarray:
    """Angular frequency (rad/s) of a photon energy ħω (eV)."""
    return np.asarray(energy_ev, dtype=float) / HBAR_EVS


def photon_energy_ev(omega: ArrayLike) -> np.ndarray:
    """Photon energy ħω (eV) of an angular frequency (rad/s)."""
    return HBAR_EVS * np.asarray(omega, dtype=float)


def drude_parameters_ev(
    eps_inf: float = EPS_INF_AG,
    hbar_omega_p_ev: float = HBAR_OMEGA_P_AG_EV,
    hbar_gamma_ev: float = HBAR_GAMMA_AG_EV,
) -> Dict[str, float]:
    """
    Drude parameters in SI, from the eV-based values usually quoted in the
    literature.

    Returns a dict ready to splat into :func:`drude_permittivity` or
    :func:`hmm_permittivities`::

        eps_t, eps_n = hmm_permittivities(
            omega, 0.3, 2.25, **drude_parameters_ev(hbar_gamma_ev=0.05)
        )
    """
    return {
        "eps_inf": float(eps_inf),
        "omega_p": float(hbar_omega_p_ev) * EV / HBAR,
        "gamma": float(hbar_gamma_ev) * EV / HBAR,
    }


# ------------------------------------------------------------------ Drude metal
def drude_permittivity(
    omega: ArrayLike,
    eps_inf: float = EPS_INF_AG,
    omega_p: float = OMEGA_P_AG,
    gamma: float = GAMMA_AG,
) -> np.ndarray:
    """
    Drude permittivity ``ε(ω) = ε_∞ − ω_p²/(ω² + iγω)``.

    Under the project's ``exp(-iωt)`` convention this is the *passive* sign
    choice: separating real and imaginary parts (multiply by the conjugate of
    the denominator) gives

        Re ε = ε_∞ − ω_p²/(ω² + γ²),
        Im ε = ω_p² γ / (ω (ω² + γ²)) > 0   for ω, γ > 0,

    so a lossy metal has ``Im ε > 0``, as required.

    Args:
        omega: Angular frequency (rad/s); scalar or array, must be > 0.
        eps_inf: Background permittivity ε_∞ (bound-electron screening).
        omega_p: Plasma frequency (rad/s).
        gamma: Collision/damping rate (rad/s).

    Returns:
        Complex permittivity, broadcast to the shape of ``omega``.
    """
    w = np.asarray(omega, dtype=float)
    if np.any(w <= 0.0):
        raise ValueError("omega must be positive")
    return eps_inf - omega_p**2 / (w**2 + 1j * gamma * w)


# --------------------------------------------------------- effective medium
def _check_fill_fraction(fill_fraction: ArrayLike) -> np.ndarray:
    f = np.asarray(fill_fraction, dtype=float)
    if np.any(f < 0.0) or np.any(f > 1.0):
        raise ValueError("fill_fraction must lie in [0, 1]")
    return f


def layered_uniaxial(
    eps_metal: ArrayLike,
    eps_dielectric: ArrayLike,
    fill_fraction: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Effective uniaxial permittivities of a metal/dielectric multilayer.

    Layers are stacked along ``z`` (optical axis ``z``), so the in-plane
    component is the arithmetic mean and the normal component the harmonic
    mean — see the module docstring for the derivation from the boundary
    conditions (tangential ``E`` continuous, normal ``D`` continuous)::

        ε_t = f ε_m + (1 − f) ε_d2
        ε_n = ε_m ε_d2 / (f ε_d2 + (1 − f) ε_m)

    Valid only for a period much smaller than the wavelength; see the module
    docstring's validity section and :func:`max_layer_period`.

    Args:
        eps_metal: Permittivity of the metal layers (complex, ``Im > 0``).
        eps_dielectric: Permittivity of the dielectric layers.
        fill_fraction: Metal filling fraction ``f = a_metal / period`` in [0, 1].

    Returns:
        ``(eps_t, eps_n)``: in-plane and normal permittivities, complex arrays
        broadcast over the three arguments.

    Raises:
        ValueError: if ``fill_fraction`` leaves [0, 1].
        ZeroDivisionError: if the harmonic denominator vanishes identically
            (only reachable in the lossless limit, at the multilayer's ENZ
            resonance ``f ε_d2 + (1 − f) ε_m = 0``).
    """
    f = _check_fill_fraction(fill_fraction)
    eps_m = np.asarray(eps_metal, dtype=complex)
    eps_d2 = np.asarray(eps_dielectric, dtype=complex)

    eps_t = f * eps_m + (1.0 - f) * eps_d2
    denom = f * eps_d2 + (1.0 - f) * eps_m
    if np.any(denom == 0):
        raise ZeroDivisionError(
            "harmonic-mean denominator f·ε_d2 + (1−f)·ε_m vanishes: the lossless "
            "ENZ resonance of the multilayer. Add loss (Im ε_m > 0) or move off "
            "the resonance."
        )
    eps_n = eps_m * eps_d2 / denom
    return eps_t, eps_n


def hmm_permittivities(
    omega: ArrayLike,
    fill_fraction: ArrayLike,
    eps_dielectric_layer: ArrayLike,
    **drude_kwargs: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ``(ε_t, ε_n)`` of a Drude-metal/dielectric multilayer at angular frequency
    ``omega``.

    Composes :func:`drude_permittivity` with :func:`layered_uniaxial`.

    Args:
        omega: Angular frequency (rad/s), scalar or array.
        fill_fraction: Metal filling fraction in [0, 1].
        eps_dielectric_layer: Permittivity of the dielectric layers
            (e.g. 2.25 for silica).
        **drude_kwargs: ``eps_inf``, ``omega_p``, ``gamma`` — defaults are the
            silver fit (see :func:`drude_parameters_ev`).

    Returns:
        ``(eps_t, eps_n)``, complex arrays broadcast over the arguments.
    """
    eps_m = drude_permittivity(omega, **drude_kwargs)
    return layered_uniaxial(eps_m, eps_dielectric_layer, fill_fraction)


def classify_anisotropy(
    eps_t: ArrayLike, eps_n: ArrayLike
) -> Union[str, np.ndarray]:
    """
    Classify the iso-frequency surface of a uniaxial medium from the signs of
    ``Re ε_t`` (in-plane) and ``Re ε_n`` (normal / optical axis).

    For a TM (extraordinary) wave the iso-frequency surface is
    ``k_∥²/ε_n + k_z²/ε_t = k₀²``, an ellipsoid when the two real parts share a
    sign and a hyperboloid when they differ:

    ============================  ==========  ==========  ==================
    class                         Re ε_t      Re ε_n      surface
    ============================  ==========  ==========  ==================
    ``'elliptic-dielectric'``     > 0         > 0         closed ellipsoid
    ``'type-I'``                  > 0         < 0         two-sheet hyperboloid
    ``'type-II'``                 < 0         > 0         one-sheet hyperboloid
    ``'elliptic-metallic'``       < 0         < 0         no propagation
    ============================  ==========  ==========  ==================

    (The type-I/type-II labels follow the usual convention keyed to the number
    of *negative* tensor components: type-I has one, type-II has two, counting
    ``ε_xx = ε_yy = ε_t`` twice.)

    A metal/dielectric multilayer with ``ε_d2 > 0`` reaches
    ``'elliptic-metallic'`` only for ``f > 1/2``: both real parts are negative
    only when ``−f ε_d2/(1−f) > Re ε_m > −(1−f) ε_d2/f``, a non-empty window
    precisely when ``f > 1/2``.

    Exact zeros (measure-zero ENZ crossings) are grouped with the positive
    branch; use :func:`transition_frequencies` to locate them.

    Args:
        eps_t: In-plane permittivity, scalar or array.
        eps_n: Normal permittivity, scalar or array.

    Returns:
        A ``str`` if both inputs are scalars, otherwise a NumPy string array
        broadcast over them.
    """
    t = np.real(np.asarray(eps_t, dtype=complex))
    n = np.real(np.asarray(eps_n, dtype=complex))
    t, n = np.broadcast_arrays(t, n)

    labels = np.where(
        t >= 0.0,
        np.where(n >= 0.0, "elliptic-dielectric", "type-I"),
        np.where(n >= 0.0, "type-II", "elliptic-metallic"),
    )
    return str(labels) if labels.ndim == 0 else labels


# ------------------------------------------------- ENZ / transition frequencies
def _bisect(func: Callable[[float], float], lo: float, hi: float, tol: float) -> float:
    """Bisection on a bracketing sign change (``func(lo)·func(hi) < 0``)."""
    f_lo = func(lo)
    for _ in range(200):
        if hi - lo <= tol:
            break
        mid = 0.5 * (lo + hi)
        f_mid = func(mid)
        if f_mid == 0.0:
            return mid
        if (f_lo < 0.0) != (f_mid < 0.0):
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def _scan_roots(
    func: Callable[[np.ndarray], np.ndarray],
    omegas: np.ndarray,
    rel_tol: float = 1e-12,
) -> np.ndarray:
    """All sign changes of a continuous ``func`` on a grid, refined by bisection."""
    values = np.asarray(func(omegas), dtype=float)
    flips = np.flatnonzero(np.sign(values[:-1]) * np.sign(values[1:]) < 0)

    def scalar(w: float) -> float:
        return float(func(np.asarray(float(w))))

    roots = [
        _bisect(scalar, float(omegas[i]), float(omegas[i + 1]), rel_tol * float(omegas[i]))
        for i in flips
    ]
    # Exact grid zeros (rare, but a lossless model can hit one).
    roots += [float(w) for w, v in zip(omegas, values, strict=True) if v == 0.0]
    return np.sort(np.asarray(roots, dtype=float))


def transition_frequencies(
    fill_fraction: float,
    eps_dielectric_layer: float,
    omega_range: Optional[Tuple[float, float]] = None,
    n_scan: int = 2001,
    **drude_kwargs: float,
) -> Dict[str, np.ndarray]:
    """
    ENZ / topological-transition frequencies of a Drude-metal multilayer: where
    ``Re ε_t`` or ``Re ε_n`` crosses zero, and the harmonic-mean resonance.

    Each crossing separates two of the classes of :func:`classify_anisotropy`,
    so together they tile the spectrum into type-I / type-II / elliptic bands.

    **``Re ε_t = 0`` is solved analytically.** ``Re ε_t = f Re ε_m +
    (1−f) ε_d2`` vanishes when ``Re ε_m = −(1−f) ε_d2 / f``, and for the Drude
    form ``Re ε_m = ε_∞ − ω_p²/(ω² + γ²)`` this inverts in closed form:

        ω² = ω_p² / (ε_∞ − Re ε_m) − γ²

    (returned only when the right-hand side is positive, i.e. when the crossing
    exists at all).

    **``Re ε_n = 0`` is solved by bisection** on the exact numerator identity
    (†) of the module docstring,

        N(ω) ≡ (1 − f) |ε_m|² + f ε_d2 Re ε_m,      Re ε_n = ε_d2 N / |D|²,

    which is continuous *through* the harmonic pole (where ``|D|²`` is merely
    small) and therefore free of the spurious sign flips a direct scan of
    ``Re ε_n`` would report. In the lossless limit ``N`` factorises as
    ``Re ε_m [(1−f) Re ε_m + f ε_d2]``, whose roots ``Re ε_m = 0`` and
    ``Re ε_m = −f ε_d2/(1−f)`` are the analytic seeds; finite damping shifts
    them slightly, hence the numerical refinement.

    **The pole** ``D = f ε_d2 + (1−f) ε_m = 0`` — the multilayer's loss
    resonance, where ``Im ε_n`` peaks — is reported at its lossless location
    ``Re ε_m = −f ε_d2/(1−f)``, again by the closed form above.

    Args:
        fill_fraction: Metal filling fraction in (0, 1).
        eps_dielectric_layer: Permittivity of the dielectric layers (real, > 0).
        omega_range: ``(ω_min, ω_max)`` search window (rad/s). Defaults to
            2000 nm – 200 nm.
        n_scan: Grid points used to bracket the ``Re ε_n`` roots.
        **drude_kwargs: ``eps_inf``, ``omega_p``, ``gamma`` (silver by default).

    Returns:
        Dict of sorted 1-D arrays, each restricted to ``omega_range``:

        * ``'eps_t_zeros'`` — ω where ``Re ε_t = 0`` (analytic);
        * ``'eps_n_zeros'`` — ω where ``Re ε_n = 0`` (bisection);
        * ``'eps_n_pole'`` — ω of the harmonic resonance (lossless location).
    """
    f = float(fill_fraction)
    if not 0.0 < f < 1.0:
        raise ValueError("fill_fraction must lie strictly inside (0, 1)")
    eps_d2 = float(eps_dielectric_layer)
    if omega_range is None:
        omega_range = (
            float(omega_from_wavelength(2000e-9)),
            float(omega_from_wavelength(200e-9)),
        )
    w_lo, w_hi = float(omega_range[0]), float(omega_range[1])
    if not 0.0 < w_lo < w_hi:
        raise ValueError("omega_range must be an increasing pair of positive frequencies")

    params = {"eps_inf": EPS_INF_AG, "omega_p": OMEGA_P_AG, "gamma": GAMMA_AG}
    params.update(drude_kwargs)
    eps_inf, omega_p, gamma = params["eps_inf"], params["omega_p"], params["gamma"]

    def omega_of_re_eps(target: float) -> Optional[float]:
        """Drude ω with ``Re ε_m = target`` (closed form), or None if none exists."""
        if eps_inf - target <= 0.0:
            return None
        w_sq = omega_p**2 / (eps_inf - target) - gamma**2
        return math.sqrt(w_sq) if w_sq > 0.0 else None

    def in_window(w: Optional[float]) -> bool:
        return w is not None and w_lo <= w <= w_hi

    # Re ε_t = 0 — analytic.
    w_t = omega_of_re_eps(-(1.0 - f) * eps_d2 / f)
    eps_t_zeros = np.array([w_t] if in_window(w_t) else [], dtype=float)

    # Re ε_n = 0 — bisection on the continuous numerator N(ω).
    def numerator(omega: np.ndarray) -> np.ndarray:
        eps_m = drude_permittivity(omega, eps_inf, omega_p, gamma)
        return (1.0 - f) * np.abs(eps_m) ** 2 + f * eps_d2 * np.real(eps_m)

    eps_n_zeros = _scan_roots(numerator, np.linspace(w_lo, w_hi, int(n_scan)))

    # Harmonic pole (lossless location) — analytic.
    w_pole = omega_of_re_eps(-f * eps_d2 / (1.0 - f))
    eps_n_pole = np.array([w_pole] if in_window(w_pole) else [], dtype=float)

    return {
        "eps_t_zeros": eps_t_zeros,
        "eps_n_zeros": eps_n_zeros,
        "eps_n_pole": eps_n_pole,
    }


# ------------------------------------------------------------- validity helper
def max_layer_period(k_max: ArrayLike, factor: float = 10.0) -> np.ndarray:
    """
    Largest layer period for which the effective-medium description is
    defensible, given the largest wavenumber ``k_max`` the mode contains.

    The homogenisation condition ``a k ≪ 1`` is made concrete as
    ``a ≤ (2π/k_max)/factor`` — the period must be at least ``factor`` times
    smaller than the shortest wavelength in the mode (``factor = 10`` is the
    usual rule of thumb; the neglected terms are ``O((a k)²) ≈ 0.4%`` there).

    Args:
        k_max: Largest relevant wavenumber (1/m), e.g.
            ``max(|k_spp|, Re κ_m, Re κ_d, |n| k₀)`` over the band of interest.
        factor: Safety factor (period-to-wavelength ratio denominator).

    Returns:
        Maximum period (m).
    """
    k = np.asarray(k_max, dtype=float)
    if np.any(k <= 0.0) or factor <= 0.0:
        raise ValueError("k_max and factor must be positive")
    return 2.0 * np.pi / (factor * k)
