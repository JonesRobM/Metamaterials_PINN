r"""
Where Does Effective-Medium Theory Break Down? — the Real Ag/Silica Multilayer
vs its Homogenised Twin

Every metamaterial result in this project so far — ``examples/dispersion_analysis.py``,
``examples/hyperbolic_metamaterial.py``, ``examples/validate_hmm_dispersion.py``
and the PINN experiments they feed — replaces the Ag/silica multilayer by a
single homogeneous uniaxial medium ``(ε_t, ε_n)`` from :mod:`src.effective_medium`.
That substitution is an *approximation* with a stated validity limit
(``a ≪ λ``, made concrete by :func:`src.effective_medium.max_layer_period` as
**33 nm** for the recommended 450–885 nm band). This study tests it directly:
it builds the actual layer stack and solves it exactly with the transfer-matrix
solver in :mod:`src.transfer_matrix`, then measures the error in the one number
the whole project cares about — the SPP wavevector ``k_spp``.

Conclusion (see the printed summary and ``emt_validity_summary.json``)
---------------------------------------------------------------------
**The homogenised results are trustworthy only for much smaller periods than
33 nm, and the reason is not the one the 33 nm rule guards against.**

1. The dominant error of the homogenised SPP is **first order in the period**,
   ``O(a)``, not the ``O((a/λ)²)`` bulk correction that ``a ≪ λ`` controls. It
   is a *surface* effect: truncating the periodic stack at the air interface
   leaves a half-cell whose polarisation the bulk averages do not describe. Two
   independent signatures confirm it — the fitted log–log slope of the error
   against ``a`` is 0.97 for the natural metal-terminated stack, and the error
   has the *same magnitude but the opposite sign* if the stack is terminated on
   a dielectric layer instead.
2. Consequently the crossovers are small. At the reference wavelength of
   633 nm the metal-terminated stack reaches

   ======================  =========  =========  =========
   quantity                1 %        5 %        10 %
   ======================  =========  =========  =========
   ``Re k_spp``            11.2 nm    > 60 nm    > 60 nm
   ``Im k_spp`` (loss)     1.0 nm     5.4 nm     11.0 nm
   ======================  =========  =========  =========

   Loss is by far the worse half: its 1 % point sits at a period of about one
   nanometre, i.e. below anything fabricable, so **no realisable Ag/silica
   multilayer has its propagation length predicted to 1 % by EMT**. At the
   30 nm period the 33 nm rule would wave through, the homogenised ``Im k_spp``
   is ~29 % wrong at the blue end of the band and ~24 % wrong at 633 nm.
3. The ``O(a)`` term can be **removed by construction**: terminating the stack
   with a *half*-thickness metal layer (``f·a/2`` against the air) puts the
   effective boundary at the centre of the first metal layer and cancels the
   surface term almost exactly. The residual error then falls as ``a²`` (fitted
   slope 2.00) and stays below 0.9 % in ``Re k_spp`` and 0.9 % in ``Im k_spp``
   across the *whole* 1–60 nm range: at a 10 nm period it is ~40× smaller in
   ``Re k_spp`` and ~380× smaller in ``Im k_spp`` than the naturally terminated
   stack. Only *that* configuration lives up to the 33 nm rule.
4. The 33 nm figure is therefore best read as what it is — a *bulk* criterion,
   necessary but not sufficient. It is justified for bulk propagation and for a
   symmetrically terminated stack; it is not a licence to quote surface-mode
   numbers to 1 % at 30 nm periods. **Practical guidance for this project:** at
   a 10 nm period the homogenised ``Re k_spp`` used by every previous experiment
   is good to ~1 % but its ``Im k_spp`` (and therefore the propagation length,
   the loss figures and the ``L/λ_spp`` quality factor) is ~10 % optimistic.
5. Two smaller results fall out of the same machinery: the error grows towards
   the **blue** end of the band (``Re k_spp``: 3.3 % at 450 nm falling to 0.4 %
   at 885 nm for a 10 nm period; ``Im k_spp`` stays at 7.5–9.4 % right across
   it), because the mode is most tightly confined there and so resolves the
   layers best; and a finite stack needs only ``N a ≳ 60 nm`` (about one
   penetration depth) to be within 1 % of the semi-infinite answer, and
   ``≳ 160 nm`` to be within 0.1 %, so nothing here is a truncation artefact.

Everything above is reported as numbers, per wavelength and per termination, in
the JSON summary.

Structure
---------
1. **Period sweep** (``emt_period_sweep.png``). At each of a few wavelengths in
   the recommended band, the true ``k_spp`` of an ``N``-period stack of fixed
   total thickness is tracked as the period ``a`` is swept from 1 to 60 nm at
   fixed metal fill fraction, for three terminations (metal / dielectric /
   half-metal). Panels (a, b) show ``Re k_spp/k₀`` and ``Im k_spp`` against the
   period with the EMT prediction as a horizontal line; panels (c, d) show the
   relative errors on log–log axes with the 1 / 5 / 10 % thresholds, the 33 nm
   ``max_layer_period`` limit, and the fitted power-law exponents.

2. **Frequency sweep** (``emt_frequency_sweep.png``). The same errors across
   450–885 nm at four fixed periods, for the metal and half-metal terminations,
   plus the crossover period against wavelength (the blue end of the band is
   the worst case: the mode is most tightly confined there, so it resolves the
   layers best).

3. **Stack-depth convergence** (``emt_convergence.png``). How many periods a
   *finite* stack needs before it behaves like the semi-infinite half-space the
   EMT describes, and confirmation that the residual EMT error is a genuine
   homogenisation error rather than a truncation artefact.

Design point and conventions
----------------------------
Design point identical to ``examples/hyperbolic_metamaterial.py``: Ag (Drude,
ε_∞ = 3.7, ħω_p = 9.1 eV, ħγ = 0.018 eV) / silica (ε_d2 = 2.25), metal fill
fraction ``f = 0.30``, air superstrate. Geometry: air fills ``z > 0``, the
multilayer ``−N a < z < 0``, and a semi-infinite Ag substrate closes the stack
from below so that the mode is strictly bound on both sides (with a silica
substrate the mode is formally leaky, since ``Re k_spp/k₀ < √ε_d2``; the two
choices agree to ~1e-9 once the stack is thick enough, which panel 3(b)
verifies). Sign convention ``exp(-iωt)``: ``Im ε > 0``, ``Im k_spp > 0``,
``Re κ > 0``.

Usage::

    python examples/emt_validity.py [--figures-dir DIR] [--n-periods-points N]
                                    [--n-wavelengths N]

Figures and ``emt_validity_summary.json`` are written to
``figures/emt_validity/``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402

from src.constants import C0  # noqa: E402
from src.effective_medium import (  # noqa: E402
    EPS_INF_AG,
    HBAR_GAMMA_AG_EV,
    HBAR_OMEGA_P_AG_EV,
    drude_permittivity,
    hmm_permittivities,
    max_layer_period,
    omega_from_wavelength,
)
from src.physics.metamaterial import MetamaterialProperties  # noqa: E402
from src.transfer_matrix import find_mode  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIGURES_DIR = REPO_ROOT / "figures" / "emt_validity"

# ------------------------------------------------------------- design point
FILL_FRACTION = 0.30  # metal filling fraction f = a_Ag / period
EPS_D2 = 2.25  # silica layers
EPS_D = 1.0  # air superstrate
LAMBDA_REF = 633e-9  # reference wavelength for the headline numbers

# Recommended band of examples/hyperbolic_metamaterial.py (figures/hyperbolic).
BAND_NM = (450.0, 885.4)
"""Recommended band of ``examples/hyperbolic_metamaterial.py`` (``figures/hyperbolic``)."""

# ---------------------------------------------------------------- numerics
TOTAL_THICKNESS = 800e-9
"""Total multilayer thickness held fixed while the period is swept.

The mode's penetration into the metamaterial is 54–132 nm across the band
(``hmm_summary.json``), so 800 nm is 6–15 e-foldings: the finite stack is
indistinguishable from the semi-infinite half-space the EMT describes to
better than 1e-6, which figure 3 verifies. Holding the *total* thickness fixed
(rather than the period count) keeps that truncation error constant along the
period sweep, so the trend measured is homogenisation error alone.
"""

MIN_PERIODS = 4
PERIOD_RANGE_NM = (1.0, 60.0)
"""Swept period range (nm).

The lower end is deliberately *below* what is fabricable — a 1 nm period is
0.3 nm of silver, for which a bulk permittivity is meaningless (the module
docstring of :mod:`src.effective_medium` asks for ≳ 5 nm). It is swept anyway
because that is the only way to locate the 1 % crossover of ``Im k_spp``
honestly: the answer, that it sits at about a nanometre, *is* the result.
"""
SWEEP_WAVELENGTHS_NM = (450.0, 550.0, 633.0, 750.0, 885.0)
FIXED_PERIODS_NM = (5.0, 10.0, 20.0, 30.0)
TERMINATIONS = ("metal", "half-metal", "dielectric")
DEFAULT_TERMINATION = "metal"
THRESHOLDS = (0.01, 0.05, 0.10)
EXPONENT_FIT_MAX_NM = 12.0  # fit the power law on the asymptotic (small-a) end
MAX_REL_DEVIATION = 0.5
"""A TMM root more than 50 % away from the EMT prediction is not "the same
mode" any more; it is recorded as NaN rather than silently compared."""

# Okabe–Ito palette; roles fixed across the three figures.
C_EMT = "#000000"
C_TERM = {"metal": "#0072B2", "half-metal": "#009E73", "dielectric": "#D55E00"}
C_PERIOD = ("#0072B2", "#009E73", "#E69F00", "#D55E00")
C_LIMIT = "#CC79A7"
C_GUIDE = "#666666"
TERM_LABEL = {
    "metal": "metal-terminated (Ag against air)",
    "half-metal": r"half-metal ($fa/2$ against air)",
    "dielectric": "dielectric-terminated (silica against air)",
}


# ============================================================ pure physics
def multilayer_stack(
    period: float,
    n_periods: int,
    fill_fraction: float = FILL_FRACTION,
    eps_metal: complex = -18.0 + 0.2j,
    eps_d2: float = EPS_D2,
    termination: str = DEFAULT_TERMINATION,
    eps_superstrate: complex = EPS_D,
    eps_substrate: Optional[complex] = None,
) -> Tuple[List[complex], List[float]]:
    r"""
    The real Ag/silica stack as ``(eps_layers, thicknesses)`` for
    :mod:`src.transfer_matrix`.

    Layers are returned **in order of increasing z**, as that module requires:
    ``eps_layers[0]`` is the semi-infinite substrate (silver by default, so the
    mode is strictly bound below), ``eps_layers[-1]`` the superstrate (air), and
    the interior is ``n_periods`` bilayers of metal thickness ``f a`` and
    dielectric thickness ``(1 − f) a``.

    Args:
        period: Layer period ``a`` (m).
        n_periods: Number of bilayers ``N``.
        fill_fraction: Metal filling fraction ``f``.
        eps_metal: Permittivity of the metal layers at this frequency.
        eps_d2: Permittivity of the dielectric layers.
        termination: What faces the superstrate —

            * ``'metal'`` — a full ``f a`` metal layer (the natural stack, and
              the default);
            * ``'dielectric'`` — a full ``(1 − f) a`` dielectric layer, so the
              bilayer order is reversed;
            * ``'half-metal'`` — a metal layer of thickness ``f a / 2``, which
              places the effective boundary at the *centre* of the first metal
              layer and cancels the leading ``O(a)`` surface error (this study's
              main empirical finding).
        eps_superstrate: Permittivity above the stack.
        eps_substrate: Semi-infinite medium below the stack; defaults to
            ``eps_metal``.

    Returns:
        ``(eps_layers, thicknesses)`` with ``len(thicknesses) ==
        len(eps_layers) - 2``.

    Raises:
        ValueError: on an unknown ``termination`` or a non-positive geometry.
    """
    if termination not in TERMINATIONS:
        raise ValueError(f"termination must be one of {TERMINATIONS}, got {termination!r}")
    if period <= 0.0 or n_periods < 1:
        raise ValueError("period must be positive and n_periods at least 1")

    a = float(period)
    d_metal, d_diel = fill_fraction * a, (1.0 - fill_fraction) * a
    metal, diel = complex(eps_metal), complex(eps_d2)

    # Built top-down (from the superstrate), then reversed into increasing z.
    top_down: List[Tuple[complex, float]] = []
    for index in range(int(n_periods)):
        if termination == "dielectric":
            top_down += [(diel, d_diel), (metal, d_metal)]
        elif termination == "half-metal" and index == 0:
            top_down += [(metal, 0.5 * d_metal), (diel, d_diel)]
        else:
            top_down += [(metal, d_metal), (diel, d_diel)]

    ordered = list(reversed(top_down))
    substrate = metal if eps_substrate is None else complex(eps_substrate)
    eps_layers = [substrate] + [eps for eps, _ in ordered] + [complex(eps_superstrate)]
    thicknesses = [d for _, d in ordered]
    return eps_layers, thicknesses


def periods_for(period: float, total_thickness: float = TOTAL_THICKNESS) -> int:
    """Bilayer count closest to a fixed total thickness (at least :data:`MIN_PERIODS`)."""
    return max(MIN_PERIODS, int(round(float(total_thickness) / float(period))))


def emt_material(
    omega: float, fill_fraction: float = FILL_FRACTION, eps_d2: float = EPS_D2
) -> MetamaterialProperties:
    """
    The homogenised half-space at one frequency.

    Constructor order (as in ``examples/hyperbolic_metamaterial.py``): with the
    optical axis along ``z`` the ``eps_parallel`` slot takes the *normal*
    component ``ε_n`` and ``eps_perpendicular`` the in-plane ``ε_t``.
    """
    eps_t, eps_n = hmm_permittivities(omega, fill_fraction, eps_d2)
    return MetamaterialProperties(
        complex(eps_n), complex(eps_t), optical_axis="z", omega=float(omega)
    )


def emt_wavevector(
    omega: float,
    fill_fraction: float = FILL_FRACTION,
    eps_d2: float = EPS_D2,
    eps_d: float = EPS_D,
) -> complex:
    """The EMT prediction for ``k_spp`` — the number every previous study used."""
    return emt_material(omega, fill_fraction, eps_d2).spp_wavevector(eps_dielectric=eps_d)


def recommended_max_period(
    band_nm: Tuple[float, float] = BAND_NM,
    n_points: int = 201,
    fill_fraction: float = FILL_FRACTION,
    eps_d2: float = EPS_D2,
    eps_d: float = EPS_D,
) -> float:
    r"""
    The project's own period limit, recomputed rather than quoted.

    :func:`src.effective_medium.max_layer_period` turns the largest wavenumber
    the mode contains into the largest defensible period, ``a ≤ (2π/k_max)/10``.
    Following ``examples/hyperbolic_metamaterial.py``, ``k_max`` is the maximum
    of ``|k_spp|``, ``Re κ_d`` and ``Re κ_m`` over the recommended band. The
    result reproduces the ``max_layer_period_nm`` written into
    ``figures/hyperbolic/hmm_summary.json``.

    Returns:
        The recommended maximum period (m).
    """
    k_max = 0.0
    for wavelength in np.linspace(band_nm[0] * 1e-9, band_nm[1] * 1e-9, int(n_points)):
        material = emt_material(float(omega_from_wavelength(wavelength)), fill_fraction, eps_d2)
        k, kappa_d, kappa_m = material.decay_constants(eps_dielectric=eps_d)
        k_max = max(k_max, abs(k), kappa_d.real, kappa_m.real)
    return float(max_layer_period(k_max))


MAX_LAYER_PERIOD_NM = recommended_max_period() * 1e9
"""``max_layer_period`` over the recommended band, in nm (≈ 33 nm)."""


def tmm_wavevector(
    omega: float,
    period: float,
    n_periods: int,
    fill_fraction: float = FILL_FRACTION,
    eps_d2: float = EPS_D2,
    termination: str = DEFAULT_TERMINATION,
    guesses: Sequence[complex] = (),
    reference: Optional[complex] = None,
    max_rel_deviation: float = MAX_REL_DEVIATION,
    **stack_kwargs: object,
) -> complex:
    r"""
    The **true** ``k_spp`` of the real multilayer, from the transfer matrix.

    Zeros of ``M₀₀`` are the bound modes (see
    :func:`src.transfer_matrix.mode_dispersion_function`); each guess in turn is
    polished with Muller's method and the first root that stays within
    ``max_rel_deviation`` of ``reference`` is returned. That guard is what makes
    a sweep safe: as the period grows the true mode moves far from the EMT seed,
    and without it the solver would eventually latch onto a *different* branch
    of the multilayer (a Bloch mode of the stack, or the substrate's own SPP)
    and quietly report it as agreement.

    Args:
        omega: Angular frequency (rad/s).
        period, n_periods, fill_fraction, eps_d2, termination: Geometry, passed
            to :func:`multilayer_stack`.
        guesses: Initial ``k_x`` guesses in priority order (typically the
            previous point of a sweep, then the EMT value).
        reference: Value the root is sanity-checked against; defaults to the
            EMT prediction.
        max_rel_deviation: Rejection radius around ``reference``.
        **stack_kwargs: Forwarded to :func:`multilayer_stack`
            (``eps_superstrate``, ``eps_substrate``).

    Returns:
        Complex ``k_spp`` (1/m), or ``nan + nan j`` when no acceptable root was
        found.
    """
    k0 = float(omega) / C0
    eps_metal = complex(drude_permittivity(omega))
    eps_layers, thicknesses = multilayer_stack(
        period, n_periods, fill_fraction, eps_metal, eps_d2, termination, **stack_kwargs
    )
    if reference is None:
        reference = emt_wavevector(omega, fill_fraction, eps_d2)

    candidates = [complex(g) for g in guesses if np.isfinite(g)]
    if not candidates or all(abs(c - reference) > 1e-30 for c in candidates):
        candidates.append(complex(reference))
    for guess in candidates:
        root = find_mode(guess, k0, eps_layers, thicknesses)
        if root is None:
            continue
        if abs(root - reference) <= max_rel_deviation * abs(reference):
            return complex(root)
    return complex("nan") + 1j * float("nan")


# ================================================================== sweeps
def relative_errors(k_tmm: np.ndarray, k_emt: complex) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    **Signed** relative errors ``(ΔRe/|Re k_emt|, ΔIm/|Im k_emt|)`` of the true
    multilayer against its homogenised prediction.

    The sign is kept because it carries the physics: a metal-terminated stack
    and a dielectric-terminated one err by the same amount in *opposite*
    directions, which is the fingerprint of a boundary-displacement term linear
    in the period. Crossovers and log–log plots take the modulus.
    """
    k_tmm = np.asarray(k_tmm, dtype=complex)
    return (
        (k_tmm.real - k_emt.real) / abs(k_emt.real),
        (k_tmm.imag - k_emt.imag) / abs(k_emt.imag),
    )


def sweep_period(
    wavelength: float,
    periods: np.ndarray,
    termination: str = DEFAULT_TERMINATION,
    fill_fraction: float = FILL_FRACTION,
    eps_d2: float = EPS_D2,
    total_thickness: float = TOTAL_THICKNESS,
) -> Dict[str, object]:
    r"""
    Track the true ``k_spp`` as the layer period is swept at fixed ``f``,
    wavelength and total thickness.

    The sweep runs from the **smallest** period upwards and seeds each solve
    with the previous root (continuation), falling back to the EMT value: at
    ``a → 0`` the EMT seed is essentially exact, and continuation then carries
    the branch out to periods where the EMT seed alone would fail.

    Args:
        wavelength: Free-space wavelength (m).
        periods: Increasing array of periods (m).
        termination: See :func:`multilayer_stack`.
        fill_fraction, eps_d2: Design point.
        total_thickness: Held fixed; the period count follows from it.

    Returns:
        Dict with ``periods``, ``n_periods``, ``k_tmm`` (complex),
        ``k_emt`` (complex scalar), ``signed_error_re``/``signed_error_im``
        and their moduli ``rel_error_re``/``rel_error_im``, plus
        ``wavelength``, ``omega``, ``k0`` and ``termination``.
    """
    omega = float(omega_from_wavelength(wavelength))
    k0 = omega / C0
    k_emt = emt_wavevector(omega, fill_fraction, eps_d2)
    periods = np.asarray(periods, dtype=float)

    counts = np.array([periods_for(a, total_thickness) for a in periods], dtype=int)
    roots = np.empty(periods.size, dtype=complex)
    previous: Optional[complex] = None
    for i, (a, n) in enumerate(zip(periods, counts, strict=True)):
        guesses = [] if previous is None else [previous]
        roots[i] = tmm_wavevector(
            omega, float(a), int(n), fill_fraction, eps_d2, termination,
            guesses=guesses, reference=k_emt,
        )
        if np.isfinite(roots[i]):
            previous = complex(roots[i])
    err_re, err_im = relative_errors(roots, k_emt)
    return {
        "wavelength": float(wavelength),
        "omega": omega,
        "k0": k0,
        "termination": termination,
        "periods": periods,
        "n_periods": counts,
        "k_tmm": roots,
        "k_emt": k_emt,
        "signed_error_re": err_re,
        "signed_error_im": err_im,
        "rel_error_re": np.abs(err_re),
        "rel_error_im": np.abs(err_im),
    }


def sweep_frequency(
    wavelengths: np.ndarray,
    period: float,
    termination: str = DEFAULT_TERMINATION,
    fill_fraction: float = FILL_FRACTION,
    eps_d2: float = EPS_D2,
    total_thickness: float = TOTAL_THICKNESS,
) -> Dict[str, object]:
    """
    Track the true ``k_spp`` across the recommended band at a fixed period.

    Continuation runs from the red end (where the mode is loosest and the EMT
    seed best) to the blue end.

    Returns:
        Dict with ``wavelengths``, ``k_tmm``, ``k_emt`` (array),
        ``rel_error_re``, ``rel_error_im``, ``period`` and ``n_periods``.
    """
    wavelengths = np.asarray(wavelengths, dtype=float)
    order = np.argsort(-wavelengths)  # red → blue
    n = periods_for(period, total_thickness)

    roots = np.empty(wavelengths.size, dtype=complex)
    k_emt = np.empty(wavelengths.size, dtype=complex)
    previous: Optional[complex] = None
    for i in order:
        omega = float(omega_from_wavelength(wavelengths[i]))
        k_emt[i] = emt_wavevector(omega, fill_fraction, eps_d2)
        guesses = [] if previous is None else [previous]
        roots[i] = tmm_wavevector(
            omega, float(period), n, fill_fraction, eps_d2, termination,
            guesses=guesses, reference=complex(k_emt[i]),
        )
        if np.isfinite(roots[i]):
            previous = complex(roots[i])

    err_re = (roots.real - k_emt.real) / np.abs(k_emt.real)
    err_im = (roots.imag - k_emt.imag) / np.abs(k_emt.imag)
    return {
        "wavelengths": wavelengths,
        "period": float(period),
        "n_periods": int(n),
        "termination": termination,
        "k_tmm": roots,
        "k_emt": k_emt,
        "signed_error_re": err_re,
        "signed_error_im": err_im,
        "rel_error_re": np.abs(err_re),
        "rel_error_im": np.abs(err_im),
    }


def sweep_n_periods(
    wavelength: float,
    period: float,
    counts: Sequence[int],
    termination: str = DEFAULT_TERMINATION,
    fill_fraction: float = FILL_FRACTION,
    eps_d2: float = EPS_D2,
    eps_substrate: Optional[complex] = None,
) -> Dict[str, object]:
    r"""
    Does a *finite* stack converge to the semi-infinite answer, and how fast?

    The reference is the deepest stack in ``counts`` (not the EMT value): this
    isolates truncation from homogenisation error, so the plateau the curve
    settles on is the genuine EMT error and the descent before it is the finite
    stack still feeling its substrate.

    Args:
        wavelength: Free-space wavelength (m).
        period: Layer period (m).
        counts: Increasing period counts to evaluate.
        termination: See :func:`multilayer_stack`.
        fill_fraction, eps_d2: Design point.
        eps_substrate: Semi-infinite medium below the stack (default: silver).

    Returns:
        Dict with ``counts``, ``thickness``, ``k_tmm``, ``k_emt``,
        ``k_reference``, ``rel_error_vs_reference``, ``rel_error_vs_emt``.
    """
    omega = float(omega_from_wavelength(wavelength))
    k_emt = emt_wavevector(omega, fill_fraction, eps_d2)
    counts = np.asarray(list(counts), dtype=int)

    roots = np.empty(counts.size, dtype=complex)
    previous: Optional[complex] = None
    for i, n in enumerate(counts):
        guesses = [] if previous is None else [previous]
        roots[i] = tmm_wavevector(
            omega, float(period), int(n), fill_fraction, eps_d2, termination,
            guesses=guesses, reference=k_emt, eps_substrate=eps_substrate,
        )
        if np.isfinite(roots[i]):
            previous = complex(roots[i])

    reference = roots[-1]
    return {
        "wavelength": float(wavelength),
        "period": float(period),
        "termination": termination,
        "counts": counts,
        "thickness": counts * float(period),
        "k_tmm": roots,
        "k_emt": k_emt,
        "k_reference": complex(reference),
        "rel_error_vs_reference": np.abs(roots - reference) / abs(reference),
        "rel_error_vs_emt": np.abs(roots - k_emt) / abs(k_emt),
    }


# ------------------------------------------------------- crossovers & slopes
def crossover_period(
    periods: np.ndarray, errors: np.ndarray, threshold: float
) -> Optional[float]:
    r"""
    The period at which the relative error first reaches ``threshold``.

    Found on the *first* upward crossing and refined by linear interpolation in
    ``(log a, log error)`` — the error is a power law in ``a``, so log–log
    interpolation is exact to the order the data supports.

    Returns ``None`` when the crossing lies **outside** the swept range, in
    either direction: the honest statement is then "< a_min" or "> a_max", not
    an extrapolated number, and the caller distinguishes the two from the errors
    at the ends of the range (:func:`crossover_table` records them).

    Args:
        periods: Increasing periods (m).
        errors: Relative errors (modulus taken), same length; NaNs are skipped.
        threshold: e.g. ``0.01`` for 1 %.

    Returns:
        The crossover period (m), or ``None``.
    """
    a = np.asarray(periods, dtype=float)
    e = np.abs(np.asarray(errors, dtype=float))
    valid = np.isfinite(a) & np.isfinite(e) & (a > 0.0)
    a, e = a[valid], e[valid]
    if a.size < 2 or e[0] >= threshold:
        return None
    for i in range(1, a.size):
        if e[i] >= threshold > e[i - 1]:
            if e[i - 1] <= 0.0 or e[i] == e[i - 1]:
                return float(a[i])
            frac = (math.log(threshold) - math.log(e[i - 1])) / (
                math.log(e[i]) - math.log(e[i - 1])
            )
            return float(math.exp(math.log(a[i - 1]) + frac * (math.log(a[i]) - math.log(a[i - 1]))))
    return None


def error_exponent(
    periods: np.ndarray, errors: np.ndarray, max_period: float = EXPONENT_FIT_MAX_NM * 1e-9
) -> Optional[float]:
    r"""
    Log–log slope ``p`` of ``error ∝ a^p`` on the small-period end.

    ``p ≈ 1`` means the error is a *surface* term (a truncated half-cell);
    ``p ≈ 2`` means it is the genuine ``O((a k)²)`` bulk homogenisation
    correction. Distinguishing the two is the whole point of this study, so the
    fit is restricted to ``a ≤ max_period`` where the asymptotics hold.

    Returns:
        The exponent, or ``None`` with fewer than three usable points.
    """
    a = np.asarray(periods, dtype=float)
    e = np.asarray(errors, dtype=float)
    e = np.abs(e)
    mask = np.isfinite(a) & np.isfinite(e) & (a > 0.0) & (e > 0.0) & (a <= max_period)
    if mask.sum() < 3:
        return None
    slope = np.polyfit(np.log(a[mask]), np.log(e[mask]), 1)[0]
    return float(slope)


def crossover_table(sweep: Dict[str, object]) -> Dict[str, Dict[str, Optional[float]]]:
    r"""
    Crossover periods (nm) at :data:`THRESHOLDS` for ``Re`` and ``Im``.

    Each sub-table also carries the fitted power-law ``exponent`` and the errors
    at the two ends of the swept range, so a ``None`` crossover can be read
    correctly: ``error_at_min_period ≥ threshold`` means the crossing is *below*
    the range, otherwise it is above it.
    """
    periods = np.asarray(sweep["periods"], dtype=float)
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for part in ("re", "im"):
        errors = np.abs(np.asarray(sweep[f"rel_error_{part}"], dtype=float))
        entry: Dict[str, Optional[float]] = {}
        for threshold in THRESHOLDS:
            a_cross = crossover_period(periods, errors, threshold)
            entry[f"{threshold * 100:g}pct_nm"] = None if a_cross is None else a_cross * 1e9
        entry["exponent"] = error_exponent(periods, errors)
        entry["min_period_nm"] = float(periods[0] * 1e9)
        entry["max_period_nm"] = float(periods[-1] * 1e9)
        finite = errors[np.isfinite(errors)]
        entry["error_at_min_period"] = float(errors[0]) if np.isfinite(errors[0]) else None
        entry["error_at_max_period"] = float(finite[-1]) if finite.size else None
        out[part] = entry
    return out


# ================================================================= plotting
def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.6,
            "lines.linewidth": 2.0,
            "legend.frameon": False,
        }
    )


def _threshold_lines(ax, label_x: float) -> None:
    """Draw and label the 1 / 5 / 10 % error thresholds on a log-y error axis."""
    for threshold in THRESHOLDS:
        ax.axhline(threshold, color=C_GUIDE, linestyle=":", linewidth=1.1)
        ax.annotate(
            f"{threshold * 100:g}%",
            xy=(label_x, threshold),
            xytext=(2, 2),
            textcoords="offset points",
            fontsize=8,
            color=C_GUIDE,
            va="bottom",
        )


def _limit_line(ax, label: bool = True) -> None:
    """Mark the ``max_layer_period`` recommendation."""
    ax.axvline(MAX_LAYER_PERIOD_NM, color=C_LIMIT, linestyle="--", linewidth=1.6)
    if label:
        ax.annotate(
            f"max_layer_period = {MAX_LAYER_PERIOD_NM:.0f} nm",
            xy=(MAX_LAYER_PERIOD_NM, 0.015),
            xycoords=("data", "axes fraction"),
            xytext=(-6, 0),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=9,
            color=C_LIMIT,
        )


def plot_period_sweep(
    sweeps: Dict[str, Dict[str, object]],
    path: Path,
    wavelength_nm: float,
    fill_fraction: float = FILL_FRACTION,
) -> None:
    """
    Figure 1: the headline. ``k_spp`` and its EMT error against the layer period
    at one wavelength, for the three terminations.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.8))
    ax_re, ax_im, ax_err_re, ax_err_im = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    any_sweep = next(iter(sweeps.values()))
    k0 = float(any_sweep["k0"])
    k_emt = complex(any_sweep["k_emt"])
    a_nm_all = np.asarray(any_sweep["periods"], dtype=float) * 1e9

    ax_re.axhline(k_emt.real / k0, color=C_EMT, linestyle="-", linewidth=1.6,
                  label=r"EMT (homogenised half-space)")
    ax_im.axhline(k_emt.imag * 1e-6, color=C_EMT, linestyle="-", linewidth=1.6,
                  label="EMT")

    for name, sweep in sweeps.items():
        a_nm = np.asarray(sweep["periods"], dtype=float) * 1e9
        k = np.asarray(sweep["k_tmm"], dtype=complex)
        colour = C_TERM[name]
        ax_re.plot(a_nm, k.real / k0, color=colour, marker="o", markersize=3.0,
                   label=TERM_LABEL[name])
        ax_im.plot(a_nm, k.imag * 1e-6, color=colour, marker="o", markersize=3.0,
                   label=TERM_LABEL[name])
        for ax, part in ((ax_err_re, "re"), (ax_err_im, "im")):
            errors = np.asarray(sweep[f"rel_error_{part}"], dtype=float)
            exponent = error_exponent(a_nm * 1e-9, errors)
            suffix = "" if exponent is None else rf"  ($p = {exponent:.2f}$)"
            ax.loglog(a_nm, errors, color=colour, marker="o", markersize=3.0,
                      label=TERM_LABEL[name] + suffix)

    ax_re.set_ylabel(r"Re $k_{\mathrm{spp}} / k_0$")
    ax_re.set_title("(a) effective index vs layer period")
    ax_im.set_ylabel(r"Im $k_{\mathrm{spp}}$ (rad/µm)")
    ax_im.set_title("(b) loss vs layer period")
    for ax in (ax_re, ax_im):
        ax.set_xlabel(r"layer period $a$ (nm)")
        ax.set_xlim(0.0, a_nm_all.max() * 1.02)
        _limit_line(ax, label=(ax is ax_re))
        ax.legend(loc="best", fontsize=8.5)

    for ax, name in (
        (ax_err_re, r"Re $k_{\mathrm{spp}}$"),
        (ax_err_im, r"Im $k_{\mathrm{spp}}$"),
    ):
        _threshold_lines(ax, a_nm_all.min())
        _limit_line(ax, label=False)
        ax.set_xlabel(r"layer period $a$ (nm)")
        ax.set_ylabel(f"relative EMT error in {name}")
        ax.set_xlim(a_nm_all.min() * 0.9, a_nm_all.max() * 1.1)
        ax.legend(loc="lower right", fontsize=8.5)
    ax_err_re.set_title(
        r"(c) EMT error in Re $k_{\mathrm{spp}}$ — slope $p$: 1 = surface term, 2 = bulk",
        pad=32,
    )
    ax_err_im.set_title(r"(d) EMT error in Im $k_{\mathrm{spp}}$ (loss is the worse half)")

    # Twin axis in a/λ on one panel: the quantity the a ≪ λ criterion talks about.
    lam_nm = wavelength_nm
    secondary = ax_err_re.secondary_xaxis(
        "top", functions=(lambda a: a / lam_nm, lambda x: x * lam_nm)
    )
    secondary.set_xlabel(r"$a / \lambda_0$", labelpad=2)

    fig.suptitle(
        rf"Effective-medium theory vs the real Ag/silica multilayer at "
        rf"$\lambda_0 = {wavelength_nm:.0f}$ nm ($f = {fill_fraction:g}$, "
        rf"{TOTAL_THICKNESS * 1e9:.0f} nm total thickness)",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)


def plot_frequency_sweep(
    frequency_sweeps: Dict[str, List[Dict[str, object]]],
    crossovers_by_wavelength: Dict[str, List[Tuple[float, Dict[str, Optional[float]]]]],
    path: Path,
) -> None:
    """
    Figure 2: the same errors across the recommended band, and the crossover
    period as a function of wavelength.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.6))
    ax_re, ax_im, ax_half, ax_cross = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    def draw(ax, rows: Sequence[Dict[str, object]], part: str, legend_loc: str) -> None:
        for row, colour in zip(rows, C_PERIOD, strict=False):
            lam_nm = np.asarray(row["wavelengths"], dtype=float) * 1e9
            ax.semilogy(lam_nm, np.asarray(row[f"rel_error_{part}"], dtype=float),
                        color=colour, label=rf"$a = {row['period'] * 1e9:.0f}$ nm")
        _threshold_lines(ax, BAND_NM[0])
        ax.set_xlim(*BAND_NM)
        ax.set_xlabel(r"free-space wavelength $\lambda_0$ (nm)")
        ax.legend(loc=legend_loc, fontsize=9, ncol=2, frameon=True, framealpha=0.92,
                  edgecolor="none")

    draw(ax_re, frequency_sweeps["metal"], "re", "upper right")
    draw(ax_im, frequency_sweeps["metal"], "im", "lower right")
    draw(ax_half, frequency_sweeps["half-metal"], "re", "upper right")
    ax_re.set_ylabel(r"relative error in Re $k_{\mathrm{spp}}$")
    ax_im.set_ylabel(r"relative error in Im $k_{\mathrm{spp}}$")
    ax_half.set_ylabel(r"relative error in Re $k_{\mathrm{spp}}$")
    ax_re.set_title(r"(a) Re $k_{\mathrm{spp}}$, metal-terminated")
    ax_im.set_title(r"(b) Im $k_{\mathrm{spp}}$, metal-terminated (the loss the project quotes)")
    ax_half.set_title(r"(c) Re $k_{\mathrm{spp}}$, half-metal termination: the $O(a)$ term gone")

    styles = {"1pct_nm": ("o", "-"), "5pct_nm": ("s", "--"), "10pct_nm": ("^", ":")}
    off_scale: List[str] = []
    for name, rows in crossovers_by_wavelength.items():
        for key, (marker, dash) in styles.items():
            lam = [lam_nm for lam_nm, table in rows if table["re"][key] is not None]
            val = [table["re"][key] for _, table in rows if table["re"][key] is not None]
            if not lam:
                off_scale.append(f"{name} {key.replace('pct_nm', '')}%")
                continue
            ax_cross.plot(lam, val, marker=marker, linestyle=dash, color=C_TERM[name],
                          linewidth=1.6, markersize=5.5,
                          label=f"{name}, {key.replace('pct_nm', '')}%")
    if off_scale:
        ax_cross.annotate(
            "off scale (never reached for $a \\leq$ "
            f"{PERIOD_RANGE_NM[1]:.0f} nm):\n" + ", ".join(off_scale),
            xy=(0.03, 0.95),
            xycoords="axes fraction",
            va="top",
            fontsize=8.5,
            color=C_TERM["half-metal"],
        )
    ax_cross.axhline(MAX_LAYER_PERIOD_NM, color=C_LIMIT, linestyle="--", linewidth=1.6)
    ax_cross.annotate(
        f"max_layer_period = {MAX_LAYER_PERIOD_NM:.0f} nm",
        xy=(0.98, MAX_LAYER_PERIOD_NM),
        xycoords=("axes fraction", "data"),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=C_LIMIT,
    )
    ax_cross.set_yscale("log")
    ax_cross.set_ylim(1.0, 300.0)
    ax_cross.set_yticks([1, 2, 5, 10, 20, MAX_LAYER_PERIOD_NM, 60, 100, 200])
    ax_cross.yaxis.set_major_formatter(ScalarFormatter())
    ax_cross.yaxis.set_minor_formatter(NullFormatter())
    ax_cross.set_xlim(*BAND_NM)
    ax_cross.set_xlabel(r"free-space wavelength $\lambda_0$ (nm)")
    ax_cross.set_ylabel(r"period at which Re $k_{\mathrm{spp}}$ error is reached (nm)")
    ax_cross.set_title("(d) the period budget, wavelength by wavelength")
    ax_cross.legend(loc="lower right", fontsize=8.5, ncol=2, frameon=True, framealpha=0.92,
                    edgecolor="none")

    fig.suptitle(
        "EMT error across the recommended 450–885 nm band: worst at the blue end, "
        "where the mode is most tightly confined",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)


def plot_convergence(
    convergence: List[Dict[str, object]],
    substrate_check: Dict[str, object],
    path: Path,
) -> None:
    """Figure 3: how deep a finite stack must be to behave like the EMT half-space."""
    fig, (ax_err, ax_index) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    for row, colour in zip(convergence, C_PERIOD, strict=False):
        counts = np.asarray(row["counts"], dtype=int)
        label = rf"$a = {row['period'] * 1e9:.0f}$ nm"
        # The last point *is* the reference, so its error is an exact zero: drop it.
        ax_err.semilogy(counts[:-1],
                        np.asarray(row["rel_error_vs_reference"], dtype=float)[:-1],
                        color=colour, marker="o", markersize=3.0, label=label)
        ax_err.axhline(
            float(np.asarray(row["rel_error_vs_emt"], dtype=float)[-1]),
            color=colour, linestyle=":", linewidth=1.4,
        )
        k0 = float(omega_from_wavelength(row["wavelength"])) / C0
        ax_index.plot(
            np.asarray(row["thickness"], dtype=float) * 1e9,
            np.asarray(row["k_tmm"], dtype=complex).real / k0,
            color=colour, marker="o", markersize=3.0, label=label,
        )

    ax_err.set_ylim(1e-16, 1.0)
    ax_err.set_xlabel("number of periods $N$")
    ax_err.set_ylabel(r"$|k_N - k_\infty| / |k_\infty|$")
    ax_err.set_title(
        "(a) truncation error vs stack depth\n"
        "(dotted: the residual EMT error each period settles on)"
    )
    ax_err.legend(loc="upper right", fontsize=9)

    k0_ref = float(omega_from_wavelength(convergence[0]["wavelength"])) / C0
    k_emt = complex(convergence[0]["k_emt"])
    ax_index.axhline(k_emt.real / k0_ref, color=C_EMT, linewidth=1.6, label="EMT half-space")
    sub_thick = np.asarray(substrate_check["thickness"], dtype=float) * 1e9
    ax_index.plot(
        sub_thick,
        np.asarray(substrate_check["k_tmm"], dtype=complex).real / k0_ref,
        color=C_GUIDE, linestyle="--", marker="x", markersize=4.0, linewidth=1.4,
        label=rf"$a = {substrate_check['period'] * 1e9:.0f}$ nm, silica substrate",
    )
    ax_index.set_xlim(0.0, 650.0)
    ax_index.set_xlabel("total multilayer thickness $Na$ (nm)")
    ax_index.set_ylabel(r"Re $k_{\mathrm{spp}} / k_0$")
    ax_index.set_title("(b) the finite stack converging to the semi-infinite answer")
    ax_index.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        rf"Finite-stack convergence at $\lambda_0 = "
        rf"{convergence[0]['wavelength'] * 1e9:.0f}$ nm: the residual gap to EMT is "
        "homogenisation error, not truncation",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path)
    plt.close(fig)


# ================================================================== summary
def _complex_record(value: complex) -> Dict[str, float]:
    return {"re": float(np.real(value)), "im": float(np.imag(value))}


def _finite_list(values: np.ndarray) -> List[Optional[float]]:
    """JSON-safe list: NaN/Inf become ``None`` (strict JSON has no such tokens)."""
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(values, dtype=float)]


def build_summary(
    period_sweeps: Dict[float, Dict[str, Dict[str, object]]],
    frequency_sweeps: Dict[str, List[Dict[str, object]]],
    convergence: List[Dict[str, object]],
    substrate_check: Dict[str, object],
    fill_fraction: float = FILL_FRACTION,
    eps_d2: float = EPS_D2,
) -> Dict[str, object]:
    """Key numbers of the study, JSON-serialisable (no NaN/Infinity tokens)."""
    omega_ref = float(omega_from_wavelength(LAMBDA_REF))
    eps_t, eps_n = hmm_permittivities(omega_ref, fill_fraction, eps_d2)

    period_records: List[Dict[str, object]] = []
    for lam_nm in sorted(period_sweeps):
        for name, sweep in period_sweeps[lam_nm].items():
            period_records.append(
                {
                    "wavelength_nm": float(lam_nm),
                    "termination": name,
                    "periods_nm": [float(a * 1e9) for a in sweep["periods"]],
                    "n_periods": [int(n) for n in sweep["n_periods"]],
                    "k_emt": _complex_record(sweep["k_emt"]),
                    "k_tmm_re": _finite_list(np.asarray(sweep["k_tmm"]).real),
                    "k_tmm_im": _finite_list(np.asarray(sweep["k_tmm"]).imag),
                    "signed_error_re": _finite_list(sweep["signed_error_re"]),
                    "signed_error_im": _finite_list(sweep["signed_error_im"]),
                    "rel_error_re": _finite_list(sweep["rel_error_re"]),
                    "rel_error_im": _finite_list(sweep["rel_error_im"]),
                    "crossovers": crossover_table(sweep),
                }
            )

    frequency_records = [
        {
            "termination": name,
            "period_nm": float(row["period"] * 1e9),
            "n_periods": int(row["n_periods"]),
            "wavelength_nm": [float(w * 1e9) for w in row["wavelengths"]],
            "signed_error_re": _finite_list(row["signed_error_re"]),
            "signed_error_im": _finite_list(row["signed_error_im"]),
            "rel_error_re": _finite_list(row["rel_error_re"]),
            "rel_error_im": _finite_list(row["rel_error_im"]),
            "worst_rel_error_re": float(np.nanmax(row["rel_error_re"])),
            "worst_rel_error_im": float(np.nanmax(row["rel_error_im"])),
        }
        for name, rows in frequency_sweeps.items()
        for row in rows
    ]

    def periods_to_converge(row: Dict[str, object], tolerance: float) -> Optional[int]:
        counts = np.asarray(row["counts"], dtype=int)
        errors = np.asarray(row["rel_error_vs_reference"], dtype=float)
        below = np.flatnonzero(np.isfinite(errors) & (errors <= tolerance))
        return None if below.size == 0 else int(counts[below[0]])

    convergence_records = [
        {
            "wavelength_nm": float(row["wavelength"] * 1e9),
            "period_nm": float(row["period"] * 1e9),
            "counts": [int(n) for n in row["counts"]],
            "thickness_nm": [float(t * 1e9) for t in row["thickness"]],
            "rel_error_vs_reference": _finite_list(row["rel_error_vs_reference"]),
            "rel_error_vs_emt": _finite_list(row["rel_error_vs_emt"]),
            "periods_for_1pct": periods_to_converge(row, 0.01),
            "periods_for_0p1pct": periods_to_converge(row, 1e-3),
            "residual_emt_error": float(np.asarray(row["rel_error_vs_emt"])[-1]),
        }
        for row in convergence
    ]

    reference = period_sweeps[float(LAMBDA_REF * 1e9)]
    headline = {
        name: crossover_table(sweep) for name, sweep in reference.items()
    }

    return {
        "design": {
            "fill_fraction": float(fill_fraction),
            "eps_dielectric_layer": float(eps_d2),
            "eps_superstrate": float(EPS_D),
            "substrate": "silver (semi-infinite)",
            "optical_axis": "z",
            "drude_model": {
                "eps_inf": EPS_INF_AG,
                "hbar_omega_p_eV": HBAR_OMEGA_P_AG_EV,
                "hbar_gamma_eV": HBAR_GAMMA_AG_EV,
            },
            "total_thickness_nm": TOTAL_THICKNESS * 1e9,
            "reference_wavelength_nm": LAMBDA_REF * 1e9,
            "band_nm": list(BAND_NM),
            "max_layer_period_nm": MAX_LAYER_PERIOD_NM,
        },
        "reference_point": {
            "wavelength_nm": LAMBDA_REF * 1e9,
            "omega": omega_ref,
            "eps_metal": _complex_record(complex(drude_permittivity(omega_ref))),
            "eps_t": _complex_record(complex(eps_t)),
            "eps_n": _complex_record(complex(eps_n)),
            "k_emt": _complex_record(emt_wavevector(omega_ref, fill_fraction, eps_d2)),
        },
        "headline_crossovers": headline,
        "period_sweeps": period_records,
        "frequency_sweeps": frequency_records,
        "convergence": convergence_records,
        "substrate_check": {
            "period_nm": float(substrate_check["period"] * 1e9),
            "substrate": "silica (semi-infinite)",
            "counts": [int(n) for n in substrate_check["counts"]],
            "k_tmm_re": _finite_list(np.asarray(substrate_check["k_tmm"]).real),
            "max_rel_difference_vs_silver": None,
        },
        "thresholds": list(THRESHOLDS),
    }


def _fmt(entry: Dict[str, Optional[float]], threshold: float) -> str:
    """Format one crossover, saying which side of the swept range it fell on."""
    value = entry[f"{threshold * 100:g}pct_nm"]
    if value is not None:
        return f"{value:.1f}"
    at_min = entry.get("error_at_min_period")
    if at_min is not None and at_min >= threshold:
        return f"< {entry['min_period_nm']:.0f}"
    return f"> {entry['max_period_nm']:.0f}"


def _print_conclusion(summary: Dict[str, object]) -> None:
    """Print the deliverable: for which periods the homogenised results hold."""
    headline = summary["headline_crossovers"]  # type: ignore[index]
    design = summary["design"]  # type: ignore[index]
    print("\n" + "=" * 88)
    print("WHERE DOES EFFECTIVE-MEDIUM THEORY BREAK DOWN?")
    print("=" * 88)
    print(
        "Structure   : Ag/silica multilayer, f = {:.2f}, eps_d2 = {:.2f}, air superstrate, "
        "{:.0f} nm thick".format(
            design["fill_fraction"], design["eps_dielectric_layer"],
            design["total_thickness_nm"],
        )
    )
    print(
        "Reference   : lambda0 = {:.0f} nm; EMT k_spp/k0 = {:.5f} + {:.5f}i".format(
            design["reference_wavelength_nm"],
            summary["reference_point"]["k_emt"]["re"]  # type: ignore[index]
            / (2.0 * math.pi / LAMBDA_REF),
            summary["reference_point"]["k_emt"]["im"]  # type: ignore[index]
            / (2.0 * math.pi / LAMBDA_REF),
        )
    )
    print(
        "\nPeriod (nm) at which the EMT error in k_spp reaches a given level, at "
        f"{design['reference_wavelength_nm']:.0f} nm:"
    )
    print(
        "  {:<34s} {:>9s} {:>9s} {:>9s}   {:>8s}".format(
            "termination / quantity", "1%", "5%", "10%", "slope p"
        )
    )
    for name in TERMINATIONS:
        table = headline.get(name)
        if table is None:
            continue
        for part, label in (("re", "Re k_spp"), ("im", "Im k_spp (loss)")):
            entry = table[part]
            exponent = entry["exponent"]
            print(
                "  {:<34s} {:>9s} {:>9s} {:>9s}   {:>8s}".format(
                    f"{name}: {label}",
                    *(_fmt(entry, t) for t in THRESHOLDS),
                    "n/a" if exponent is None else f"{exponent:.2f}",
                )
            )
    print(
        f"\nThe project's stated limit is max_layer_period = "
        f"{design['max_layer_period_nm']:.0f} nm (a <= lambda/10 for the mode's largest k)."
    )
    metal = headline.get(DEFAULT_TERMINATION, {})
    half = headline.get("half-metal", {})
    print(
        "VERDICT\n"
        "  The naturally terminated stack errs at FIRST order in the period (slope p ~ 1): the\n"
        "  truncated half-cell at the air interface is a surface term the bulk averages do not\n"
        "  contain. Metal- and dielectric-terminated stacks err by the same amount with OPPOSITE\n"
        "  sign, which is the fingerprint of that term. Hence, at {:.0f} nm:\n"
        "      Re k_spp reaches 1% at a = {} nm,  Im k_spp (the loss) at a = {} nm.\n"
        "  So the 33 nm rule is a BULK criterion: necessary, but it does NOT certify the surface\n"
        "  mode, and every homogenised loss figure this project quotes for a 20-30 nm period is\n"
        "  tens of per cent out.\n"
        "  Cutting the first metal layer to HALF thickness puts the effective boundary at the\n"
        "  centre of that layer and cancels the O(a) term (slope p ~ {}), pushing the 1% point\n"
        "  in Re k_spp out to a = {} nm and in Im k_spp to a = {} nm. That is the only\n"
        "  configuration for which the 33 nm limit is defensible.".format(
            design["reference_wavelength_nm"],
            _fmt(metal["re"], 0.01) if metal else "?",
            _fmt(metal["im"], 0.01) if metal else "?",
            "n/a"
            if not half or half["re"]["exponent"] is None
            else f"{half['re']['exponent']:.1f}",
            _fmt(half["re"], 0.01) if half else "?",
            _fmt(half["im"], 0.01) if half else "?",
        )
    )
    print("=" * 88 + "\n")


# ===================================================================== main
def main(argv: Optional[List[str]] = None) -> Dict[str, object]:
    parser = argparse.ArgumentParser(
        description="Effective-medium validity study (analytics only): the true SPP of the "
        "Ag/silica multilayer, by transfer matrix, against its homogenised prediction."
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help=f"output directory for figures and JSON (default: {DEFAULT_FIGURES_DIR})",
    )
    parser.add_argument(
        "--n-periods-points",
        type=int,
        default=22,
        help="points along the layer-period axis (default: 22)",
    )
    parser.add_argument(
        "--n-wavelengths",
        type=int,
        default=31,
        help="points along the wavelength axis of the frequency sweep (default: 31)",
    )
    parser.add_argument(
        "--fill-fraction",
        type=float,
        default=FILL_FRACTION,
        help=f"metal filling fraction (default: {FILL_FRACTION})",
    )
    args = parser.parse_args(argv)
    figures_dir: Path = args.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    fill_fraction = float(args.fill_fraction)
    _apply_style()

    periods = np.geomspace(
        PERIOD_RANGE_NM[0] * 1e-9, PERIOD_RANGE_NM[1] * 1e-9, max(4, int(args.n_periods_points))
    )

    # 1. Period sweep at several wavelengths and all three terminations.
    period_sweeps: Dict[float, Dict[str, Dict[str, object]]] = {}
    for lam_nm in SWEEP_WAVELENGTHS_NM:
        period_sweeps[float(lam_nm)] = {
            name: sweep_period(lam_nm * 1e-9, periods, name, fill_fraction, EPS_D2)
            for name in TERMINATIONS
        }
    plot_period_sweep(
        period_sweeps[float(LAMBDA_REF * 1e9)],
        figures_dir / "emt_period_sweep.png",
        LAMBDA_REF * 1e9,
        fill_fraction,
    )

    # 2. Frequency sweep across the recommended band at a few fixed periods.
    wavelengths = np.linspace(BAND_NM[0] * 1e-9, BAND_NM[1] * 1e-9, max(4, int(args.n_wavelengths)))
    frequency_sweeps = {
        name: [
            sweep_frequency(wavelengths, a * 1e-9, name, fill_fraction, EPS_D2)
            for a in FIXED_PERIODS_NM
        ]
        for name in ("metal", "half-metal")
    }
    crossovers_by_wavelength = {
        name: [
            (float(lam_nm), crossover_table(period_sweeps[float(lam_nm)][name]))
            for lam_nm in SWEEP_WAVELENGTHS_NM
        ]
        for name in TERMINATIONS
    }
    plot_frequency_sweep(
        frequency_sweeps, crossovers_by_wavelength, figures_dir / "emt_frequency_sweep.png"
    )

    # 3. Stack-depth convergence, plus a substrate cross-check.
    counts = np.unique(np.round(np.geomspace(1, 120, 20)).astype(int))
    convergence = [
        sweep_n_periods(LAMBDA_REF, a * 1e-9, counts, DEFAULT_TERMINATION, fill_fraction, EPS_D2)
        for a in (5.0, 10.0, 20.0)
    ]
    substrate_check = sweep_n_periods(
        LAMBDA_REF, 10e-9, counts, DEFAULT_TERMINATION, fill_fraction, EPS_D2,
        eps_substrate=EPS_D2,
    )
    plot_convergence(convergence, substrate_check, figures_dir / "emt_convergence.png")

    summary = build_summary(
        period_sweeps, frequency_sweeps, convergence, substrate_check, fill_fraction, EPS_D2
    )
    silver = np.asarray(convergence[1]["k_tmm"], dtype=complex)
    silica = np.asarray(substrate_check["k_tmm"], dtype=complex)
    with np.errstate(invalid="ignore"):
        difference = np.abs(silver - silica) / np.abs(silver)
    summary["substrate_check"]["max_rel_difference_vs_silver"] = float(  # type: ignore[index]
        np.nanmax(difference[-3:])
    )

    summary_path = figures_dir / "emt_validity_summary.json"
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Figures written to {figures_dir}")
    for record in summary["frequency_sweeps"]:  # type: ignore[index]
        print(
            "  {:<11s} a = {:4.0f} nm:  worst error over 450-885 nm  "
            "Re {:6.2f}%   Im {:6.2f}%".format(
                record["termination"], record["period_nm"],
                100.0 * record["worst_rel_error_re"], 100.0 * record["worst_rel_error_im"],
            )
        )
    for record in summary["convergence"]:  # type: ignore[index]
        print(
            "  a = {:4.0f} nm: {} periods ({:.0f} nm) reach 1% of the semi-infinite answer, "
            "{} reach 0.1%; residual EMT error {:.2%}".format(
                record["period_nm"], record["periods_for_1pct"],
                (record["periods_for_1pct"] or 0) * record["period_nm"],
                record["periods_for_0p1pct"], record["residual_emt_error"],
            )
        )
    print(
        "  substrate cross-check (silver vs silica below the stack): "
        "{:.2e} relative".format(summary["substrate_check"]["max_rel_difference_vs_silver"])  # type: ignore[index]
    )
    _print_conclusion(summary)
    print(f"Summary written to {summary_path}")
    return summary


if __name__ == "__main__":
    main()
