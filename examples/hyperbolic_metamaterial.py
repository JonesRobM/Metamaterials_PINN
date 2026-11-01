r"""
Hyperbolic Metamaterial Design Study — Choosing the Band for a Dispersive SPP PINN

An analytics-only companion to ``examples/dispersion_analysis.py`` (no
training, runtime seconds). Where that study swept *hand-chosen* uniaxial
permittivities, this one builds them from a real structure: an Ag/silica
multilayer stacked along ``z``, homogenised by
:mod:`src.effective_medium` into a uniaxial pair ``(ε_t, ε_n)`` that is
genuinely dispersive, and pushed through the benchmark-validated closed-form
SPP dispersion of :class:`src.physics.metamaterial.MetamaterialProperties`.

**Deliverable.** The headline output is a *recommended frequency band* for the
next PINN experiment (printed and written to ``hmm_summary.json``): a
contiguous ω-interval on which a bound SPP exists throughout, ``k_spp(ω)`` is
appreciably and *non-linearly* dispersive, and all the mode's decay lengths
stay within about one order of magnitude so a single network's collocation
sampling can cover the whole band. This is the ingredient that
``examples/validate_spp_dispersion.py`` idealises away: holding ε fixed across
the band makes ``k_spp(ω) = (ω/c)·n_spp`` an exact straight line through the
origin and the mode family self-similar. With ε(ω) from the multilayer the
curve bends, and the band chosen here quantifies by how much.

Structure
---------
1. **Permittivity spectrum** (``hmm_permittivities.png``). ``Re ε`` and
   ``Im ε`` of both components over ħω = 0.8–5.0 eV (1550–248 nm, so the
   suggested 400–1500 nm window plus the UV where the topological transitions
   actually live), with the type-I / type-II / elliptic regions shaded and the
   ENZ crossings from
   :func:`src.effective_medium.transition_frequencies` annotated. The
   arithmetic mean ``ε_t`` crosses zero once (the type-II onset); the harmonic
   mean ``ε_n`` crosses zero twice, the lower crossing sitting on the
   multilayer's loss resonance where ``Im ε_n`` spikes by three orders of
   magnitude (see the module docstring of :mod:`src.effective_medium`, identity
   (†): ``Im ε_n = f ε_d2² Im ε_m / |D|²`` stays positive but blows up as the
   harmonic denominator ``D`` collapses).

2. **SPP dispersion** (``hmm_spp_dispersion.png``). The ω–k diagram of the
   HMM/air interface, the residual of ``Re k_spp(ω)`` about a straight line
   (the nonlinearity the PINN has to learn), the propagation length, and the
   two decay constants. The bound sub-band and the recommended band are marked
   on every panel.

3. **Fill-fraction sweep** (``hmm_fill_fraction_map.png``). How the picture
   moves with the metal filling fraction ``f``: an ``(f, ħω)`` map coloured by
   anisotropy class with the analytic transition curves overlaid, and the same
   plane coloured by ``Re k_spp/k₀`` inside the bound region, with the
   recommended band boxed.

Design point
------------
Ag (Drude, ε_∞ = 3.7, ħω_p = 9.1 eV, ħγ = 0.018 eV — the same fit as
``examples/dispersion_analysis.py``) / silica (ε_d2 = 2.25) with metal filling
fraction ``f = 0.30``, against an air superstrate (ε_d = 1). ``f = 0.30`` sits
at the low end of the conventional 0.3–0.5 multilayer range, and the band
recommendation is re-run at a spread of fill fractions
(``fill_fraction_scan`` in the JSON, printed by ``main``) to show that the
achievable bandwidth is a *shallow* optimum around ``f ≈ 0.2–0.3`` and falls
away above it: a larger metal fraction drives ``ε_t`` strongly negative, which
pushes the mode back onto the light line. Figure 3 shows the same story as a
map — the type-II window and the bound-SPP region sliding to the blue as ``f``
grows.

Bound-mode criterion
--------------------
As established in ``examples/dispersion_analysis.py``: ``is_spp_supported``
(the unsquared matching condition ``κ_d/ε_d + κ_m/ε_t = 0`` on the ``Re κ > 0``
branch) **plus** the non-radiative gate ``Re k_spp > √ε_d k₀``, because with
loss the matching condition also admits radiative quasi-roots below the light
line. The sweeps evaluate this with a vectorised transcription of
:class:`~src.physics.metamaterial.MetamaterialProperties` (the 2-D map is
40 000 points); ``tests/examples/test_hyperbolic_metamaterial.py`` pins the
transcription to the class itself, which stays the authority.

Homogenisation caveat
---------------------
Everything here treats the multilayer as a homogeneous uniaxial medium. That is
an approximation, and ``max_layer_period`` reports only its *bulk* validity
limit (``a·k ≪ 1``, ~33 nm here). ``examples/emt_validity.py`` measures the
error directly against a transfer-matrix solve of the real layered stack and
finds that for this *surface* mode the leading term is O(a), not O((a/λ)²) —
a termination effect, evidenced by metal- and dielectric-terminated stacks
erring by equal magnitudes with opposite signs. Consequently ``Re k_spp`` is
within ~1 % only for periods ≲ 11 nm, and ``Im k_spp`` — every propagation
length and loss figure quoted here — is ~9 % optimistic already at 10 nm.
Read the band recommendation below as a statement about the homogenised model;
consult that study before treating any of it as fabrication advice.

Sign convention: ``exp(-iωt)``, so ``Im ε > 0``, ``Im k_spp > 0``, ``Re κ > 0``.

Usage::

    python examples/hyperbolic_metamaterial.py [--figures-dir DIR] [--n-points N]

Figures and ``hmm_summary.json`` are written to ``figures/hyperbolic/``.
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
from matplotlib.colors import ListedColormap, Normalize  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from src.constants import C0  # noqa: E402
from src.effective_medium import (  # noqa: E402
    ANISOTROPY_CLASSES,
    EPS_INF_AG,
    HBAR_EVS,
    HBAR_GAMMA_AG_EV,
    HBAR_OMEGA_P_AG_EV,
    classify_anisotropy,
    hmm_permittivities,
    max_layer_period,
    omega_from_photon_energy_ev,
    omega_from_wavelength,
    photon_energy_ev,
    transition_frequencies,
    wavelength_from_omega,
)
from src.physics.metamaterial import MetamaterialProperties  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIGURES_DIR = REPO_ROOT / "figures" / "hyperbolic"

HC_EV_NM = 1239.841984332  # h·c in eV·nm, for the wavelength twin axes

# ------------------------------------------------------------- design point
FILL_FRACTION = 0.30  # metal filling fraction f = a_Ag / period
EPS_D2 = 2.25  # silica layers
EPS_D = 1.0  # air superstrate above the interface
LAMBDA_REF = 633e-9  # reference wavelength: ω₀ = 2πc/λ_ref (repo convention)
OMEGA_REF = 2.0 * math.pi * C0 / LAMBDA_REF

# Figure 1/2 spectral window (eV). Wide enough to contain both ENZ crossings.
EV_MIN, EV_MAX = 0.8, 5.0

# ------------------------------------------------ band-search window & criteria
# The Drude fit for silver has no interband transitions and degrades rapidly
# below ~450 nm (see examples/dispersion_analysis.py), so the search never
# recommends a band whose blue edge relies on it.
LAMBDA_SEARCH = (450e-9, 1500e-9)
MAX_KAPPA_SPREAD = 10.0  # max/min over {Re κ_d, Re κ_m} across the band
MIN_NONLINEARITY_PCT = 15.0  # max |residual| of Re k_spp about a line, % of range
MIN_NEFF_RATIO = 1.15  # max/min of Re k_spp/k₀ across the band
MIN_L_OVER_LAMBDA_SPP = 10.0  # mode must survive ≥ 10 of its own wavelengths
FILL_SCAN = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60)  # f values re-searched

# Okabe–Ito palette; roles fixed across the three figures.
C_T = "#0072B2"  # blue: in-plane ε_t and its quantities
C_N = "#D55E00"  # vermillion: normal ε_n
C_SPP = "#0072B2"  # blue: the SPP branch
C_DIEL = "#009E73"  # green: dielectric-side κ_d
C_METAL = "#D55E00"  # vermillion: metamaterial-side κ_m
C_LIGHT = "#555555"  # grey: light line and guides
C_BAND = "#CC79A7"  # purple: the recommended band and its annotations

CLASS_COLOURS = {
    "type-I": "#56B4E9",
    "type-II": "#E69F00",
    "elliptic-dielectric": "#009E73",
    "elliptic-metallic": "#666666",
}
CLASS_ALPHA = 0.20


# ================================================================ pure physics
def hmm_interface(
    omega: float, fill_fraction: float = FILL_FRACTION, eps_d2: float = EPS_D2
) -> MetamaterialProperties:
    """
    The multilayer as a uniaxial half-space at one frequency.

    Note the constructor order: with the optical axis along ``z`` (the stacking
    direction) ``eps_parallel`` is the component *along the axis*, i.e. the
    normal component ``ε_n``, and ``eps_perpendicular`` is the in-plane ``ε_t``.
    """
    eps_t, eps_n = hmm_permittivities(omega, fill_fraction, eps_d2)
    return MetamaterialProperties(
        complex(eps_n), complex(eps_t), optical_axis="z", omega=float(omega)
    )


def _root_decaying(z2: np.ndarray) -> np.ndarray:
    """Vectorised ``Re > 0`` branch (evanescent decay away from the interface)."""
    r = np.sqrt(np.asarray(z2, dtype=complex))
    flip = (r.real < 0) | ((r.real == 0) & (r.imag < 0))
    return np.where(flip, -r, r)


def _root_propagating(z2: np.ndarray) -> np.ndarray:
    """Vectorised ``Im > 0`` branch (decay along the propagation direction)."""
    r = np.sqrt(np.asarray(z2, dtype=complex))
    flip = (r.imag < 0) | ((r.imag == 0) & (r.real < 0))
    return np.where(flip, -r, r)


def spp_metrics(
    eps_t: np.ndarray,
    eps_n: np.ndarray,
    omega: np.ndarray,
    eps_d: float = EPS_D,
    rel_tol: float = 1e-6,
    bound_tol: float = 1e-3,
) -> Dict[str, np.ndarray]:
    """
    Vectorised TM SPP mode metrics at a uniaxial/dielectric interface.

    A term-for-term transcription of
    :meth:`~src.physics.metamaterial.MetamaterialProperties.spp_wavevector`,
    ``_decay_constants`` and ``is_spp_supported`` into NumPy (the scalar class
    is the authority; the equivalence is asserted in the tests), plus the
    non-radiative gate ``Re k_spp > √ε_d k₀``.

    Args:
        eps_t, eps_n: In-plane and normal permittivities (broadcastable).
        omega: Angular frequency (rad/s), broadcastable with the above.
        eps_d: Superstrate permittivity.
        rel_tol: Tolerance on the unsquared matching condition.
        bound_tol: A decay constant counts as bound only when
            ``Re κ > bound_tol·|κ|``.

    Returns:
        Dict of broadcast arrays: ``k0``, ``n_eff`` (``Re k_spp/k₀``),
        ``k_spp`` (complex), ``kappa_d``, ``kappa_m`` (complex, ``Re > 0``
        branch), ``L`` (propagation length, m), ``bound`` (bool mask). Every
        real quantity is NaN where the mode is not bound.
    """
    eps_t = np.asarray(eps_t, dtype=complex)
    eps_n = np.asarray(eps_n, dtype=complex)
    k0 = np.asarray(omega, dtype=float) / C0
    eps_t, eps_n, k0 = np.broadcast_arrays(eps_t, eps_n, k0)

    denom = eps_t * eps_n - eps_d**2
    with np.errstate(divide="ignore", invalid="ignore"):
        n_sq = np.where(denom == 0, np.nan, eps_d * eps_n * (eps_t - eps_d) / denom)
        n = _root_propagating(n_sq)
        kappa_d = _root_decaying(n_sq - eps_d)
        kappa_m = _root_decaying(eps_t * (n_sq / eps_n - 1.0))
        matching = kappa_d / eps_d + kappa_m / eps_t
        scale = np.abs(kappa_d / eps_d) + np.abs(kappa_m / eps_t)

    bound = (
        (kappa_d.real > bound_tol * np.abs(kappa_d))
        & (kappa_m.real > bound_tol * np.abs(kappa_m))
        & (np.abs(matching) <= rel_tol * scale)
        & (n.real > math.sqrt(eps_d))
    )

    def masked(values: np.ndarray) -> np.ndarray:
        return np.where(bound, values, np.nan)

    k_spp = n * k0
    return {
        "k0": k0,
        "bound": bound,
        "n_eff": masked(n.real),
        "k_spp": np.where(bound, k_spp, np.nan + 0j),
        "kappa_d": np.where(bound, kappa_d * k0, np.nan + 0j),
        "kappa_m": np.where(bound, kappa_m * k0, np.nan + 0j),
        "L": masked(np.where(k_spp.imag > 0, 1.0 / (2.0 * np.abs(k_spp.imag)), np.inf)),
    }


def sweep_spectrum(
    omegas: np.ndarray, fill_fraction: float = FILL_FRACTION, eps_d2: float = EPS_D2,
    eps_d: float = EPS_D,
) -> Dict[str, np.ndarray]:
    """
    Permittivities, anisotropy class and SPP metrics over an ω grid.

    Returns the :func:`spp_metrics` dict plus ``omega``, ``eps_t``, ``eps_n``
    and ``anisotropy`` (array of class labels).
    """
    omegas = np.asarray(omegas, dtype=float)
    eps_t, eps_n = hmm_permittivities(omegas, fill_fraction, eps_d2)
    out = spp_metrics(eps_t, eps_n, omegas, eps_d)
    out.update(omega=omegas, eps_t=eps_t, eps_n=eps_n,
               anisotropy=np.asarray(classify_anisotropy(eps_t, eps_n)))
    return out


def pole_resonance(
    transitions: Dict[str, np.ndarray],
    fill_fraction: float = FILL_FRACTION,
    eps_d2: float = EPS_D2,
) -> Optional[Dict[str, float]]:
    """
    The multilayer's harmonic-mean resonance, evaluated *at* the pole rather
    than read off a grid (the line is narrow enough that a uniform sweep
    under-reports its height).

    Returns ``None`` when the pole lies outside the searched window.
    """
    poles = transitions["eps_n_pole"]
    if len(poles) == 0:
        return None
    omega = float(poles[0])
    _, eps_n = hmm_permittivities(omega, fill_fraction, eps_d2)
    return {
        "omega": omega,
        "photon_energy_ev": float(photon_energy_ev(omega)),
        "wavelength_nm": float(wavelength_from_omega(omega) * 1e9),
        "re_eps_n": float(eps_n.real),
        "im_eps_n": float(eps_n.imag),
    }


def refine_grid(
    omegas: np.ndarray,
    transitions: Dict[str, np.ndarray],
    n_local: int = 161,
    rel_halfwidth: float = 0.02,
) -> np.ndarray:
    """
    Add a locally dense cluster of frequencies around each ENZ crossing and the
    harmonic pole.

    The ``Im ε_n`` resonance is narrow (its width is set by ``Im ε_m``, a part
    in ten thousand of ω here), so a uniform grid renders it at whatever height
    it happens to sample. Refining makes the figures resolution-independent
    where it matters and costs nothing — every sweep here is closed-form.
    """
    extra = [np.asarray(omegas, dtype=float)]
    for key in ("eps_t_zeros", "eps_n_zeros", "eps_n_pole"):
        for w in transitions.get(key, ()):
            extra.append(np.linspace(w * (1 - rel_halfwidth), w * (1 + rel_halfwidth), n_local))
    grid = np.unique(np.concatenate(extra))
    lo, hi = float(np.min(omegas)), float(np.max(omegas))
    return grid[(grid >= lo) & (grid <= hi)]


def linear_fit_residual(
    omega: np.ndarray, k_real: np.ndarray, through_origin: bool = True
) -> Tuple[float, np.ndarray]:
    """
    Departure of ``Re k_spp(ω)`` from a straight line, as a percentage of the
    total range of ``k``.

    Two different references, because they answer two different questions:

    ``through_origin=True`` (default) fits ``k = aω``, the shape a
    *non-dispersive* ε produces (``k = n ω/c`` for constant ``n``). The metric
    is then "how far is this from the fixed-ε idealisation", which is the
    comparison this project cares about and the criterion the band search uses.

    ``through_origin=False`` fits the best general line ``k = aω + b``, so the
    metric isolates *curvature* alone, with any constant offset removed. This
    is the stricter reading of the word "nonlinear" and is always the smaller
    of the two.

    Both are identically zero for a fixed-ε dispersion.

    .. note::
       The general fit scales ``ω`` before fitting. Fitting the raw design
       matrix ``[ω, 1]`` is rank-deficient in practice — ω ~ 3e15 makes the
       intercept column's singular value fall below ``lstsq``'s default cutoff,
       so it is silently discarded and the "general" fit collapses onto the
       origin fit without warning. That defect previously made this function
       return the ``through_origin=True`` number whichever branch was asked
       for; :func:`test_general_fit_recovers_a_nonzero_intercept` guards it.

    The fit weights every sample equally, so the number is grid-dependent
    unless ``omega`` is evenly spaced — which is why the band search runs on a
    grid uniform in ω rather than in wavelength.

    Returns:
        ``(percent, residual)``.
    """
    omega = np.asarray(omega, dtype=float)
    k_real = np.asarray(k_real, dtype=float)
    if through_origin:
        slope = float(np.dot(omega, k_real) / np.dot(omega, omega))
        residual = k_real - slope * omega
    else:
        scale = float(np.mean(omega)) or 1.0
        coefficients = np.polyfit(omega / scale, k_real, 1)
        residual = k_real - np.polyval(coefficients, omega / scale)
    span = float(k_real.max() - k_real.min())
    if span <= 0.0:
        return 0.0, residual
    return 100.0 * float(np.abs(residual).max()) / span, residual


def recommend_band(
    sweep: Dict[str, np.ndarray],
    max_kappa_spread: float = MAX_KAPPA_SPREAD,
    min_nonlinearity_pct: float = MIN_NONLINEARITY_PCT,
    min_neff_ratio: float = MIN_NEFF_RATIO,
    min_l_over_lambda: float = MIN_L_OVER_LAMBDA_SPP,
) -> Optional[Dict[str, object]]:
    """
    The **widest** contiguous ω-band that a frequency-conditioned PINN could
    train on, or ``None`` if the sweep contains no such band.

    Constraints, all of which must hold at every grid point of the band:

    (a) a bound SPP exists (``is_spp_supported`` *and* the non-radiative gate),
        and it survives at least ``min_l_over_lambda`` of its own wavelengths
        (``L ≥ min_l_over_lambda · λ_spp``), so the mode is a real feature of
        the field and not an over-damped resonance;
    (b) ``Re k_spp`` varies appreciably (``max/min of Re k_spp/k₀ ≥
        min_neff_ratio``) and *non-linearly*: the largest residual about a
        straight line is at least ``min_nonlinearity_pct`` % of the range of
        ``Re k_spp`` — the fixed-ε idealisation scores exactly 0 %;
    (c) the mode's inverse length scales stay together: the ratio of the
        largest to the smallest of ``{Re κ_d, Re κ_m}`` over the whole band is
        at most ``max_kappa_spread``, so one collocation sampling covers the
        band (the deepest field tail is within an order of magnitude of the
        tightest).

    *Why widest:* nonlinearity-as-objective is degenerate — the percentage
    metric grows monotonically as the band narrows around the curved blue end
    — so bandwidth is maximised instead and the nonlinearity is imposed as a
    floor and reported. In practice the red edge is set by (c) and the blue
    edge by the 450 nm Drude-validity floor of the search window.

    Args:
        sweep: Output of :func:`sweep_spectrum` on an **increasing** ω grid.
        max_kappa_spread, min_nonlinearity_pct, min_neff_ratio,
        min_l_over_lambda: The thresholds above.

    Returns:
        Dict describing the band (indices, ω/λ edges, k and κ ranges,
        nonlinearity, quality factors), or ``None``.
    """
    omega = sweep["omega"]
    n_pts = omega.size
    kappa_d, kappa_m = sweep["kappa_d"].real, sweep["kappa_m"].real
    lambda_spp = np.where(sweep["bound"], 2.0 * np.pi / np.abs(sweep["k_spp"].real), np.nan)
    with np.errstate(invalid="ignore"):
        quality = sweep["L"] / lambda_spp
        usable = sweep["bound"] & np.isfinite(quality) & (quality >= min_l_over_lambda)

    best: Optional[Tuple[int, int, float]] = None
    for i in range(n_pts):
        if not usable[i]:
            continue
        # Widen while the monotone constraints (usable everywhere, κ spread) hold.
        lo = min(kappa_d[i], kappa_m[i])
        hi = max(kappa_d[i], kappa_m[i])
        j = i
        while j + 1 < n_pts and usable[j + 1]:
            new_lo = min(lo, kappa_d[j + 1], kappa_m[j + 1])
            new_hi = max(hi, kappa_d[j + 1], kappa_m[j + 1])
            if new_hi / new_lo > max_kappa_spread:
                break
            lo, hi, j = new_lo, new_hi, j + 1
        # Shrink until the non-monotone criteria (b) are met as well.
        while j > i:
            window = slice(i, j + 1)
            n_eff = sweep["n_eff"][window]
            pct, _ = linear_fit_residual(omega[window], sweep["k_spp"].real[window])
            if n_eff.max() / n_eff.min() >= min_neff_ratio and pct >= min_nonlinearity_pct:
                width = (omega[j] - omega[i]) / (0.5 * (omega[i] + omega[j]))
                if best is None or width > best[2]:
                    best = (i, j, width)
                break
            j -= 1

    if best is None:
        return None

    i, j, width = best
    window = slice(i, j + 1)
    omega_band = omega[window]
    k_real = sweep["k_spp"].real[window]
    pct, residual = linear_fit_residual(omega_band, k_real)
    kappa_all = np.concatenate([kappa_d[window], kappa_m[window]])
    quality_band = quality[window]
    k_max = float(max(np.abs(sweep["k_spp"][window]).max(), kappa_all.max()))
    return {
        "index_range": [i, j],
        "omega": [float(omega_band[0]), float(omega_band[-1])],
        "omega_over_omega_ref": [
            float(omega_band[0] / OMEGA_REF),
            float(omega_band[-1] / OMEGA_REF),
        ],
        "photon_energy_ev": [
            float(photon_energy_ev(omega_band[0])),
            float(photon_energy_ev(omega_band[-1])),
        ],
        "wavelength_nm": [
            float(wavelength_from_omega(omega_band[-1]) * 1e9),
            float(wavelength_from_omega(omega_band[0]) * 1e9),
        ],
        "relative_width": float(width),
        "n_eff_range": [float(sweep["n_eff"][window].min()), float(sweep["n_eff"][window].max())],
        "k_spp_re_range_per_um": [float(k_real.min() * 1e-6), float(k_real.max() * 1e-6)],
        "kappa_d_range_per_um": [
            float(kappa_d[window].min() * 1e-6),
            float(kappa_d[window].max() * 1e-6),
        ],
        "kappa_m_range_per_um": [
            float(kappa_m[window].min() * 1e-6),
            float(kappa_m[window].max() * 1e-6),
        ],
        "kappa_spread": float(kappa_all.max() / kappa_all.min()),
        "decay_length_range_nm": [
            float(1e9 / kappa_all.max()),
            float(1e9 / kappa_all.min()),
        ],
        # Departure from the fixed-ε form k ∝ ω (the band-selection criterion)
        "nonlinearity_percent": float(pct),
        # Strict curvature: departure from the best general line k = aω + b
        "curvature_percent": float(linear_fit_residual(omega_band, k_real, through_origin=False)[0]),
        "max_residual_per_um": float(np.abs(residual).max() * 1e-6),
        "propagation_length_um_range": [
            float(sweep["L"][window].min() * 1e6),
            float(sweep["L"][window].max() * 1e6),
        ],
        "L_over_lambda_spp_range": [float(quality_band.min()), float(quality_band.max())],
        "max_layer_period_nm": float(max_layer_period(k_max) * 1e9),
    }


def band_vs_fill_fraction(
    search_omegas: np.ndarray,
    fills: Sequence[float],
    eps_d2: float = EPS_D2,
) -> List[Dict[str, object]]:
    """
    Re-run the band recommendation at several metal filling fractions.

    This is what justifies the design point: the achievable bandwidth is a
    shallow optimum in ``f``, so the choice can be made on other grounds
    (fabrication, staying inside the conventional 0.3–0.5 range) without
    costing much.

    Returns:
        One row per fill fraction, with the band's headline numbers or
        ``None`` where no band qualifies.
    """
    rows: List[Dict[str, object]] = []
    for f in fills:
        band = recommend_band(sweep_spectrum(search_omegas, float(f), eps_d2))
        rows.append(
            {
                "fill_fraction": float(f),
                # NB: this is the *qualifying* band — the criteria in
                # `recommend_band` (kappa spread, nonlinearity, n_eff ratio,
                # L/lambda_spp) have already been applied, so it is strictly
                # narrower than the region where a bound mode merely exists.
                # `bound_mode_regions` at the top level is the unfiltered one.
                "band_is_criteria_filtered": True,
                "band": None
                if band is None
                else {
                    key: band[key]
                    for key in (
                        "wavelength_nm",
                        "omega_over_omega_ref",
                        "relative_width",
                        "nonlinearity_percent",
                        "n_eff_range",
                        "kappa_spread",
                    )
                },
            }
        )
    return rows


def sweep_fill_fraction(
    fills: np.ndarray, omegas: np.ndarray, eps_d2: float = EPS_D2, eps_d: float = EPS_D
) -> Dict[str, np.ndarray]:
    """
    ``(f, ω)`` maps of anisotropy class and SPP effective index.

    Grids are indexed ``[i_f, i_omega]`` so they plot directly with
    ``pcolormesh(omegas, fills, grid)``.
    """
    fills = np.asarray(fills, dtype=float)
    omegas = np.asarray(omegas, dtype=float)
    eps_t, eps_n = hmm_permittivities(omegas[None, :], fills[:, None], eps_d2)
    metrics = spp_metrics(eps_t, eps_n, omegas[None, :], eps_d)
    labels = np.asarray(classify_anisotropy(eps_t, eps_n))
    class_index = np.select(
        [labels == name for name in ANISOTROPY_CLASSES],
        list(range(len(ANISOTROPY_CLASSES))),
        default=-1,
    )
    return {
        "fills": fills,
        "omega": omegas,
        "class_index": class_index,
        "n_eff": metrics["n_eff"],
        "bound": metrics["bound"],
    }


def class_spans(
    omegas: np.ndarray, labels: np.ndarray
) -> List[Tuple[float, float, str]]:
    """Contiguous ``(ω_start, ω_end, class)`` runs of an anisotropy-class array."""
    spans: List[Tuple[float, float, str]] = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            spans.append((float(omegas[start]), float(omegas[i - 1]), str(labels[start])))
            start = i
    return spans


def bound_spans(omegas: np.ndarray, bound: np.ndarray) -> List[Tuple[float, float]]:
    """Contiguous ``(ω_start, ω_end)`` runs where a bound mode exists."""
    spans: List[Tuple[float, float]] = []
    start: Optional[int] = None
    for i, flag in enumerate(bound):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            spans.append((float(omegas[start]), float(omegas[i - 1])))
            start = None
    if start is not None:
        spans.append((float(omegas[start]), float(omegas[-1])))
    return spans


# ================================================================== plotting
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


def _wavelength_axis(ax, ticks_nm: Sequence[float] = (250, 300, 400, 600, 1000, 1500)) -> None:
    """Twin top axis in wavelength for an eV x-axis."""

    def to_nm(energy_ev):
        return HC_EV_NM / np.maximum(np.asarray(energy_ev, dtype=float), 1e-9)

    secondary = ax.secondary_xaxis("top", functions=(to_nm, to_nm))
    secondary.set_xticks(list(ticks_nm))
    secondary.set_xlabel(r"free-space wavelength $\lambda_0$ (nm)", labelpad=6)


def _shade_classes(ax, spans: Sequence[Tuple[float, float, str]]) -> None:
    """Shade anisotropy-class regions on an eV x-axis."""
    for w_lo, w_hi, name in spans:
        ax.axvspan(
            photon_energy_ev(w_lo),
            photon_energy_ev(w_hi),
            color=CLASS_COLOURS.get(name, "#cccccc"),
            alpha=CLASS_ALPHA,
            linewidth=0,
            zorder=0,
        )


def _mark_band(ax, band: Optional[Dict[str, object]], label: bool = False) -> None:
    """Outline the recommended band on an eV x-axis."""
    if band is None:
        return
    lo, hi = band["photon_energy_ev"]  # type: ignore[index]
    ax.axvspan(lo, hi, facecolor="none", edgecolor=C_BAND, linewidth=1.6,
               linestyle="--", zorder=5)
    if label:
        ax.annotate(
            "recommended\nPINN band",
            xy=(0.5 * (lo + hi), 0.97),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="top",
            fontsize=9,
            color=C_BAND,
        )


def plot_permittivities(
    sweep: Dict[str, np.ndarray],
    transitions: Dict[str, np.ndarray],
    path: Path,
    fill_fraction: float = FILL_FRACTION,
    eps_d2: float = EPS_D2,
) -> None:
    """Figure 1: ε_t(ω), ε_n(ω) with anisotropy regions and ENZ crossings."""
    energy = photon_energy_ev(sweep["omega"])
    spans = class_spans(sweep["omega"], sweep["anisotropy"])

    fig, (ax_re, ax_im) = plt.subplots(2, 1, figsize=(8.4, 8.2), sharex=True)

    for ax in (ax_re, ax_im):
        _shade_classes(ax, spans)
        for w in transitions["eps_t_zeros"]:
            ax.axvline(photon_energy_ev(w), color=C_T, linestyle=":", linewidth=1.4)
        for w in transitions["eps_n_zeros"]:
            ax.axvline(photon_energy_ev(w), color=C_N, linestyle=":", linewidth=1.4)

    ax_re.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax_re.plot(energy, sweep["eps_t"].real, color=C_T, label=r"Re $\varepsilon_t$ (in-plane)")
    ax_re.plot(energy, sweep["eps_n"].real, color=C_N, label=r"Re $\varepsilon_n$ (normal)")
    ax_re.set_yscale("symlog", linthresh=1.0)
    ax_re.set_ylabel(r"Re $\varepsilon$")
    ax_re.set_title(
        rf"(a) real parts — Ag/silica multilayer, $f = {fill_fraction:g}$, "
        rf"$\varepsilon_{{d2}} = {eps_d2:g}$ (symlog, $|\varepsilon| < 1$ linear)"
    )
    ax_re.legend(loc="upper left")

    ax_im.semilogy(energy, sweep["eps_t"].imag, color=C_T, label=r"Im $\varepsilon_t$")
    ax_im.semilogy(energy, sweep["eps_n"].imag, color=C_N, label=r"Im $\varepsilon_n$")
    ax_im.set_ylabel(r"Im $\varepsilon$  (loss; $>0$ throughout)")
    ax_im.set_xlabel(r"photon energy $\hbar\omega$ (eV)")
    ax_im.set_title(r"(b) imaginary parts — passive everywhere, resonant at the pole")
    ax_im.legend(loc="upper left")

    # Label the crossings along their guide lines, on the (emptier) loss panel.
    for key, colour, symbol in (
        ("eps_t_zeros", C_T, r"\varepsilon_t"),
        ("eps_n_zeros", C_N, r"\varepsilon_n"),
    ):
        for w in transitions[key]:
            ax_im.annotate(
                rf"Re ${symbol} = 0$  ({wavelength_from_omega(w) * 1e9:.0f} nm)",
                xy=(photon_energy_ev(w), 0.03),
                xycoords=("data", "axes fraction"),
                xytext=(-4, 0),
                textcoords="offset points",
                fontsize=9,
                color=colour,
                rotation=90,
                ha="right",
                va="bottom",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
            )
    resonance = pole_resonance(transitions, fill_fraction, eps_d2)
    if resonance is not None:
        peak = resonance["im_eps_n"]
        ax_im.annotate(
            "harmonic-mean pole:\n"
            rf"Im $\varepsilon_n$ peaks at {peak:.0f}" "\n(loss resonance, still passive)",
            xy=(resonance["photon_energy_ev"], peak),
            xytext=(-14, -4),
            textcoords="offset points",
            fontsize=9,
            color=C_N,
            ha="right",
            va="top",
            arrowprops={"arrowstyle": "->", "color": C_N, "linewidth": 1.0},
        )

    handles = [
        Patch(facecolor=CLASS_COLOURS[name], alpha=CLASS_ALPHA, label=name)
        for name in ANISOTROPY_CLASSES
        if any(span[2] == name for span in spans)
    ]
    ax_im.legend(
        handles=[*ax_im.get_legend_handles_labels()[0], *handles],
        loc="upper left",
        fontsize=9,
    )

    _wavelength_axis(ax_re)
    fig.suptitle(
        "Effective permittivities of an Ag/silica hyperbolic metamaterial "
        "(layers stacked along z)",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)


def plot_spp_dispersion(
    sweep: Dict[str, np.ndarray],
    search: Dict[str, np.ndarray],
    band: Optional[Dict[str, object]],
    path: Path,
    fill_fraction: float = FILL_FRACTION,
) -> None:
    """
    Figure 2: ω–k diagram, nonlinearity residual, L(ω) and the decay constants.

    ``sweep`` is the wide spectral survey (panels a, c, d); ``band`` indexes
    into ``search``, the grid the recommendation was made on, so every number
    drawn here is the one written to the JSON summary.
    """
    energy = photon_energy_ev(sweep["omega"])
    bound = sweep["bound"]
    spans = class_spans(sweep["omega"], sweep["anisotropy"])
    band_energy = photon_energy_ev(search["omega"])

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.6))
    ax_wk, ax_res, ax_len, ax_kappa = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # ---- (a) ω–k diagram: k on x, ħω on y (classic layout).
    k_um = sweep["k_spp"].real * 1e-6
    k_axis = np.linspace(0.0, 1.05 * float(np.nanmax(k_um)), 200)
    ax_wk.plot(
        k_axis,
        HBAR_EVS * C0 * (k_axis * 1e6) / math.sqrt(EPS_D),
        color=C_LIGHT,
        linestyle="--",
        linewidth=1.5,
        label=r"light line $\omega = ck/\sqrt{\varepsilon_d}$",
    )
    ax_wk.plot(k_um, energy, color=C_SPP, label="HMM/air SPP branch")
    if band is not None:
        i, j = band["index_range"]  # type: ignore[index]
        window = slice(i, j + 1)
        ax_wk.plot(search["k_spp"].real[window] * 1e-6, band_energy[window], color=C_BAND,
                   linewidth=3.4, solid_capstyle="round", label="recommended PINN band",
                   zorder=4)
        # The fixed-ε idealisation of examples/validate_spp_dispersion.py:
        # ε constant ⇒ k = n·ω/c, a straight line through the origin.
        omega_band = search["omega"][window]
        i_mid = i + int(np.argmin(np.abs(omega_band - 0.5 * (omega_band[0] + omega_band[-1]))))
        n_mid = float(search["n_eff"][i_mid])
        e_edges = np.array([band_energy[i], band_energy[j]])
        ax_wk.plot(
            n_mid * e_edges / (HBAR_EVS * C0) * 1e-6,
            e_edges,
            color="black",
            linestyle=":",
            linewidth=1.6,
            label=rf"fixed-$\varepsilon$ idealisation ($n = {n_mid:.2f}$)",
        )
    ax_wk.set_xlabel(r"Re $k_{\mathrm{spp}}$ (rad/µm)")
    ax_wk.set_ylabel(r"photon energy $\hbar\omega$ (eV)")
    ax_wk.set_xlim(0.0, k_axis.max())
    ax_wk.set_ylim(energy.min(), energy.max())
    ax_wk.set_title("(a) SPP dispersion of the multilayer")
    ax_wk.legend(loc="upper left", fontsize=9)

    # ---- (b) residual about the straight-line fit over the recommended band.
    if band is not None:
        i, j = band["index_range"]  # type: ignore[index]
        window = slice(i, j + 1)
        pct, residual = linear_fit_residual(
            search["omega"][window], search["k_spp"].real[window]
        )
        ax_res.axhline(0.0, color="black", linewidth=1.0, linestyle=":")
        ax_res.plot(band_energy[window], residual * 1e-6, color=C_BAND)
        ax_res.fill_between(band_energy[window], 0.0, residual * 1e-6, color=C_BAND, alpha=0.25)
        curvature_pct, _ = linear_fit_residual(
            search["omega"][window], search["k_spp"].real[window], through_origin=False
        )
        ax_res.set_title(
            rf"(b) departure from the fixed-$\varepsilon$ form $k \propto \omega$: "
            rf"{pct:.1f}% of the $k_{{\mathrm{{spp}}}}$ range "
            rf"(pure curvature {curvature_pct:.1f}%)"
        )
        ax_res.annotate(
            "fixed-ε idealisation\nwould be flat at 0",
            xy=(0.03, 0.06),
            xycoords="axes fraction",
            fontsize=9,
            color=C_LIGHT,
        )
        ax_res.set_xlim(band_energy[window].min(), band_energy[window].max())
    ax_res.set_ylabel(r"Re $k_{\mathrm{spp}}$ − linear fit (rad/µm)")
    ax_res.set_xlabel(r"photon energy $\hbar\omega$ (eV)  (recommended band only)")

    # ---- (c) propagation length.
    ax_len.semilogy(energy, np.where(bound, sweep["L"] * 1e6, np.nan), color=C_SPP)
    ax_len.set_ylabel(r"$L = 1/(2\,\mathrm{Im}\,k_{\mathrm{spp}})$ (µm)")
    ax_len.set_xlabel(r"photon energy $\hbar\omega$ (eV)")
    ax_len.set_title("(c) propagation length")

    # ---- (d) decay constants.
    ax_kappa.semilogy(energy, sweep["kappa_d"].real * 1e-6, color=C_DIEL,
                      label=r"Re $\kappa_d$ (air)")
    ax_kappa.semilogy(energy, sweep["kappa_m"].real * 1e-6, color=C_METAL,
                      label=r"Re $\kappa_m$ (metamaterial)")
    if band is not None:
        lo_nm, hi_nm = band["decay_length_range_nm"]  # type: ignore[index]
        e_lo, e_hi = band["photon_energy_ev"]  # type: ignore[index]
        ax_kappa.fill_between(
            [e_lo, e_hi], 1e3 / hi_nm, 1e3 / lo_nm, color=C_BAND, alpha=0.13, linewidth=0
        )
        ax_kappa.annotate(
            rf"in-band spread {band['kappa_spread']:.1f}$\times$"  # type: ignore[index]
            "\n" rf"($1/\kappa$ from {lo_nm:.0f} to {hi_nm:.0f} nm)",
            xy=(0.03, 0.06),
            xycoords="axes fraction",
            fontsize=9,
            color=C_BAND,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )
    ax_kappa.set_ylim(0.3, 60.0)
    ax_kappa.set_ylabel(r"decay constant (rad/µm)")
    ax_kappa.set_xlabel(r"photon energy $\hbar\omega$ (eV)")
    ax_kappa.set_title("(d) field confinement on both sides")
    ax_kappa.legend(loc="upper left", fontsize=9)

    for ax in (ax_len, ax_kappa):
        _shade_classes(ax, spans)
        for _, span_hi in bound_spans(sweep["omega"], bound):
            ax.axvline(photon_energy_ev(span_hi), color=C_LIGHT, linestyle="-.", linewidth=1.1)
        ax.set_xlim(energy.min(), energy.max())
        _mark_band(ax, band)

    # Name the bound-mode edge once, on the propagation-length panel.
    edges = bound_spans(sweep["omega"], bound)
    if edges:
        ax_len.annotate(
            "bound-mode edge\n(type-II onset)",
            xy=(photon_energy_ev(edges[-1][1]), 0.5),
            xycoords=("data", "axes fraction"),
            xytext=(-6, 0),
            textcoords="offset points",
            rotation=90,
            ha="right",
            va="center",
            fontsize=9,
            color=C_LIGHT,
        )

    fig.suptitle(
        rf"SPP dispersion of the Ag/silica HMM against air ($f = {fill_fraction:g}$): "
        "dispersive ε makes the branch genuinely nonlinear",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)


def plot_fill_fraction_map(
    maps: Dict[str, np.ndarray],
    band: Optional[Dict[str, object]],
    path: Path,
    eps_d2: float = EPS_D2,
    fill_fraction: float = FILL_FRACTION,
) -> None:
    """Figure 3: ``(f, ħω)`` maps of anisotropy class and of Re k_spp/k₀."""
    energy = photon_energy_ev(maps["omega"])
    fills = maps["fills"]

    fig, (ax_class, ax_index) = plt.subplots(1, 2, figsize=(13.6, 5.2))
    for ax in (ax_class, ax_index):
        ax.grid(False)
        ax.set_xlabel(r"photon energy $\hbar\omega$ (eV)")
        ax.set_ylabel(r"metal filling fraction $f$")

    cmap = ListedColormap([CLASS_COLOURS[name] for name in ANISOTROPY_CLASSES])
    ax_class.pcolormesh(
        energy, fills, maps["class_index"], cmap=cmap,
        norm=Normalize(vmin=-0.5, vmax=len(ANISOTROPY_CLASSES) - 0.5),
        shading="nearest", rasterized=True,
    )
    ax_class.set_title("(a) anisotropy class")

    # Analytic transition curves, f by f.
    curves: Dict[str, List[Tuple[float, float]]] = {"eps_t_zeros": [], "eps_n_zeros": []}
    window = (float(maps["omega"].min()), float(maps["omega"].max()))
    for f in fills:
        transitions = transition_frequencies(float(f), eps_d2, omega_range=window)
        for key in curves:
            for w in transitions[key]:
                curves[key].append((photon_energy_ev(w), float(f)))
    for key, colour, label in (
        ("eps_t_zeros", C_T, r"Re $\varepsilon_t = 0$"),
        ("eps_n_zeros", C_N, r"Re $\varepsilon_n = 0$"),
    ):
        if curves[key]:
            xs, ys = zip(*curves[key], strict=True)
            ax_class.plot(xs, ys, ".", color=colour, markersize=2.0, label=label)

    present = sorted(set(np.unique(maps["class_index"]).tolist()))
    ax_class.legend(
        handles=[
            *[
                Patch(facecolor=CLASS_COLOURS[ANISOTROPY_CLASSES[k]],
                      label=ANISOTROPY_CLASSES[k])
                for k in present
                if 0 <= k < len(ANISOTROPY_CLASSES)
            ],
            *ax_class.get_legend_handles_labels()[0],
        ],
        loc="upper left",
        fontsize=9,
        framealpha=0.9,
        markerscale=4,
    )

    n_eff = maps["n_eff"]
    finite = np.isfinite(n_eff)
    mesh = ax_index.pcolormesh(
        energy, fills, n_eff, cmap="viridis",
        vmin=1.0, vmax=float(np.nanpercentile(n_eff[finite], 99)) if finite.any() else 2.0,
        shading="nearest", rasterized=True,
    )
    fig.colorbar(mesh, ax=ax_index, label=r"Re $k_{\mathrm{spp}}/k_0$")
    ax_index.set_facecolor("#ececec")
    ax_index.set_title(r"(b) bound-SPP effective index (grey: no bound mode)")

    for ax in (ax_class, ax_index):
        ax.axhline(fill_fraction, color="white", linestyle="--", linewidth=1.2)
        ax.annotate(
            rf"design point $f = {fill_fraction:g}$",
            xy=(energy.min() + 0.05 * np.ptp(energy), fill_fraction),
            xytext=(0, 5),
            textcoords="offset points",
            fontsize=9,
            color="white",
        )
        if band is not None:
            lo, hi = band["photon_energy_ev"]  # type: ignore[index]
            ax.add_patch(
                Rectangle(
                    (lo, fills.min()), hi - lo, np.ptp(fills),
                    facecolor="none", edgecolor=C_BAND, linewidth=1.6, linestyle="--",
                )
            )
    if band is not None:
        lo, hi = band["photon_energy_ev"]  # type: ignore[index]
        ax_index.annotate(
            "recommended band",
            xy=(0.5 * (lo + hi), fills.max()),
            xytext=(0, -12),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=9,
            color=C_BAND,
        )

    fig.suptitle(
        rf"Fill-fraction sweep of the Ag/silica multilayer ($\varepsilon_{{d2}} = {eps_d2:g}$, "
        r"air superstrate)",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path)
    plt.close(fig)


# ================================================================== summary
def build_summary(
    sweep: Dict[str, np.ndarray],
    transitions: Dict[str, np.ndarray],
    band: Optional[Dict[str, object]],
    fill_scan: List[Dict[str, object]],
    n_points: int,
    fill_fraction: float = FILL_FRACTION,
    eps_d2: float = EPS_D2,
) -> Dict[str, object]:
    """Key numbers of the study, JSON-serialisable (no NaN/Infinity tokens)."""

    def as_ev_nm(omegas: np.ndarray) -> List[Dict[str, float]]:
        return [
            {
                "omega": float(w),
                "photon_energy_ev": float(photon_energy_ev(w)),
                "wavelength_nm": float(wavelength_from_omega(w) * 1e9),
            }
            for w in omegas
        ]

    spans = [
        {
            "class": name,
            "photon_energy_ev": [float(photon_energy_ev(lo)), float(photon_energy_ev(hi))],
            "wavelength_nm": [
                float(wavelength_from_omega(hi) * 1e9),
                float(wavelength_from_omega(lo) * 1e9),
            ],
        }
        for lo, hi, name in class_spans(sweep["omega"], sweep["anisotropy"])
    ]
    bound = [
        {
            "photon_energy_ev": [float(photon_energy_ev(lo)), float(photon_energy_ev(hi))],
            "wavelength_nm": [
                float(wavelength_from_omega(hi) * 1e9),
                float(wavelength_from_omega(lo) * 1e9),
            ],
        }
        for lo, hi in bound_spans(sweep["omega"], sweep["bound"])
    ]

    return {
        "design": {
            "fill_fraction": float(fill_fraction),
            "eps_dielectric_layer": float(eps_d2),
            "eps_superstrate": float(EPS_D),
            "optical_axis": "z",
            "drude_model": {
                "eps_inf": EPS_INF_AG,
                "hbar_omega_p_eV": HBAR_OMEGA_P_AG_EV,
                "hbar_gamma_eV": HBAR_GAMMA_AG_EV,
            },
            "omega_ref": OMEGA_REF,
            "lambda_ref_nm": LAMBDA_REF * 1e9,
        },
        "spectral_window": {
            "photon_energy_ev": [float(photon_energy_ev(sweep["omega"]).min()),
                                 float(photon_energy_ev(sweep["omega"]).max())],
            "wavelength_nm": [float(wavelength_from_omega(sweep["omega"]).min() * 1e9),
                              float(wavelength_from_omega(sweep["omega"]).max() * 1e9)],
        },
        "transitions": {
            "eps_t_zeros": as_ev_nm(transitions["eps_t_zeros"]),
            "eps_n_zeros": as_ev_nm(transitions["eps_n_zeros"]),
            "eps_n_pole": as_ev_nm(transitions["eps_n_pole"]),
            "pole_resonance": pole_resonance(transitions, fill_fraction, eps_d2),
        },
        "anisotropy_regions": spans,
        "bound_mode_regions": bound,
        "recommended_band": band,
        "fill_fraction_scan": fill_scan,
        "band_criteria": {
            "search_wavelength_nm": [LAMBDA_SEARCH[0] * 1e9, LAMBDA_SEARCH[1] * 1e9],
            "max_kappa_spread": MAX_KAPPA_SPREAD,
            "min_nonlinearity_percent": MIN_NONLINEARITY_PCT,
            "min_neff_ratio": MIN_NEFF_RATIO,
            "min_L_over_lambda_spp": MIN_L_OVER_LAMBDA_SPP,
        },
        "n_points": int(n_points),
    }


def _print_recommendation(summary: Dict[str, object]) -> None:
    """Print the key deliverable: the band the next PINN experiment should use."""
    band = summary["recommended_band"]  # type: ignore[index]
    design = summary["design"]  # type: ignore[index]
    print("\n" + "=" * 78)
    print("RECOMMENDED BAND FOR THE PINN EXPERIMENT")
    print("=" * 78)
    if band is None:
        print("No band satisfies the criteria in the search window.")
        return
    print(
        "Structure     : Ag/silica multilayer, f = {:.2f}, eps_d2 = {:.2f}, "
        "air superstrate (eps_d = {:.1f})".format(
            design["fill_fraction"], design["eps_dielectric_layer"], design["eps_superstrate"]
        )
    )
    print(
        "Band          : omega/omega0 = [{:.4f}, {:.4f}]  (omega0 = 2*pi*c/{:.0f} nm)".format(
            *band["omega_over_omega_ref"], design["lambda_ref_nm"]
        )
    )
    print(
        "                lambda = [{:.1f}, {:.1f}] nm   =   hbar*omega = [{:.3f}, {:.3f}] eV"
        "   (relative width {:.0%})".format(
            *band["wavelength_nm"], *band["photon_energy_ev"], band["relative_width"]
        )
    )
    print(
        "k_spp         : Re k_spp/k0 = [{:.3f}, {:.3f}],  Re k_spp = [{:.3f}, {:.3f}] rad/um".format(
            *band["n_eff_range"], *band["k_spp_re_range_per_um"]
        )
    )
    print(
        "Nonlinearity  : max departure from a straight line = {:.1f}% of the k_spp range "
        "({:.4f} rad/um); the fixed-eps idealisation scores 0%".format(
            band["nonlinearity_percent"], band["max_residual_per_um"]
        )
    )
    print(
        "kappa_d       : [{:.3f}, {:.3f}] rad/um     kappa_m: [{:.3f}, {:.3f}] rad/um".format(
            *band["kappa_d_range_per_um"], *band["kappa_m_range_per_um"]
        )
    )
    print(
        "                decay lengths [{:.0f}, {:.0f}] nm, overall spread {:.1f}x "
        "(criterion <= {:.0f}x)".format(
            *band["decay_length_range_nm"],
            band["kappa_spread"],
            summary["band_criteria"]["max_kappa_spread"],  # type: ignore[index]
        )
    )
    print(
        "Propagation   : L = [{:.1f}, {:.1f}] um  =  [{:.0f}, {:.0f}] SPP wavelengths".format(
            *band["propagation_length_um_range"], *band["L_over_lambda_spp_range"]
        )
    )
    print(
        "Fabrication   : bulk effective-medium validity needs a period <= {:.1f} nm "
        "(i.e. <= {:.1f} nm Ag + {:.1f} nm silica)".format(
            band["max_layer_period_nm"],
            band["max_layer_period_nm"] * design["fill_fraction"],
            band["max_layer_period_nm"] * (1.0 - design["fill_fraction"]),
        )
    )
    print(
        "                BUT that is a BULK criterion, O((a/lambda)^2). For this\n"
        "                SURFACE mode the leading error is O(a) — a termination\n"
        "                effect — so it is far stricter: see examples/emt_validity.py,\n"
        "                which measures it against a transfer-matrix solve of the real\n"
        "                multilayer. At a 10 nm period Re k_spp is good to ~1% but\n"
        "                Im k_spp (hence propagation length) is ~9% optimistic; at\n"
        "                30 nm Im k_spp is ~24% out. Terminating the stack with a\n"
        "                half-thickness metal layer cancels the O(a) term and\n"
        "                restores O(a^2), keeping the error < 1% out to 60 nm."
    )
    print("=" * 78 + "\n")


# ===================================================================== main
def main(argv: Optional[List[str]] = None) -> Dict[str, object]:
    parser = argparse.ArgumentParser(
        description="Hyperbolic-metamaterial design study (analytics only): effective "
        "permittivities, SPP dispersion, and the recommended band for a "
        "frequency-conditioned PINN."
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help=f"output directory for figures and JSON (default: {DEFAULT_FIGURES_DIR})",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=401,
        help="points along the frequency axis (the f-sweep uses n/4 rows) (default: 401)",
    )
    parser.add_argument(
        "--fill-fraction",
        type=float,
        default=FILL_FRACTION,
        help=f"metal filling fraction of the design point (default: {FILL_FRACTION})",
    )
    args = parser.parse_args(argv)
    figures_dir: Path = args.figures_dir
    n: int = args.n_points
    fill_fraction: float = args.fill_fraction
    figures_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    # 1. Permittivity spectrum over the wide window, refined at the resonances.
    uniform = omega_from_photon_energy_ev(np.linspace(EV_MIN, EV_MAX, n))
    transitions = transition_frequencies(
        fill_fraction, EPS_D2, omega_range=(float(uniform.min()), float(uniform.max()))
    )
    omegas = refine_grid(uniform, transitions)
    sweep = sweep_spectrum(omegas, fill_fraction, EPS_D2)
    plot_permittivities(
        sweep, transitions, figures_dir / "hmm_permittivities.png", fill_fraction
    )

    # 2. Band search on the Drude-trustworthy window, then the dispersion figure.
    # Uniform in ω (not λ): the nonlinearity metric is a least-squares fit in ω,
    # so an even sampling in ω keeps it independent of the grid.
    search_omegas = np.linspace(
        float(omega_from_wavelength(LAMBDA_SEARCH[1])),
        float(omega_from_wavelength(LAMBDA_SEARCH[0])),
        n,
    )
    search = sweep_spectrum(search_omegas, fill_fraction, EPS_D2)
    band = recommend_band(search)
    plot_spp_dispersion(
        sweep, search, band, figures_dir / "hmm_spp_dispersion.png", fill_fraction
    )

    # 3. Fill-fraction map (uniform grid: a 2-D map cannot resolve the pole anyway).
    maps = sweep_fill_fraction(np.linspace(0.05, 0.95, max(5, n // 4)), uniform, EPS_D2)
    plot_fill_fraction_map(
        maps, band, figures_dir / "hmm_fill_fraction_map.png", EPS_D2, fill_fraction
    )

    fill_scan = band_vs_fill_fraction(search_omegas, FILL_SCAN, EPS_D2)
    summary = build_summary(
        sweep, transitions, band, fill_scan, n, fill_fraction, EPS_D2
    )
    summary_path = figures_dir / "hmm_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Figures written to {figures_dir}")
    regions = ", ".join(
        "{}: {:.0f}–{:.0f} nm".format(r["class"], *r["wavelength_nm"])
        for r in summary["anisotropy_regions"]  # type: ignore[index]
    )
    print(f"Anisotropy regions (f = {fill_fraction:g}): {regions}")
    resonance = summary["transitions"]["pole_resonance"]  # type: ignore[index]
    if resonance is not None:
        print(
            "Harmonic-mean pole at {:.0f} nm: Im ε_n = {:.0f} (a loss resonance — still "
            "passive), Re ε_n = {:.2f}".format(
                resonance["wavelength_nm"], resonance["im_eps_n"], resonance["re_eps_n"]
            )
        )
    print("Band width vs metal filling fraction (why f is a shallow choice):")
    for row in fill_scan:
        if row["band"] is None:
            print(f"    f = {row['fill_fraction']:.2f}:  no qualifying band")
        else:
            print(
                "    f = {:.2f}:  {:.0f}–{:.0f} nm, relative width {:.0%}, "
                "nonlinearity {:.0f}%".format(
                    row["fill_fraction"],
                    *row["band"]["wavelength_nm"],
                    row["band"]["relative_width"],
                    row["band"]["nonlinearity_percent"],
                )
            )
    _print_recommendation(summary)
    print(f"Summary written to {summary_path}")
    return summary


if __name__ == "__main__":
    main()
