"""
Analytical SPP Dispersion Study — Drude Silver and Uniaxial Metamaterials

A training-free companion to the PINN validation examples: everything here is
computed from the benchmark-validated closed-form dispersion in
:class:`src.physics.metamaterial.MetamaterialProperties` (machine-precision
anchored in ``tests/test_benchmark_spp.py``). Three studies are produced, each
as a publication-quality figure, plus a machine-readable summary
(``dispersion_summary.json``) of the key numbers.

1. **Drude silver dispersion** (``spp_dispersion_silver.png``,
   ``spp_length_scales_silver.png``). Silver is modelled with the simple Drude
   fit::

       ε(ω) = ε_∞ − ω_p² / (ω² + iγω)

   with ε_∞ = 3.7, ħω_p = 9.1 eV, ħγ = 0.018 eV — standard literature values
   for the free-electron response of silver in the red/near-IR. *Limitations
   vs Johnson & Christy (Phys. Rev. B 6, 4370 (1972))*: a Drude fit contains
   no interband transitions, so it degrades rapidly below ~450 nm where the
   d-band absorption of silver switches on, and the small free-electron
   damping ħγ = 0.018 eV under-represents the measured loss (at 633 nm it
   gives Im ε ≈ 0.2 against the J&C value 0.55, i.e. Drude silver propagates
   further than real silver). Re ε, which controls the dispersion curve and
   the confinement, agrees with J&C to a few per cent across the sweep — the
   complex ε at 633 nm is within 15% of the J&C anchor −18.3 + 0.55j used in
   the benchmark tests. The free-space wavelength is swept over 400–1000 nm
   and the classic ω–k diagram is drawn: the SPP branch bends below the light
   line ω = ck/√ε_d toward the surface-plasmon asymptote

       ω_sp = ω_p / √(ε_∞ + ε_d)

   (the textbook ω_p/√(1 + ε_d) with the background ε_∞ restored; for
   ε_∞ = 3.7, ε_d = 1 this is ħω_sp ≈ 4.20 eV ≈ 295 nm). The second figure
   shows the wavelength dependence of the mode's three length scales: the
   intensity propagation length L = 1/(2 Im k_spp) and the field penetration
   depths δ_d = 1/Re κ_d (dielectric) and δ_m = 1/Re κ_m (metal).

2. **Uniaxial design-space maps** (``metamaterial_design_space.png``). At
   fixed λ₀ = 633 nm against air (ε_d = 1), the (Re ε_t, Re ε_n) plane is
   swept with small fixed losses Im ε_t = 0.1, Im ε_n = 0.05 (ε_t in-plane
   along propagation, ε_n normal to the interface, optical axis ⟂ interface).
   Mapped: (a) where a bound TM surface mode exists
   (:meth:`~src.physics.metamaterial.MetamaterialProperties.is_spp_supported`,
   which checks the *unsquared* matching condition κ_d/ε_d + κ_m/ε_t = 0 on
   the bound branch), (b) the effective index Re k_spp/k₀, (c) the
   propagation length. In the lossless limit the support region is
   ``ε_t < 0`` and (``ε_n > ε_d`` or (``ε_n < 0`` and ``ε_t ε_n > ε_d²``)) —
   notably the isotropic threshold ε_m < −ε_d disappears: any ε_t < 0
   supports a bound mode once ε_n > ε_d.

3. **Anisotropy cut** (``anisotropy_cut.png``). A 1-D slice at fixed
   ε_t = −4 + 0.2j sweeping Re ε_n across the design space, crossing both
   support boundaries: the resonance ε_t ε_n = ε_d² near Re ε_n = ε_d²/ε_t ≈
   −0.25, where k_spp diverges (denominator of the closed form vanishes —
   guarded here, since ``spp_wavevector`` raises ``ZeroDivisionError`` on the
   exact singularity), and the bound-mode onset near ε_n = ε_d.

Sign convention throughout (as in the module under study): time dependence
``exp(-iωt)``, lossy media ``Im ε > 0``, decaying propagation
``Im k_spp > 0``, bound decay constants ``Re κ > 0``.

Usage::

    python examples/dispersion_analysis.py [--figures-dir DIR] [--n-points N]

Figures and ``dispersion_summary.json`` are written to
``figures/dispersion/`` by default. No GPU, no training — runtime is seconds.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap, LogNorm  # noqa: E402

from src.constants import C0  # noqa: E402
from src.physics.metamaterial import MetamaterialProperties  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIGURES_DIR = REPO_ROOT / "figures" / "dispersion"

# ------------------------------------------------------------------ constants
EV = 1.602176634e-19  # J per eV (exact, SI 2019)
HBAR = 1.054571817e-34  # J s
HBAR_EVS = HBAR / EV  # eV s — converts ω (rad/s) to ħω (eV)

# Drude fit for silver: ε(ω) = ε_∞ − ω_p²/(ω² + iγω). Standard values; see
# the module docstring for their limitations vs Johnson & Christy.
EPS_INF = 3.7
HBAR_OMEGA_P_EV = 9.1
HBAR_GAMMA_EV = 0.018
OMEGA_P = HBAR_OMEGA_P_EV * EV / HBAR  # rad/s
GAMMA = HBAR_GAMMA_EV * EV / HBAR  # rad/s

EPS_AG_JC_633NM = -18.3 + 0.55j  # Johnson & Christy anchor (tests/test_benchmark_spp.py)

# Study parameters
LAMBDA_MIN, LAMBDA_MAX = 400e-9, 1000e-9  # Drude sweep band (m)
LAMBDA0_MAP = 633e-9  # fixed frequency for the design-space maps (m)
EPS_D = 1.0  # air superstrate throughout
EPS_T_RANGE = (-10.0, 2.0)  # Re ε_t window (in-plane, along propagation)
EPS_N_RANGE = (-4.0, 8.0)  # Re ε_n window (normal to the interface)
IM_EPS_T = 0.1  # small fixed losses for the maps
IM_EPS_N = 0.05
CUT_EPS_T = -4.0 + 0.2j  # anisotropy cut: fixed in-plane permittivity
BENCHMARK_POINT = (-4.0, 3.0)  # (Re ε_t, Re ε_n) anchored in the benchmark tests

# Okabe–Ito colourblind-safe palette, fixed roles across all figures.
C_SPP = "#0072B2"  # blue: the SPP mode / propagation length
C_LIGHT = "#555555"  # grey: light line, guides
C_DIEL = "#009E73"  # green: dielectric-side quantities
C_METAL = "#D55E00"  # vermillion: metamaterial-side quantities
C_MARK = "#CC79A7"  # purple: annotation markers


# ================================================================ pure physics
def omega_from_wavelength(wavelength: float) -> float:
    """Angular frequency (rad/s) of a free-space wavelength (m)."""
    return 2.0 * math.pi * C0 / wavelength


def drude_permittivity(omega: float) -> complex:
    """
    Drude silver ε(ω) = ε_∞ − ω_p²/(ω² + iγω) under the ``exp(-iωt)``
    convention (lossy ⇒ Im ε > 0).
    """
    return EPS_INF - OMEGA_P**2 / (omega**2 + 1j * GAMMA * omega)


def surface_plasmon_energy_ev(eps_d: float = EPS_D) -> float:
    """ħω_sp (eV) of the asymptote ω_sp = ω_p/√(ε_∞ + ε_d) (Re ε(ω_sp) = −ε_d)."""
    return HBAR_OMEGA_P_EV / math.sqrt(EPS_INF + eps_d)


def silver_interface(omega: float) -> MetamaterialProperties:
    """Isotropic Drude-silver half-space (ε_t = ε_n = ε_Drude(ω))."""
    eps = drude_permittivity(omega)
    return MetamaterialProperties(eps, eps, optical_axis="z", omega=omega)


def uniaxial_interface(eps_t: complex, eps_n: complex, omega: float) -> MetamaterialProperties:
    """
    Uniaxial half-space with optical axis normal to the interface ('z'):
    the constructor's ``eps_parallel`` is the component *along the axis*,
    i.e. ε_n; ``eps_perpendicular`` is the in-plane ε_t.
    """
    return MetamaterialProperties(eps_n, eps_t, optical_axis="z", omega=omega)


def bound_mode_metrics(
    material: MetamaterialProperties, eps_d: float = EPS_D
) -> Optional[dict]:
    """
    Mode metrics of the bound TM SPP at a material/dielectric interface, or
    ``None`` when no bound mode exists.

    Every sweep point is gated by ``is_spp_supported`` and additionally
    guarded against the closed form's resonance ε_t ε_n = ε_d², where
    ``spp_wavevector`` raises ``ZeroDivisionError``.

    On top of ``is_spp_supported`` the classic *non-radiative* criterion
    ``Re k_spp > √ε_d · k₀`` (beyond the light line) is enforced. With finite
    losses the matching condition κ_d/ε_d + κ_m/ε_t = 0 acquires roots with
    ``Re n_eff < 1`` whose decay constants have small positive real parts
    (e.g. ε_t = 2 + 0.1i, ε_n = 2 + 0.05i gives n_eff ≈ 0.82 + 0.01i): these
    are strongly damped quasi-modes that radiate into the dielectric, not
    bound SPPs, and vanish in the lossless limit.
    """
    try:
        if not material.is_spp_supported(eps_dielectric=eps_d):
            return None
        k = material.spp_wavevector(eps_dielectric=eps_d)
        if k.real <= math.sqrt(eps_d) * material.k0:
            return None
        return {
            "k_spp": k,
            "n_eff": k.real / material.k0,
            "L": material.propagation_length(eps_dielectric=eps_d),
            "delta_d": material.penetration_depth_dielectric(eps_dielectric=eps_d),
            "delta_m": material.penetration_depth_metamaterial(eps_dielectric=eps_d),
        }
    except ZeroDivisionError:  # exact ε_t ε_n = ε_d² grid point
        return None


# =================================================================== sweeps
def sweep_silver_dispersion(wavelengths: np.ndarray, eps_d: float = EPS_D) -> dict:
    """
    Drude-silver SPP dispersion over an array of free-space wavelengths (m).

    Returns arrays (NaN where no bound mode): ``wavelength``, ``omega``,
    ``k0``, ``eps`` (Drude ε), ``k_spp`` (complex), ``n_eff``, ``L``,
    ``delta_d``, ``delta_m`` and the boolean ``supported`` mask.
    """
    n = len(wavelengths)
    out = {
        "wavelength": np.asarray(wavelengths, dtype=float),
        "omega": np.empty(n),
        "k0": np.empty(n),
        "eps": np.empty(n, dtype=complex),
        "k_spp": np.full(n, np.nan, dtype=complex),
        "n_eff": np.full(n, np.nan),
        "L": np.full(n, np.nan),
        "delta_d": np.full(n, np.nan),
        "delta_m": np.full(n, np.nan),
        "supported": np.zeros(n, dtype=bool),
    }
    for i, lam in enumerate(wavelengths):
        omega = omega_from_wavelength(lam)
        out["omega"][i] = omega
        out["k0"][i] = omega / C0
        out["eps"][i] = drude_permittivity(omega)
        metrics = bound_mode_metrics(silver_interface(omega), eps_d)
        if metrics is not None:
            out["supported"][i] = True
            out["k_spp"][i] = metrics["k_spp"]
            out["n_eff"][i] = metrics["n_eff"]
            out["L"][i] = metrics["L"]
            out["delta_d"][i] = metrics["delta_d"]
            out["delta_m"][i] = metrics["delta_m"]
    return out


def sweep_design_space(
    lambda0: float,
    eps_t_re: np.ndarray,
    eps_n_re: np.ndarray,
    im_eps_t: float = IM_EPS_T,
    im_eps_n: float = IM_EPS_N,
    eps_d: float = EPS_D,
) -> dict:
    """
    Map the uniaxial (Re ε_t, Re ε_n) design plane at fixed frequency.

    Grids are indexed ``[i_n, i_t]`` (rows = ε_n, columns = ε_t) so they plot
    directly with ``pcolormesh(eps_t_re, eps_n_re, grid)``. ``n_eff`` and
    ``L`` are NaN outside the supported region.
    """
    omega = omega_from_wavelength(lambda0)
    shape = (len(eps_n_re), len(eps_t_re))
    supported = np.zeros(shape, dtype=bool)
    n_eff = np.full(shape, np.nan)
    length = np.full(shape, np.nan)
    for i, en in enumerate(eps_n_re):
        for j, et in enumerate(eps_t_re):
            m = uniaxial_interface(et + 1j * im_eps_t, en + 1j * im_eps_n, omega)
            metrics = bound_mode_metrics(m, eps_d)
            if metrics is not None:
                supported[i, j] = True
                n_eff[i, j] = metrics["n_eff"]
                length[i, j] = metrics["L"]
    return {
        "eps_t_re": np.asarray(eps_t_re, dtype=float),
        "eps_n_re": np.asarray(eps_n_re, dtype=float),
        "supported": supported,
        "n_eff": n_eff,
        "L": length,
        "lambda0": lambda0,
        "eps_d": eps_d,
        "im_eps_t": im_eps_t,
        "im_eps_n": im_eps_n,
    }


def sweep_anisotropy_cut(
    lambda0: float,
    eps_n_re: np.ndarray,
    eps_t: complex = CUT_EPS_T,
    im_eps_n: float = IM_EPS_N,
    eps_d: float = EPS_D,
) -> dict:
    """1-D cut across the design space: ε_n sweeps at fixed complex ε_t."""
    omega = omega_from_wavelength(lambda0)
    n = len(eps_n_re)
    out = {
        "eps_n_re": np.asarray(eps_n_re, dtype=float),
        "eps_t": eps_t,
        "n_eff": np.full(n, np.nan),
        "L": np.full(n, np.nan),
        "delta_d": np.full(n, np.nan),
        "delta_m": np.full(n, np.nan),
        "supported": np.zeros(n, dtype=bool),
    }
    for i, en in enumerate(eps_n_re):
        m = uniaxial_interface(eps_t, en + 1j * im_eps_n, omega)
        metrics = bound_mode_metrics(m, eps_d)
        if metrics is not None:
            out["supported"][i] = True
            out["n_eff"][i] = metrics["n_eff"]
            out["L"][i] = metrics["L"]
            out["delta_d"][i] = metrics["delta_d"]
            out["delta_m"][i] = metrics["delta_m"]
    return out


def support_boundaries(eps_n_re: np.ndarray, supported: np.ndarray) -> list[float]:
    """Re ε_n midpoints where the supported/unsupported flag flips."""
    flips = np.flatnonzero(np.diff(supported.astype(int)) != 0)
    return [0.5 * (eps_n_re[i] + eps_n_re[i + 1]) for i in flips]


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


def plot_silver_dispersion(res: dict, path: Path, eps_d: float = EPS_D) -> None:
    """ω–k diagram: Drude-silver SPP branch, light line, ω_sp asymptote."""
    sup = res["supported"]
    k_um = res["k_spp"].real[sup] * 1e-6  # rad/µm
    hw = HBAR_EVS * res["omega"][sup]  # eV

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    k_axis = np.linspace(0.0, 1.05 * k_um.max(), 200)
    ax.plot(
        k_axis,
        HBAR_EVS * C0 * (k_axis * 1e6) / math.sqrt(eps_d),
        color=C_LIGHT,
        linestyle="--",
        linewidth=1.5,
        label=r"light line $\omega = ck/\sqrt{\varepsilon_d}$",
    )
    ax.plot(k_um, hw, color=C_SPP, label="SPP branch (Drude Ag / air)")

    hw_sp = surface_plasmon_energy_ev(eps_d)
    ax.axhline(hw_sp, color=C_MARK, linestyle=":", linewidth=1.5)
    ax.annotate(
        r"$\hbar\omega_{sp} = \hbar\omega_p/\sqrt{\varepsilon_\infty + \varepsilon_d}"
        rf" = {hw_sp:.2f}\,$eV",
        xy=(0.03 * k_axis.max(), hw_sp),
        xytext=(0.03 * k_axis.max(), hw_sp - 0.28),
        color=C_MARK,
        fontsize=10,
    )

    # Mark the Johnson & Christy anchor wavelength.
    i633 = int(np.argmin(np.abs(res["wavelength"] - 633e-9)))
    if res["supported"][i633]:
        ax.plot(
            res["k_spp"].real[i633] * 1e-6,
            HBAR_EVS * res["omega"][i633],
            marker="o",
            markersize=7,
            color=C_MARK,
            linestyle="none",
            label=r"$\lambda_0 = 633\,$nm (J&C anchor)",
        )

    lam_min_nm = res["wavelength"].min() * 1e9
    lam_max_nm = res["wavelength"].max() * 1e9
    ax.set_xlabel(r"Re $k_{\mathrm{spp}}$ (rad/µm)")
    ax.set_ylabel(r"photon energy $\hbar\omega$ (eV)")
    ax.set_title(
        "SPP dispersion: Drude silver / air "
        rf"($\lambda_0 = {lam_min_nm:.0f}$–${lam_max_nm:.0f}\,$nm)"
    )
    ax.set_xlim(0.0, k_axis.max())
    ax.set_ylim(0.0, max(1.12 * hw.max(), hw_sp + 0.45))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_silver_length_scales(res: dict, path: Path) -> None:
    """L(λ) and the two penetration depths δ_d(λ), δ_m(λ) for Drude silver."""
    sup = res["supported"]
    lam_nm = res["wavelength"][sup] * 1e9

    fig, (ax_l, ax_d) = plt.subplots(1, 2, figsize=(10.5, 4.3), sharex=True)

    ax_l.semilogy(lam_nm, res["L"][sup] * 1e6, color=C_SPP)
    ax_l.set_xlabel(r"free-space wavelength $\lambda_0$ (nm)")
    ax_l.set_ylabel(r"$L = 1/(2\,\mathrm{Im}\,k_{\mathrm{spp}})$ (µm)")
    ax_l.set_title("Propagation length")

    ax_d.semilogy(lam_nm, res["delta_d"][sup] * 1e9, color=C_DIEL)
    ax_d.semilogy(lam_nm, res["delta_m"][sup] * 1e9, color=C_METAL)
    # Direct labels beside each curve, kept inside the axes.
    i_mid = int(0.55 * (len(lam_nm) - 1))
    ax_d.annotate(
        r"$\delta_d$ (air)",
        xy=(lam_nm[i_mid], res["delta_d"][sup][i_mid] * 1e9),
        xytext=(0, -16),
        textcoords="offset points",
        ha="center",
        va="top",
        color=C_DIEL,
    )
    ax_d.annotate(
        r"$\delta_m$ (silver)",
        xy=(lam_nm[i_mid], res["delta_m"][sup][i_mid] * 1e9),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        color=C_METAL,
    )
    ax_d.set_xlabel(r"free-space wavelength $\lambda_0$ (nm)")
    ax_d.set_ylabel(r"penetration depth $1/\mathrm{Re}\,\kappa$ (nm)")
    ax_d.set_title("Field penetration depths")

    fig.suptitle("Drude silver / air SPP length scales", y=1.0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _lossless_support_boundaries(ax, maps: dict) -> None:
    """Dashed lossless-limit boundaries of the support region (see docstring)."""
    et_lo = maps["eps_t_re"].min()
    en_lo, en_hi = maps["eps_n_re"].min(), maps["eps_n_re"].max()
    eps_d = maps["eps_d"]
    style = {"color": "black", "linestyle": "--", "linewidth": 1.1, "alpha": 0.75}
    # ε_t = 0: no bound mode for ε_t > 0 (matching needs opposite-sign ε).
    ax.plot([0.0, 0.0], [en_lo, en_hi], **style)
    # ε_n = ε_d: bound-mode onset for ε_t < 0.
    ax.plot([et_lo, 0.0], [eps_d, eps_d], **style)
    # Resonance hyperbola ε_t ε_n = ε_d² bounding the both-negative branch.
    et = np.linspace(et_lo, eps_d**2 / en_lo, 200)
    ax.plot(et, eps_d**2 / et, **style)


def plot_design_space(maps: dict, path: Path) -> None:
    """Centrepiece: support map, effective index and propagation length maps."""
    et, en = maps["eps_t_re"], maps["eps_n_re"]
    lam0_nm = maps["lambda0"] * 1e9

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))
    for ax in axes:
        ax.grid(False)
        ax.set_xlabel(r"Re $\varepsilon_t$ (in-plane)")
        ax.set_ylabel(r"Re $\varepsilon_n$ (normal)")

    # (a) bound-mode existence.
    ax = axes[0]
    cmap_bin = ListedColormap(["#ececec", "#4477aa"])
    pm = ax.pcolormesh(
        et, en, maps["supported"].astype(float), cmap=cmap_bin, vmin=0, vmax=1,
        shading="nearest", rasterized=True,
    )
    cbar = fig.colorbar(pm, ax=ax, ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(["no bound\nmode", "SPP\nsupported"], fontsize=9)
    _lossless_support_boundaries(ax, maps)
    ax.annotate(
        r"$\varepsilon_n = \varepsilon_d$", xy=(-9.6, 1.25), fontsize=9, color="black"
    )
    ax.annotate(
        r"$\varepsilon_t \varepsilon_n = \varepsilon_d^2$",
        xy=(-9.6, -1.6),
        fontsize=9,
        color="black",
    )
    ax.plot(
        *BENCHMARK_POINT, marker="*", markersize=13, color=C_MARK,
        markeredgecolor="black", markeredgewidth=0.5, linestyle="none",
    )
    ax.annotate(
        "benchmark point", xy=BENCHMARK_POINT, xytext=(6, 6),
        textcoords="offset points", fontsize=9, color=C_MARK,
    )
    ax.set_title(r"(a) bound TM mode exists")

    # (b) effective index, masked to the supported region. The range is
    # narrow (~1–2.5 away from the resonance), so a linear scale clipped at
    # the 99th percentile reads better than a log one.
    ax = axes[1]
    n_eff = maps["n_eff"]
    finite = np.isfinite(n_eff)
    pm = ax.pcolormesh(
        et, en, n_eff, cmap="viridis",
        vmin=1.0, vmax=np.nanpercentile(n_eff[finite], 99),
        shading="nearest", rasterized=True,
    )
    fig.colorbar(pm, ax=ax, label=r"Re $k_{\mathrm{spp}}/k_0$")
    ax.set_title(r"(b) mode confinement Re $k_{\mathrm{spp}}/k_0$")

    # (c) propagation length, masked, log scale.
    ax = axes[2]
    length_um = maps["L"] * 1e6
    finite = np.isfinite(length_um)
    norm = LogNorm(
        vmin=np.nanpercentile(length_um[finite], 1),
        vmax=np.nanpercentile(length_um[finite], 99),
    )
    pm = ax.pcolormesh(et, en, length_um, cmap="magma", norm=norm,
                       shading="nearest", rasterized=True)
    fig.colorbar(pm, ax=ax, label=r"$L$ (µm)")
    ax.set_title(r"(c) propagation length $L$")

    # Tie panels (b, c) to the 1-D anisotropy cut. The label runs along the
    # line over the both-negative region, where both colour maps are dark.
    for ax in axes[1:]:
        ax.axvline(CUT_EPS_T.real, color="white", linestyle="--", linewidth=1.1)
        ax.annotate(
            "anisotropy cut",
            xy=(CUT_EPS_T.real, en.min() + 0.15),
            xytext=(-4, 0),
            textcoords="offset points",
            fontsize=8,
            color="white",
            rotation=90,
            ha="right",
            va="bottom",
        )

    fig.suptitle(
        rf"Uniaxial metamaterial / air SPP design space at $\lambda_0 = {lam0_nm:.0f}\,$nm "
        rf"(Im $\varepsilon_t = {maps['im_eps_t']}$, Im $\varepsilon_n = {maps['im_eps_n']}$)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def plot_anisotropy_cut(cut: dict, path: Path, lambda0: float = LAMBDA0_MAP) -> None:
    """1-D cut: mode metrics vs Re ε_n at fixed ε_t, boundaries annotated."""
    en = cut["eps_n_re"]
    sup = cut["supported"]
    boundaries = support_boundaries(en, sup)

    fig, (ax_k, ax_l, ax_d) = plt.subplots(3, 1, figsize=(7.2, 9.2), sharex=True)

    def shade_unsupported(ax):
        # Contiguous unsupported runs as grey bands.
        edges = [en[0], *boundaries, en[-1]]
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            mid = 0.5 * (lo + hi)
            if not sup[int(np.argmin(np.abs(en - mid)))]:
                ax.axvspan(lo, hi, color="#dddddd", alpha=0.6, zorder=0)
        for b in boundaries:
            ax.axvline(b, color=C_LIGHT, linestyle=":", linewidth=1.2)

    for ax in (ax_k, ax_l, ax_d):
        shade_unsupported(ax)

    ax_k.semilogy(en, np.where(sup, cut["n_eff"], np.nan), color=C_SPP)
    ax_k.set_ylabel(r"Re $k_{\mathrm{spp}}/k_0$")
    ax_k.set_title("Mode confinement (diverges at the resonance)")

    ax_l.semilogy(en, np.where(sup, cut["L"] * 1e6, np.nan), color=C_SPP)
    ax_l.set_ylabel(r"$L$ (µm)")
    ax_l.set_title("Propagation length")

    ax_d.semilogy(en, np.where(sup, cut["delta_d"] * 1e9, np.nan), color=C_DIEL)
    ax_d.semilogy(en, np.where(sup, cut["delta_m"] * 1e9, np.nan), color=C_METAL)
    idx = np.flatnonzero(sup)
    ax_d.annotate(
        r"$\delta_d$ (air)",
        xy=(en[idx[-1]], cut["delta_d"][idx[-1]] * 1e9),
        xytext=(-8, 8),
        textcoords="offset points",
        ha="right",
        color=C_DIEL,
    )
    ax_d.annotate(
        r"$\delta_m$ (metamaterial)",
        xy=(en[idx[-1]], cut["delta_m"][idx[-1]] * 1e9),
        xytext=(-8, -14),
        textcoords="offset points",
        ha="right",
        color=C_METAL,
    )
    ax_d.set_ylabel(r"penetration depth (nm)")
    ax_d.set_xlabel(r"Re $\varepsilon_n$ (normal component)")
    ax_d.set_title("Field penetration depths")

    # Annotate the physics of each boundary on the top panel.
    eps_d = EPS_D
    resonance = (eps_d**2 / cut["eps_t"]).real
    for b in boundaries:
        if abs(b - resonance) < 0.5:
            text = (
                r"resonance $\varepsilon_t\varepsilon_n = \varepsilon_d^2$"
                + rf" ($\varepsilon_n \approx {resonance:.2f}$)"
            )
            xytext = (5, -6)  # inside the grey no-mode band, clear of the peak
            ha = "left"
        else:
            text = r"onset $\varepsilon_n \approx \varepsilon_d$"
            xytext = (5, -6)
            ha = "left"
        ax_k.annotate(
            text,
            xy=(b, ax_k.get_ylim()[1]),
            xytext=xytext,
            textcoords="offset points",
            fontsize=8.5,
            color=C_LIGHT,
            rotation=90,
            va="top",
            ha=ha,
        )

    et = cut["eps_t"]
    fig.suptitle(
        rf"Anisotropy cut at $\varepsilon_t = {et.real:g} + {et.imag:g}i$, "
        rf"$\lambda_0 = {lambda0 * 1e9:.0f}\,$nm, $\varepsilon_d = {EPS_D:g}$ "
        "(grey: no bound mode)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)


# ================================================================== summary
def _finite_range(values: np.ndarray) -> list[float]:
    finite = values[np.isfinite(values)]
    return [float(finite.min()), float(finite.max())]


def build_summary(silver: dict, maps: dict, cut: dict, n_points: int) -> dict:
    """Key numbers of the three studies, JSON-serialisable."""
    i633 = int(np.argmin(np.abs(silver["wavelength"] - 633e-9)))
    eps_633 = silver["eps"][i633]
    rel_dev = abs(eps_633 - EPS_AG_JC_633NM) / abs(EPS_AG_JC_633NM)
    return {
        "drude_model": {
            "eps_inf": EPS_INF,
            "hbar_omega_p_eV": HBAR_OMEGA_P_EV,
            "hbar_gamma_eV": HBAR_GAMMA_EV,
        },
        "silver_at_633nm": {
            "eps_drude": [eps_633.real, eps_633.imag],
            "eps_johnson_christy": [EPS_AG_JC_633NM.real, EPS_AG_JC_633NM.imag],
            "relative_deviation_vs_jc": float(rel_dev),
            "n_eff": float(silver["n_eff"][i633]),
            "propagation_length_um": float(silver["L"][i633] * 1e6),
            "penetration_depth_dielectric_nm": float(silver["delta_d"][i633] * 1e9),
            "penetration_depth_metal_nm": float(silver["delta_m"][i633] * 1e9),
        },
        "surface_plasmon_asymptote": {
            "hbar_omega_sp_eV": surface_plasmon_energy_ev(),
            "wavelength_nm": 2.0 * math.pi * C0 * HBAR_EVS / surface_plasmon_energy_ev() * 1e9,
        },
        "silver_sweep": {
            "wavelength_range_nm": [
                float(silver["wavelength"].min() * 1e9),
                float(silver["wavelength"].max() * 1e9),
            ],
            "supported_fraction": float(silver["supported"].mean()),
            "propagation_length_um_range": _finite_range(silver["L"] * 1e6),
        },
        "design_space": {
            "lambda0_nm": float(maps["lambda0"] * 1e9),
            "eps_d": maps["eps_d"],
            "im_eps_t": maps["im_eps_t"],
            "im_eps_n": maps["im_eps_n"],
            "eps_t_re_range": _finite_range(maps["eps_t_re"]),
            "eps_n_re_range": _finite_range(maps["eps_n_re"]),
            "supported_fraction": float(maps["supported"].mean()),
            "n_eff_range": _finite_range(maps["n_eff"]),
            "propagation_length_um_range": _finite_range(maps["L"] * 1e6),
        },
        "anisotropy_cut": {
            "eps_t": [cut["eps_t"].real, cut["eps_t"].imag],
            "supported_fraction": float(cut["supported"].mean()),
            "support_boundaries_re_eps_n": [
                float(b) for b in support_boundaries(cut["eps_n_re"], cut["supported"])
            ],
        },
        "n_points": n_points,
    }


# ===================================================================== main
def main(argv: Optional[list[str]] = None) -> dict:
    parser = argparse.ArgumentParser(
        description="Analytical SPP dispersion study (no training): Drude silver "
        "and uniaxial metamaterial design-space maps."
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
        default=201,
        help="points per sweep axis: wavelength sweep, each map axis, cut (default: 201)",
    )
    args = parser.parse_args(argv)
    figures_dir: Path = args.figures_dir
    n: int = args.n_points
    figures_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    # 1. Drude silver dispersion sweep.
    wavelengths = np.linspace(LAMBDA_MIN, LAMBDA_MAX, n)
    silver = sweep_silver_dispersion(wavelengths)
    plot_silver_dispersion(silver, figures_dir / "spp_dispersion_silver.png")
    plot_silver_length_scales(silver, figures_dir / "spp_length_scales_silver.png")

    # 2. Uniaxial design-space maps at 633 nm.
    maps = sweep_design_space(
        LAMBDA0_MAP,
        np.linspace(*EPS_T_RANGE, n),
        np.linspace(*EPS_N_RANGE, n),
    )
    plot_design_space(maps, figures_dir / "metamaterial_design_space.png")

    # 3. Anisotropy cut across the support boundaries.
    cut = sweep_anisotropy_cut(LAMBDA0_MAP, np.linspace(*EPS_N_RANGE, n))
    plot_anisotropy_cut(cut, figures_dir / "anisotropy_cut.png")

    summary = build_summary(silver, maps, cut, n)
    summary_path = figures_dir / "dispersion_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    ag = summary["silver_at_633nm"]
    print(f"Figures written to {figures_dir}")
    print(
        "Drude ε(633 nm) = {:.3f} + {:.3f}i  (J&C: {:.2f} + {:.2f}i, |Δε|/|ε_JC| = {:.1%})".format(
            ag["eps_drude"][0],
            ag["eps_drude"][1],
            EPS_AG_JC_633NM.real,
            EPS_AG_JC_633NM.imag,
            ag["relative_deviation_vs_jc"],
        )
    )
    print(
        "SPP @ 633 nm (Drude Ag/air): n_eff = {:.4f}, L = {:.1f} µm, "
        "δ_d = {:.0f} nm, δ_m = {:.1f} nm".format(
            ag["n_eff"],
            ag["propagation_length_um"],
            ag["penetration_depth_dielectric_nm"],
            ag["penetration_depth_metal_nm"],
        )
    )
    print(
        "Surface-plasmon asymptote: ħω_sp = {:.2f} eV (λ ≈ {:.0f} nm)".format(
            summary["surface_plasmon_asymptote"]["hbar_omega_sp_eV"],
            summary["surface_plasmon_asymptote"]["wavelength_nm"],
        )
    )
    print(
        "Design space @ 633 nm: {:.1%} of the (ε_t, ε_n) window supports a bound SPP".format(
            summary["design_space"]["supported_fraction"]
        )
    )
    print(
        "Anisotropy cut (ε_t = −4 + 0.2i): support boundaries at Re ε_n ≈ "
        + ", ".join(
            f"{b:.2f}" for b in summary["anisotropy_cut"]["support_boundaries_re_eps_n"]
        )
    )
    print(f"Summary written to {summary_path}")
    return summary


if __name__ == "__main__":
    main()
