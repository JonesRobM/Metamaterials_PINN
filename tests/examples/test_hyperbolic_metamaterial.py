"""
Tests for examples/hyperbolic_metamaterial.py (analytics only, no training).

The load-bearing claim of the study is its *recommended band*, so most of what
is checked here re-derives that band's properties independently — through
:class:`src.physics.metamaterial.MetamaterialProperties` rather than the
example's vectorised transcription of it.
"""

import json
import math

import numpy as np
import pytest

from examples import hyperbolic_metamaterial as hmm
from src.constants import C0
from src.effective_medium import hmm_permittivities, omega_from_wavelength
from src.physics.metamaterial import MetamaterialProperties

FIGURE_NAMES = [
    "hmm_permittivities.png",
    "hmm_spp_dispersion.png",
    "hmm_fill_fraction_map.png",
]


def reference_mode(omega: float, fill_fraction: float = hmm.FILL_FRACTION) -> dict:
    """
    Independent bound-mode check straight from ``MetamaterialProperties``.

    Mirrors ``examples/dispersion_analysis.py``: ``is_spp_supported`` plus the
    non-radiative gate ``Re k_spp > √ε_d k₀``. Returns ``{}`` when unbound.
    """
    eps_t, eps_n = hmm_permittivities(omega, fill_fraction, hmm.EPS_D2)
    # Constructor order: eps_parallel is along the optical axis ('z'), i.e. ε_n.
    material = MetamaterialProperties(
        complex(eps_n), complex(eps_t), optical_axis="z", omega=omega
    )
    if not material.is_spp_supported(eps_dielectric=hmm.EPS_D):
        return {}
    k = material.spp_wavevector(eps_dielectric=hmm.EPS_D)
    if k.real <= math.sqrt(hmm.EPS_D) * material.k0:
        return {}
    return {
        "k_spp": k,
        "n_eff": k.real / material.k0,
        "L": material.propagation_length(eps_dielectric=hmm.EPS_D),
        "delta_d": material.penetration_depth_dielectric(eps_dielectric=hmm.EPS_D),
        "delta_m": material.penetration_depth_metamaterial(eps_dielectric=hmm.EPS_D),
    }


@pytest.fixture(scope="module")
def search():
    """The sweep the band recommendation is made on, at the example's defaults."""
    omegas = np.linspace(
        float(omega_from_wavelength(hmm.LAMBDA_SEARCH[1])),
        float(omega_from_wavelength(hmm.LAMBDA_SEARCH[0])),
        401,
    )
    return hmm.sweep_spectrum(omegas)


@pytest.fixture(scope="module")
def band(search):
    return hmm.recommend_band(search)


# ------------------------------------------- vectorised transcription vs the class
class TestSppMetricsAgreesWithTheClass:
    """``spp_metrics`` must reproduce ``MetamaterialProperties`` exactly."""

    @pytest.mark.parametrize("lam_nm", [450.0, 500.0, 633.0, 800.0, 1200.0])
    def test_bound_points(self, lam_nm):
        omega = float(omega_from_wavelength(lam_nm * 1e-9))
        eps_t, eps_n = hmm_permittivities(omega, hmm.FILL_FRACTION, hmm.EPS_D2)
        metrics = hmm.spp_metrics(eps_t, eps_n, omega)
        reference = reference_mode(omega)
        assert bool(metrics["bound"]) is bool(reference)
        assert complex(metrics["k_spp"]) == pytest.approx(reference["k_spp"], rel=1e-12)
        assert float(metrics["L"]) == pytest.approx(reference["L"], rel=1e-12)
        assert 1.0 / complex(metrics["kappa_d"]).real == pytest.approx(
            reference["delta_d"], rel=1e-12
        )
        assert 1.0 / complex(metrics["kappa_m"]).real == pytest.approx(
            reference["delta_m"], rel=1e-12
        )

    @pytest.mark.parametrize("lam_nm", [300.0, 350.0, 405.0])
    def test_unbound_points_agree_too(self, lam_nm):
        """Above the type-II onset neither route finds a bound mode."""
        omega = float(omega_from_wavelength(lam_nm * 1e-9))
        eps_t, eps_n = hmm_permittivities(omega, hmm.FILL_FRACTION, hmm.EPS_D2)
        assert not bool(hmm.spp_metrics(eps_t, eps_n, omega)["bound"])
        assert reference_mode(omega) == {}

    def test_unbound_entries_are_masked(self, search):
        unbound = ~search["bound"]
        if unbound.any():
            assert np.all(np.isnan(search["n_eff"][unbound]))
            assert np.all(np.isnan(search["L"][unbound]))

    def test_hmm_interface_uses_the_constructor_order(self):
        """eps_parallel is along the optical axis, so it must receive ε_n."""
        omega = float(omega_from_wavelength(633e-9))
        eps_t, eps_n = hmm_permittivities(omega, hmm.FILL_FRACTION, hmm.EPS_D2)
        material = hmm.hmm_interface(omega)
        assert material.eps_along("z") == pytest.approx(complex(eps_n), rel=1e-15)
        assert material.eps_along("x") == pytest.approx(complex(eps_t), rel=1e-15)


# ------------------------------------------------------------ nonlinearity metric
class TestLinearFitResidual:
    def test_exactly_zero_for_a_straight_line(self):
        """The fixed-ε idealisation k = n ω/c must score 0%."""
        omega = np.linspace(2e15, 4e15, 51)
        pct, residual = hmm.linear_fit_residual(omega, 1.3 * omega / C0)
        assert pct == pytest.approx(0.0, abs=1e-9)
        assert np.abs(residual).max() < 1e-6 * (1.3 * omega / C0).max()

    def test_detects_curvature(self):
        omega = np.linspace(1.0, 2.0, 101)
        pct, _ = hmm.linear_fit_residual(omega, omega**2)
        assert pct > 5.0

    def test_general_fit_recovers_a_nonzero_intercept(self):
        """Regression guard for a silent rank truncation.

        Fitting the raw design matrix ``[ω, 1]`` at optical ω (~3e15) puts the
        intercept column's singular value below ``lstsq``'s default cutoff, so
        it is dropped without warning and the general fit collapses onto the
        through-origin fit. Here the data lie exactly on a line with a large
        non-zero intercept: a genuine two-parameter fit has zero residual,
        while the truncated fit does not.
        """
        omega = np.linspace(2.1e15, 4.2e15, 101)
        intercept = 7.0e6
        k = 3.4e-9 * omega + intercept

        pct_general, residual = hmm.linear_fit_residual(omega, k, through_origin=False)
        assert pct_general == pytest.approx(0.0, abs=1e-6)
        assert np.abs(residual).max() < 1e-6 * float(np.abs(k).max())

        # ...whereas the through-origin reference cannot absorb the intercept
        pct_origin, _ = hmm.linear_fit_residual(omega, k, through_origin=True)
        assert pct_origin > 1.0

    def test_curvature_is_never_larger_than_departure_from_fixed_eps(self):
        """The general line has one more free parameter, so it always fits at
        least as well as the line through the origin."""
        omega = np.linspace(2.1e15, 4.2e15, 101)
        k = 3.4e-9 * omega + 7.0e6 + 4.0e-25 * (omega - 3e15) ** 2
        pct_origin, _ = hmm.linear_fit_residual(omega, k, through_origin=True)
        pct_general, _ = hmm.linear_fit_residual(omega, k, through_origin=False)
        assert pct_general <= pct_origin + 1e-9

    def test_flat_input_is_not_a_division_by_zero(self):
        pct, _ = hmm.linear_fit_residual(np.linspace(1.0, 2.0, 5), np.ones(5))
        assert pct == 0.0


# ------------------------------------------------------------- recommended band
class TestRecommendedBand:
    def test_band_exists(self, band):
        assert band is not None
        lo, hi = band["omega"]
        assert 0.0 < lo < hi

    def test_edges_are_inside_the_search_window(self, band):
        lam_lo, lam_hi = band["wavelength_nm"]
        assert hmm.LAMBDA_SEARCH[0] * 1e9 - 1e-6 <= lam_lo < lam_hi
        assert lam_hi <= hmm.LAMBDA_SEARCH[1] * 1e9 + 1e-6

    def test_bound_mode_at_the_endpoints_and_midpoint(self, band):
        """Verified independently via MetamaterialProperties + the light-line gate."""
        lo, hi = band["omega"]
        for omega in (lo, 0.5 * (lo + hi), hi):
            mode = reference_mode(float(omega))
            assert mode, f"no bound mode at omega = {omega:.4e}"
            assert mode["n_eff"] > 1.0
            assert mode["L"] > 0.0

    def test_bound_mode_everywhere_inside_the_band(self, band):
        """Not just at three points: the whole interval must be bound."""
        lo, hi = band["omega"]
        for omega in np.linspace(lo, hi, 25):
            assert reference_mode(float(omega)), f"unbound at omega = {omega:.4e}"

    def test_reported_k_and_kappa_ranges_are_right(self, band):
        lo, hi = band["omega"]
        modes = [reference_mode(float(w)) for w in np.linspace(lo, hi, 41)]
        n_eff = np.array([m["n_eff"] for m in modes])
        assert band["n_eff_range"][0] == pytest.approx(n_eff.min(), rel=5e-3)
        assert band["n_eff_range"][1] == pytest.approx(n_eff.max(), rel=5e-3)
        kappa = np.concatenate(
            [1.0 / np.array([m["delta_d"] for m in modes]),
             1.0 / np.array([m["delta_m"] for m in modes])]
        )
        assert band["kappa_spread"] == pytest.approx(kappa.max() / kappa.min(), rel=5e-2)

    def test_nonlinearity_exceeds_the_floor(self, band):
        assert band["nonlinearity_percent"] >= hmm.MIN_NONLINEARITY_PCT
        # Strict curvature removes the intercept, so it is the smaller number
        assert 0.0 < band["curvature_percent"] <= band["nonlinearity_percent"]
        # The whole point of the study: comfortably above a straight line.
        assert band["nonlinearity_percent"] > 20.0

    def test_nonlinearity_recomputed_from_the_class(self, band):
        """Re-derive the metric from MetamaterialProperties on a fresh grid."""
        lo, hi = band["omega"]
        omegas = np.linspace(lo, hi, 121)
        k_real = np.array([reference_mode(float(w))["k_spp"].real for w in omegas])
        pct, _ = hmm.linear_fit_residual(omegas, k_real)
        assert pct == pytest.approx(band["nonlinearity_percent"], rel=0.05)

    def test_k_spp_varies_appreciably(self, band):
        lo, hi = band["n_eff_range"]
        assert hi / lo >= hmm.MIN_NEFF_RATIO

    def test_decay_scales_stay_within_an_order_of_magnitude(self, band):
        assert band["kappa_spread"] <= hmm.MAX_KAPPA_SPREAD * (1.0 + 1e-9)
        short, long = band["decay_length_range_nm"]
        assert 0.0 < short < long
        assert long / short == pytest.approx(band["kappa_spread"], rel=1e-9)

    def test_mode_propagates_many_wavelengths(self, band):
        assert band["L_over_lambda_spp_range"][0] >= hmm.MIN_L_OVER_LAMBDA_SPP

    def test_band_is_wide_enough_to_condition_on(self, band):
        assert band["relative_width"] > 0.3

    def test_omega_over_omega_ref_is_consistent(self, band):
        lo, hi = band["omega"]
        assert band["omega_over_omega_ref"][0] == pytest.approx(lo / hmm.OMEGA_REF, rel=1e-12)
        assert band["omega_over_omega_ref"][1] == pytest.approx(hi / hmm.OMEGA_REF, rel=1e-12)

    def test_effective_medium_period_is_fabricable(self, band):
        """A few-nm-scale period: the EMT caveat translated into a real number."""
        assert 5.0 < band["max_layer_period_nm"] < 100.0

    def test_returns_none_when_nothing_qualifies(self, search):
        assert hmm.recommend_band(search, min_nonlinearity_pct=1e6) is None


@pytest.fixture(scope="module")
def scan():
    """The band recommendation re-run over a spread of fill fractions."""
    omegas = np.linspace(
        float(omega_from_wavelength(hmm.LAMBDA_SEARCH[1])),
        float(omega_from_wavelength(hmm.LAMBDA_SEARCH[0])),
        201,
    )
    return hmm.band_vs_fill_fraction(omegas, (0.2, 0.3, 0.4, 0.6))


class TestBandVsFillFraction:
    def test_one_row_per_fill_fraction(self, scan):
        assert [row["fill_fraction"] for row in scan] == [0.2, 0.3, 0.4, 0.6]

    def test_bandwidth_falls_away_at_high_filling(self, scan):
        """The design point is a shallow optimum; heavy metal filling loses it."""
        widths = {
            row["fill_fraction"]: (
                0.0 if row["band"] is None else row["band"]["relative_width"]
            )
            for row in scan
        }
        assert widths[0.2] > widths[0.4] > widths[0.6]
        assert widths[0.3] > 0.5

    def test_band_moves_to_the_blue_with_filling(self, scan):
        """More metal ⇒ the type-II onset (and the band) shifts to shorter λ."""
        red_edges = [
            row["band"]["wavelength_nm"][1] for row in scan if row["band"] is not None
        ]
        assert red_edges == sorted(red_edges, reverse=True)


# ------------------------------------------------------------ spectral structure
@pytest.fixture(scope="module")
def sweep():
    """The wide spectral survey used by figures 1 and 2."""
    omegas = hmm.omega_from_photon_energy_ev(np.linspace(hmm.EV_MIN, hmm.EV_MAX, 601))
    return hmm.sweep_spectrum(omegas)


@pytest.fixture(scope="module")
def maps():
    """A coarse (f, ω) map, as figure 3 builds."""
    return hmm.sweep_fill_fraction(
        np.linspace(0.05, 0.95, 19),
        hmm.omega_from_photon_energy_ev(np.linspace(hmm.EV_MIN, hmm.EV_MAX, 61)),
    )


class TestSpectrum:
    def test_type_ii_covers_the_visible_and_near_ir(self, sweep):
        """At f = 0.3 the multilayer is type-II from ~408 nm out to the IR."""
        lam_nm = 1e9 * 2.0 * np.pi * C0 / sweep["omega"]
        red = lam_nm > 450.0
        assert np.all(sweep["anisotropy"][red] == "type-II")

    def test_bound_modes_coincide_with_the_type_ii_window(self, sweep):
        """ε_t < 0 with ε_n > ε_d is exactly where the bound TM mode lives."""
        type_ii = sweep["anisotropy"] == "type-II"
        assert np.all(sweep["bound"][type_ii])
        assert not np.any(sweep["bound"][~type_ii])

    def test_all_four_classes_are_ordered_in_frequency(self, sweep):
        spans = hmm.class_spans(sweep["omega"], sweep["anisotropy"])
        names = [name for _, _, name in spans]
        assert names[0] == "type-II"  # lowest frequency
        assert "type-I" in names
        assert names.count("elliptic-dielectric") >= 1

    def test_bound_spans_is_a_single_contiguous_run(self, sweep):
        spans = hmm.bound_spans(sweep["omega"], sweep["bound"])
        assert len(spans) == 1
        lo, hi = spans[0]
        assert lo < hi

    def test_refine_grid_resolves_the_resonance(self):
        """The narrow Im ε_n line must survive coarse sampling."""
        uniform = hmm.omega_from_photon_energy_ev(np.linspace(hmm.EV_MIN, hmm.EV_MAX, 51))
        transitions = hmm.transition_frequencies(
            hmm.FILL_FRACTION, hmm.EPS_D2,
            omega_range=(float(uniform.min()), float(uniform.max())),
        )
        refined = hmm.refine_grid(uniform, transitions)
        assert refined.size > uniform.size
        assert np.all(np.diff(refined) > 0)
        peak = hmm.pole_resonance(transitions)["im_eps_n"]
        _, eps_n = hmm_permittivities(refined, hmm.FILL_FRACTION, hmm.EPS_D2)
        assert eps_n.imag.max() > 0.5 * peak

    def test_pole_resonance_is_passive_and_finite(self):
        transitions = hmm.transition_frequencies(hmm.FILL_FRACTION, hmm.EPS_D2)
        resonance = hmm.pole_resonance(transitions)
        assert resonance["im_eps_n"] > 50.0
        assert np.isfinite(resonance["im_eps_n"])
        assert 250.0 < resonance["wavelength_nm"] < 350.0


class TestFillFractionSweep:
    def test_shapes_and_indexing(self, maps):
        assert maps["class_index"].shape == (19, 61)
        assert maps["n_eff"].shape == (19, 61)
        assert maps["class_index"].min() >= 0

    def test_rows_match_a_single_fill_fraction_sweep(self, maps):
        row = 7
        single = hmm.sweep_spectrum(maps["omega"], float(maps["fills"][row]))
        np.testing.assert_allclose(
            maps["n_eff"][row], single["n_eff"], rtol=1e-12, equal_nan=True
        )

    def test_elliptic_metallic_only_above_half_filling(self, maps):
        metallic = hmm.ANISOTROPY_CLASSES.index("elliptic-metallic")
        rows = np.flatnonzero((maps["class_index"] == metallic).any(axis=1))
        assert rows.size > 0
        assert maps["fills"][rows].min() > 0.5


# ------------------------------------------------------------ end-to-end main
@pytest.fixture(scope="module")
def main_run(tmp_path_factory):
    """Run main() once at low resolution; share the output across tests."""
    out_dir = tmp_path_factory.mktemp("hyperbolic_figures")
    summary = hmm.main(["--figures-dir", str(out_dir), "--n-points", "61"])
    return out_dir, summary


class TestMainEndToEnd:
    def test_creates_all_figures(self, main_run):
        out_dir, _ = main_run
        for name in FIGURE_NAMES:
            path = out_dir / name
            assert path.is_file(), name
            assert path.stat().st_size > 10_000  # non-trivial PNG

    def test_summary_json_schema(self, main_run):
        out_dir, summary = main_run
        with open(out_dir / "hmm_summary.json") as fh:
            on_disk = json.load(fh)
        assert on_disk == summary

        assert set(on_disk) == {
            "design",
            "spectral_window",
            "transitions",
            "anisotropy_regions",
            "bound_mode_regions",
            "recommended_band",
            "fill_fraction_scan",
            "band_criteria",
            "n_points",
        }

        scan = on_disk["fill_fraction_scan"]
        assert [row["fill_fraction"] for row in scan] == list(hmm.FILL_SCAN)
        design_row = next(
            row for row in scan if row["fill_fraction"] == pytest.approx(hmm.FILL_FRACTION)
        )
        assert design_row["band"]["wavelength_nm"] == pytest.approx(
            on_disk["recommended_band"]["wavelength_nm"]
        )
        # The scan's "band" is criteria-filtered, so it must be no wider than
        # the region where a bound mode actually exists; the flag says so.
        assert design_row["band_is_criteria_filtered"] is True
        bound_lo, bound_hi = (
            on_disk["bound_mode_regions"][0]["wavelength_nm"][0],
            on_disk["bound_mode_regions"][0]["wavelength_nm"][1],
        )
        band_lo, band_hi = sorted(design_row["band"]["wavelength_nm"])
        assert min(bound_lo, bound_hi) <= band_lo and band_hi <= max(bound_lo, bound_hi)

        design = on_disk["design"]
        assert set(design) == {
            "fill_fraction",
            "eps_dielectric_layer",
            "eps_superstrate",
            "optical_axis",
            "drude_model",
            "omega_ref",
            "lambda_ref_nm",
        }
        assert design["fill_fraction"] == pytest.approx(hmm.FILL_FRACTION)
        assert design["eps_dielectric_layer"] == pytest.approx(2.25)
        assert design["optical_axis"] == "z"
        assert design["drude_model"] == {
            "eps_inf": 3.7,
            "hbar_omega_p_eV": 9.1,
            "hbar_gamma_eV": 0.018,
        }

        transitions = on_disk["transitions"]
        assert set(transitions) == {
            "eps_t_zeros",
            "eps_n_zeros",
            "eps_n_pole",
            "pole_resonance",
        }
        assert len(transitions["eps_t_zeros"]) == 1
        assert set(transitions["eps_t_zeros"][0]) == {
            "omega",
            "photon_energy_ev",
            "wavelength_nm",
        }
        assert transitions["eps_t_zeros"][0]["wavelength_nm"] == pytest.approx(407.6, abs=1.0)
        assert transitions["pole_resonance"]["im_eps_n"] > 50.0

        assert on_disk["n_points"] == 61
        assert {r["class"] for r in on_disk["anisotropy_regions"]} <= set(
            hmm.ANISOTROPY_CLASSES
        )
        assert len(on_disk["bound_mode_regions"]) == 1

    def test_recommended_band_schema(self, main_run):
        _, summary = main_run
        band = summary["recommended_band"]
        assert band is not None
        assert set(band) == {
            "index_range",
            "omega",
            "omega_over_omega_ref",
            "photon_energy_ev",
            "wavelength_nm",
            "relative_width",
            "n_eff_range",
            "k_spp_re_range_per_um",
            "kappa_d_range_per_um",
            "kappa_m_range_per_um",
            "kappa_spread",
            "decay_length_range_nm",
            "nonlinearity_percent",
            "curvature_percent",
            "max_residual_per_um",
            "propagation_length_um_range",
            "L_over_lambda_spp_range",
            "max_layer_period_nm",
        }
        for key in ("omega", "wavelength_nm", "n_eff_range", "kappa_d_range_per_um"):
            lo, hi = band[key]
            assert lo < hi, key
        assert band["nonlinearity_percent"] >= hmm.MIN_NONLINEARITY_PCT
        assert band["n_eff_range"][0] > 1.0

    def test_recommended_band_is_bound_at_its_endpoints(self, main_run):
        """The deliverable, re-checked against the class on the low-res run."""
        _, summary = main_run
        lo, hi = summary["recommended_band"]["omega"]
        for omega in (lo, 0.5 * (lo + hi), hi):
            assert reference_mode(float(omega))

    def test_summary_is_valid_strict_json(self, main_run):
        """No NaN/Infinity tokens (invalid in strict JSON) in the file."""
        out_dir, _ = main_run
        text = (out_dir / "hmm_summary.json").read_text()
        json.loads(text, parse_constant=lambda name: pytest.fail(f"non-finite: {name}"))

    def test_custom_fill_fraction_is_honoured(self, tmp_path):
        summary = hmm.main(
            ["--figures-dir", str(tmp_path), "--n-points", "41", "--fill-fraction", "0.6"]
        )
        assert summary["design"]["fill_fraction"] == pytest.approx(0.6)
