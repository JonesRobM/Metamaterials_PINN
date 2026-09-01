"""
Tests for ``examples/emt_validity.py`` (analytics only, no training).

The study's load-bearing claims are (i) that the transfer-matrix answer
converges to the effective-medium one as the period vanishes — otherwise the
comparison would be measuring a bug, not a physical error — and (ii) that the
crossover periods it reports are real, ordered, and reproducible. Those are what
is checked here, together with the JSON contract and an end-to-end ``main()``
run at low resolution.

The headline structural claim, that the naturally terminated stack errs at
``O(a)`` while a half-metal termination errs at ``O(a²)``, is pinned as a
numerical fact on the fitted log–log exponents.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from examples import emt_validity as emt
from src.effective_medium import drude_permittivity, omega_from_wavelength

FIGURE_NAMES = [
    "emt_period_sweep.png",
    "emt_frequency_sweep.png",
    "emt_convergence.png",
]

PERIODS = np.geomspace(1e-9, 60e-9, 12)
SMALL = PERIODS <= 25e-9


@pytest.fixture(scope="module")
def sweeps():
    """Period sweeps at the reference wavelength for the two key terminations."""
    return {
        name: emt.sweep_period(emt.LAMBDA_REF, PERIODS, name)
        for name in ("metal", "half-metal")
    }


# ------------------------------------------------------------------ geometry
class TestMultilayerStack:
    @pytest.mark.parametrize("termination", emt.TERMINATIONS)
    def test_layer_counts_and_thicknesses(self, termination):
        a, n = 12e-9, 7
        eps_layers, thicknesses = emt.multilayer_stack(
            a, n, 0.30, -18.0 + 0.2j, 2.25, termination
        )
        assert len(thicknesses) == len(eps_layers) - 2 == 2 * n
        assert all(d > 0.0 for d in thicknesses)
        expected = n * a - (0.5 * 0.30 * a if termination == "half-metal" else 0.0)
        assert sum(thicknesses) == pytest.approx(expected)

    @pytest.mark.parametrize("termination", emt.TERMINATIONS)
    def test_layers_alternate_and_end_on_air(self, termination):
        eps_layers, _ = emt.multilayer_stack(10e-9, 5, 0.30, -18.0 + 0.2j, 2.25, termination)
        assert eps_layers[-1] == complex(emt.EPS_D)
        assert eps_layers[0] == complex(-18.0 + 0.2j)  # silver substrate by default
        interior = eps_layers[1:-1]
        assert all(a != b for a, b in zip(interior, interior[1:], strict=False)), (
            "adjacent layers must differ, otherwise the period is not what it claims"
        )

    def test_termination_selects_the_layer_facing_air(self):
        for termination, expected in (
            ("metal", -18.0 + 0.2j),
            ("half-metal", -18.0 + 0.2j),
            ("dielectric", 2.25),
        ):
            eps_layers, thicknesses = emt.multilayer_stack(
                10e-9, 4, 0.30, -18.0 + 0.2j, 2.25, termination
            )
            assert eps_layers[-2] == complex(expected)
        # ...and the half-metal top layer is genuinely half as thick.
        _, full = emt.multilayer_stack(10e-9, 4, 0.30, -18.0 + 0.2j, 2.25, "metal")
        _, half = emt.multilayer_stack(10e-9, 4, 0.30, -18.0 + 0.2j, 2.25, "half-metal")
        assert half[-1] == pytest.approx(0.5 * full[-1])

    @pytest.mark.parametrize(
        "kwargs",
        [{"termination": "nonsense"}, {"period": 0.0}, {"n_periods": 0}],
    )
    def test_invalid_geometry_rejected(self, kwargs):
        args = {"period": 10e-9, "n_periods": 4, "eps_metal": -18.0 + 0.2j}
        args.update(kwargs)
        with pytest.raises(ValueError):
            emt.multilayer_stack(**args)

    def test_periods_for_holds_the_total_thickness(self):
        for a_nm in (2.0, 7.5, 30.0):
            n = emt.periods_for(a_nm * 1e-9)
            assert abs(n * a_nm * 1e-9 - emt.TOTAL_THICKNESS) <= 0.5 * a_nm * 1e-9
        assert emt.periods_for(1e-3) == emt.MIN_PERIODS  # floor for absurd periods


# --------------------------------------------------- the thin-period limit
class TestThinPeriodLimit:
    """EMT and TMM must agree as ``a → 0``: that is what makes the study a test."""

    @pytest.mark.parametrize("termination", ["metal", "half-metal"])
    def test_error_vanishes_at_the_smallest_period(self, sweeps, termination):
        sweep = sweeps[termination]
        tolerance = 2e-3 if termination == "metal" else 1e-5
        assert sweep["rel_error_re"][0] < tolerance
        assert sweep["rel_error_im"][0] < 10.0 * tolerance

    @pytest.mark.parametrize("termination", ["metal", "half-metal"])
    def test_error_is_monotone_in_the_period(self, sweeps, termination):
        sweep = sweeps[termination]
        for part in ("re", "im"):
            errors = np.asarray(sweep[f"rel_error_{part}"])[SMALL]
            assert np.all(np.isfinite(errors))
            assert np.all(np.diff(errors) > 0.0), f"{termination}/{part} not monotone"

    def test_every_root_was_found(self, sweeps):
        for sweep in sweeps.values():
            assert np.all(np.isfinite(sweep["k_tmm"]))
            assert np.all(np.asarray(sweep["k_tmm"]).imag > 0.0)  # exp(-iwt): lossy

    def test_metal_and_dielectric_terminations_err_with_opposite_signs(self):
        """The signature of a boundary term: same size, opposite direction."""
        periods = np.geomspace(2e-9, 10e-9, 4)
        metal = emt.sweep_period(emt.LAMBDA_REF, periods, "metal")
        dielectric = emt.sweep_period(emt.LAMBDA_REF, periods, "dielectric")
        for part in ("re", "im"):
            a = np.asarray(metal[f"signed_error_{part}"])
            b = np.asarray(dielectric[f"signed_error_{part}"])
            assert np.all(a * b < 0.0), f"{part}: signs should be opposite"
            assert np.allclose(np.abs(a), np.abs(b), rtol=0.25)

    def test_half_metal_termination_removes_the_leading_error(self, sweeps):
        """The study's main structural claim: p ≈ 1 becomes p ≈ 2."""
        metal = emt.error_exponent(PERIODS, sweeps["metal"]["rel_error_re"])
        half = emt.error_exponent(PERIODS, sweeps["half-metal"]["rel_error_re"])
        assert metal == pytest.approx(1.0, abs=0.12)
        assert half == pytest.approx(2.0, abs=0.12)
        # ...and it is a large practical difference, not a formal one.
        ten_nm = int(np.argmin(np.abs(PERIODS - 10e-9)))
        ratio = (
            sweeps["metal"]["rel_error_re"][ten_nm]
            / sweeps["half-metal"]["rel_error_re"][ten_nm]
        )
        assert ratio > 30.0

    def test_tmm_matches_the_class_at_vanishing_period(self):
        """One direct comparison against ``MetamaterialProperties`` itself."""
        omega = float(omega_from_wavelength(emt.LAMBDA_REF))
        k_emt = emt.emt_wavevector(omega)
        k_tmm = emt.tmm_wavevector(
            omega, 1e-9, emt.periods_for(1e-9), termination="half-metal"
        )
        assert abs(k_tmm - k_emt) / abs(k_emt) < 1e-4
        assert emt.emt_material(omega).is_spp_supported(eps_dielectric=emt.EPS_D)


# --------------------------------------------------------------- crossovers
class TestCrossovers:
    def test_ordered_and_finite(self, sweeps):
        table = emt.crossover_table(sweeps["metal"])
        values = [table["im"][f"{t * 100:g}pct_nm"] for t in emt.THRESHOLDS]
        assert all(v is not None for v in values), "Im crossings must lie inside the sweep"
        assert all(math.isfinite(v) and v > 0.0 for v in values)
        assert values[0] < values[1] < values[2]
        # Re k_spp is the better-behaved half: its 1 % point is later.
        assert table["re"]["1pct_nm"] > table["im"]["1pct_nm"]

    def test_re_crossover_is_a_sane_number(self, sweeps):
        table = emt.crossover_table(sweeps["metal"])
        assert 5.0 < table["re"]["1pct_nm"] < 25.0

    def test_half_metal_never_crosses_inside_the_range(self, sweeps):
        table = emt.crossover_table(sweeps["half-metal"])
        assert table["re"]["1pct_nm"] is None
        assert table["re"]["error_at_max_period"] < 0.01

    def test_crossover_period_on_a_synthetic_power_law(self):
        a = np.geomspace(1e-9, 100e-9, 60)
        errors = 1e-3 * (a / 1e-9)  # 1 % exactly at 10 nm
        found = emt.crossover_period(a, errors, 0.01)
        assert found == pytest.approx(10e-9, rel=1e-3)
        assert emt.crossover_period(a, errors, 10.0) is None  # above the range
        assert emt.crossover_period(a, errors + 1.0, 0.01) is None  # below the range

    def test_error_exponent_on_synthetic_data(self):
        a = np.geomspace(1e-9, 12e-9, 20)
        assert emt.error_exponent(a, 3.0 * a**2) == pytest.approx(2.0, abs=1e-9)
        assert emt.error_exponent(a, 3.0 * a) == pytest.approx(1.0, abs=1e-9)
        assert emt.error_exponent(a[:2], a[:2]) is None

    def test_recommended_max_period_matches_the_hyperbolic_study(self):
        """33 nm is recomputed here, not copied, and must reproduce hmm_summary.json."""
        assert emt.recommended_max_period() * 1e9 == pytest.approx(33.07, abs=0.05)
        assert emt.MAX_LAYER_PERIOD_NM == pytest.approx(33.07, abs=0.05)


# --------------------------------------------------------- solver guardrails
class TestSolverGuardrails:
    def test_absurd_reference_yields_nan_not_a_wrong_mode(self):
        omega = float(omega_from_wavelength(emt.LAMBDA_REF))
        value = emt.tmm_wavevector(omega, 10e-9, 40, reference=1.0 + 0.0j)
        assert not np.isfinite(value)

    def test_frequency_sweep_tracks_the_branch_across_the_band(self):
        wavelengths = np.linspace(450e-9, 885e-9, 6)
        row = emt.sweep_frequency(wavelengths, 10e-9)
        assert np.all(np.isfinite(row["k_tmm"]))
        # Error grows towards the blue, where the mode is most tightly confined.
        errors = np.asarray(row["rel_error_re"])
        assert errors[0] > errors[-1]

    def test_finite_stack_converges_to_the_semi_infinite_answer(self):
        row = emt.sweep_n_periods(emt.LAMBDA_REF, 10e-9, [2, 4, 8, 16, 32, 64])
        errors = np.asarray(row["rel_error_vs_reference"])[:-1]
        assert np.all(np.diff(errors) < 0.0)
        assert errors[-1] < 1e-5
        # The plateau is the homogenisation error, and it does not go away.
        assert np.asarray(row["rel_error_vs_emt"])[-1] > 1e-3

    def test_substrate_choice_does_not_matter_for_a_thick_stack(self):
        silver = emt.sweep_n_periods(emt.LAMBDA_REF, 10e-9, [40, 60])
        silica = emt.sweep_n_periods(
            emt.LAMBDA_REF, 10e-9, [40, 60], eps_substrate=emt.EPS_D2
        )
        difference = abs(silver["k_tmm"][-1] - silica["k_tmm"][-1]) / abs(silver["k_tmm"][-1])
        assert difference < 1e-6

    def test_drude_metal_is_passive_at_the_reference_point(self):
        omega = float(omega_from_wavelength(emt.LAMBDA_REF))
        assert complex(drude_permittivity(omega)).imag > 0.0


# ------------------------------------------------------------- end-to-end
@pytest.fixture(scope="module")
def main_run(tmp_path_factory):
    """One low-resolution end-to-end run, shared by the schema tests."""
    out_dir = tmp_path_factory.mktemp("emt_validity")
    summary = emt.main(
        ["--figures-dir", str(out_dir), "--n-periods-points", "5", "--n-wavelengths", "5"]
    )
    return out_dir, summary


@pytest.mark.slow
class TestMain:
    """A full low-resolution run (~10 s: the 1 nm period needs an 800-period stack)."""

    def test_figures_written(self, main_run):
        out_dir, _ = main_run
        for name in FIGURE_NAMES:
            path = out_dir / name
            assert path.exists() and path.stat().st_size > 10_000

    def test_summary_top_level_schema(self, main_run):
        out_dir, summary = main_run
        on_disk = json.loads((out_dir / "emt_validity_summary.json").read_text())
        assert set(on_disk) == {
            "design",
            "reference_point",
            "headline_crossovers",
            "period_sweeps",
            "frequency_sweeps",
            "convergence",
            "substrate_check",
            "thresholds",
        }
        assert set(on_disk) == set(summary)
        design = on_disk["design"]
        assert design["fill_fraction"] == pytest.approx(0.30)
        assert design["eps_dielectric_layer"] == pytest.approx(2.25)
        assert design["reference_wavelength_nm"] == pytest.approx(633.0)
        assert design["max_layer_period_nm"] == pytest.approx(33.07, abs=0.05)
        assert design["drude_model"] == {
            "eps_inf": 3.7,
            "hbar_omega_p_eV": 9.1,
            "hbar_gamma_eV": 0.018,
        }
        assert on_disk["thresholds"] == [0.01, 0.05, 0.10]

    def test_reference_point_schema(self, main_run):
        _, summary = main_run
        reference = summary["reference_point"]
        assert set(reference) == {
            "wavelength_nm", "omega", "eps_metal", "eps_t", "eps_n", "k_emt"
        }
        for key in ("eps_metal", "eps_t", "eps_n", "k_emt"):
            assert set(reference[key]) == {"re", "im"}
        assert reference["eps_metal"]["re"] < 0.0 and reference["eps_metal"]["im"] > 0.0
        assert reference["eps_t"]["re"] < 0.0 < reference["eps_n"]["re"]  # type-II
        assert reference["k_emt"]["im"] > 0.0

    def test_headline_crossovers_schema(self, main_run):
        _, summary = main_run
        headline = summary["headline_crossovers"]
        assert set(headline) == set(emt.TERMINATIONS)
        for table in headline.values():
            assert set(table) == {"re", "im"}
            for entry in table.values():
                assert set(entry) == {
                    "1pct_nm", "5pct_nm", "10pct_nm", "exponent",
                    "min_period_nm", "max_period_nm",
                    "error_at_min_period", "error_at_max_period",
                }
                assert entry["error_at_min_period"] < entry["error_at_max_period"]

    def test_period_sweep_records(self, main_run):
        _, summary = main_run
        records = summary["period_sweeps"]
        assert len(records) == len(emt.SWEEP_WAVELENGTHS_NM) * len(emt.TERMINATIONS)
        for record in records:
            n = len(record["periods_nm"])
            for key in ("n_periods", "k_tmm_re", "k_tmm_im", "rel_error_re",
                        "rel_error_im", "signed_error_re", "signed_error_im"):
                assert len(record[key]) == n, key
            assert record["termination"] in emt.TERMINATIONS
            assert record["periods_nm"] == sorted(record["periods_nm"])
            assert record["k_emt"]["im"] > 0.0

    def test_convergence_and_substrate_records(self, main_run):
        _, summary = main_run
        for record in summary["convergence"]:
            assert record["periods_for_1pct"] is not None
            assert record["periods_for_0p1pct"] >= record["periods_for_1pct"]
            assert 0.0 < record["residual_emt_error"] < 1.0
        assert summary["substrate_check"]["max_rel_difference_vs_silver"] < 1e-6

    def test_summary_is_valid_strict_json(self, main_run):
        """No NaN/Infinity tokens (invalid in strict JSON) in the file."""
        out_dir, _ = main_run
        text = (out_dir / "emt_validity_summary.json").read_text()
        json.loads(text, parse_constant=lambda name: pytest.fail(f"non-finite: {name}"))

    def test_conclusion_survives_the_low_resolution_run(self, main_run):
        """The headline finding must not be an artefact of the sweep resolution."""
        _, summary = main_run
        headline = summary["headline_crossovers"]
        assert headline["metal"]["re"]["exponent"] == pytest.approx(1.0, abs=0.2)
        assert headline["half-metal"]["re"]["exponent"] == pytest.approx(2.0, abs=0.2)
        assert headline["metal"]["re"]["1pct_nm"] < emt.MAX_LAYER_PERIOD_NM
