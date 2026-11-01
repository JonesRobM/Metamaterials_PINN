"""Tests for examples/dispersion_analysis.py (analytics only, no training)."""

import json
import math

import numpy as np
import pytest

from examples import dispersion_analysis as da
from src.constants import C0

FIGURE_NAMES = [
    "spp_dispersion_silver.png",
    "spp_length_scales_silver.png",
    "metamaterial_design_space.png",
    "anisotropy_cut.png",
]


# ---------------------------------------------------------------- Drude model
class TestDrudeSilver:
    def test_permittivity_at_633nm_near_johnson_christy(self):
        """The Drude fit must sit within ~15% of the J&C benchmark anchor."""
        eps = da.drude_permittivity(da.omega_from_wavelength(633e-9))
        rel = abs(eps - da.EPS_AG_JC_633NM) / abs(da.EPS_AG_JC_633NM)
        print(f"Drude eps(633 nm) = {eps:.4f}, |deviation| vs J&C = {rel:.2%}")
        assert rel < 0.15
        # Loose band on the real part too (dominates the dispersion).
        assert -21.0 < eps.real < -15.5

    def test_lossy_sign_convention(self):
        """exp(-iωt) convention: Im ε > 0 for a lossy medium."""
        for lam in (400e-9, 633e-9, 1000e-9):
            assert da.drude_permittivity(da.omega_from_wavelength(lam)).imag > 0

    def test_surface_plasmon_asymptote(self):
        """ħω_sp = ħω_p / sqrt(ε_∞ + ε_d): Re ε(ω_sp) = −ε_d."""
        hw_sp = da.surface_plasmon_energy_ev(eps_d=1.0)
        assert hw_sp == pytest.approx(9.1 / np.sqrt(4.7), rel=1e-12)
        omega_sp = hw_sp / da.HBAR_EVS
        eps_sp = da.drude_permittivity(omega_sp)
        assert eps_sp.real == pytest.approx(-1.0, abs=1e-3)


# ------------------------------------------------------- silver dispersion sweep
@pytest.fixture(scope="module")
def silver_sweep():
    return da.sweep_silver_dispersion(np.linspace(400e-9, 1000e-9, 61))


class TestSilverDispersionSweep:
    def test_whole_band_supports_spp(self, silver_sweep):
        assert silver_sweep["supported"].all()

    def test_dispersion_bends_below_light_line(self, silver_sweep):
        """Bound mode: Re k_spp > sqrt(ε_d) k0 wherever supported."""
        sup = silver_sweep["supported"]
        assert np.all(
            silver_sweep["k_spp"].real[sup] > np.sqrt(da.EPS_D) * silver_sweep["k0"][sup]
        )

    def test_propagation_length_increases_toward_red(self, silver_sweep):
        """Drude loss falls to the red: L(λ) strictly increasing."""
        sup = silver_sweep["supported"]
        length = silver_sweep["L"][sup]
        assert np.all(np.isfinite(length))
        assert np.all(np.diff(length) > 0)

    def test_penetration_depths_physical(self, silver_sweep):
        """δ_m stays tens of nm; δ_d exceeds it everywhere in the band."""
        sup = silver_sweep["supported"]
        delta_d = silver_sweep["delta_d"][sup]
        delta_m = silver_sweep["delta_m"][sup]
        assert np.all(delta_m > 10e-9) and np.all(delta_m < 60e-9)
        assert np.all(delta_d > delta_m)


# --------------------------------------------------------- design-space maps
class TestDesignSpaceMap:
    def test_support_map_reference_points(self):
        """False at (ε_t=+2, ε_n=+2); True at the benchmark (ε_t=−4, ε_n=+3)."""
        maps = da.sweep_design_space(
            633e-9, np.array([-4.0, 2.0]), np.array([2.0, 3.0])
        )
        # Grids are indexed [i_n, i_t].
        assert maps["supported"][0, 1] == False  # noqa: E712  (ε_t=+2, ε_n=+2)
        assert bool(maps["supported"][1, 0]) is True  # (ε_t=−4, ε_n=+3)
        assert np.isfinite(maps["n_eff"][1, 0])
        assert np.isfinite(maps["L"][1, 0])
        # Unsupported cells stay masked.
        assert np.isnan(maps["n_eff"][0, 1]) and np.isnan(maps["L"][0, 1])

    def test_no_bound_mode_for_positive_eps_t(self):
        """TM matching κ_d/ε_d + κ_m/ε_t = 0 needs ε_t < 0."""
        maps = da.sweep_design_space(
            633e-9, np.array([0.5, 1.0, 2.0]), np.array([-2.0, 2.0, 6.0])
        )
        assert not maps["supported"].any()

    def test_singular_point_is_guarded(self):
        """ε_t ε_n = ε_d² (lossless) raises inside spp_wavevector; the sweep
        must swallow it and report the point as unsupported."""
        maps = da.sweep_design_space(
            633e-9, np.array([-2.0]), np.array([-0.5]), im_eps_t=0.0, im_eps_n=0.0
        )
        assert maps["supported"][0, 0] == False  # noqa: E712
        assert np.isnan(maps["L"][0, 0])


# ------------------------------------------------------------ anisotropy cut
class TestAnisotropyCut:
    def test_crosses_support_boundaries(self):
        """At ε_t = −4+0.2j the cut is supported for ε_n < ε_d²/ε_t ≈ −0.25
        and for ε_n > ε_d, unsupported in between."""
        en = np.array([-2.0, -1.0, 0.0, 0.5, 2.0, 5.0])
        cut = da.sweep_anisotropy_cut(633e-9, en)
        assert list(cut["supported"]) == [True, True, False, False, True, True]
        boundaries = da.support_boundaries(en, cut["supported"])
        assert len(boundaries) == 2
        assert -1.0 < boundaries[0] < 0.0  # near the resonance ε_n ≈ −0.25
        assert 0.5 < boundaries[1] < 2.0  # near the onset ε_n ≈ ε_d

    def test_metrics_masked_outside_support(self):
        cut = da.sweep_anisotropy_cut(633e-9, np.array([0.0, 3.0]))
        assert np.isnan(cut["L"][0]) and np.isnan(cut["n_eff"][0])
        assert np.isfinite(cut["L"][1]) and cut["n_eff"][1] > 1.0


# ------------------------------------------------------------ end-to-end main
@pytest.fixture(scope="module")
def main_run(tmp_path_factory):
    """Run main() once at tiny resolution; share the output across tests."""
    out_dir = tmp_path_factory.mktemp("dispersion_figures")
    summary = da.main(["--figures-dir", str(out_dir), "--n-points", "25"])
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
        with open(out_dir / "dispersion_summary.json") as fh:
            on_disk = json.load(fh)
        assert on_disk == summary

        assert set(on_disk) == {
            "drude_model",
            "silver_at_633nm",
            "surface_plasmon_asymptote",
            "silver_sweep",
            "design_space",
            "anisotropy_cut",
            "n_points",
        }
        ag = on_disk["silver_at_633nm"]
        assert set(ag) == {
            "eps_drude",
            "eps_johnson_christy",
            "relative_deviation_vs_jc",
            "wavelength_nm",
            "n_eff",
            "propagation_length_um",
            "penetration_depth_dielectric_nm",
            "penetration_depth_metal_nm",
        }
        assert len(ag["eps_drude"]) == 2 and ag["eps_drude"][0] < 0
        assert 0.0 <= ag["relative_deviation_vs_jc"] < 0.15
        assert ag["n_eff"] > 1.0
        assert ag["propagation_length_um"] > 0.0

        # The anchor must be evaluated at exactly 633 nm, not snapped to the
        # nearest sweep grid point, so it does not drift with --n-points.
        assert ag["wavelength_nm"] == 633.0
        omega_633 = 2.0 * math.pi * C0 / 633e-9
        eps_exact = complex(da.drude_permittivity(omega_633))
        assert ag["eps_drude"][0] == pytest.approx(eps_exact.real, rel=1e-12)
        assert ag["eps_drude"][1] == pytest.approx(eps_exact.imag, rel=1e-12)

        design = on_disk["design_space"]
        assert design["lambda0_nm"] == pytest.approx(633.0)
        assert 0.0 < design["supported_fraction"] < 1.0
        lo, hi = design["n_eff_range"]
        assert 1.0 < lo < hi

        cut = on_disk["anisotropy_cut"]
        assert cut["eps_t"] == [-4.0, 0.2]
        assert len(cut["support_boundaries_re_eps_n"]) == 2
        assert on_disk["n_points"] == 25

    def test_summary_is_valid_strict_json(self, main_run):
        """No NaN/Infinity tokens (invalid in strict JSON) in the file."""
        out_dir, _ = main_run
        text = (out_dir / "dispersion_summary.json").read_text()
        json.loads(text, parse_constant=lambda name: pytest.fail(f"non-finite: {name}"))
