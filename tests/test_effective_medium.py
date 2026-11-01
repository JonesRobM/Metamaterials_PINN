"""
Tests for :mod:`src.effective_medium` — the layered-multilayer effective-medium
theory behind ``examples/hyperbolic_metamaterial.py``.

Conventions checked throughout: ``exp(-iωt)``, so a passive medium has
``Im ε > 0``.
"""

import numpy as np
import pytest

from src import effective_medium as em

EPS_AG_JC_633NM = -18.3 + 0.55j  # Johnson & Christy anchor (tests/test_benchmark_spp.py)
EPS_SILICA = 2.25


# ---------------------------------------------------------------- Drude metal
class TestDrude:
    def test_matches_johnson_christy_at_633nm(self):
        """Same 15% agreement the existing dispersion study asserts."""
        eps = em.drude_permittivity(em.omega_from_wavelength(633e-9))
        rel = abs(eps - EPS_AG_JC_633NM) / abs(EPS_AG_JC_633NM)
        print(f"Drude eps(633 nm) = {eps:.4f}, |deviation| vs J&C = {rel:.2%}")
        assert rel < 0.15
        assert -21.0 < eps.real < -15.5

    def test_passivity_across_the_band(self):
        """exp(-iωt): Im ε > 0 everywhere a lossy metal is evaluated."""
        omega = em.omega_from_wavelength(np.linspace(200e-9, 2000e-9, 401))
        eps = em.drude_permittivity(omega)
        assert np.all(eps.imag > 0.0)
        # ... and it is a strictly decreasing function of wavelength-frequency:
        # Im ε = ω_p² γ / (ω(ω² + γ²)) falls monotonically with ω.
        order = np.argsort(omega)
        assert np.all(np.diff(eps.imag[order]) < 0.0)

    def test_real_imaginary_split_identity(self):
        """ε = [ε_∞ − ω_p²/(ω²+γ²)] + i[ω_p²γ/(ω(ω²+γ²))], as documented."""
        omega = em.omega_from_wavelength(np.array([400e-9, 633e-9, 1500e-9]))
        eps = em.drude_permittivity(omega)
        w_p, gamma, eps_inf = em.OMEGA_P_AG, em.GAMMA_AG, em.EPS_INF_AG
        np.testing.assert_allclose(
            eps.real, eps_inf - w_p**2 / (omega**2 + gamma**2), rtol=1e-14
        )
        np.testing.assert_allclose(
            eps.imag, w_p**2 * gamma / (omega * (omega**2 + gamma**2)), rtol=1e-14
        )

    def test_lossless_limit_is_real(self):
        eps = em.drude_permittivity(em.omega_from_wavelength(633e-9), gamma=0.0)
        assert eps.imag == 0.0
        assert eps.real < 0.0

    def test_ev_parameters_helper(self):
        """The eV helper reproduces the module's silver constants."""
        params = em.drude_parameters_ev()
        assert params["eps_inf"] == em.EPS_INF_AG
        assert params["omega_p"] == pytest.approx(em.OMEGA_P_AG, rel=1e-15)
        assert params["gamma"] == pytest.approx(em.GAMMA_AG, rel=1e-15)
        omega = em.omega_from_wavelength(633e-9)
        assert em.drude_permittivity(omega, **params) == em.drude_permittivity(omega)

    def test_rejects_non_positive_frequency(self):
        with pytest.raises(ValueError):
            em.drude_permittivity(0.0)


class TestUnitConversions:
    def test_wavelength_round_trip(self):
        lam = np.array([250e-9, 633e-9, 1550e-9])
        np.testing.assert_allclose(
            em.wavelength_from_omega(em.omega_from_wavelength(lam)), lam, rtol=1e-15
        )

    def test_photon_energy_round_trip(self):
        ev = np.array([0.8, 2.0, 4.7])
        np.testing.assert_allclose(
            em.photon_energy_ev(em.omega_from_photon_energy_ev(ev)), ev, rtol=1e-15
        )

    def test_633nm_is_1p96_ev(self):
        assert float(em.photon_energy_ev(em.omega_from_wavelength(633e-9))) == pytest.approx(
            1.9587, abs=1e-3
        )


# ------------------------------------------------------------ effective medium
class TestLayeredUniaxial:
    @pytest.mark.parametrize("eps_m", [-18.3 + 0.55j, -2.0 + 0.1j, 4.0 + 0.0j])
    def test_pure_dielectric_limit(self, eps_m):
        """f → 0: both components collapse onto the dielectric layer."""
        eps_t, eps_n = em.layered_uniaxial(eps_m, EPS_SILICA, 0.0)
        assert abs(eps_t - EPS_SILICA) < 1e-12
        assert abs(eps_n - EPS_SILICA) < 1e-12

    @pytest.mark.parametrize("eps_m", [-18.3 + 0.55j, -2.0 + 0.1j, 4.0 + 0.0j])
    def test_pure_metal_limit(self, eps_m):
        """f → 1: both components collapse onto the metal."""
        eps_t, eps_n = em.layered_uniaxial(eps_m, EPS_SILICA, 1.0)
        assert abs(eps_t - eps_m) < 1e-12
        assert abs(eps_n - eps_m) < 1e-12

    def test_isotropic_when_layers_match(self):
        """Identical constituents cannot be anisotropic, at any f."""
        for f in (0.1, 0.5, 0.9):
            eps_t, eps_n = em.layered_uniaxial(2.25 + 0.01j, 2.25 + 0.01j, f)
            assert abs(eps_t - eps_n) < 1e-14

    def test_arithmetic_and_harmonic_forms(self):
        """The two means, spelt out independently of the implementation."""
        eps_m, f = -18.3 + 0.55j, 0.3
        eps_t, eps_n = em.layered_uniaxial(eps_m, EPS_SILICA, f)
        assert eps_t == pytest.approx(f * eps_m + (1 - f) * EPS_SILICA, rel=1e-15)
        assert eps_n == pytest.approx(1.0 / (f / eps_m + (1 - f) / EPS_SILICA), rel=1e-13)

    def test_ordering_of_the_two_means(self):
        """For a positive pair the harmonic mean never exceeds the arithmetic one."""
        eps_t, eps_n = em.layered_uniaxial(9.0, 1.0, 0.5)
        assert eps_n.real < eps_t.real

    def test_rejects_fill_fraction_outside_unit_interval(self):
        for bad in (-0.01, 1.01):
            with pytest.raises(ValueError):
                em.layered_uniaxial(-2.0 + 0.1j, EPS_SILICA, bad)

    def test_lossless_pole_raises(self):
        """f ε_d2 + (1−f) ε_m = 0 exactly: only reachable without loss."""
        f = 0.3
        eps_m = -f * EPS_SILICA / (1.0 - f)  # real, exactly on the pole
        with pytest.raises(ZeroDivisionError):
            em.layered_uniaxial(eps_m, EPS_SILICA, f)


class TestPassivityAndThePole:
    """
    What actually happens at the harmonic mean's pole ``D = f ε_d2 + (1−f) ε_m → 0``.

    It is *not* a passivity violation and *not* a divergence: with loss present
    the pole is a Lorentzian-like resonance of finite height, ``Im ε_n`` stays
    positive and peaks like ``1/|D|²``, and ``Re ε_n`` sweeps smoothly through
    zero — the multilayer's physical ENZ resonance.
    """

    F = 0.3

    def _band(self, n=2001):
        """A dense frequency band straddling the pole (≈ 294 nm for f = 0.3)."""
        pole = em.transition_frequencies(self.F, EPS_SILICA)["eps_n_pole"][0]
        return np.linspace(0.97 * pole, 1.03 * pole, n), pole

    def test_arithmetic_mean_is_always_passive(self):
        """Im ε_t = f Im ε_m ≥ 0 — trivially, and over a wide band."""
        omega = em.omega_from_wavelength(np.linspace(200e-9, 2000e-9, 501))
        eps_t, _ = em.hmm_permittivities(omega, self.F, EPS_SILICA)
        assert np.all(eps_t.imag >= 0.0)
        np.testing.assert_allclose(
            eps_t.imag, self.F * em.drude_permittivity(omega).imag, rtol=1e-14
        )

    def test_harmonic_mean_stays_passive_through_the_pole(self):
        omega, _ = self._band()
        _, eps_n = em.hmm_permittivities(omega, self.F, EPS_SILICA)
        assert np.all(eps_n.imag > 0.0), "Im ε_n must not change sign at the pole"

    def test_harmonic_mean_imaginary_part_identity(self):
        """Im ε_n = f ε_d2² Im ε_m / |D|², the identity (†) of the module docstring."""
        omega, _ = self._band(401)
        eps_m = em.drude_permittivity(omega)
        _, eps_n = em.hmm_permittivities(omega, self.F, EPS_SILICA)
        denom = self.F * EPS_SILICA + (1.0 - self.F) * eps_m
        expected = self.F * EPS_SILICA**2 * eps_m.imag / np.abs(denom) ** 2
        np.testing.assert_allclose(eps_n.imag, expected, rtol=1e-12)

    def test_pole_is_a_finite_resonance_not_a_divergence(self):
        """
        Peak height is set by the loss: |ε_n| ≈ ε_d2|ε_m| / ((1−f) Im ε_m), and
        Im ε_n at the pole ≈ f ε_d2² / ((1−f)² Im ε_m).
        """
        omega, pole = self._band()
        _, eps_n = em.hmm_permittivities(omega, self.F, EPS_SILICA)
        eps_m_pole = em.drude_permittivity(pole)
        predicted_im = self.F * EPS_SILICA**2 / ((1 - self.F) ** 2 * eps_m_pole.imag)
        peak = float(eps_n.imag.max())
        print(f"Im eps_n peak = {peak:.1f} (predicted {predicted_im:.1f})")
        assert np.isfinite(peak)
        assert peak == pytest.approx(predicted_im, rel=0.02)
        # Two orders of magnitude above the off-resonance loss, but bounded.
        off_resonance = min(float(eps_n.imag[0]), float(eps_n.imag[-1]))
        assert peak > 100.0 * off_resonance

    def test_real_part_crosses_zero_at_the_pole(self):
        """Re ε_n changes sign across the resonance: the physical ENZ crossing."""
        omega, pole = self._band()
        _, eps_n = em.hmm_permittivities(omega, self.F, EPS_SILICA)
        assert eps_n.real[0] * eps_n.real[-1] < 0.0
        crossings = np.flatnonzero(np.diff(np.sign(eps_n.real)) != 0)
        assert len(crossings) == 1
        assert omega[crossings[0]] == pytest.approx(pole, rel=1e-3)

    def test_reported_pole_matches_the_minimum_of_the_denominator(self):
        omega, pole = self._band()
        eps_m = em.drude_permittivity(omega)
        denom = np.abs(self.F * EPS_SILICA + (1.0 - self.F) * eps_m)
        assert omega[int(np.argmin(denom))] == pytest.approx(pole, rel=1e-3)


class TestHmmPermittivities:
    def test_composes_drude_and_layered(self):
        omega = em.omega_from_wavelength(633e-9)
        expected = em.layered_uniaxial(em.drude_permittivity(omega), EPS_SILICA, 0.3)
        got = em.hmm_permittivities(omega, 0.3, EPS_SILICA)
        assert got[0] == expected[0] and got[1] == expected[1]

    def test_drude_kwargs_are_forwarded(self):
        omega = em.omega_from_wavelength(633e-9)
        lossless = em.hmm_permittivities(omega, 0.3, EPS_SILICA, gamma=0.0)
        assert lossless[0].imag == 0.0 and lossless[1].imag == 0.0

    def test_type_ii_in_the_visible_at_f_030(self):
        """The design point of examples/hyperbolic_metamaterial.py."""
        eps_t, eps_n = em.hmm_permittivities(em.omega_from_wavelength(633e-9), 0.3, EPS_SILICA)
        assert eps_t.real < 0.0 and eps_n.real > 0.0
        assert em.classify_anisotropy(eps_t, eps_n) == "type-II"


class TestVectorisation:
    """Array calls must agree with a scalar loop, elementwise."""

    def test_drude(self):
        omega = em.omega_from_wavelength(np.linspace(300e-9, 1500e-9, 37))
        loop = np.array([complex(em.drude_permittivity(float(w))) for w in omega])
        np.testing.assert_allclose(em.drude_permittivity(omega), loop, rtol=1e-15)

    def test_hmm_permittivities(self):
        omega = em.omega_from_wavelength(np.linspace(300e-9, 1500e-9, 37))
        eps_t, eps_n = em.hmm_permittivities(omega, 0.35, EPS_SILICA)
        for i, w in enumerate(omega):
            t_i, n_i = em.hmm_permittivities(float(w), 0.35, EPS_SILICA)
            assert eps_t[i] == pytest.approx(complex(t_i), rel=1e-15)
            assert eps_n[i] == pytest.approx(complex(n_i), rel=1e-15)

    def test_classification(self):
        omega = em.omega_from_wavelength(np.linspace(250e-9, 1500e-9, 53))
        eps_t, eps_n = em.hmm_permittivities(omega, 0.35, EPS_SILICA)
        labels = em.classify_anisotropy(eps_t, eps_n)
        assert labels.shape == omega.shape
        for i in range(omega.size):
            assert labels[i] == em.classify_anisotropy(complex(eps_t[i]), complex(eps_n[i]))

    def test_broadcasting_over_fill_fraction(self):
        """A fill column and a frequency row give a 2-D map."""
        fills = np.linspace(0.1, 0.9, 5)[:, None]
        omega = em.omega_from_wavelength(np.linspace(300e-9, 1500e-9, 7))[None, :]
        eps_t, eps_n = em.hmm_permittivities(omega, fills, EPS_SILICA)
        assert eps_t.shape == eps_n.shape == (5, 7)
        for i in range(5):
            row_t, row_n = em.hmm_permittivities(omega[0], float(fills[i, 0]), EPS_SILICA)
            np.testing.assert_allclose(eps_t[i], row_t, rtol=1e-15)
            np.testing.assert_allclose(eps_n[i], row_n, rtol=1e-15)

    def test_scalar_input_returns_scalar_string(self):
        assert isinstance(em.classify_anisotropy(1.0, 1.0), str)


# ------------------------------------------------------------- classification
class TestClassifyAnisotropy:
    @pytest.mark.parametrize(
        "eps_t, eps_n, expected",
        [
            (2.0 + 0.1j, 3.0 + 0.1j, "elliptic-dielectric"),
            (2.0 + 0.1j, -3.0 + 0.1j, "type-I"),
            (-4.0 + 0.2j, 3.0 + 0.05j, "type-II"),  # the repo's benchmark point
            (-4.0 + 0.2j, -3.0 + 0.05j, "elliptic-metallic"),
            (0.0, 5.0, "elliptic-dielectric"),  # exact zeros join the positive branch
            (0.0, -5.0, "type-I"),
        ],
    )
    def test_hand_built_cases(self, eps_t, eps_n, expected):
        assert em.classify_anisotropy(eps_t, eps_n) == expected

    def test_imaginary_parts_are_irrelevant(self):
        assert em.classify_anisotropy(-4.0 + 99j, 3.0 + 99j) == "type-II"

    def test_every_class_is_declared(self):
        labels = {
            em.classify_anisotropy(t, n)
            for t in (-1.0, 1.0)
            for n in (-1.0, 1.0)
        }
        assert labels == set(em.ANISOTROPY_CLASSES)

    def test_elliptic_metallic_needs_f_above_one_half(self):
        """
        A metal/dielectric multilayer is doubly negative only for f > 1/2 (both
        real parts negative requires −f ε_d2/(1−f) > Re ε_m > −(1−f) ε_d2/f).
        """
        omega = em.omega_from_wavelength(np.linspace(250e-9, 1500e-9, 1201))

        def has_metallic(f):
            eps_t, eps_n = em.hmm_permittivities(omega, f, EPS_SILICA)
            return "elliptic-metallic" in set(em.classify_anisotropy(eps_t, eps_n).tolist())

        assert not has_metallic(0.30)
        assert not has_metallic(0.50)
        assert has_metallic(0.70)


# -------------------------------------------------------- transition frequencies
F_DESIGN = 0.3


@pytest.fixture(scope="module")
def transitions():
    return em.transition_frequencies(F_DESIGN, EPS_SILICA)


class TestTransitionFrequencies:
    F = F_DESIGN

    def test_eps_t_crossing_is_a_zero_of_re_eps_t(self, transitions):
        assert len(transitions["eps_t_zeros"]) == 1
        for omega in transitions["eps_t_zeros"]:
            eps_t, _ = em.hmm_permittivities(omega, self.F, EPS_SILICA)
            assert abs(eps_t.real) < 1e-10 * abs(eps_t)

    def test_eps_t_crossing_matches_the_closed_form(self, transitions):
        """ω² = ω_p²/(ε_∞ − Re ε_m) − γ² with Re ε_m = −(1−f) ε_d2/f."""
        target = -(1 - self.F) * EPS_SILICA / self.F
        omega = np.sqrt(em.OMEGA_P_AG**2 / (em.EPS_INF_AG - target) - em.GAMMA_AG**2)
        assert transitions["eps_t_zeros"][0] == pytest.approx(omega, rel=1e-12)
        assert em.wavelength_from_omega(omega) * 1e9 == pytest.approx(407.6, abs=1.0)

    def test_eps_n_crossings_are_zeros_of_re_eps_n(self, transitions):
        assert len(transitions["eps_n_zeros"]) == 2
        for omega in transitions["eps_n_zeros"]:
            _, eps_n = em.hmm_permittivities(omega, self.F, EPS_SILICA)
            assert abs(eps_n.real) < 1e-8 * abs(eps_n), f"Re eps_n = {eps_n.real} at {omega}"

    def test_eps_n_crossings_bracket_the_lossless_roots(self, transitions):
        """Lossless seeds: Re ε_m = 0 and Re ε_m = −f ε_d2/(1−f)."""
        seeds = sorted(
            np.sqrt(em.OMEGA_P_AG**2 / (em.EPS_INF_AG - target) - em.GAMMA_AG**2)
            for target in (0.0, -self.F * EPS_SILICA / (1 - self.F))
        )
        for found, seed in zip(sorted(transitions["eps_n_zeros"]), seeds, strict=True):
            assert found == pytest.approx(seed, rel=1e-3)

    def test_transitions_separate_the_classes(self, transitions):
        """Each crossing has different anisotropy classes on either side."""
        for key in ("eps_t_zeros", "eps_n_zeros"):
            for omega in transitions[key]:
                before = em.classify_anisotropy(
                    *em.hmm_permittivities(omega * 0.999, self.F, EPS_SILICA)
                )
                after = em.classify_anisotropy(
                    *em.hmm_permittivities(omega * 1.001, self.F, EPS_SILICA)
                )
                assert before != after, f"{key} at {omega} separates nothing"

    def test_pole_is_reported(self, transitions):
        assert len(transitions["eps_n_pole"]) == 1
        pole = transitions["eps_n_pole"][0]
        eps_m = em.drude_permittivity(pole)
        assert eps_m.real == pytest.approx(-self.F * EPS_SILICA / (1 - self.F), rel=1e-12)

    def test_window_restricts_the_results(self):
        """A window containing only the type-II onset returns only that root."""
        transitions = em.transition_frequencies(
            self.F,
            EPS_SILICA,
            omega_range=(
                float(em.omega_from_wavelength(1500e-9)),
                float(em.omega_from_wavelength(350e-9)),
            ),
        )
        assert len(transitions["eps_t_zeros"]) == 1
        assert len(transitions["eps_n_zeros"]) == 0
        assert len(transitions["eps_n_pole"]) == 0

    def test_no_crossing_when_none_exists(self):
        """A metal that never gets negative enough has no ε_t crossing."""
        transitions = em.transition_frequencies(
            0.05, EPS_SILICA, omega_range=(1e15, 5e15), omega_p=1e15
        )
        assert len(transitions["eps_t_zeros"]) == 0

    def test_rejects_degenerate_arguments(self):
        with pytest.raises(ValueError):
            em.transition_frequencies(0.0, EPS_SILICA)
        with pytest.raises(ValueError):
            em.transition_frequencies(0.3, EPS_SILICA, omega_range=(5e15, 1e15))


# ------------------------------------------------------------- validity helper
class TestMaxLayerPeriod:
    def test_ten_times_smaller_than_the_mode_wavelength(self):
        k = 2.0 * np.pi / 500e-9  # a 500 nm mode
        assert float(em.max_layer_period(k)) == pytest.approx(50e-9, rel=1e-12)

    def test_factor_scales_inversely(self):
        k = 1e7
        assert float(em.max_layer_period(k, factor=20.0)) == pytest.approx(
            0.5 * float(em.max_layer_period(k, factor=10.0)), rel=1e-12
        )

    def test_vectorised(self):
        k = np.array([1e7, 2e7])
        np.testing.assert_allclose(em.max_layer_period(k), 2 * np.pi / (10.0 * k), rtol=1e-15)

    def test_rejects_non_positive(self):
        with pytest.raises(ValueError):
            em.max_layer_period(0.0)
