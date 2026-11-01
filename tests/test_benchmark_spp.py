"""
Benchmark tests anchoring the SPP analytics to independent ground truth.

Every reference value here is computed *independently* of
:mod:`src.physics.metamaterial` (own ``cmath`` implementations, standard
textbook formulas) or taken from the literature:

* Isotropic silver at λ₀ = 633 nm, ε_Ag ≈ −18.3 + 0.55j (Johnson & Christy,
  *Phys. Rev. B* 6, 4370 (1972)) against air and glass — effective index,
  propagation length and penetration depths per Maier, *Plasmonics:
  Fundamentals and Applications*, Ch. 2, and Raether, *Surface Plasmons*.
* Uniaxial closed form ``k² = k₀² ε_d ε_n (ε_t − ε_d)/(ε_t ε_n − ε_d²)``
  verified numerically against the unsquared TM matching condition
  ``κ_d/ε_d + κ_m/ε_t = 0`` and an independent complex secant root-find.

Sign conventions (see the module under test): ``exp(-iωt)``, lossy
``Im(ε) > 0``, bound/decaying ``Im(k_spp) > 0``, decay constants
``Re(κ) > 0``.
"""

from __future__ import annotations

import cmath
import math

import pytest

from src.constants import C0
from src.physics.metamaterial import MetamaterialProperties

LAMBDA0 = 633e-9
K0 = 2.0 * math.pi / LAMBDA0
OMEGA = K0 * C0

EPS_AG = -18.3 + 0.55j  # Johnson & Christy silver at 633 nm


# --------------------------------------------------------------------- helpers
def propagating_root(z_sq: complex) -> complex:
    """Independent branch choice: ``Im > 0`` (decay along propagation)."""
    root = cmath.sqrt(z_sq)
    if root.imag < 0 or (root.imag == 0 and root.real < 0):
        root = -root
    return root


def decaying_root(z_sq: complex) -> complex:
    """Independent branch choice: ``Re > 0`` (evanescent decay from interface)."""
    root = cmath.sqrt(z_sq)
    if root.real < 0 or (root.real == 0 and root.imag < 0):
        root = -root
    return root


def isotropic_k_spp(eps_m: complex, eps_d: complex, k0: float = K0) -> complex:
    """Textbook isotropic SPP wavevector k₀·sqrt(ε_m ε_d / (ε_m + ε_d))."""
    return k0 * propagating_root(eps_m * eps_d / (eps_m + eps_d))


def anisotropic_k_spp(
    eps_t: complex, eps_n: complex, eps_d: complex, k0: float = K0
) -> complex:
    """Uniaxial closed form (ε_t in-plane along propagation, ε_n normal)."""
    denom = eps_t * eps_n - eps_d**2
    k_sq = k0**2 * eps_d * eps_n * (eps_t - eps_d) / denom
    return propagating_root(k_sq)


def decay_constants(
    k: complex, eps_t: complex, eps_n: complex, eps_d: complex, k0: float = K0
) -> tuple[complex, complex]:
    """(κ_d, κ_m) on the ``Re > 0`` branch for in-plane wavevector ``k``."""
    kappa_d = decaying_root(k**2 - eps_d * k0**2)
    kappa_m = decaying_root(eps_t * (k**2 / eps_n - k0**2))
    return kappa_d, kappa_m


def matching_residual(
    k: complex, eps_t: complex, eps_n: complex, eps_d: complex, k0: float = K0
) -> float:
    """Relative residual of the unsquared TM condition κ_d/ε_d + κ_m/ε_t = 0."""
    kappa_d, kappa_m = decay_constants(k, eps_t, eps_n, eps_d, k0)
    residual = kappa_d / eps_d + kappa_m / eps_t
    scale = abs(kappa_d / eps_d) + abs(kappa_m / eps_t)
    return abs(residual) / scale


def secant_root(
    eps_t: complex, eps_n: complex, eps_d: complex, k_start: complex, k0: float = K0
) -> complex:
    """Complex secant iteration on the unsquared matching function."""

    def f(k: complex) -> complex:
        kappa_d, kappa_m = decay_constants(k, eps_t, eps_n, eps_d, k0)
        return kappa_d / eps_d + kappa_m / eps_t

    k_prev, k_curr = k_start, 0.96 * k_start
    f_prev, f_curr = f(k_prev), f(k_curr)
    for _ in range(200):
        if f_curr == f_prev:
            break
        k_next = k_curr - f_curr * (k_curr - k_prev) / (f_curr - f_prev)
        k_prev, f_prev = k_curr, f_curr
        k_curr, f_curr = k_next, f(k_next)
        if abs(f_curr) < 1e-13 * k0:
            break
    return k_curr


def make_uniaxial(eps_t: complex, eps_n: complex) -> MetamaterialProperties:
    """Interface-normal optical axis 'z': ε_∥ = ε_zz = ε_n, ε_⊥ = ε_xx = ε_t."""
    return MetamaterialProperties(eps_n, eps_t, optical_axis="z", omega=OMEGA)


# ============================================================ A. literature anchor
class TestSilverLiteratureAnchor:
    """Isotropic silver benchmarks at λ₀ = 633 nm (Johnson & Christy)."""

    @pytest.mark.parametrize("eps_d", [1.0, 2.25], ids=["air", "glass"])
    def test_spp_wavevector_matches_independent_formula(self, eps_d):
        k_ref = isotropic_k_spp(EPS_AG, eps_d)
        m = MetamaterialProperties(EPS_AG, EPS_AG, "z", omega=OMEGA)
        k = m.spp_wavevector(eps_dielectric=eps_d)
        assert abs(k - k_ref) / abs(k_ref) < 1e-10
        assert k.imag > 0  # decaying along propagation, exp(-iωt) convention

    def test_effective_index_silver_air(self):
        """Re(n_eff) ≈ 1.028 for Ag/air at 633 nm (Maier Ch. 2, Raether)."""
        m = MetamaterialProperties(EPS_AG, EPS_AG, "z", omega=OMEGA)
        n_eff = m.spp_wavevector(eps_dielectric=1.0).real / K0
        assert 1.0277 < n_eff < 1.0290

    def test_effective_index_silver_glass(self):
        """Independent value: Re sqrt(ε_m ε_d/(ε_m+ε_d)) ≈ 1.6016 for ε_d = 2.25."""
        m = MetamaterialProperties(EPS_AG, EPS_AG, "z", omega=OMEGA)
        n_eff = m.spp_wavevector(eps_dielectric=2.25).real / K0
        assert n_eff == pytest.approx(1.6016, abs=2e-3)

    def test_propagation_length_silver_air(self):
        """L = 1/(2 Im k_spp) ≈ 56–60 µm for Ag/air at 633 nm (Maier, Raether)."""
        m = MetamaterialProperties(EPS_AG, EPS_AG, "z", omega=OMEGA)
        length = m.propagation_length(eps_dielectric=1.0)
        assert 50e-6 < length < 70e-6
        # And exactly the independent value:
        k_ref = isotropic_k_spp(EPS_AG, 1.0)
        assert length == pytest.approx(1.0 / (2.0 * k_ref.imag), rel=1e-10)

    def test_propagation_length_silver_glass(self):
        """Loss grows with confinement: L(glass) ≈ 15 µm < L(air)."""
        m = MetamaterialProperties(EPS_AG, EPS_AG, "z", omega=OMEGA)
        length = m.propagation_length(eps_dielectric=2.25)
        assert 12e-6 < length < 18e-6

    @pytest.mark.parametrize(
        "eps_d, band_d, band_m",
        [
            (1.0, (400e-9, 450e-9), (22e-9, 25e-9)),
            (2.25, (160e-9, 200e-9), (21e-9, 24e-9)),
        ],
        ids=["air", "glass"],
    )
    def test_penetration_depths(self, eps_d, band_d, band_m):
        """δ_i = 1/Re κ_i with κ_i = sqrt(k_spp² − ε_i k₀²), derived independently.

        Ag/air at 633 nm: δ_d ≈ 419 nm, δ_m ≈ 23 nm (Maier Ch. 2 quotes
        ~24 nm in the metal and hundreds of nm in air at visible frequencies).
        """
        k_ref = isotropic_k_spp(EPS_AG, eps_d)
        kappa_d = decaying_root(k_ref**2 - eps_d * K0**2)
        kappa_m = decaying_root(k_ref**2 - EPS_AG * K0**2)
        delta_d_ref = 1.0 / kappa_d.real
        delta_m_ref = 1.0 / kappa_m.real
        assert band_d[0] < delta_d_ref < band_d[1]
        assert band_m[0] < delta_m_ref < band_m[1]

        m = MetamaterialProperties(EPS_AG, EPS_AG, "z", omega=OMEGA)
        assert m.penetration_depth_dielectric(eps_dielectric=eps_d) == pytest.approx(
            delta_d_ref, rel=1e-10
        )
        assert m.penetration_depth_metamaterial(eps_dielectric=eps_d) == pytest.approx(
            delta_m_ref, rel=1e-10
        )

    @pytest.mark.parametrize("eps_d", [1.0, 2.25], ids=["air", "glass"])
    def test_silver_interface_supports_spp(self, eps_d):
        m = MetamaterialProperties(EPS_AG, EPS_AG, "z", omega=OMEGA)
        assert m.is_spp_supported(eps_dielectric=eps_d) is True

    def test_field_enhancement_factor_isotropic(self):
        """|E_z|/|E_x| on the dielectric side ≈ sqrt(|ε_m|/ε_d) (Maier Sec. 2.2)."""
        m = MetamaterialProperties(EPS_AG, EPS_AG, "z", omega=OMEGA)
        ratio = m.field_enhancement_factor(eps_dielectric=1.0)
        assert ratio == pytest.approx(math.sqrt(abs(EPS_AG)), rel=5e-3)


# ===================================================== B. anisotropic verification
ANISO_SETS = [
    pytest.param(-4.0 + 0.2j, 3.0 + 0.05j, 1.0, id="metal-dielectric-hyperbolic"),
    pytest.param(-18.3 + 0.55j, -18.0 + 0.5j, 1.0, id="near-isotropic-silver"),
    pytest.param(-2.5 + 0.3j, -12.0 + 1.0j, 2.25, id="strongly-anisotropic-glass"),
    pytest.param(-6.0 + 0.1j, 8.0 + 0.02j, 1.5, id="hyperbolic-type-II-like"),
]


class TestAnisotropicMatchingCondition:
    @pytest.mark.parametrize("eps_t, eps_n, eps_d", ANISO_SETS)
    def test_closed_form_satisfies_tm_matching(self, eps_t, eps_n, eps_d):
        """κ_d/ε_d + κ_m/ε_t = 0 (unsquared) at the repo's k_spp, Re κ > 0 branches."""
        m = make_uniaxial(eps_t, eps_n)
        k = m.spp_wavevector(eps_dielectric=eps_d)
        assert matching_residual(k, eps_t, eps_n, eps_d) < 1e-10

    @pytest.mark.parametrize("eps_t, eps_n, eps_d", ANISO_SETS)
    def test_repo_matches_independent_closed_form(self, eps_t, eps_n, eps_d):
        m = make_uniaxial(eps_t, eps_n)
        k = m.spp_wavevector(eps_dielectric=eps_d)
        k_ref = anisotropic_k_spp(eps_t, eps_n, eps_d)
        assert abs(k - k_ref) / abs(k_ref) < 1e-12
        assert k.imag > 0

    @pytest.mark.parametrize("eps_t, eps_n, eps_d", ANISO_SETS)
    def test_secant_root_find_recovers_closed_form(self, eps_t, eps_n, eps_d):
        """Newton/secant on the matching function from 1.2× the closed form
        converges back to the same root (no spurious squared-equation root)."""
        k_ref = anisotropic_k_spp(eps_t, eps_n, eps_d)
        k_root = secant_root(eps_t, eps_n, eps_d, k_start=1.2 * k_ref)
        assert abs(k_root - k_ref) / abs(k_ref) < 1e-8

    def test_near_isotropic_limit_agrees_with_isotropic_formula(self):
        """ε_t ≈ ε_n must reproduce the isotropic SPP dispersion."""
        eps_t, eps_n = -18.3 + 0.55j, -18.0 + 0.5j
        k_aniso = anisotropic_k_spp(eps_t, eps_n, 1.0)
        k_iso = isotropic_k_spp(eps_t, 1.0)
        assert abs(k_aniso - k_iso) / abs(k_iso) < 1e-3
        # Exactly isotropic parameters: identical to machine precision.
        k_exact = anisotropic_k_spp(EPS_AG, EPS_AG, 1.0)
        assert abs(k_exact - isotropic_k_spp(EPS_AG, 1.0)) / abs(k_exact) < 1e-14


class TestOpticalAxisOrientation:
    """`_spp_components` must give ε_t along propagation, ε_n along z."""

    EPS_PAR = 3.0 + 0.05j  # along the optical axis
    EPS_PERP = -4.0 + 0.2j  # perpendicular to the optical axis

    def test_axis_z_is_in_plane_isotropic(self):
        """Optical axis normal to the interface: k identical for 'x' and 'y'
        propagation (in-plane rotational symmetry), with ε_t = ε_⊥, ε_n = ε_∥."""
        m = MetamaterialProperties(self.EPS_PAR, self.EPS_PERP, "z", omega=OMEGA)
        kx = m.spp_wavevector(propagation_direction="x")
        ky = m.spp_wavevector(propagation_direction="y")
        assert kx == ky
        k_ref = anisotropic_k_spp(self.EPS_PERP, self.EPS_PAR, 1.0)
        assert abs(kx - k_ref) / abs(k_ref) < 1e-12

    @pytest.mark.parametrize("axis", ["x", "y"])
    def test_in_plane_axis_is_direction_dependent(self, axis):
        """In-plane optical axis: propagation along the axis mixes ε_∥ (in-plane)
        with ε_⊥ (normal); propagation across it sees only ε_⊥ — the ordinary/
        isotropic limit k₀ sqrt(ε_⊥ ε_d/(ε_⊥+ε_d))."""
        m = MetamaterialProperties(self.EPS_PAR, self.EPS_PERP, axis, omega=OMEGA)
        other = "y" if axis == "x" else "x"
        k_along = m.spp_wavevector(propagation_direction=axis)
        k_across = m.spp_wavevector(propagation_direction=other)

        # Across the axis: TM fields never sample ε_∥ -> isotropic ε_⊥ result.
        k_iso_perp = isotropic_k_spp(self.EPS_PERP, 1.0)
        assert abs(k_across - k_iso_perp) / abs(k_iso_perp) < 1e-12

        # Along the axis: extraordinary case with ε_t = ε_∥, ε_n = ε_⊥.
        k_extra = anisotropic_k_spp(self.EPS_PAR, self.EPS_PERP, 1.0)
        assert abs(k_along - k_extra) / abs(k_extra) < 1e-12

        # Anisotropy must actually show up.
        assert abs(k_along - k_across) / abs(k_across) > 1e-2

    def test_axis_x_and_y_are_mirror_images(self):
        """Swapping the in-plane axis and the propagation direction together
        leaves the physics unchanged."""
        mx = MetamaterialProperties(self.EPS_PAR, self.EPS_PERP, "x", omega=OMEGA)
        my = MetamaterialProperties(self.EPS_PAR, self.EPS_PERP, "y", omega=OMEGA)
        assert mx.spp_wavevector(propagation_direction="x") == my.spp_wavevector(
            propagation_direction="y"
        )
        assert mx.spp_wavevector(propagation_direction="y") == my.spp_wavevector(
            propagation_direction="x"
        )


# ======================================================== C. branch / edge behaviour
class TestResonanceSingularity:
    """ε_t ε_n → ε_d² makes the closed-form denominator vanish."""

    def test_exact_singularity_raises(self):
        # ε_t ε_n = (-2)(-0.5) = 1 = ε_d² exactly.
        m = make_uniaxial(-2.0, -0.5)
        with pytest.raises(ZeroDivisionError):
            m.spp_wavevector(eps_dielectric=1.0)

    @pytest.mark.parametrize("delta", [1e-1, 1e-2, 1e-4, 1e-6, 1e-8])
    def test_lossless_approach_stays_finite_and_bound(self, delta):
        """k² = 1.5(1+δ)/δ · k₀² → +∞ as δ → 0⁺: finite, Im ≥ 0, matching holds."""
        eps_t, eps_n = -2.0, -0.5 * (1.0 + delta)
        m = make_uniaxial(eps_t, eps_n)
        k = m.spp_wavevector(eps_dielectric=1.0)
        assert math.isfinite(abs(k))
        assert k.imag >= 0
        assert k.real > K0  # beyond the light line, increasingly confined
        assert matching_residual(k, eps_t, eps_n, 1.0) < 1e-10

    @pytest.mark.parametrize("delta", [1e-1, 1e-2, 1e-3])
    def test_lossy_approach_keeps_decaying_branch(self, delta):
        """With Im(ε_t) > 0 the root just off the singularity must keep Im(k) > 0."""
        eps_t = -2.0 + 0.05j
        eps_n = (1.0 + delta) / eps_t  # ε_t ε_n = 1 + δ, singular at δ = 0
        m = make_uniaxial(eps_t, eps_n)
        k = m.spp_wavevector(eps_dielectric=1.0)
        assert math.isfinite(abs(k))
        assert k.imag > 0
        assert matching_residual(k, eps_t, eps_n, 1.0) < 1e-10

    def test_diverging_confinement_towards_singularity(self):
        """|k| grows monotonically as the singularity is approached."""
        deltas = [1e-1, 1e-2, 1e-3, 1e-4]
        mags = []
        for delta in deltas:
            m = make_uniaxial(-2.0, -0.5 * (1.0 + delta))
            mags.append(abs(m.spp_wavevector(eps_dielectric=1.0)))
        assert mags == sorted(mags)


class TestLightLineUnbinding:
    """Isotropic lossless family: a bound SPP exists iff ε_m < −ε_d."""

    @pytest.mark.parametrize("eps_d", [1.0, 2.25], ids=["air", "glass"])
    def test_supported_flag_flips_at_minus_eps_d(self, eps_d):
        ratios_bound = [-1.05, -1.02, -1.001]
        ratios_unbound = [-0.999, -0.98, -0.95]
        for r in ratios_bound:
            m = MetamaterialProperties(r * eps_d, r * eps_d, "z", omega=OMEGA)
            assert m.is_spp_supported(eps_dielectric=eps_d) is True, r
        for r in ratios_unbound:
            m = MetamaterialProperties(r * eps_d, r * eps_d, "z", omega=OMEGA)
            assert m.is_spp_supported(eps_dielectric=eps_d) is False, r

    def test_flip_point_located_at_minus_eps_d(self):
        """Bisection on ε_m places the True/False transition at −ε_d ± 1e-9."""
        eps_d = 1.0

        def supported(eps_m: float) -> bool:
            m = MetamaterialProperties(eps_m, eps_m, "z", omega=OMEGA)
            return m.is_spp_supported(eps_dielectric=eps_d)

        lo, hi = -1.05, -0.95  # supported(lo) is True, supported(hi) is False
        assert supported(lo) and not supported(hi)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if supported(mid):
                lo = mid
            else:
                hi = mid
        assert hi - lo < 1e-12
        assert 0.5 * (lo + hi) == pytest.approx(-eps_d, abs=1e-9)

    def test_wavevector_crosses_light_line(self):
        """For ε_m < −ε_d the mode lies beyond the light line (k > sqrt(ε_d)k₀,
        bound); for −ε_d < ε_m < 0 the closed form gives k² < 0 (no bound mode)."""
        m_bound = MetamaterialProperties(-1.05, -1.05, "z", omega=OMEGA)
        k_bound = m_bound.spp_wavevector(eps_dielectric=1.0)
        assert k_bound.real > K0 and k_bound.imag == 0

        m_unbound = MetamaterialProperties(-0.95, -0.95, "z", omega=OMEGA)
        k_unbound = m_unbound.spp_wavevector(eps_dielectric=1.0)
        assert k_unbound.real == 0 and k_unbound.imag > 0  # purely evanescent in x
