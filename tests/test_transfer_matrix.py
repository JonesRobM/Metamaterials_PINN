r"""
Validation of :mod:`src.transfer_matrix` against everything the repository
already trusts.

The solver is new, so nothing here is a self-consistency check against itself.
Each group pins it to an independent authority:

(a) **Fresnel.** The TM amplitude reflection of a single interface is rebuilt
    from scratch in this file (the same algebra as
    ``tests/test_benchmark_fresnel.py``, H-amplitude convention) and compared to
    ``reflection_coefficient`` at several angles, for lossless, lossy and
    metallic second media. Agreement is required at ~1e-12, and Brewster's angle
    must come out as an exact zero.

(b) **Single-interface SPP.** With a silver half-space against air,
    ``find_mode`` must return exactly
    :meth:`~src.physics.metamaterial.MetamaterialProperties.spp_wavevector` —
    the benchmark-validated closed form — to ~1e-9 relative, from any of
    several seeds, and the mode's normal wavevectors must reproduce that class's
    ``decay_constants``. The same root is also read off
    :func:`src.analytical.analytical_spp_fields` through the phase advance of
    ``H_y`` along ``x``. This is the key correctness anchor: it is the case in
    which the transfer-matrix mode condition ``M₀₀ = 0`` reduces analytically to
    the repo's matching condition ``κ_d/ε_d + κ_m/ε_m = 0``.

(c) **Thin metal film.** A metal film between identical dielectrics supports the
    coupled long-range/short-range SPP pair. The two branches are tracked by
    continuation in the film thickness and their textbook behaviour asserted
    quantitatively: distinct at every thickness, both converging to the
    single-interface value as the film thickens, and splitting monotonically as
    it thins with the long-range branch moving *towards* the light line and
    *down* in loss.

(d) **Structural identities.** A zero-thickness layer is the identity, splitting
    a layer in two is exact, and a lossless stack conserves energy.

(e) **Piecewise ε(z) and the mode's field profile.** ``permittivity_profile``
    must return the stack layer by layer (including the right-continuous choice
    exactly *on* an interface), and ``mode_field_profile`` must (i) reduce to
    :func:`src.analytical.analytical_spp_fields` for a single interface to
    ~1e-9, (ii) keep ``H_y``, ``E_x`` and ``D_z`` continuous across every one of
    a 13-interface Ag/silica stack's boundaries while ``E_z`` jumps by exactly
    ``ε_below/ε_above``, (iii) decay in both half-spaces with the ``κ = Im k_z``
    the branch predicts, and (iv) satisfy ``H_y'' = (k_x² − ε k₀²) H_y`` inside
    each layer.

Sign convention throughout: ``exp(-iωt)``, ``Im ε > 0``, ``Im k_z ≥ 0``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.analytical import analytical_spp_fields
from src.constants import C0 as C_LIGHT
from src.constants import EPS0
from src.physics.metamaterial import MetamaterialProperties
from src.transfer_matrix import (
    find_mode,
    find_modes,
    interface_matrix,
    layer_boundaries,
    layer_index_at,
    mode_dispersion_function,
    mode_field_profile,
    muller,
    normal_wavevector,
    permittivity_profile,
    power_coefficients,
    reflection_coefficient,
    reflection_transmission,
    scan_modes,
    stack_matrix,
    tm_ratio,
)

LAM = 633e-9
# Derive k0 the way the repo does (k0 = omega / C0) so that every module in the
# comparison sees the *same* float; src.constants.C0 is 1/sqrt(eps0*mu0), which
# differs from the SI-exact 299792458 in the last few ulp.
OMEGA = 2.0 * math.pi * C_LIGHT / LAM
K0 = OMEGA / C_LIGHT

EPS_AIR = 1.0
EPS_SILICA = 2.25
EPS_SILICA_LOSSY = 2.25 + 0.1j
EPS_AG = -18.3 + 0.55j  # Johnson & Christy silver at 633 nm
EPS_AG_800 = -30.0 + 1.2j

ANGLES_DEG = [0.0, 15.0, 30.0, 55.0, 75.0]


# --------------------------------------------------------------------------
# Independent Fresnel construction (no repo physics code)
# --------------------------------------------------------------------------
def ref_kz(eps: complex, kx: complex, k0: float = K0) -> complex:
    """z-wavenumber on the ``Im >= 0`` branch (``Re >= 0`` when real)."""
    v = complex(np.sqrt(complex(eps) * k0**2 - complex(kx) ** 2 + 0j))
    if v.imag < 0 or (v.imag == 0 and v.real < 0):
        v = -v
    return v


def ref_fresnel_tm(eps1: complex, eps2: complex, kx: complex, k0: float = K0):
    """``(r_p, t_p)`` in the H-amplitude convention: ``r = (ε₂k₁ − ε₁k₂)/(ε₂k₁ + ε₁k₂)``."""
    k1z, k2z = ref_kz(eps1, kx, k0), ref_kz(eps2, kx, k0)
    r = (eps2 * k1z - eps1 * k2z) / (eps2 * k1z + eps1 * k2z)
    return complex(r), complex(1.0 + r)


def kx_of_angle(eps1: complex, theta_deg: float, k0: float = K0) -> float:
    return float(np.sqrt(complex(eps1).real) * k0 * math.sin(math.radians(theta_deg)))


# ==========================================================================
# (a) Fresnel
# ==========================================================================
class TestFresnel:
    @pytest.mark.parametrize(
        "eps2", [EPS_SILICA, EPS_SILICA_LOSSY, EPS_AG], ids=["lossless", "lossy", "metal"]
    )
    @pytest.mark.parametrize("theta", ANGLES_DEG)
    def test_single_interface_matches_analytic_tm_fresnel(self, theta, eps2):
        kx = kx_of_angle(EPS_AIR, theta)
        r_ref, t_ref = ref_fresnel_tm(EPS_AIR, eps2, kx)
        r, t = reflection_transmission(kx, K0, [EPS_AIR, eps2])
        assert complex(r) == pytest.approx(r_ref, rel=1e-12, abs=1e-13)
        assert complex(t) == pytest.approx(t_ref, rel=1e-12, abs=1e-13)

    @pytest.mark.parametrize("eps1,eps2", [(EPS_SILICA, EPS_AIR), (EPS_SILICA, EPS_AG)])
    @pytest.mark.parametrize("theta", ANGLES_DEG)
    def test_reversed_and_dense_incidence(self, eps1, eps2, theta):
        """Incidence from the denser medium, including beyond the critical angle."""
        kx = kx_of_angle(eps1, theta)
        r_ref, _ = ref_fresnel_tm(eps1, eps2, kx)
        r = reflection_coefficient(kx, K0, [eps1, eps2])
        assert complex(r) == pytest.approx(r_ref, rel=1e-12, abs=1e-13)

    def test_brewster_angle_is_an_exact_zero(self):
        n1, n2 = math.sqrt(EPS_AIR), math.sqrt(EPS_SILICA)
        kx = n1 * K0 * math.sin(math.atan(n2 / n1))
        assert abs(complex(reflection_coefficient(kx, K0, [EPS_AIR, EPS_SILICA]))) < 1e-14

    def test_total_internal_reflection_has_unit_modulus(self):
        theta_c = math.degrees(math.asin(math.sqrt(EPS_AIR / EPS_SILICA)))
        kx = kx_of_angle(EPS_SILICA, theta_c + 10.0)
        r = complex(reflection_coefficient(kx, K0, [EPS_SILICA, EPS_AIR]))
        assert abs(r) == pytest.approx(1.0, abs=1e-14)

    def test_normal_wavevector_branch_matches_the_repo_convention(self):
        """``Im k_z >= 0`` — equivalently ``Re(-i k_z) >= 0``, the decaying branch."""
        for eps in (EPS_AIR, EPS_SILICA, EPS_SILICA_LOSSY, EPS_AG):
            for theta in ANGLES_DEG:
                kx = kx_of_angle(EPS_SILICA, theta)
                kz = complex(normal_wavevector(eps, kx, K0))
                assert kz == pytest.approx(ref_kz(eps, kx), rel=1e-14, abs=1e-14)
                assert kz.imag >= 0.0
                assert (-1j * kz).real >= -1e-30

    def test_vectorised_over_kx(self):
        kx = np.linspace(0.0, 0.9 * K0, 17)
        r = reflection_coefficient(kx, K0, [EPS_AIR, EPS_AG])
        assert r.shape == kx.shape
        assert r.dtype == np.complex128
        for i, value in enumerate(kx):
            assert complex(r[i]) == pytest.approx(
                ref_fresnel_tm(EPS_AIR, EPS_AG, value)[0], rel=1e-12, abs=1e-13
            )

    def test_multilayer_reflection_matches_a_hand_rolled_two_interface_sum(self):
        """A single film against the closed-form Airy sum, built independently here."""
        eps1, eps_film, eps3, d = EPS_AIR, EPS_SILICA, 4.0, 137e-9
        kx = kx_of_angle(eps1, 35.0)
        r12, _ = ref_fresnel_tm(eps1, eps_film, kx)
        r23, _ = ref_fresnel_tm(eps_film, eps3, kx)
        phase = np.exp(2j * ref_kz(eps_film, kx) * d)
        airy = (r12 + r23 * phase) / (1.0 + r12 * r23 * phase)
        r = reflection_coefficient(kx, K0, [eps1, eps_film, eps3], [d])
        assert complex(r) == pytest.approx(complex(airy), rel=1e-12, abs=1e-13)


# ==========================================================================
# (b) Single-interface SPP — the correctness anchor
# ==========================================================================
def isotropic_material(eps_metal: complex, omega: float = OMEGA) -> MetamaterialProperties:
    """``MetamaterialProperties`` degenerated to an isotropic metal."""
    return MetamaterialProperties(eps_metal, eps_metal, optical_axis="z", omega=omega)


def wavenumbers(lam: float):
    """``(omega, k0)`` consistent with ``MetamaterialProperties._resolve_k0``."""
    omega = 2.0 * math.pi * C_LIGHT / lam
    return omega, omega / C_LIGHT


class TestSingleInterfaceSPP:
    CASES = [(EPS_AG, LAM), (EPS_AG_800, 800e-9), (-9.0 + 0.3j, 500e-9)]

    @pytest.mark.parametrize("eps_metal,lam", CASES)
    @pytest.mark.parametrize("seed_scale", [0.85, 0.999, 1.0, 1.05, 1.3])
    def test_find_mode_recovers_spp_wavevector(self, eps_metal, lam, seed_scale):
        omega, k0 = wavenumbers(lam)
        material = isotropic_material(eps_metal, omega)
        k_ref = material.spp_wavevector(eps_dielectric=EPS_AIR)
        # Layers in increasing z: metal below (z < 0), air above.
        root = find_mode(k_ref * seed_scale, k0, [eps_metal, EPS_AIR])
        assert root is not None
        assert abs(root - k_ref) / abs(k_ref) < 1e-9

    @pytest.mark.parametrize("eps_metal,lam", CASES)
    def test_mode_decay_constants_match_the_class(self, eps_metal, lam):
        """``κ = −i k_z`` on both sides must equal ``decay_constants``."""
        omega, k0 = wavenumbers(lam)
        material = isotropic_material(eps_metal, omega)
        k_ref, kappa_d_ref, kappa_m_ref = material.decay_constants(eps_dielectric=EPS_AIR)
        root = find_mode(k_ref, k0, [eps_metal, EPS_AIR])
        kappa_m = -1j * complex(normal_wavevector(eps_metal, root, k0))
        kappa_d = -1j * complex(normal_wavevector(EPS_AIR, root, k0))
        assert kappa_d == pytest.approx(kappa_d_ref, rel=1e-9)
        assert kappa_m == pytest.approx(kappa_m_ref, rel=1e-9)
        assert kappa_d.real > 0.0 and kappa_m.real > 0.0

    def test_mode_condition_is_the_repo_matching_condition(self):
        """``M₀₀ = 0`` ⟺ ``κ_d/ε_d + κ_m/ε_m = 0`` for a bare interface."""
        material = isotropic_material(EPS_AG)
        k = material.spp_wavevector(eps_dielectric=EPS_AIR)
        _, kappa_d, kappa_m = material.decay_constants(eps_dielectric=EPS_AIR)
        matching = kappa_d / EPS_AIR + kappa_m / EPS_AG
        scale = abs(kappa_d / EPS_AIR) + abs(kappa_m / EPS_AG)
        assert abs(matching) < 1e-9 * scale
        # ...and the transfer-matrix element vanishes there to machine precision.
        value = complex(mode_dispersion_function(k, K0, [EPS_AG, EPS_AIR]))
        away = complex(mode_dispersion_function(k * 1.02, K0, [EPS_AG, EPS_AIR]))
        assert abs(value) < 1e-14 * abs(away)

    def test_matches_the_analytical_spp_field_phase(self):
        """Read ``k_spp`` off :func:`analytical_spp_fields` and compare to the root."""
        material = isotropic_material(EPS_AG)
        k_ref = material.spp_wavevector(eps_dielectric=EPS_AIR)
        root = find_mode(k_ref * 1.1, K0, [EPS_AG, EPS_AIR])

        step = 1e-8  # metres, well inside one SPP wavelength
        coords = torch.tensor(
            [[0.0, 0.0, 1e-9], [step, 0.0, 1e-9]], dtype=torch.float64
        )
        _, H = analytical_spp_fields(coords, OMEGA, EPS_AG, EPS_AG, EPS_AIR)
        ratio = complex(H[1, 1] / H[0, 1])
        k_from_fields = -1j * np.log(ratio) / step
        assert abs(k_from_fields - root) / abs(root) < 1e-9

    def test_scan_finds_the_mode_without_a_seed(self):
        material = isotropic_material(EPS_AG)
        k_ref = material.spp_wavevector(eps_dielectric=EPS_AIR)
        roots = scan_modes(
            K0, [EPS_AG, EPS_AIR], k_x_re_range=(1.001 * K0, 3.0 * K0), n_re=400
        )
        assert roots.size >= 1
        assert min(abs(r - k_ref) for r in roots) / abs(k_ref) < 1e-9

    def test_no_bound_mode_between_two_dielectrics(self):
        """TM surface modes need a negative permittivity; the guard must hold."""
        roots = scan_modes(
            K0, [EPS_SILICA, EPS_AIR], k_x_re_range=(1.001 * K0, 4.0 * K0), n_re=400
        )
        assert roots.size == 0

    def test_leaky_root_rejected_unless_requested(self):
        """A metal/air mode seen from a dense substrate is not bound; the flag decides."""
        material = isotropic_material(EPS_AG)
        k_ref = material.spp_wavevector(eps_dielectric=EPS_AIR)
        # n_eff ≈ 1.03 < sqrt(4.0): the field oscillates in the substrate.
        stack = ([4.0, EPS_AG, EPS_AIR], [400e-9])
        assert find_mode(k_ref, K0, *stack, require_bound=True) is None
        leaky = find_mode(k_ref, K0, *stack, require_bound=False)
        assert leaky is not None
        assert abs(leaky - k_ref) / abs(k_ref) < 1e-3


# ==========================================================================
# (c) Thin metal film: the coupled long-range / short-range pair
# ==========================================================================
THICKNESSES = np.geomspace(10e-9, 400e-9, 24)
SEPARATED = THICKNESSES <= 130e-9  # branches resolvable well above float64 noise


def _imi_stack(thickness: float):
    return [EPS_AIR, EPS_AG, EPS_AIR], [float(thickness)]


def _long_range_seed(thickness: float) -> complex:
    r"""
    Thin-film asymptote of the *symmetric* (long-range) branch.

    ``tanh(κ_m d/2) = −ε_m κ_d/(ε_d κ_m)`` with a small argument gives
    ``κ_d ≈ −ε_d κ_m² d /(2 ε_m)`` and ``κ_m² ≈ (ε_d − ε_m) k₀²``, i.e. a mode
    just above the light line — the defining feature of the long-range branch.
    """
    kappa_d = 0.5 * EPS_AIR * (EPS_AIR - EPS_AG) * K0**2 * thickness / (-EPS_AG)
    return complex(np.sqrt(EPS_AIR * K0**2 + kappa_d**2))


def _track(thicknesses, seed) -> np.ndarray:
    """Follow one branch along ``thicknesses``, extrapolating the guess in log d."""
    history: list[tuple[float, complex]] = []
    roots = np.empty(len(thicknesses), dtype=complex)
    for i, d in enumerate(thicknesses):
        if not history:
            guess = seed
        elif len(history) == 1:
            guess = history[-1][1]
        else:
            (d1, r1), (d2, r2) = history[-2], history[-1]
            guess = r2 + (r2 - r1) * (math.log(d) - math.log(d2)) / (
                math.log(d2) - math.log(d1)
            )
        root = find_mode(complex(guess), K0, *_imi_stack(d))
        assert root is not None, f"lost the branch at d = {d * 1e9:.1f} nm"
        roots[i] = root
        history.append((float(d), complex(root)))
    return roots


@pytest.fixture(scope="module")
def imi_branches():
    """``(long_range, short_range)`` roots over :data:`THICKNESSES`."""
    long_range = _track(THICKNESSES, _long_range_seed(float(THICKNESSES[0])))
    seeded = scan_modes(
        K0, *_imi_stack(float(THICKNESSES[0])),
        k_x_re_range=(1.0001 * K0, 60.0 * K0), n_re=4000,
    )
    assert seeded.size >= 1
    short_range = _track(THICKNESSES, complex(seeded[-1]))
    return long_range, short_range


@pytest.fixture(scope="module")
def k_single():
    """The bare Ag/air SPP both film branches must converge to."""
    return isotropic_material(EPS_AG).spp_wavevector(eps_dielectric=EPS_AIR)


class TestThinMetalFilm:
    def test_two_distinct_roots_exist(self, imi_branches):
        long_range, short_range = imi_branches
        separation = np.abs(long_range - short_range) / np.abs(long_range)
        assert np.all(separation[SEPARATED] > 1e-4)

    def test_both_branches_converge_to_the_single_interface_mode(
        self, imi_branches, k_single
    ):
        long_range, short_range = imi_branches
        for branch in imi_branches:
            error = np.abs(branch - k_single) / abs(k_single)
            assert error[-1] < 1e-8, "thick film must reproduce the bare interface"
            # ...and does so monotonically over the well-separated range.
            resolved = error[SEPARATED]
            assert np.all(np.diff(resolved) < 0.0)
        assert long_range[0] != short_range[0]

    def test_long_range_branch_moves_to_the_light_line_and_loses_less(
        self, imi_branches, k_single
    ):
        long_range, _ = imi_branches
        resolved = long_range[SEPARATED]
        # Thinner film ⇒ smaller Re k (towards the light line) and smaller Im k.
        assert np.all(np.diff(resolved.real) > 0.0)
        assert np.all(np.diff(resolved.imag) > 0.0)
        assert np.all(resolved.real > math.sqrt(EPS_AIR) * K0)
        assert np.all(resolved.real < k_single.real)
        assert np.all(resolved.imag < k_single.imag)
        # "Long range" means literally that: the propagation length grows.
        length = 1.0 / (2.0 * resolved.imag)
        assert length[0] / length[-1] > 50.0

    def test_short_range_branch_confines_and_loses_more(self, imi_branches, k_single):
        _, short_range = imi_branches
        resolved = short_range[SEPARATED]
        assert np.all(np.diff(resolved.real) < 0.0)
        assert np.all(np.diff(resolved.imag) < 0.0)
        assert np.all(resolved.real > k_single.real)
        assert np.all(resolved.imag > k_single.imag)

    def test_the_pair_straddles_the_single_interface_mode(self, imi_branches, k_single):
        long_range, short_range = imi_branches
        assert np.all(long_range[SEPARATED].imag < k_single.imag)
        assert np.all(short_range[SEPARATED].imag > k_single.imag)
        assert np.all(long_range[SEPARATED].imag < short_range[SEPARATED].imag)

    def test_splitting_shrinks_exponentially_with_thickness(self, imi_branches):
        """Coupling through an evanescent film ⇒ splitting ∝ exp(−2 κ_m d)."""
        long_range, short_range = imi_branches
        splitting = np.abs(long_range - short_range)[SEPARATED]
        assert np.all(np.diff(splitting) < 0.0)
        assert splitting[0] / splitting[-1] > 100.0

    def test_find_modes_deduplicates_and_sorts(self, k_single):
        d = 30e-9
        seeds = [_long_range_seed(d), _long_range_seed(d) * 1.0000001, 3.0 * K0]
        roots = find_modes(seeds, K0, *_imi_stack(d))
        assert roots.size == 2
        assert roots[0].real < roots[1].real


# ==========================================================================
# (d) Structural identities
# ==========================================================================
STACK_EPS = [EPS_AIR, EPS_SILICA, EPS_AG, 4.0]
STACK_D = [50e-9, 30e-9]


class TestStackAlgebra:
    @pytest.mark.parametrize("kx_over_k0", [0.0, 0.7, 1.4, 3.0])
    def test_zero_thickness_layer_is_the_identity(self, kx_over_k0):
        """``D(a→b) D(b→c) = D(a→c)``: an interposed film of no thickness is invisible."""
        kx = kx_over_k0 * K0
        base = stack_matrix(kx, K0, STACK_EPS, STACK_D)
        padded = stack_matrix(
            kx, K0, [EPS_AIR, EPS_SILICA, 7.0 - 2.0j, EPS_AG, 4.0], [50e-9, 0.0, 30e-9]
        )
        scale = np.abs(base.matrix).max()
        assert np.abs(padded.matrix - base.matrix).max() < 1e-13 * scale
        assert padded.log_scale == pytest.approx(base.log_scale, abs=1e-30)

    @pytest.mark.parametrize("kx_over_k0", [0.0, 0.7, 1.4, 3.0])
    def test_splitting_a_layer_in_half_changes_nothing(self, kx_over_k0):
        kx = kx_over_k0 * K0
        base = stack_matrix(kx, K0, STACK_EPS, STACK_D)
        split = stack_matrix(
            kx, K0, [EPS_AIR, EPS_SILICA, EPS_SILICA, EPS_AG, 4.0], [25e-9, 25e-9, 30e-9]
        )
        scale = np.abs(base.matrix).max()
        assert np.abs(split.matrix - base.matrix).max() < 1e-13 * scale
        assert complex(split.log_scale) == pytest.approx(complex(base.log_scale), rel=1e-14)

    def test_interface_matrices_compose(self):
        p = [tm_ratio(normal_wavevector(e, 0.6 * K0, K0), e) for e in (1.0, 2.25, 12.0)]
        direct = interface_matrix(p[0], p[2])
        composed = interface_matrix(p[0], p[1]) @ interface_matrix(p[1], p[2])
        assert np.abs(composed - direct).max() < 1e-14

    @pytest.mark.parametrize("theta", [0.0, 20.0, 40.0, 60.0])
    def test_energy_conservation_for_a_lossless_stack(self, theta):
        kx = kx_of_angle(EPS_AIR, theta)
        reflectance, transmittance = power_coefficients(
            kx, K0, [EPS_AIR, EPS_SILICA, 4.0, EPS_SILICA], [80e-9, 120e-9]
        )
        assert float(reflectance) + float(transmittance) == pytest.approx(1.0, abs=1e-13)

    @pytest.mark.parametrize("theta", [0.0, 30.0, 60.0])
    def test_lossy_stack_absorbs(self, theta):
        kx = kx_of_angle(EPS_AIR, theta)
        reflectance, transmittance = power_coefficients(
            kx, K0, [EPS_AIR, EPS_AG, EPS_SILICA], [20e-9]
        )
        total = float(reflectance) + float(transmittance)
        assert 0.0 < total < 1.0

    def test_unscaled_matrix_reproduces_the_scaled_one(self):
        kx = 0.6 * K0
        stack = stack_matrix(kx, K0, STACK_EPS, STACK_D)
        expected = stack.matrix * np.exp(-stack.log_scale)
        assert np.abs(stack.unscaled - expected).max() == 0.0
        # r = M10/M00 is invariant under the rescaling, by construction.
        ratio = stack.unscaled[1, 0] / stack.unscaled[0, 0]
        assert complex(ratio) == pytest.approx(
            complex(reflection_coefficient(kx, K0, STACK_EPS, STACK_D)), rel=1e-12
        )

    def test_batched_kx_matches_scalar_calls(self):
        kx = np.array([0.0, 0.5, 1.1, 2.4]) * K0
        batched = mode_dispersion_function(kx, K0, STACK_EPS, STACK_D)
        assert batched.shape == kx.shape
        for i, value in enumerate(kx):
            scalar = mode_dispersion_function(value, K0, STACK_EPS, STACK_D)
            assert complex(batched[i]) == pytest.approx(complex(scalar), rel=1e-14)

    @pytest.mark.parametrize(
        "eps_layers,thicknesses",
        [([1.0], []), ([1.0, 2.0], [10e-9]), ([1.0, 2.0, 3.0], []), ([1.0, 2.0, 3.0], [-1e-9])],
    )
    def test_malformed_stacks_are_rejected(self, eps_layers, thicknesses):
        with pytest.raises(ValueError):
            stack_matrix(0.5 * K0, K0, eps_layers, thicknesses)


class TestMuller:
    def test_finds_a_complex_polynomial_root(self):
        """``z³ − 1`` from a real seed: Muller must leave the real axis on its own."""
        root, converged, _ = muller(lambda z: z**3 - 1.0, 0.4 + 0.2j)
        assert converged
        assert abs(root**3 - 1.0) < 1e-12

    def test_reports_failure_on_a_function_without_roots(self):
        root, converged, _ = muller(lambda z: np.exp(z) + 2.0 + 0j, 1.0, tol=1e-14, max_iter=6)
        assert not converged or abs(np.exp(root) + 2.0) > 1e-9


# ==========================================================================
# (e) Piecewise ε(z) and the mode's field profile
# ==========================================================================
FILL = 0.30
PERIOD = 30e-9
N_PERIODS = 6


def _multilayer(n_periods: int = N_PERIODS, period: float = PERIOD):
    """Ag/silica stack, metal-terminated against air, on a semi-infinite Ag substrate.

    Ordered by increasing z, as :func:`stack_matrix` requires, and identical in
    construction to ``examples/emt_validity.multilayer_stack`` for the
    ``'metal'`` termination.
    """
    d_metal, d_diel = FILL * period, (1.0 - FILL) * period
    eps_layers = [EPS_AG]
    thicknesses = []
    for _ in range(n_periods):
        eps_layers += [EPS_SILICA, EPS_AG]
        thicknesses += [d_diel, d_metal]
    eps_layers += [EPS_AIR]
    return eps_layers, thicknesses


class TestPermittivityProfile:
    def test_boundaries_are_the_running_sum_of_the_thicknesses(self):
        bounds = layer_boundaries([10e-9, 25e-9, 5e-9], z0=-40e-9)
        assert bounds == pytest.approx([-40e-9, -30e-9, -5e-9, 0.0])

    def test_single_interface_has_one_boundary_at_z0(self):
        assert layer_boundaries([], z0=0.0) == pytest.approx([0.0])

    def test_profile_is_the_stack_layer_by_layer(self):
        eps_layers, thicknesses = _multilayer(2)
        z0 = -sum(thicknesses)
        bounds = layer_boundaries(thicknesses, z0)
        # A point in the middle of every finite layer, plus one in each
        # semi-infinite medium.
        mid = 0.5 * (bounds[:-1] + bounds[1:])
        probes = np.concatenate([[bounds[0] - 50e-9], mid, [bounds[-1] + 50e-9]])
        got = permittivity_profile(probes, eps_layers, thicknesses, z0=z0)
        assert got == pytest.approx(np.asarray(eps_layers, dtype=complex))

    def test_exactly_on_an_interface_takes_the_medium_above(self):
        """The documented right-continuous convention, and the one the PINN uses."""
        eps_layers, thicknesses = _multilayer(2)
        z0 = -sum(thicknesses)
        bounds = layer_boundaries(thicknesses, z0)
        on_interface = permittivity_profile(bounds, eps_layers, thicknesses, z0=z0)
        just_above = permittivity_profile(
            bounds + 1e-15, eps_layers, thicknesses, z0=z0
        )
        just_below = permittivity_profile(
            bounds - 1e-15, eps_layers, thicknesses, z0=z0
        )
        assert on_interface == pytest.approx(just_above)
        assert np.all(on_interface != just_below)
        # ... including the topmost interface at z = 0, which must read as air.
        assert complex(on_interface[-1]) == EPS_AIR

    def test_layer_index_spans_every_medium_exactly_once(self):
        eps_layers, thicknesses = _multilayer(3)
        z0 = -sum(thicknesses)
        z = np.linspace(z0 - 20e-9, 20e-9, 5000)
        idx = layer_index_at(z, thicknesses, z0)
        assert idx.min() == 0
        assert idx.max() == len(eps_layers) - 1
        assert np.all(np.diff(idx) >= 0)  # monotone in z

    def test_zero_thickness_layer_is_never_selected(self):
        """A layer of no thickness occupies no z, so the profile skips it."""
        eps_layers = [EPS_AIR, EPS_SILICA, 7.0 - 2.0j, EPS_AG, 4.0]
        thicknesses = [50e-9, 0.0, 30e-9]
        z = np.linspace(-20e-9, 100e-9, 401)
        assert not np.any(permittivity_profile(z, eps_layers, thicknesses) == 7.0 - 2.0j)


@pytest.fixture(scope="module")
def multilayer_mode():
    """``(k_x, eps_layers, thicknesses, z0)`` of the Ag/silica stack's bound SPP."""
    eps_layers, thicknesses = _multilayer()
    z0 = -sum(thicknesses)  # topmost interface at z = 0
    root = find_mode(1.1 * K0, K0, eps_layers, thicknesses)
    assert root is not None
    return complex(root), eps_layers, thicknesses, z0


class TestModeFieldProfile:
    def test_reduces_to_the_analytical_single_interface_mode(self):
        """The correctness anchor: one interface must reproduce ``analytical_spp_fields``."""
        material = isotropic_material(EPS_AG)
        k_ref = material.spp_wavevector(eps_dielectric=EPS_AIR)
        root = find_mode(k_ref, K0, [EPS_AG, EPS_AIR])
        assert root is not None

        z = np.linspace(-150e-9, 400e-9, 501)
        profile = mode_field_profile(root, K0, [EPS_AG, EPS_AIR], (), z, omega=OMEGA)

        coords = torch.tensor(
            np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1), dtype=torch.float64
        )
        E, H = analytical_spp_fields(coords, OMEGA, EPS_AG, EPS_AG, EPS_AIR, H0=1.0)
        for got, want in (
            (profile.H_y, H[:, 1].numpy()),
            (profile.E_x, E[:, 0].numpy()),
            (profile.E_z, E[:, 2].numpy()),
        ):
            assert np.linalg.norm(got - want) / np.linalg.norm(want) < 1e-9

    def test_H_y_E_x_and_D_z_are_continuous_at_every_interface(self, multilayer_mode):
        r"""
        The methodological point of the layered PINN: only ``E_z`` jumps.

        ``H_y`` and ``E_x`` are what the interface matrix matches, and
        ``D_z = ε₀ ε E_z = −k_x H_y / ω`` inherits the continuity of ``H_y``.
        """
        k_x, eps_layers, thicknesses, z0 = multilayer_mode
        bounds = layer_boundaries(thicknesses, z0)
        delta = 1e-15  # ~1e-8 of the fields' own variation scale
        below = mode_field_profile(
            k_x, K0, eps_layers, thicknesses, bounds - delta, z0=z0, omega=OMEGA
        )
        above = mode_field_profile(
            k_x, K0, eps_layers, thicknesses, bounds + delta, z0=z0, omega=OMEGA
        )
        for name in ("H_y", "E_x", "D_z"):
            lo, hi = getattr(below, name), getattr(above, name)
            jump = np.abs(hi - lo) / np.abs(lo).max()
            assert jump.max() < 1e-7, f"{name} jumps by {jump.max():.2e}"

    def test_E_z_jumps_by_exactly_the_permittivity_ratio(self, multilayer_mode):
        k_x, eps_layers, thicknesses, z0 = multilayer_mode
        bounds = layer_boundaries(thicknesses, z0)
        delta = 1e-15
        below = mode_field_profile(
            k_x, K0, eps_layers, thicknesses, bounds - delta, z0=z0, omega=OMEGA
        )
        above = mode_field_profile(
            k_x, K0, eps_layers, thicknesses, bounds + delta, z0=z0, omega=OMEGA
        )
        ratio = above.E_z / below.E_z
        expected = below.eps / above.eps
        assert np.abs(ratio / expected - 1.0).max() < 1e-6
        # Every interface of this stack is a genuine jump, not a cosmetic one.
        assert np.abs(ratio - 1.0).min() > 0.5

    def test_D_z_equals_minus_kx_Hy_over_omega(self, multilayer_mode):
        """``∇×H`` in the ``z`` direction, with no ``ε`` left in it."""
        k_x, eps_layers, thicknesses, z0 = multilayer_mode
        z = np.linspace(z0 - 30e-9, 300e-9, 777)
        p = mode_field_profile(k_x, K0, eps_layers, thicknesses, z, z0=z0, omega=OMEGA)
        assert np.abs(p.D_z - (-k_x * p.H_y / OMEGA)).max() < 1e-14 * np.abs(p.D_z).max()
        assert np.abs(p.D_z - EPS0 * p.eps * p.E_z).max() < 1e-14 * np.abs(p.D_z).max()

    def test_decays_with_the_expected_kappa_in_both_half_spaces(self, multilayer_mode):
        k_x, eps_layers, thicknesses, z0 = multilayer_mode
        bounds = layer_boundaries(thicknesses, z0)
        for span, medium, sign in (
            ((bounds[-1] + 100e-9, bounds[-1] + 400e-9), -1, -1.0),
            ((bounds[0] - 60e-9, bounds[0] - 5e-9), 0, +1.0),
        ):
            z = np.linspace(*span, 200)
            p = mode_field_profile(k_x, K0, eps_layers, thicknesses, z, z0=z0, omega=OMEGA)
            slope = np.polyfit(z, np.log(np.abs(p.H_y)), 1)[0]
            kappa = p.k_z[medium].imag  # κ = −i k_z, so Re κ = Im k_z
            assert slope == pytest.approx(sign * kappa, rel=1e-6)
            assert kappa > 0.0  # bound on this branch

    def test_leakage_vanishes_at_a_mode_and_not_beside_one(self, multilayer_mode):
        k_x, eps_layers, thicknesses, z0 = multilayer_mode
        z = np.array([0.0])
        at_mode = mode_field_profile(
            k_x, K0, eps_layers, thicknesses, z, z0=z0, omega=OMEGA
        )
        beside = mode_field_profile(
            1.02 * k_x, K0, eps_layers, thicknesses, z, z0=z0, omega=OMEGA
        )
        assert at_mode.leakage < 1e-10
        assert beside.leakage > 1e-3

    def test_normalisation_puts_H0_at_the_top_interface(self, multilayer_mode):
        k_x, eps_layers, thicknesses, z0 = multilayer_mode
        bounds = layer_boundaries(thicknesses, z0)
        p = mode_field_profile(
            k_x, K0, eps_layers, thicknesses, bounds[-1:], z0=z0, omega=OMEGA, H0=2.5
        )
        assert complex(p.H_y[0]) == pytest.approx(2.5 + 0j, abs=1e-12)
        # ... and elsewhere if asked.
        q = mode_field_profile(
            k_x, K0, eps_layers, thicknesses, np.array([-45e-9]), z0=z0, omega=OMEGA,
            h0_at=-45e-9,
        )
        assert complex(q.H_y[0]) == pytest.approx(1.0 + 0j, abs=1e-12)

    def test_profile_satisfies_the_wave_equation_inside_each_layer(self, multilayer_mode):
        r"""``d²H_y/dz² = (k_x² − ε k₀²) H_y`` — checked by finite differences."""
        k_x, eps_layers, thicknesses, z0 = multilayer_mode
        bounds = layer_boundaries(thicknesses, z0)
        centres = 0.5 * (bounds[:-1] + bounds[1:])
        h = 1e-11
        z = np.concatenate([centres - h, centres, centres + h])
        p = mode_field_profile(k_x, K0, eps_layers, thicknesses, z, z0=z0, omega=OMEGA)
        n = centres.size
        second = (p.H_y[:n] - 2.0 * p.H_y[n : 2 * n] + p.H_y[2 * n :]) / h**2
        expected = (k_x**2 - p.eps[n : 2 * n] * K0**2) * p.H_y[n : 2 * n]
        assert np.abs(second - expected).max() < 1e-6 * np.abs(expected).max()

    def test_the_permittivity_it_reports_is_the_stack_profile(self, multilayer_mode):
        k_x, eps_layers, thicknesses, z0 = multilayer_mode
        z = np.linspace(z0 - 30e-9, 200e-9, 999)
        p = mode_field_profile(k_x, K0, eps_layers, thicknesses, z, z0=z0, omega=OMEGA)
        assert p.eps == pytest.approx(
            permittivity_profile(z, eps_layers, thicknesses, z0=z0)
        )

    def test_default_grid_spans_the_stack(self, multilayer_mode):
        k_x, eps_layers, thicknesses, z0 = multilayer_mode
        p = mode_field_profile(k_x, K0, eps_layers, thicknesses, z0=z0, omega=OMEGA)
        assert p.z.min() < p.boundaries[0]
        assert p.z.max() > p.boundaries[-1]

    def test_rejects_a_singular_normalisation_point(self, multilayer_mode):
        k_x, eps_layers, thicknesses, z0 = multilayer_mode
        with pytest.raises(ValueError, match="cannot normalise"):
            mode_field_profile(
                k_x, K0, eps_layers, thicknesses, np.array([0.0]), z0=z0, omega=OMEGA,
                h0_at=1e30,  # the excluded exponential overflows this far out
            )

    def test_omega_defaults_to_k0_times_c(self, multilayer_mode):
        k_x, eps_layers, thicknesses, z0 = multilayer_mode
        z = np.linspace(-100e-9, 100e-9, 51)
        default = mode_field_profile(k_x, K0, eps_layers, thicknesses, z, z0=z0)
        explicit = mode_field_profile(
            k_x, K0, eps_layers, thicknesses, z, z0=z0, omega=K0 * C_LIGHT
        )
        assert np.abs(default.E_x - explicit.E_x).max() == 0.0
