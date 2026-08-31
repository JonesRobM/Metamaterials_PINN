"""
Unit tests for the physics module.

Covers Maxwell's equations, the autograd differential operators, metamaterial
constitutive relations / SPP dispersion and the boundary conditions, all under
the ``exp(-iωt)`` convention (``∇×E = iωμ₀H``, ``∇×H = -iωε₀εᵣE``, lossy
``Im ε > 0``, decaying ``Im k > 0``).
"""

import cmath

import numpy as np
import pytest
import torch

from src.analytical import analytical_plane_wave
from src.constants import C0, EPS0, ETA0, MU0
from src.physics.boundary_conditions import BoundaryConditions
from src.physics.differential_ops import (
    curl,
    curl_complex,
    divergence,
    divergence_complex,
    gradient,
    jacobian,
)
from src.physics.maxwell_equations import MaxwellEquations
from src.physics.metamaterial import MetamaterialProperties
from tests.conftest import K0, OMEGA, sample_coords


# =========================================================================== Maxwell
class TestMaxwellEquations:
    """Test suite for Maxwell equations implementation."""

    @pytest.fixture
    def maxwell_solver(self):
        return MaxwellEquations(OMEGA)

    @pytest.fixture
    def sample_fields(self):
        torch.manual_seed(42)
        coords = torch.randn(10, 3, requires_grad=True)
        E_field = torch.complex(torch.randn(10, 3), torch.randn(10, 3)).requires_grad_(True)
        H_field = torch.complex(torch.randn(10, 3), torch.randn(10, 3)).requires_grad_(True)
        return coords, E_field, H_field

    def test_initialization(self, maxwell_solver):
        assert maxwell_solver.omega == OMEGA
        assert maxwell_solver.c == pytest.approx(C0)
        assert maxwell_solver.k0 == pytest.approx(K0)

    def test_invalid_omega(self):
        with pytest.raises(ValueError):
            MaxwellEquations(0.0)

    def test_curl_operator_shape(self, maxwell_solver, sample_fields):
        coords, E_field, _ = sample_fields
        assert maxwell_solver.curl_operator(E_field, coords).shape == E_field.shape

    def test_curl_of_gradient_is_zero(self, maxwell_solver):
        coords = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        phi = coords[:, 0] ** 2 + coords[:, 1] ** 2 + coords[:, 2] ** 2
        grad_phi = torch.autograd.grad(phi, coords, create_graph=True)[0]
        curl_grad = maxwell_solver.curl_operator(torch.complex(grad_phi, torch.zeros_like(grad_phi)), coords)
        assert torch.allclose(curl_grad, torch.zeros_like(curl_grad), atol=1e-6)

    def test_residual_shapes(self, maxwell_solver, sample_fields):
        coords, E_field, H_field = sample_fields
        eps = torch.eye(3, dtype=torch.complex64).unsqueeze(0).expand(10, -1, -1)
        assert maxwell_solver.curl_E_residual(E_field, H_field, coords).shape == (10, 6)
        assert maxwell_solver.curl_H_residual(E_field, H_field, coords, eps).shape == (10, 6)
        assert maxwell_solver.divergence_E_residual(E_field, coords, eps).shape == (10, 2)
        assert maxwell_solver.divergence_B_residual(H_field, coords).shape == (10, 2)
        assert maxwell_solver.total_residual(E_field, H_field, coords, eps).shape == (10, 16)

    def test_poynting_vector_shape_and_dtype(self, maxwell_solver, sample_fields):
        _, E_field, H_field = sample_fields
        S = maxwell_solver.poynting_vector(E_field, H_field)
        assert S.shape == E_field.shape
        assert S.dtype == torch.float32

    def test_curl_of_constant_field_is_zero(self, maxwell_solver):
        coords = torch.randn(5, 3, requires_grad=True)
        constant_field = torch.complex(torch.full((5, 3), 1.0), torch.full((5, 3), 2.0))
        curl_val = maxwell_solver.curl_operator(constant_field, coords)
        assert torch.allclose(curl_val, torch.zeros_like(curl_val), atol=1e-6)

    @pytest.mark.parametrize("imag", [False, True])
    def test_curl_of_linear_field(self, maxwell_solver, imag):
        """F = (0, 0, x) (or i x) has curl (0, -1, 0) (or (0, -i, 0))."""
        coords = torch.randn(5, 3, requires_grad=True)
        part = torch.zeros_like(coords)
        part[:, 2] = coords[:, 0]
        zero = torch.zeros_like(coords)
        field = torch.complex(zero, part) if imag else torch.complex(part, zero)
        expected = torch.zeros(5, 3, dtype=torch.complex64)
        expected[:, 1] = -1j if imag else -1.0
        assert torch.allclose(maxwell_solver.curl_operator(field, coords), expected, atol=1e-6)

    @pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-9), (torch.float32, 1e-4)])
    def test_plane_wave_residuals_vanish(self, dtype, tol):
        """The analytical vacuum plane wave satisfies all four Maxwell residuals."""
        coords = sample_coords(48, dtype=dtype)
        cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        k_vec = torch.tensor([0.6, 0.8, 0.0], dtype=dtype) * K0
        E0 = torch.tensor([0.0, 0.0, 1.0], dtype=cdtype)
        E, H = analytical_plane_wave(coords, k_vec, E0, OMEGA)
        eps = torch.eye(3, dtype=cdtype).unsqueeze(0).expand(48, -1, -1)

        solver = MaxwellEquations(OMEGA)
        res = solver.total_residual(E, H, coords, eps)
        assert res.shape == (48, 16)
        # Normalise every residual by k0 and the amplitude of the field it constrains.
        scale_E = K0 * E.abs().max()
        scale_H = K0 * H.abs().max()
        assert float(res[:, :6].abs().max()) / scale_E < tol
        assert float(res[:, 6:12].abs().max()) / scale_H < tol
        assert float(res[:, 12:14].abs().max()) / scale_E < tol
        assert float(res[:, 14:16].abs().max()) / scale_H < tol

    def test_plane_wave_wrong_wavenumber_has_residual(self):
        """A wave with |k| != k0 violates Ampère's law."""
        coords = sample_coords(16)
        k_vec = torch.tensor([0.5 * K0, 0.0, 0.0], dtype=torch.float64)
        E0 = torch.tensor([0.0, 1.0, 0.0], dtype=torch.complex128)
        E, H = analytical_plane_wave(coords, k_vec, E0, OMEGA)
        eps = torch.eye(3, dtype=torch.complex128).unsqueeze(0).expand(16, -1, -1)
        solver = MaxwellEquations(OMEGA)
        res_H = solver.curl_H_residual(E, H, coords, eps)
        # |curl H + iωε0 E| = ωε0 |1 - k²/k0²| |E| = 0.75 ωε0
        expected = 0.75 * OMEGA * EPS0
        assert torch.norm(res_H, dim=1).mean().item() == pytest.approx(expected, rel=1e-6)

    def test_poynting_vector_of_plane_wave(self):
        """S = |E|² / (2 η0) k̂ for a vacuum plane wave."""
        coords = sample_coords(20)
        khat = torch.tensor([0.0, 0.6, 0.8], dtype=torch.float64)
        E0 = torch.tensor([2.0, 0.0, 0.0], dtype=torch.complex128)
        E, H = analytical_plane_wave(coords, K0 * khat, E0, OMEGA)
        S = MaxwellEquations.poynting_vector(E, H)
        expected = (abs(E0[0]) ** 2 / (2 * ETA0)) * khat
        assert torch.allclose(S, expected.unsqueeze(0).expand(20, -1), rtol=1e-9)

    def test_lossy_medium_dissipates_power(self):
        """∇·S = -½ ω ε0 Im(ε) |E|² < 0 for Im(ε) > 0 -- a consistency check on sign conventions."""
        # Plane wave in a lossy medium: k = k0 sqrt(eps) with Im k > 0 (decaying along +x).
        eps = 2.0 + 0.5j
        k = K0 * cmath.sqrt(eps)
        assert k.imag > 0
        coords = sample_coords(16)
        x = coords[:, 0]
        phase = torch.exp(1j * k * x).to(torch.complex128)
        E = torch.zeros(16, 3, dtype=torch.complex128)
        E[:, 1] = phase
        H = torch.zeros(16, 3, dtype=torch.complex128)
        H[:, 2] = k * phase / (OMEGA * MU0)  # H = (k x E)/(ω μ0)
        S = MaxwellEquations.poynting_vector(E, H)
        div_S = divergence(S, coords)
        expected = -0.5 * OMEGA * EPS0 * eps.imag * (E.abs() ** 2).sum(dim=1)
        assert torch.allclose(div_S, expected, rtol=1e-8)


# =========================================================================== differential ops
class TestDifferentialOps:
    @pytest.fixture
    def coords(self):
        return sample_coords(10, extent=1.0)

    def test_gradient_of_quadratic(self, coords):
        phi = (coords**2).sum(dim=1)
        g = gradient(phi, coords)
        assert torch.allclose(g, 2 * coords, atol=1e-12)

    def test_gradient_rejects_complex(self, coords):
        with pytest.raises(TypeError):
            gradient(torch.complex(coords[:, 0], coords[:, 1]), coords)

    def test_gradient_pads_2d_coords(self):
        c = sample_coords(5, dim=2, extent=1.0)
        g = gradient(c[:, 0] * c[:, 1], c)
        assert g.shape == (5, 3)
        assert torch.allclose(g[:, 0], c[:, 1]) and torch.allclose(g[:, 1], c[:, 0])
        assert torch.all(g[:, 2] == 0)

    def test_gradient_of_constant_is_zero(self, coords):
        g = gradient(torch.ones(10, dtype=coords.dtype), coords)
        assert torch.all(g == 0)

    def test_jacobian_of_linear_map(self, coords):
        A = torch.tensor([[1.0, 2.0, 3.0], [0.0, -1.0, 4.0], [5.0, 0.5, 0.0]], dtype=torch.float64)
        F = coords @ A.T
        J = jacobian(F, coords)
        assert J.shape == (10, 3, 3)
        assert torch.allclose(J, A.unsqueeze(0).expand(10, -1, -1))

    def test_curl_of_rotation_field(self, coords):
        """curl(-y, x, 0) = (0, 0, 2)."""
        F = torch.stack([-coords[:, 1], coords[:, 0], torch.zeros_like(coords[:, 0])], dim=1)
        c = curl(F, coords)
        expected = torch.tensor([0.0, 0.0, 2.0], dtype=torch.float64).expand(10, -1)
        assert torch.allclose(c, expected, atol=1e-12)

    def test_divergence_of_position(self, coords):
        """div(x, y, z) = 3."""
        d = divergence(coords, coords)
        assert torch.allclose(d, torch.full((10,), 3.0, dtype=torch.float64), atol=1e-12)

    def test_divergence_of_curl_is_zero(self, coords):
        F = torch.stack([coords[:, 1] * coords[:, 2], coords[:, 0] ** 2, torch.sin(coords[:, 0])], dim=1)
        d = divergence(curl(F, coords), coords)
        assert torch.allclose(d, torch.zeros(10, dtype=torch.float64), atol=1e-10)

    def test_complex_wrappers(self, coords):
        F_re = torch.stack([-coords[:, 1], coords[:, 0], torch.zeros_like(coords[:, 0])], dim=1)
        F = torch.complex(F_re, coords)  # imaginary part = position vector
        c = curl_complex(F, coords)
        d = divergence_complex(F, coords)
        assert torch.allclose(c.real, torch.tensor([0.0, 0.0, 2.0], dtype=torch.float64).expand(10, -1), atol=1e-12)
        assert torch.allclose(c.imag, torch.zeros(10, 3, dtype=torch.float64), atol=1e-12)
        assert torch.allclose(d.real, torch.zeros(10, dtype=torch.float64), atol=1e-12)
        assert torch.allclose(d.imag, torch.full((10,), 3.0, dtype=torch.float64), atol=1e-12)
        # Real inputs pass straight through
        assert torch.equal(curl_complex(F_re, coords), curl(F_re, coords))

    def test_shape_validation(self, coords):
        with pytest.raises(ValueError):
            divergence(coords[:, :2], coords)
        with pytest.raises(ValueError):
            jacobian(coords[:, :2], coords)


# =========================================================================== metamaterial
class TestMetamaterialProperties:
    @pytest.fixture
    def anisotropic(self):
        return MetamaterialProperties(eps_parallel=-5 + 0.5j, eps_perpendicular=2.0 + 0.1j,
                                      optical_axis="z", omega=OMEGA)

    def test_initialisation(self, anisotropic):
        assert anisotropic.eps_par == -5 + 0.5j
        assert anisotropic.eps_perp == 2.0 + 0.1j
        assert anisotropic.optical_axis == "z"
        assert anisotropic.omega == OMEGA
        assert anisotropic.k0 == pytest.approx(K0)
        assert "MetamaterialProperties" in repr(anisotropic)

    def test_invalid_axis(self):
        with pytest.raises(ValueError):
            MetamaterialProperties(1.0, 1.0, optical_axis="w")

    def test_omega_and_wavelength_exclusive(self):
        with pytest.raises(ValueError):
            MetamaterialProperties(1.0, 1.0, omega=OMEGA, wavelength=1e-6)

    def test_wavelength_sets_omega(self):
        lam = 800e-9
        m = MetamaterialProperties(1.0, 1.0, wavelength=lam)
        assert m.omega == pytest.approx(2 * np.pi * C0 / lam)
        assert m.k0 == pytest.approx(2 * np.pi / lam)

    def test_missing_frequency_raises(self):
        m = MetamaterialProperties(-2 + 0.1j, 4 + 0.05j)
        assert m.k0 is None
        with pytest.raises(ValueError, match="No frequency"):
            m.spp_wavevector()
        # ...but an explicit omega works
        assert m.spp_wavevector(omega=OMEGA).imag > 0

    def test_eps_along(self, anisotropic):
        assert anisotropic.eps_along("z") == anisotropic.eps_par
        assert anisotropic.eps_along("x") == anisotropic.eps_perp
        assert anisotropic.eps_along("Y") == anisotropic.eps_perp
        with pytest.raises(ValueError):
            anisotropic.eps_along("q")

    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_permittivity_tensor(self, axis):
        m = MetamaterialProperties(-3 + 0.2j, 2.0, optical_axis=axis)
        coords = torch.zeros(4, 3)
        t = m.permittivity_tensor(coords)
        assert t.shape == (4, 3, 3)
        assert t.dtype == torch.complex64
        diag = torch.diagonal(t[0])
        idx = "xyz".index(axis)
        for i in range(3):
            assert complex(diag[i]) == pytest.approx(-3 + 0.2j if i == idx else 2.0)
        off = t[0] - torch.diag(diag)
        assert torch.all(off == 0)
        assert m.permittivity_tensor(torch.zeros(2, 3, dtype=torch.float64)).dtype == torch.complex128

    def test_effective_permittivity_isotropic_limit(self):
        m = MetamaterialProperties(2.0, 2.0, omega=OMEGA)
        kx = torch.tensor([0.0, 0.5, 1.0]) * K0
        eps_eff = m.effective_permittivity(kx, torch.zeros(3))
        assert torch.allclose(eps_eff, torch.full((3,), 2.0, dtype=eps_eff.dtype))

    def test_effective_permittivity_normal_incidence(self, anisotropic):
        eps_eff = anisotropic.effective_permittivity(torch.tensor([0.0]), torch.tensor([0.0]))
        assert complex(eps_eff[0]) == pytest.approx(anisotropic.eps_perp)

    def test_effective_permittivity_in_plane_axis(self):
        m = MetamaterialProperties(-3.0, 2.0, optical_axis="x", omega=OMEGA)
        # k along x sees eps_xx = eps_par; eps_n = eps_zz = eps_perp
        k = torch.tensor([K0])
        eps_eff = m.effective_permittivity(k, torch.zeros(1))
        expected = -3.0 + 1.0 * (1.0 - (-3.0) / 2.0)
        assert complex(eps_eff[0]) == pytest.approx(expected)
        # k = 0 falls back to eps_xx
        assert complex(m.effective_permittivity(torch.zeros(1), torch.zeros(1))[0]) == pytest.approx(-3.0)

    def test_isotropic_spp_matches_closed_form(self):
        eps_m, eps_d = -20 + 1j, 2.25
        m = MetamaterialProperties(eps_m, eps_m, optical_axis="z", omega=OMEGA)
        k = m.spp_wavevector(eps_dielectric=eps_d)
        expected = K0 * cmath.sqrt(eps_m * eps_d / (eps_m + eps_d))
        if expected.imag < 0:
            expected = -expected
        assert k == pytest.approx(expected, rel=1e-12)
        assert k.imag > 0
        assert k.real > K0 * np.sqrt(eps_d)  # bound: slower than light in the dielectric

    def test_spp_dispersion_relation_returns_parts(self):
        m = MetamaterialProperties(-20 + 1j, -20 + 1j, omega=OMEGA)
        re, im = m.spp_dispersion_relation(eps_dielectric=2.25)
        k = m.spp_wavevector(eps_dielectric=2.25)
        assert (re, im) == (k.real, k.imag)

    def test_spp_wavevector_k0_override(self):
        m = MetamaterialProperties(-20 + 1j, -20 + 1j)
        k_a = m.spp_wavevector(k0=K0)
        k_b = m.spp_wavevector(omega=OMEGA)
        assert k_a == pytest.approx(k_b)
        assert m.spp_wavevector(k0=2 * K0) == pytest.approx(2 * k_a)

    def test_spp_propagation_direction(self):
        m = MetamaterialProperties(-5 + 0.2j, -20 + 1j, optical_axis="x", omega=OMEGA)
        kx = m.spp_wavevector(propagation_direction="x")
        ky = m.spp_wavevector(propagation_direction="y")
        assert kx != ky
        with pytest.raises(ValueError):
            m.spp_wavevector(propagation_direction="z")

    def test_spp_singular_dispersion(self):
        # eps_t * eps_n = eps_d^2 with eps_d = 1
        m = MetamaterialProperties(0.5, 2.0, omega=OMEGA)
        with pytest.raises(ZeroDivisionError):
            m.spp_wavevector(eps_dielectric=1.0)

    @pytest.mark.parametrize(
        "eps_par,eps_perp,expected",
        [
            (4 + 0.05j, -2 + 0.1j, True),
            (-2 + 0.1j, 4 + 0.05j, False),
            (2.25, 2.25, False),
            (-20 + 1j, -20 + 1j, True),
        ],
    )
    def test_is_spp_supported(self, eps_par, eps_perp, expected):
        m = MetamaterialProperties(eps_par, eps_perp, optical_axis="z")
        assert m.is_spp_supported(eps_dielectric=1.0) is expected

    def test_is_spp_supported_is_frequency_independent(self):
        a = MetamaterialProperties(4 + 0.05j, -2 + 0.1j, omega=OMEGA).is_spp_supported()
        b = MetamaterialProperties(4 + 0.05j, -2 + 0.1j, omega=3 * OMEGA).is_spp_supported()
        assert a is b is True

    def test_is_spp_supported_singular(self):
        assert MetamaterialProperties(0.5, 2.0).is_spp_supported(eps_dielectric=1.0) is False

    def test_propagation_length(self):
        m = MetamaterialProperties(-20 + 1j, -20 + 1j, omega=OMEGA)
        k = m.spp_wavevector(eps_dielectric=2.25)
        assert m.propagation_length(eps_dielectric=2.25) == pytest.approx(1.0 / (2.0 * k.imag))

    def test_propagation_length_lossless_is_infinite(self):
        m = MetamaterialProperties(-20.0, -20.0, omega=OMEGA)
        assert m.propagation_length(eps_dielectric=2.25) == float("inf")

    def test_penetration_depths_isotropic(self):
        eps_m, eps_d = -20.0, 2.25
        m = MetamaterialProperties(eps_m, eps_m, omega=OMEGA)
        k = m.spp_wavevector(eps_dielectric=eps_d)
        kappa_d = cmath.sqrt(k**2 - eps_d * K0**2)
        kappa_m = cmath.sqrt(k**2 - eps_m * K0**2)
        assert m.penetration_depth_dielectric(eps_dielectric=eps_d) == pytest.approx(1 / kappa_d.real)
        assert m.penetration_depth_metamaterial(eps_dielectric=eps_d) == pytest.approx(1 / kappa_m.real)
        # Field extends further into the dielectric than into the metal
        assert m.penetration_depth_dielectric(eps_dielectric=eps_d) > m.penetration_depth_metamaterial(eps_dielectric=eps_d)

    def test_field_enhancement_factor_isotropic(self):
        eps_m, eps_d = -20.0, 2.25
        m = MetamaterialProperties(eps_m, eps_m, omega=OMEGA)
        assert m.field_enhancement_factor(eps_dielectric=eps_d) == pytest.approx(np.sqrt(abs(eps_m) / eps_d))

    def test_anisotropic_spp_differs_from_isotropic(self):
        """Changing only the normal component moves k_spp (anisotropy matters)."""
        iso = MetamaterialProperties(-8 + 0.3j, -8 + 0.3j, omega=OMEGA)
        aniso = MetamaterialProperties(-3 + 0.3j, -8 + 0.3j, optical_axis="z", omega=OMEGA)
        k_iso = iso.spp_wavevector(eps_dielectric=1.5)
        k_aniso = aniso.spp_wavevector(eps_dielectric=1.5)
        assert k_iso != pytest.approx(k_aniso)
        assert k_aniso.imag > 0


# =========================================================================== boundary conditions
class TestBoundaryConditions:
    @pytest.fixture
    def bc(self):
        return BoundaryConditions(interface_normal=(0.0, 0.0, 1.0))

    @pytest.fixture
    def fields(self):
        g = torch.Generator().manual_seed(3)
        E = torch.complex(torch.randn(6, 3, generator=g), torch.randn(6, 3, generator=g))
        H = torch.complex(torch.randn(6, 3, generator=g), torch.randn(6, 3, generator=g))
        return E, H

    def test_normal_is_normalised(self):
        bc = BoundaryConditions(interface_normal=(0.0, 3.0, 4.0))
        assert torch.allclose(bc.interface_normal, torch.tensor([0.0, 0.6, 0.8]))
        assert "BoundaryConditions" in repr(bc)

    def test_zero_normal_raises(self):
        with pytest.raises(ValueError):
            BoundaryConditions(interface_normal=(0.0, 0.0, 0.0))

    def test_cross_product(self, bc):
        a = torch.tensor([[1.0, 0.0, 0.0]])
        b = torch.tensor([0.0, 1.0, 0.0])
        c = bc.cross_product(a, b)
        assert torch.allclose(c, torch.tensor([[0.0, 0.0, 1.0]]))
        cz = bc.cross_product(a, torch.tensor([0.0, 1j, 0.0]))
        assert cz.is_complex()
        assert torch.allclose(cz, torch.tensor([[0.0, 0.0, 1j]]))

    def test_tangential_continuity_zero_for_identical_fields(self, bc, fields):
        E, H = fields
        assert torch.all(bc.tangential_E_continuity(E, E) == 0)
        assert torch.all(bc.tangential_H_continuity(H, H) == 0)
        assert bc.tangential_E_continuity(E, E).shape == (6, 6)

    def test_tangential_continuity_nonzero_for_different_fields(self, bc, fields):
        E, H = fields
        E2 = E.clone()
        E2[:, 0] += 1.0  # tangential (x) jump
        res = bc.tangential_E_continuity(E, E2)
        assert res.abs().sum() > 0
        # n × (Δx x̂) = ẑ × x̂ = ŷ
        assert torch.allclose(res[:, 1], torch.ones(6))
        assert torch.allclose(res[:, [0, 2, 3, 4, 5]], torch.zeros(6, 5))

    def test_tangential_continuity_ignores_normal_jump(self, bc, fields):
        E, _ = fields
        E2 = E.clone()
        E2[:, 2] += 5.0  # normal (z) jump is allowed
        assert torch.allclose(bc.tangential_E_continuity(E, E2), torch.zeros(6, 6), atol=1e-6)

    def test_normal_D_continuity(self, bc, fields):
        E, _ = fields
        n = E.shape[0]
        eye = torch.eye(3, dtype=torch.complex64).unsqueeze(0).expand(n, -1, -1)
        eps1 = eye * 2.0
        eps2 = eye * 4.0
        # Same E on both sides but different eps -> D jump = eps0 * 2 * Ez
        res = bc.normal_D_continuity(E, E, eps1, eps2)
        assert res.shape == (n, 2)
        expected = 2.0 * EPS0 * E[:, 2]
        assert torch.allclose(torch.complex(res[:, 0], res[:, 1]), expected, rtol=1e-5)
        # relative=True drops eps0
        res_rel = bc.normal_D_continuity(E, E, eps1, eps2, relative=True)
        assert torch.allclose(torch.complex(res_rel[:, 0], res_rel[:, 1]), 2.0 * E[:, 2], rtol=1e-5)
        # Continuous D: E2 = E1 * eps1 / eps2 in the normal component
        E2 = E.clone()
        E2[:, 2] = E[:, 2] * 0.5
        assert torch.allclose(bc.normal_D_continuity(E, E2, eps1, eps2), torch.zeros(n, 2), atol=1e-16)

    def test_normal_B_continuity(self, bc, fields):
        _, H = fields
        assert torch.all(bc.normal_B_continuity(H, H) == 0)
        H2 = H.clone()
        H2[:, 2] += 1.0
        res = bc.normal_B_continuity(H, H2)
        assert torch.allclose(res[:, 0], torch.full((6,), MU0), rtol=1e-5)
        res_rel = bc.normal_B_continuity(H, H2, relative=True)
        assert torch.allclose(res_rel[:, 0], torch.ones(6), rtol=1e-6)

    def test_spp_boundary_conditions_shape_and_zero(self, bc, fields):
        E, H = fields
        n = E.shape[0]
        eps_mm = torch.eye(3, dtype=torch.complex64).unsqueeze(0).expand(n, -1, -1) * 2.25
        res = bc.spp_boundary_conditions(E, H, E, H, eps_mm, eps_dielectric=2.25)
        assert res.shape == (n, 16)
        assert torch.allclose(res, torch.zeros_like(res), atol=1e-12)

    def test_perfect_conductor_zero_for_normal_field(self, bc):
        E = torch.zeros(4, 3, dtype=torch.complex64)
        E[:, 2] = torch.tensor([1.0, 2j, -3.0, 0.5 + 0.5j])
        res = bc.perfect_conductor_boundary(E)
        assert res.shape == (4, 6)
        assert torch.allclose(res, torch.zeros(4, 6), atol=1e-7)

    def test_perfect_conductor_nonzero_for_tangential_field(self, bc):
        E = torch.zeros(4, 3, dtype=torch.complex64)
        E[:, 0] = 1.0
        E[:, 1] = 1j
        res = bc.perfect_conductor_boundary(E)
        assert torch.allclose(res[:, 0], torch.ones(4))  # Re Ex
        assert torch.allclose(res[:, 4], torch.ones(4))  # Im Ey
        assert torch.allclose(res[:, [2, 5]], torch.zeros(4, 2))  # no normal component

    def test_perfect_conductor_with_oblique_normal(self):
        bc = BoundaryConditions(interface_normal=(1.0, 1.0, 0.0))
        E = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.complex64)  # parallel to n
        assert torch.allclose(bc.perfect_conductor_boundary(E), torch.zeros(1, 6), atol=1e-6)

    def test_impedance_boundary_condition(self, bc):
        # n = z. E = Ex x̂ -> n × E = Ex ŷ. H tangential = Hy ŷ. Residual = (Ex - Zs Hy) ŷ
        Zs = 50.0 + 10j
        Hy = 1.0 + 0.5j
        E = torch.zeros(3, 3, dtype=torch.complex64)
        H = torch.zeros(3, 3, dtype=torch.complex64)
        E[:, 0] = Zs * Hy
        H[:, 1] = Hy
        res = bc.impedance_boundary_condition(E, H, Zs)
        assert res.shape == (3, 6)
        assert torch.allclose(res, torch.zeros(3, 6), atol=1e-4)
        H[:, 1] = 2 * Hy
        assert bc.impedance_boundary_condition(E, H, Zs).abs().sum() > 0

    def test_radiation_boundary_condition(self, bc):
        """An outgoing wave exp(+ikr) satisfies (∂r - ik)F = 0."""
        k0 = 2.0
        E = torch.complex(torch.randn(5, 3), torch.randn(5, 3))
        H = torch.complex(torch.randn(5, 3), torch.randn(5, 3))
        dE, dH = 1j * k0 * E, 1j * k0 * H
        res = bc.radiation_boundary_condition(E, H, k0, normal_derivative_E=dE, normal_derivative_H=dH)
        assert res.shape == (5, 12)
        assert torch.allclose(res, torch.zeros(5, 12), atol=1e-6)
        # Without derivatives the residual is -ikF
        res0 = bc.radiation_boundary_condition(E, H, k0)
        assert torch.allclose(torch.complex(res0[:, :3], res0[:, 6:9]), -1j * k0 * E, atol=1e-6)
        # Background permittivity rescales k
        res_bg = bc.radiation_boundary_condition(E, H, k0, eps_background=4.0)
        assert torch.allclose(torch.complex(res_bg[:, :3], res_bg[:, 6:9]), -2j * k0 * E, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
