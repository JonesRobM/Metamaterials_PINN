"""
Tests for src.design — the differentiable torch SPP dispersion.

The correctness anchor is parity with the scalar reference implementation
:class:`src.physics.metamaterial.MetamaterialProperties` (itself validated to
machine precision in ``tests/test_benchmark_spp.py``): every derived quantity
must agree to 1e-10 relative across randomized supported parameter sets, with
finite gradients, batched == scalar-loop behaviour, and no NaN gradients near
the branch seam.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import pytest
import torch

from src import design
from src.constants import C0
from src.physics.metamaterial import MetamaterialProperties

LAMBDA0 = 633e-9
OMEGA = 2 * math.pi * C0 / LAMBDA0
K0 = OMEGA / C0
REL = 1e-10


def _reference(eps_t: complex, eps_n: complex) -> MetamaterialProperties:
    """Scalar reference (optical axis z: eps_parallel is the normal component)."""
    return MetamaterialProperties(
        eps_parallel=eps_n, eps_perpendicular=eps_t, optical_axis="z", omega=OMEGA
    )


def _random_supported(n: int = 24, seed: int = 0) -> List[Tuple[complex, complex]]:
    """
    ``n`` random ``(eps_t, eps_n)`` pairs that the scalar gate accepts,
    alternating metal-like (both real parts negative) and type-II hyperbolic
    (``Re eps_t < 0 < Re eps_n``) draws.
    """
    rng = torch.Generator().manual_seed(seed)
    out: List[Tuple[complex, complex]] = []
    while len(out) < n:
        u = torch.rand(4, generator=rng, dtype=torch.float64).tolist()
        if len(out) % 2 == 0:
            re_t = -(1.5 + 18.0 * u[0])
            eps_t = complex(re_t, 0.01 + 0.6 * u[1])
            eps_n = complex(re_t * (0.4 + u[2]), 0.01 + 0.6 * u[3])
        else:
            eps_t = complex(-(1.5 + 8.0 * u[0]), 0.01 + 0.6 * u[1])
            eps_n = complex(0.5 + 4.0 * u[2], 0.005 + 0.1 * u[3])
        if _reference(eps_t, eps_n).is_spp_supported(eps_dielectric=1.0):
            out.append((eps_t, eps_n))
    return out


def _rel_err(a: complex, b: complex) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


# --------------------------------------------------------------------- parity
def test_parity_with_scalar_reference():
    """All derived quantities match MetamaterialProperties to 1e-10 relative."""
    sets = _random_supported(24)
    assert len(sets) >= 20
    for eps_t, eps_n in sets:
        m = _reference(eps_t, eps_n)
        k_ref = m.spp_wavevector(eps_dielectric=1.0)
        _, kd_ref, km_ref = m._decay_constants(OMEGA, 1.0, "x")
        et = torch.tensor(eps_t, dtype=torch.complex128)
        en = torch.tensor(eps_n, dtype=torch.complex128)

        k = design.spp_wavevector_torch(et, en, OMEGA)
        assert _rel_err(complex(k), k_ref) < REL
        assert k.imag >= 0  # propagating branch

        k2, kd, km = design.decay_constants_torch(et, en, OMEGA)
        assert _rel_err(complex(k2), k_ref) < REL
        assert _rel_err(complex(kd), kd_ref) < REL
        assert _rel_err(complex(km), km_ref) < REL
        assert kd.real > 0 and km.real > 0  # bound-mode branch

        length = design.propagation_length_torch(et, en, OMEGA)
        assert _rel_err(float(length), m.propagation_length()) < REL
        dd, dm = design.penetration_depths_torch(et, en, OMEGA)
        assert _rel_err(float(dd), m.penetration_depth_dielectric()) < REL
        assert _rel_err(float(dm), m.penetration_depth_metamaterial()) < REL
        fe = design.field_enhancement_torch(et, en, OMEGA)
        assert _rel_err(float(fe), m.field_enhancement_factor()) < REL


def test_parity_nonunit_dielectric():
    """The eps_d argument matches the scalar reference too (glass superstrate)."""
    eps_d = 2.25
    for eps_t, eps_n in [(-8.0 + 0.3j, -6.0 + 0.1j), (-20.0 + 0.5j, -15.0 + 0.2j)]:
        m = _reference(eps_t, eps_n)
        if not m.is_spp_supported(eps_dielectric=eps_d):
            continue
        k_ref = m.spp_wavevector(eps_dielectric=eps_d)
        _, kd_ref, km_ref = m._decay_constants(OMEGA, eps_d, "x")
        et = torch.tensor(eps_t, dtype=torch.complex128)
        en = torch.tensor(eps_n, dtype=torch.complex128)
        k, kd, km = design.decay_constants_torch(et, en, OMEGA, eps_d)
        assert _rel_err(complex(k), k_ref) < REL
        assert _rel_err(complex(kd), kd_ref) < REL
        assert _rel_err(complex(km), km_ref) < REL


# --------------------------------------------------------------------- gradients
def test_gradients_exist_and_are_finite():
    """backward through k_spp (and derived scalars) gives finite grads on all 4 leaves."""
    for eps_t, eps_n in _random_supported(20):
        leaves = [
            torch.tensor(v, dtype=torch.float64, requires_grad=True)
            for v in (eps_t.real, eps_t.imag, eps_n.real, eps_n.imag)
        ]
        re_t, im_t, re_n, im_n = leaves
        et = design.make_eps(re_t, im_t)
        en = design.make_eps(re_n, im_n)
        k = design.spp_wavevector_torch(et, en, OMEGA)
        (k.real / K0 + k.imag / K0).backward()
        for leaf in leaves:
            assert leaf.grad is not None
            assert torch.isfinite(leaf.grad)
        assert any(leaf.grad.abs() > 0 for leaf in leaves)


def test_gradients_through_derived_quantities():
    re_t = torch.tensor(-4.0, dtype=torch.float64, requires_grad=True)
    re_n = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    et = design.make_eps(re_t, 0.2)
    en = design.make_eps(re_n, 0.05)
    length = design.propagation_length_torch(et, en, OMEGA)
    dd, dm = design.penetration_depths_torch(et, en, OMEGA)
    fe = design.field_enhancement_torch(et, en, OMEGA)
    pen = design.support_penalty_torch(et, en)
    total = length / LAMBDA0 + (dd + dm) / LAMBDA0 + fe + pen
    total.backward()
    assert torch.isfinite(re_t.grad) and torch.isfinite(re_n.grad)
    assert re_t.grad.abs() > 0 and re_n.grad.abs() > 0


# --------------------------------------------------------------------- batching
def test_batched_matches_scalar_loop():
    sets = _random_supported(20, seed=1)
    et = torch.tensor([s[0] for s in sets], dtype=torch.complex128)
    en = torch.tensor([s[1] for s in sets], dtype=torch.complex128)
    k_b, kd_b, km_b = design.decay_constants_torch(et, en, OMEGA)
    L_b = design.propagation_length_torch(et, en, OMEGA)
    dd_b, dm_b = design.penetration_depths_torch(et, en, OMEGA)
    fe_b = design.field_enhancement_torch(et, en, OMEGA)
    sup_b = design.is_spp_supported_torch(et, en)
    assert k_b.shape == (len(sets),)
    for i, (eps_t, eps_n) in enumerate(sets):
        e_t = torch.tensor(eps_t, dtype=torch.complex128)
        e_n = torch.tensor(eps_n, dtype=torch.complex128)
        k, kd, km = design.decay_constants_torch(e_t, e_n, OMEGA)
        assert _rel_err(complex(k_b[i]), complex(k)) < 1e-14
        assert _rel_err(complex(kd_b[i]), complex(kd)) < 1e-14
        assert _rel_err(complex(km_b[i]), complex(km)) < 1e-14
        assert _rel_err(float(L_b[i]), float(design.propagation_length_torch(e_t, e_n, OMEGA))) < 1e-14
        dd, dm = design.penetration_depths_torch(e_t, e_n, OMEGA)
        assert _rel_err(float(dd_b[i]), float(dd)) < 1e-14
        assert _rel_err(float(dm_b[i]), float(dm)) < 1e-14
        assert _rel_err(float(fe_b[i]), float(design.field_enhancement_torch(e_t, e_n, OMEGA))) < 1e-14
        assert bool(sup_b[i]) == bool(design.is_spp_supported_torch(e_t, e_n))


def test_batched_gradients():
    """Batched eps built from real leaf tensors backprops to every element."""
    sets = _random_supported(8, seed=2)
    re_t = torch.tensor([s[0].real for s in sets], dtype=torch.float64, requires_grad=True)
    im_t = torch.tensor([s[0].imag for s in sets], dtype=torch.float64, requires_grad=True)
    re_n = torch.tensor([s[1].real for s in sets], dtype=torch.float64, requires_grad=True)
    im_n = torch.tensor([s[1].imag for s in sets], dtype=torch.float64, requires_grad=True)
    k = design.spp_wavevector_torch(design.make_eps(re_t, im_t), design.make_eps(re_n, im_n), OMEGA)
    (k.real.sum() / K0).backward()
    for leaf in (re_t, im_t, re_n, im_n):
        assert torch.all(torch.isfinite(leaf.grad))
        assert torch.all(leaf.grad.abs() > 0)


# --------------------------------------------------------------------- branch selection
def test_branch_seam_continuity_and_grads():
    """
    Near the principal branch cut (k² on the negative real axis: eps_t = -3,
    eps_n = -0.2 gives (k/k0)² = -2 for lossless media) the Im >= 0 flip keeps
    the selected root continuous and its gradients finite on both sides.
    """
    ks = []
    for sign in (+1.0, -1.0):
        re_t = torch.tensor(-3.0, dtype=torch.float64, requires_grad=True)
        im_t = torch.tensor(sign * 1e-5, dtype=torch.float64, requires_grad=True)
        et = design.make_eps(re_t, im_t)
        en = torch.tensor(-0.2 + 0j, dtype=torch.complex128)
        k = design.spp_wavevector_torch(et, en, OMEGA)
        assert k.imag > 0  # Im >= 0 branch held on both sides of the seam
        (k.real + k.imag).backward()
        assert torch.isfinite(re_t.grad) and torch.isfinite(im_t.grad)
        ks.append(complex(k.detach()))
    # Continuity across the seam: the two roots differ only at O(perturbation).
    assert abs(ks[0] - ks[1]) < 1e-3 * abs(ks[0])


def test_branch_seam_positive_axis():
    """Near-lossless propagating modes (k² near the positive real axis): finite grads."""
    for im in (1e-8, 1e-4):
        re_t = torch.tensor(-4.0, dtype=torch.float64, requires_grad=True)
        re_n = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        k = design.spp_wavevector_torch(
            design.make_eps(re_t, im), design.make_eps(re_n, im / 4), OMEGA
        )
        (k.real + k.imag).backward()
        assert torch.isfinite(re_t.grad) and torch.isfinite(re_n.grad)
        assert k.imag >= 0


def test_lossless_limits():
    """Real permittivities: Im k = 0 so L = inf; unbound sides give inf depths."""
    et = torch.tensor(-4.0 + 0j, dtype=torch.complex128)
    en = torch.tensor(3.0 + 0j, dtype=torch.complex128)
    m = _reference(-4.0, 3.0)
    assert math.isinf(float(design.propagation_length_torch(et, en, OMEGA)))
    assert math.isinf(m.propagation_length())
    # Radiative configuration: kappa_d purely imaginary -> infinite "depth" on both.
    et_r = torch.tensor(2.0 + 0j, dtype=torch.complex128)
    en_r = torch.tensor(2.0 + 0j, dtype=torch.complex128)
    dd, _ = design.penetration_depths_torch(et_r, en_r, OMEGA)
    assert math.isinf(float(dd))
    assert math.isinf(_reference(2.0, 2.0).penetration_depth_dielectric())


# --------------------------------------------------------------------- supported gate
def test_is_spp_supported_parity_random():
    """The boolean gate agrees with the scalar bound_tol/rel_tol logic everywhere."""
    rng = torch.Generator().manual_seed(3)
    pairs = []
    for _ in range(200):
        u = torch.rand(4, generator=rng, dtype=torch.float64).tolist()
        eps_t = complex(-20.0 + 25.0 * u[0], 0.6 * u[1])
        eps_n = complex(-10.0 + 20.0 * u[2], 0.3 * u[3])
        pairs.append((eps_t, eps_n))
    et = torch.tensor([p[0] for p in pairs], dtype=torch.complex128)
    en = torch.tensor([p[1] for p in pairs], dtype=torch.complex128)
    got = design.is_spp_supported_torch(et, en)
    assert got.dtype == torch.bool
    expected = [_reference(*p).is_spp_supported(eps_dielectric=1.0) for p in pairs]
    assert got.tolist() == expected
    assert any(expected) and not all(expected)  # the sample covers both outcomes


def test_is_spp_supported_singular_dispersion():
    """eps_t * eps_n = eps_d² is rejected, mirroring the scalar ZeroDivision guard."""
    et = torch.tensor(2.0 + 0j, dtype=torch.complex128)
    en = torch.tensor(0.5 + 0j, dtype=torch.complex128)
    assert not bool(design.is_spp_supported_torch(et, en))


def test_support_penalty_zero_inside_positive_outside():
    for eps_t, eps_n in _random_supported(10, seed=4):
        pen = design.support_penalty_torch(
            torch.tensor(eps_t, dtype=torch.complex128),
            torch.tensor(eps_n, dtype=torch.complex128),
        )
        assert float(pen) < 1e-12
    # Radiative (dielectric/dielectric) configuration is penalised.
    pen = design.support_penalty_torch(
        torch.tensor(2.0 + 0j, dtype=torch.complex128),
        torch.tensor(2.0 + 0j, dtype=torch.complex128),
    )
    assert float(pen) > 0.05
    # Singular dispersion is maximally penalised.
    pen = design.support_penalty_torch(
        torch.tensor(2.0 + 0j, dtype=torch.complex128),
        torch.tensor(0.5 + 0j, dtype=torch.complex128),
    )
    assert float(pen) == pytest.approx(10.0)


# --------------------------------------------------------------------- parametrisation
def test_make_eps_values_and_grads():
    re = torch.tensor([1.0, -4.0], dtype=torch.float64, requires_grad=True)
    im = torch.tensor([0.2, 0.05], dtype=torch.float64, requires_grad=True)
    eps = design.make_eps(re, im)
    assert eps.dtype == torch.complex128
    assert torch.allclose(eps.real, re.detach()) and torch.allclose(eps.imag, im.detach())
    eps.real.sum().backward()
    assert torch.allclose(re.grad, torch.ones(2, dtype=torch.float64))
    # Scalars and mixed number/tensor inputs broadcast.
    assert complex(design.make_eps(-4.0, 0.2)) == -4.0 + 0.2j
    assert design.make_eps(torch.zeros(3, dtype=torch.float64), 0.1).shape == (3,)


def test_constrain_im_passivity():
    raw = torch.tensor([-50.0, 0.0, 3.0], dtype=torch.float64, requires_grad=True)
    im = design.constrain_im(raw, im_min=1e-3)
    assert torch.all(im >= 1e-3)
    im.sum().backward()
    assert torch.all(torch.isfinite(raw.grad))
    # Round trip through the inverse at representative values.
    for target in (0.05, 0.2, 2.0):
        raw_val = design.constrain_im_inverse(target, im_min=1e-3)
        assert float(design.constrain_im(raw_val, im_min=1e-3)) == pytest.approx(target, rel=1e-12)
    with pytest.raises(ValueError):
        design.constrain_im(0.0, im_min=0.0)
    with pytest.raises(ValueError):
        design.constrain_im_inverse(1e-4, im_min=1e-3)


def test_float_inputs_accepted():
    """Plain python numbers work for every argument."""
    k = design.spp_wavevector_torch(-4 + 0.2j, 3 + 0.05j, OMEGA, 1.0)
    m = _reference(-4 + 0.2j, 3 + 0.05j)
    assert _rel_err(complex(k), m.spp_wavevector(eps_dielectric=1.0)) < REL
