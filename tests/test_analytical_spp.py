"""Machine-precision checks for ``src.analytical.analytical_spp_fields``.

The mode is verified with machinery that is itself benchmark-validated
(tests/test_benchmark_anisotropic.py, tests/test_benchmark_fresnel.py):

* Maxwell residuals per half-space via ``MaxwellEquations.total_residual`` with
  the correct (tensor) permittivity of that side,
* interface continuity (tangential E/H, normal D and B) via ``BoundaryConditions``,
* k_spp / decay constants against ``MetamaterialProperties``.

Cases: isotropic silver/air at 633 nm and a uniaxial (type-II) metamaterial.
"""

import math

import pytest
import torch

from src.analytical import analytical_spp_fields
from src.constants import C0, ETA0
from src.physics.boundary_conditions import BoundaryConditions
from src.physics.maxwell_equations import MaxwellEquations
from src.physics.metamaterial import MetamaterialProperties

OMEGA = 2 * math.pi * C0 / 633e-9  # lambda0 = 633 nm
EPS_AG = -18.3 + 0.55j

CASES = {
    "isotropic-silver-air": dict(eps_t=EPS_AG, eps_n=EPS_AG, eps_d=1.0),
    "isotropic-silver-glass": dict(eps_t=EPS_AG, eps_n=EPS_AG, eps_d=2.25),
    "uniaxial-type-II": dict(eps_t=-4.0 + 0.2j, eps_n=3.0 + 0.05j, eps_d=1.0),
}


def _coords(side: str, n: int = 40, seed: int = 0) -> torch.Tensor:
    """Random float64 points confined to one half-space (|z| in a decade of nm scales)."""
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(n, generator=g, dtype=torch.float64) * 2 - 1) * 600e-9
    y = (torch.rand(n, generator=g, dtype=torch.float64) * 2 - 1) * 600e-9
    z = torch.rand(n, generator=g, dtype=torch.float64) * 80e-9 + 2e-9
    if side == "metal":
        z = -z
    return torch.stack([x, y, z], dim=1)


def _fields_on(coords: torch.Tensor, case: dict):
    return analytical_spp_fields(
        coords, OMEGA, case["eps_t"], case["eps_n"], eps_dielectric=case["eps_d"]
    )


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("side", ["dielectric", "metal"])
def test_mode_satisfies_maxwell_per_halfspace(name, side):
    case = CASES[name]
    coords = _coords(side).requires_grad_(True)
    E, H = _fields_on(coords, case)

    if side == "dielectric":
        eps = torch.eye(3, dtype=torch.complex128) * case["eps_d"]
    else:
        eps = torch.diag(
            torch.tensor([case["eps_t"], case["eps_t"], case["eps_n"]], dtype=torch.complex128)
        )
    eps = eps.unsqueeze(0).expand(coords.shape[0], -1, -1)

    maxwell = MaxwellEquations(OMEGA)
    residual = maxwell.total_residual(E, H, coords, eps)
    # Natural scale of the curl-E block: k0 |E|
    k0 = OMEGA / C0
    scale = k0 * float(E.abs().max().detach())
    assert float(residual.abs().max()) / scale < 1e-9


@pytest.mark.parametrize("name", CASES)
def test_interface_continuity(name):
    case = CASES[name]
    g = torch.Generator().manual_seed(1)
    xy = (torch.rand(50, 2, generator=g, dtype=torch.float64) * 2 - 1) * 600e-9
    # One-sided limits: kappa_m * delta must be << tolerance (kappa_m ~ 4e7 /m),
    # since the envelope difference at +-delta is physical, not an error.
    delta = 1e-18
    up = torch.stack([xy[:, 0], xy[:, 1], torch.full((50,), delta, dtype=torch.float64)], dim=1)
    dn = torch.stack([xy[:, 0], xy[:, 1], torch.full((50,), -delta, dtype=torch.float64)], dim=1)

    E1, H1 = _fields_on(dn, case)  # metal side (medium 1)
    E2, H2 = _fields_on(up, case)  # dielectric side (medium 2)

    bc = BoundaryConditions(interface_normal=(0.0, 0.0, 1.0))
    field_scale = float(E2.abs().max())

    tE = bc.tangential_E_continuity(E1, E2)
    tH = bc.tangential_H_continuity(H1, H2)
    assert float(tE.abs().max()) / field_scale < 1e-9
    assert float(tH.abs().max()) / float(H2.abs().max()) < 1e-9

    eps1 = torch.diag(
        torch.tensor([case["eps_t"], case["eps_t"], case["eps_n"]], dtype=torch.complex128)
    ).unsqueeze(0).expand(50, -1, -1)
    eps2 = torch.eye(3, dtype=torch.complex128).unsqueeze(0).expand(50, -1, -1) * case["eps_d"]
    nD = bc.normal_D_continuity(E1, E2, eps1, eps2)
    assert float(nD.abs().max()) / field_scale < 1e-9

    # Normal E itself must be DIScontinuous (the mode has E_z jumping by eps ratio)
    jump = (E2[:, 2] - E1[:, 2]).abs().max()
    assert float(jump) / field_scale > 0.1


@pytest.mark.parametrize("name", CASES)
def test_wavevector_and_decay_match_metamaterial_module(name):
    case = CASES[name]
    m = MetamaterialProperties(case["eps_n"], case["eps_t"], "z", omega=OMEGA)
    # optical axis z: eps_parallel is the normal component, eps_perpendicular in-plane
    k_repo = m.spp_wavevector(eps_dielectric=case["eps_d"])

    coords = _coords("dielectric", n=2)
    E, H = _fields_on(coords, case)
    # Recover k_spp from the phase between two points at the same z
    x0 = torch.tensor([[0.0, 0.0, 10e-9]], dtype=torch.float64)
    x1 = torch.tensor([[100e-9, 0.0, 10e-9]], dtype=torch.float64)
    _, H0 = _fields_on(x0, case)
    _, H1 = _fields_on(x1, case)
    ratio = complex(H1[0, 1] / H0[0, 1])
    k_measured = complex(torch.log(torch.tensor(ratio)) / 1j) / 100e-9
    # log/exp round-trip over a 100 nm baseline limits recovery to ~1e-8 relative
    assert abs(k_measured - k_repo) / abs(k_repo) < 1e-6

    # Decay constants from amplitude ratios on each side
    for _side, sign, kappa_expected in [
        ("dielectric", 1.0, m._decay_constants(OMEGA, case["eps_d"], "x")[1]),
        ("metal", -1.0, m._decay_constants(OMEGA, case["eps_d"], "x")[2]),
    ]:
        za, zb = sign * 5e-9, sign * 25e-9
        pa = torch.tensor([[0.0, 0.0, za]], dtype=torch.float64)
        pb = torch.tensor([[0.0, 0.0, zb]], dtype=torch.float64)
        _, Ha = _fields_on(pa, case)
        _, Hb = _fields_on(pb, case)
        kappa_measured = complex(
            torch.log(Ha[0, 1] / Hb[0, 1])
        ) / (abs(zb) - abs(za))
        assert abs(kappa_measured - kappa_expected) / abs(kappa_expected) < 1e-6


def test_field_scale_is_impedance_like():
    """|E| / |H| for the mode is of order eta0 (sanity for anchor scaling)."""
    case = CASES["isotropic-silver-air"]
    coords = _coords("dielectric")
    E, H = _fields_on(coords, case)
    ratio = float(E.abs().max() / H.abs().max())
    assert 0.3 * ETA0 < ratio < 3 * ETA0
