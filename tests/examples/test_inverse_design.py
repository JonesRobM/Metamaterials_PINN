"""Tests for examples/inverse_design.py (reduced steps, looser tolerances)."""

from __future__ import annotations

import json
import math

import pytest
import torch

from examples import inverse_design as idm
from src.physics.metamaterial import MetamaterialProperties


def _reference(re_t: float, re_n: float) -> MetamaterialProperties:
    return MetamaterialProperties(
        eps_parallel=complex(re_n, idm.IM_N),
        eps_perpendicular=complex(re_t, idm.IM_T),
        optical_axis="z",
        omega=idm.OMEGA,
    )


def test_constants():
    assert idm.OMEGA == pytest.approx(2 * math.pi * idm.C0 / 633e-9)
    assert idm.K0 == pytest.approx(idm.OMEGA / idm.C0)
    assert idm.EPS_D == 1.0
    assert (idm.IM_T, idm.IM_N) == (0.2, 0.05)
    assert idm.DELTA_D_MAX_NM in idm.PARETO_BOUNDS_NM
    assert 5 <= len(idm.PARETO_BOUNDS_NM) <= 8


def test_describe_design_torch_reference_parity():
    """The torch metrics and the scalar-reference metrics agree to 1e-8 relative."""
    design = idm.describe_design(-4.0, 3.0)
    for key in ("n_eff", "L_um", "delta_d_nm", "delta_m_nm", "enhancement"):
        ref = design[f"reference_{key}"]
        assert design[key] == pytest.approx(ref, rel=1e-8)
    assert design["supported"] is True and design["reference_supported"] is True
    assert design["eps_t"] == [-4.0, idm.IM_T]
    assert design["eps_n"] == [3.0, idm.IM_N]


def test_wavevector_two_distinct_solutions(tmp_path):
    """Both inits hit the target; the degeneracy demo returns two distinct designs."""
    torch.manual_seed(0)
    res = idm.run_wavevector(steps=500, figures_dir=tmp_path)
    sols = res["solutions"]
    assert len(sols) == 2
    for sol in sols:
        assert abs(sol["n_eff"] - res["target_n_eff"]) < 0.01
        assert sol["supported"] and sol["reference_supported"]
        # Achieved metric recomputed with the scalar reference matches the torch value.
        assert sol["n_eff"] == pytest.approx(sol["reference_n_eff"], rel=1e-8)
        m = _reference(sol["eps_t"][0], sol["eps_n"][0])
        n_eff_ref = m.spp_wavevector(eps_dielectric=idm.EPS_D).real / idm.K0
        assert sol["n_eff"] == pytest.approx(n_eff_ref, rel=1e-8)
    # Two genuinely different points on the solution contour.
    assert res["solution_separation"] > 1.0
    assert (tmp_path / "wavevector_convergence.png").exists()
    assert (tmp_path / "wavevector_map.png").exists()


def test_propagation_constraint_and_pareto(tmp_path):
    """The confinement constraint is met, parity holds, and the front is monotone."""
    torch.manual_seed(0)
    res = idm.run_propagation(steps=900, figures_dir=tmp_path, bounds_nm=(200.0, 300.0, 450.0))
    sol = res["solution"]
    assert res["constraint_delta_d_nm"] == 300.0
    assert sol["bound_nm"] == 300.0
    assert sol["supported"] and sol["reference_supported"]
    # Constraint satisfied (small penalty-method slack).
    assert sol["delta_d_nm"] <= 300.0 * 1.02
    # Torch metric == scalar reference metric.
    assert sol["L_um"] == pytest.approx(sol["reference_L_um"], rel=1e-8)
    m = _reference(sol["eps_t"][0], sol["eps_n"][0])
    assert sol["L_um"] == pytest.approx(m.propagation_length(eps_dielectric=idm.EPS_D) * 1e6,
                                        rel=1e-8)
    assert sol["delta_d_nm"] == pytest.approx(
        m.penetration_depth_dielectric(eps_dielectric=idm.EPS_D) * 1e9, rel=1e-8
    )
    # Pareto front: looser confinement buys strictly longer propagation.
    pareto = res["pareto"]
    assert [p["bound_nm"] for p in pareto] == [200.0, 300.0, 450.0]
    lengths = [p["L_um"] for p in pareto]
    assert lengths[0] < lengths[1] < lengths[2]
    for p in pareto:
        assert p["supported"]
        assert p["delta_d_nm"] <= p["bound_nm"] * 1.02
    assert (tmp_path / "propagation_convergence.png").exists()
    assert (tmp_path / "propagation_pareto.png").exists()


def test_propagation_rejects_bad_primary(tmp_path):
    with pytest.raises(ValueError):
        idm.run_propagation(steps=1, figures_dir=tmp_path, bounds_nm=(200.0,), primary_nm=300.0)


def test_enhancement_hits_target(tmp_path):
    torch.manual_seed(0)
    res = idm.run_enhancement(steps=500, figures_dir=tmp_path)
    sol = res["solution"]
    assert abs(sol["enhancement"] - res["target_enhancement"]) < 0.01
    assert sol["supported"] and sol["reference_supported"]
    assert sol["enhancement"] == pytest.approx(sol["reference_enhancement"], rel=1e-8)
    m = _reference(sol["eps_t"][0], sol["eps_n"][0])
    assert sol["enhancement"] == pytest.approx(
        m.field_enhancement_factor(eps_dielectric=idm.EPS_D), rel=1e-8
    )
    assert (tmp_path / "enhancement_convergence.png").exists()


def test_parse_args_defaults_and_choices():
    args = idm.parse_args([])
    assert args.problem == "all" and args.steps == idm.DEFAULT_STEPS
    args = idm.parse_args(["--problem", "propagation", "--steps", "100"])
    assert args.problem == "propagation" and args.steps == 100
    with pytest.raises(SystemExit):
        idm.parse_args(["--problem", "spin"])


def test_main_all_writes_json_schema(tmp_path):
    """main --problem all produces design_results.json with the full schema."""
    results = idm.main(["--problem", "all", "--figures-dir", str(tmp_path), "--steps", "300"])
    path = tmp_path / "design_results.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data.keys() == results.keys()
    for key in ("lambda0_nm", "eps_d", "im_eps_t", "im_eps_n", "steps",
                "wavevector", "propagation", "enhancement"):
        assert key in data
    wv = data["wavevector"]
    assert set(wv) == {"target_n_eff", "solutions", "solution_separation", "figures"}
    for sol in wv["solutions"]:
        for k in ("init", "eps_t", "eps_n", "n_eff", "reference_n_eff", "supported",
                  "final_loss", "target_n_eff"):
            assert k in sol
    prop = data["propagation"]
    assert set(prop) == {"constraint_delta_d_nm", "solution", "pareto", "figures"}
    assert len(prop["pareto"]) == len(idm.PARETO_BOUNDS_NM)
    for entry in prop["pareto"]:
        for k in ("bound_nm", "eps_t", "eps_n", "L_um", "reference_L_um",
                  "delta_d_nm", "reference_delta_d_nm", "supported"):
            assert k in entry
    enh = data["enhancement"]
    assert set(enh) == {"target_enhancement", "solution", "figures"}
    for k in ("enhancement", "reference_enhancement", "target_enhancement"):
        assert k in enh["solution"]
    # Every referenced figure file exists.
    for problem in ("wavevector", "propagation", "enhancement"):
        for fig in data[problem]["figures"].values():
            assert (tmp_path / fig.split("/")[-1]).exists()
