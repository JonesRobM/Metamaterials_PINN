"""Tests for the domain samplers in ``src.data.domain_sampler``."""

import numpy as np
import pytest
import torch

from src.data.domain_sampler import (
    AdaptiveSampler,
    InterfaceSampler,
    SamplingRegion,
    SamplingStrategy,
    SPPDomainSampler,
    StratifiedSampler,
    UniformSampler,
)

BOUNDS_3D = [(-1e-6, 1e-6), (-2e-6, 2e-6), (-3e-6, 3e-6)]
BOUNDS_2D = [(0.0, 1.0), (-1.0, 1.0)]


def _in_bounds(points: torch.Tensor, bounds) -> bool:
    lows = torch.tensor([lo for lo, _ in bounds], dtype=points.dtype)
    highs = torch.tensor([hi for _, hi in bounds], dtype=points.dtype)
    return bool(torch.all(points >= lows) and torch.all(points <= highs))


# --------------------------------------------------------------------------- base class
class TestDomainSamplerBase:
    def test_invalid_bounds_raise(self):
        with pytest.raises(ValueError):
            UniformSampler([(1.0, 0.0)])

    def test_volume_and_center(self):
        s = UniformSampler(BOUNDS_3D)
        assert s.spatial_dim == 3
        assert s.domain_volume == pytest.approx(2e-6 * 4e-6 * 6e-6)
        assert torch.allclose(s.domain_center, torch.zeros(3))

    def test_dtype_propagates(self):
        s = UniformSampler(BOUNDS_2D, dtype=torch.float64)
        pts = s.sample_points(10)["points"]
        assert pts.dtype == torch.float64

    def test_sampling_strategy_enum(self):
        assert SamplingStrategy("uniform") is SamplingStrategy.UNIFORM
        assert len(SamplingStrategy) == 5


# --------------------------------------------------------------------------- uniform
class TestUniformSampler:
    def test_output_keys_and_shapes(self):
        s = UniformSampler(BOUNDS_3D)
        out = s.sample_points(50)
        assert set(out) == {"points", "regions", "weights"}
        assert out["points"].shape == (50, 3)
        assert out["regions"].shape == (50,)
        assert out["regions"].dtype == torch.long
        assert out["weights"].shape == (50,)
        assert torch.all(out["weights"] == 1)

    def test_points_within_bounds(self):
        s = UniformSampler(BOUNDS_3D)
        pts = s.sample_points(500)["points"]
        assert _in_bounds(pts, BOUNDS_3D)

    def test_seed_reproducible(self):
        a = UniformSampler(BOUNDS_2D, seed=123).sample_points(20)["points"]
        b = UniformSampler(BOUNDS_2D, seed=123).sample_points(20)["points"]
        assert torch.equal(a, b)

    def test_different_seeds_differ(self):
        a = UniformSampler(BOUNDS_2D, seed=1).sample_points(20)["points"]
        b = UniformSampler(BOUNDS_2D, seed=2).sample_points(20)["points"]
        assert not torch.equal(a, b)

    def test_covers_domain(self):
        """Every octant of the box receives samples for a large draw."""
        s = UniformSampler(BOUNDS_3D, seed=0)
        pts = s.sample_points(2000)["points"]
        signs = (pts > 0).long()
        octants = signs[:, 0] * 4 + signs[:, 1] * 2 + signs[:, 2]
        assert len(torch.unique(octants)) == 8

    def test_exclusion_region_respected(self):
        excl = SamplingRegion("hole", [(0.25, 0.75), (-0.5, 0.5)])
        s = UniformSampler(BOUNDS_2D, exclusion_regions=[excl], seed=0)
        out = s.sample_points(200)
        pts = out["points"]
        assert pts.shape == (200, 2)
        inside = (pts[:, 0] >= 0.25) & (pts[:, 0] <= 0.75) & (pts[:, 1] >= -0.5) & (pts[:, 1] <= 0.5)
        assert not inside.any()
        assert _in_bounds(pts, BOUNDS_2D)


# --------------------------------------------------------------------------- stratified
class TestStratifiedSampler:
    @pytest.fixture
    def regions(self):
        return [
            SamplingRegion("left", [(0.0, 0.5), (-1.0, 1.0)], weight=3.0),
            SamplingRegion("right", [(0.5, 1.0), (-1.0, 1.0)], weight=1.0),
        ]

    def test_weights_normalised(self, regions):
        StratifiedSampler(BOUNDS_2D, regions)
        assert sum(r.weight for r in regions) == pytest.approx(1.0)
        assert regions[0].weight == pytest.approx(0.75)

    def test_output_shapes_and_region_info(self, regions):
        s = StratifiedSampler(BOUNDS_2D, regions, seed=0)
        out = s.sample_points(200)
        assert out["points"].shape == (200, 2)
        assert out["regions"].shape == (200,)
        assert out["weights"].shape == (200,)
        assert out["region_info"] == ["left", "right"]
        assert _in_bounds(out["points"], BOUNDS_2D)

    def test_region_coverage_matches_weights(self, regions):
        s = StratifiedSampler(BOUNDS_2D, regions, seed=0)
        out = s.sample_points(400)
        pts, ids = out["points"], out["regions"]
        # Region labels are consistent with the region bounds
        assert torch.all(pts[ids == 0, 0] <= 0.5)
        assert torch.all(pts[ids == 1, 0] >= 0.5)
        # 75 % of points should land in the left region
        frac_left = float((ids == 0).float().mean())
        assert frac_left == pytest.approx(0.75, abs=0.05)

    def test_density_function_biases_samples(self):
        """A density function concentrated on y > 0 should skew the samples."""
        region = SamplingRegion(
            "dense", [(0.0, 1.0), (-1.0, 1.0)], weight=1.0,
            density_function=lambda p: 1.0 if p[1] > 0 else 0.05,
        )
        s = StratifiedSampler(BOUNDS_2D, [region], seed=0)
        pts = s.sample_points(300)["points"]
        assert pts.shape == (300, 2)
        assert float((pts[:, 1] > 0).float().mean()) > 0.8

    def test_padding_when_regions_under_sample(self):
        """Weights that do not sum to one leave a deficit that is padded uniformly."""
        region = SamplingRegion("half", [(0.0, 0.5), (-1.0, 1.0)], weight=0.5)
        s = StratifiedSampler(BOUNDS_2D, [region], normalize_weights=False, seed=0)
        out = s.sample_points(100)
        assert out["points"].shape == (100, 2)
        assert out["regions"].shape == (100,)


# --------------------------------------------------------------------------- interface
class TestInterfaceSampler:
    def test_output_keys(self):
        s = InterfaceSampler(BOUNDS_3D, [{"type": "plane", "normal_axis": 2, "position": 0.0}],
                             interface_thickness=1e-8, seed=0)
        out = s.sample_points(100, interface_fraction=0.5)
        assert set(out) == {"points", "interface_labels", "weights", "n_interface", "n_bulk"}
        assert out["points"].shape == (100, 3)
        assert out["n_interface"] == 50 and out["n_bulk"] == 50
        assert int(out["interface_labels"].sum()) == 50
        assert torch.all(out["weights"][out["interface_labels"]] == 2.0)
        assert torch.all(out["weights"][~out["interface_labels"]] == 1.0)

    def test_interface_points_cluster_near_plane(self):
        thickness = 1e-8
        s = InterfaceSampler(BOUNDS_3D, [{"type": "plane", "normal_axis": 2, "position": 0.5e-6}],
                             interface_thickness=thickness, seed=0)
        out = s.sample_points(400, interface_fraction=0.5)
        pts, labels = out["points"], out["interface_labels"]
        z_iface = pts[labels, 2]
        assert torch.all((z_iface - 0.5e-6).abs() <= thickness + 1e-12)
        # Bulk points are spread over the whole z range
        z_bulk = pts[~labels, 2]
        assert float(z_bulk.std()) > 10 * thickness
        assert _in_bounds(pts, BOUNDS_3D)

    def test_interface_clamped_to_domain(self):
        """Interface at the domain edge does not produce out-of-bounds points."""
        s = InterfaceSampler(BOUNDS_3D, [{"type": "plane", "normal_axis": 2, "position": 3e-6}],
                             interface_thickness=1e-7, seed=0)
        pts = s.sample_points(100)["points"]
        assert _in_bounds(pts, BOUNDS_3D)

    def test_cylindrical_interface(self):
        radius, thick = 1e-6, 1e-8
        s = InterfaceSampler(
            [(-2e-6, 2e-6), (-2e-6, 2e-6), (0.0, 1e-6)],
            [{"type": "cylinder", "radius": radius, "axis": 2, "center": [0.0, 0.0]}],
            interface_thickness=thick, seed=0,
        )
        out = s.sample_points(200, interface_fraction=1.0)
        pts = out["points"][out["interface_labels"]]
        r = torch.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        assert torch.all((r - radius).abs() <= thick * 1.01)

    def test_unknown_interface_type_warns(self):
        s = InterfaceSampler(BOUNDS_3D, [{"type": "sphere"}], seed=0)
        with pytest.warns(UserWarning, match="Unknown interface type"):
            out = s.sample_points(20)
        assert out["n_interface"] == 0
        assert out["points"].shape == (10, 3)


# --------------------------------------------------------------------------- SPP
class TestSPPDomainSampler:
    BOUNDS = [(-1e-6, 1e-6), (-1e-6, 1e-6), (-2e-6, 2e-6)]

    def test_regions_created(self):
        s = SPPDomainSampler(self.BOUNDS, interface_position=0.0, spp_decay_length=2e-7,
                             metamaterial_bounds=[(-1e-6, 1e-6), (-1e-6, 1e-6), (-2e-6, 0.0)])
        names = [r.name for r in s.spp_regions]
        assert names == ["spp_interface", "metamaterial_spp", "dielectric"]
        iface = s.spp_regions[0].bounds[2]
        assert iface == (-2e-7, 2e-7)

    def test_output_metadata(self):
        s = SPPDomainSampler(self.BOUNDS, spp_decay_length=2e-7, seed=0)
        out = s.sample_points(120)
        assert out["points"].shape == (120, 3)
        assert out["interface_position"] == 0.0
        assert out["decay_length"] == 2e-7
        assert out["spp_regions"] == ["spp_interface", "dielectric"]
        assert _in_bounds(out["points"], self.BOUNDS)

    def test_density_higher_near_interface(self):
        L = 2e-7
        s = SPPDomainSampler(self.BOUNDS, spp_decay_length=L, seed=0)
        z = s.sample_points(600)["points"][:, 2].abs()
        near = float((z < L).float().mean())
        far = float((z > 1e-6).float().mean())
        # Uniform sampling would give near ≈ 0.1, far ≈ 0.5.
        assert near > 0.5
        assert far < 0.15

    def test_density_functions(self):
        s = SPPDomainSampler(self.BOUNDS, spp_decay_length=1e-7)
        assert s._spp_interface_density(np.array([0.0, 0.0, 0.0])) == pytest.approx(1.0)
        assert s._spp_interface_density(np.array([0.0, 0.0, 1e-7])) < 1e-3
        assert s._spp_decay_density(np.array([0.0, 0.0, -1e-7])) == pytest.approx(np.exp(-1))
        assert s._spp_decay_density(np.array([0.0, 0.0, 1e-7])) == 0.1
        assert s._dielectric_decay_density(np.array([0.0, 0.0, 1e-7])) == pytest.approx(np.exp(-1))
        assert s._dielectric_decay_density(np.array([0.0, 0.0, -1e-7])) == 0.1


# --------------------------------------------------------------------------- adaptive
class TestAdaptiveSampler:
    def test_initial_sampling_uses_base(self):
        s = AdaptiveSampler(BOUNDS_2D, seed=0)
        out = s.sample_points(30)
        assert out["points"].shape == (30, 2)
        assert "adaptive_iteration" not in out

    def test_update_residuals_limits_memory(self):
        s = AdaptiveSampler(BOUNDS_2D, memory_length=2, seed=0)
        pts = torch.rand(10, 2)
        for _ in range(4):
            s.update_residuals(pts, torch.rand(10))
        assert len(s.residual_history) == 2
        assert s.adaptation_count == 4
        assert callable(s.density_estimate)

    def test_adaptive_sampling_moves_points_to_high_residual(self):
        s = AdaptiveSampler(BOUNDS_2D, seed=0)
        # Residual is large only in the right half of the domain.
        pts = s.sample_points(400)["points"]
        residuals = torch.where(pts[:, 0] > 0.5, torch.full((400,), 10.0), torch.full((400,), 0.01))
        s.update_residuals(pts, residuals)
        out = s.sample_points(300)
        assert out["sampling_method"] == "adaptive"
        assert out["adaptive_iteration"] == 1
        assert out["points"].shape == (300, 2)
        assert _in_bounds(out["points"], BOUNDS_2D)
        frac_right = float((out["points"][:, 0] > 0.5).float().mean())
        assert frac_right > 0.6

    def test_vector_residuals_accepted(self):
        s = AdaptiveSampler(BOUNDS_2D, seed=0)
        pts = torch.rand(50, 2)
        s.update_residuals(pts, torch.rand(50, 6))
        assert s.density_estimate is not None
        val = s.density_estimate(np.array([0.5, 0.0]))
        assert 0.0 <= val <= 10.0

    def test_refine_without_history_falls_back(self):
        s = AdaptiveSampler(BOUNDS_2D, seed=0)
        new = s.refine_around_high_residuals(25)
        assert new.shape == (25, 2)
        assert _in_bounds(new, BOUNDS_2D)

    def test_refine_zero_points(self):
        s = AdaptiveSampler(BOUNDS_2D, seed=0)
        assert s.refine_around_high_residuals(0).shape == (0, 2)

    def test_refine_clusters_around_high_residuals(self):
        s = AdaptiveSampler(BOUNDS_2D, seed=0)
        pts = torch.rand(200, 2)
        pts[:, 1] = pts[:, 1] * 2 - 1
        # Single hot spot at (0.9, 0.9)
        residuals = torch.exp(-((pts[:, 0] - 0.9) ** 2 + (pts[:, 1] - 0.9) ** 2) / 0.005)
        s.update_residuals(pts, residuals)
        new = s.refine_around_high_residuals(100, residual_threshold=0.9, radius_factor=0.05)
        assert new.shape == (100, 2)
        assert _in_bounds(new, BOUNDS_2D)
        centre = torch.tensor([0.9, 0.9])
        dist = torch.norm(new - centre, dim=1)
        assert float(dist.median()) < 0.3
        assert float(dist.mean()) < float(torch.norm(pts - centre, dim=1).mean())


# --------------------------------------------------------------------------- boundary
class TestSampleDomainBoundary:
    @pytest.mark.parametrize("n", [1, 7, 60])
    def test_points_lie_on_faces(self, n):
        s = UniformSampler(BOUNDS_3D, seed=0)
        pts = s.sample_domain_boundary(n)
        assert pts.shape == (n, 3)
        assert _in_bounds(pts, BOUNDS_3D)
        lows = torch.tensor([lo for lo, _ in BOUNDS_3D])
        highs = torch.tensor([hi for _, hi in BOUNDS_3D])
        on_face = (pts == lows) | (pts == highs)
        assert torch.all(on_face.any(dim=1))

    def test_all_faces_represented(self):
        s = UniformSampler(BOUNDS_3D, seed=0)
        pts = s.sample_domain_boundary(60)
        lows = torch.tensor([lo for lo, _ in BOUNDS_3D])
        highs = torch.tensor([hi for _, hi in BOUNDS_3D])
        for d in range(3):
            assert (pts[:, d] == lows[d]).sum() >= 10
            assert (pts[:, d] == highs[d]).sum() >= 10

    def test_zero_points(self):
        s = UniformSampler(BOUNDS_2D)
        assert s.sample_domain_boundary(0).shape == (0, 2)
