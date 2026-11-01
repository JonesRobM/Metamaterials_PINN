"""Tests for ``src.utils.metrics`` and smoke tests for ``src.utils.plotting``."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from src.constants import ETA0  # noqa: E402
from src.utils.metrics import (  # noqa: E402
    BoundaryConditionMetrics,
    EnergyConservationMetrics,
    FieldAccuracyMetrics,
    MaxwellResidualMetrics,
    MetricResult,
    MetricsCollector,
    TrainingMetrics,
)
from src.utils.plotting import (  # noqa: E402
    ComplexFieldVisualizer,
    DispersionPlotter,
    EMFieldPlotter,
    PlotConfig,
    SPPAnalysisPlotter,
    TrainingPlotter,
)
from tests.conftest import K0, OMEGA, make_plane_wave, sample_coords  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# =========================================================================== metrics
class TestMaxwellResidualMetrics:
    def test_plane_wave_residuals_small(self, plane_wave_net):
        coords = sample_coords(32)
        metric = MaxwellResidualMetrics(OMEGA)
        res = metric.compute(network=plane_wave_net, coords=coords)
        assert set(res) == {"curl_E", "curl_H", "div_E", "div_B"}
        for r in res.values():
            assert isinstance(r, MetricResult)
            assert np.isfinite(r.value) and r.std is not None
        assert res["curl_E"].value / K0 < 1e-9
        assert res["curl_H"].value / (K0 / ETA0) < 1e-9
        assert res["div_E"].value / K0 < 1e-9
        assert res["div_B"].value / (K0 / ETA0) < 1e-9

    def test_wrong_wavenumber_detected(self):
        net = make_plane_wave(k_scale=0.5)
        res = MaxwellResidualMetrics(OMEGA).compute(network=net, coords=sample_coords(16))
        assert res["curl_E"].value / K0 < 1e-9
        # |curl H + iωε0 E| = 0.75 k0 / η0
        assert res["curl_H"].value == pytest.approx(0.75 * K0 / ETA0, rel=1e-6)

    def test_custom_epsilon_tensor(self):
        net = make_plane_wave(k_scale=1.5)  # satisfies Maxwell for eps = 2.25
        coords = sample_coords(16)
        eps = torch.eye(3, dtype=torch.complex128).unsqueeze(0).expand(16, -1, -1) * 2.25
        res = MaxwellResidualMetrics(OMEGA).compute(network=net, coords=coords, epsilon_tensor=eps)
        assert res["curl_H"].value / (K0 / ETA0) < 1e-9

    def test_history_and_trend(self):
        m = MaxwellResidualMetrics(OMEGA)
        assert m.get_trend() == "insufficient_data"
        for v in np.linspace(1.0, 0.1, 10):
            m.update_history(MetricResult("x", float(v)))
        assert m.get_trend() == "decreasing"
    def test_network_without_get_fields(self, plane_wave_net):
        class Plain(torch.nn.Module):
            def forward(self, c):
                return plane_wave_net(c)

        res = MaxwellResidualMetrics(OMEGA).compute(network=Plain(), coords=sample_coords(8))
        assert res["curl_E"].value / K0 < 1e-9


class TestFieldAccuracyMetrics:
    def test_exact_prediction(self, plane_wave_net):
        coords = sample_coords(20).detach()
        E, H = plane_wave_net.fields(coords)
        fields = torch.cat([E.real, E.imag, H.real, H.imag], dim=1)  # columns 0-2 = Re E
        res = FieldAccuracyMetrics().compute(predicted_fields=fields, reference_fields=fields.clone())
        assert res["mse"].value == 0.0
        assert res["mae"].value == 0.0
        assert res["relative_error"].value == pytest.approx(0.0, abs=1e-12)
        for name in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
            assert f"{name}_mse" in res and f"{name}_correlation" in res
        assert res["Ey_correlation"].value == pytest.approx(1.0, abs=1e-6)
        assert res["Ex_correlation"].value == 0.0  # constant (zero) component -> NaN mapped to 0

    def test_scaled_prediction(self, plane_wave_net):
        coords = sample_coords(20).detach()
        E, H = plane_wave_net.fields(coords)
        ref = torch.cat([E.real, E.imag, H.real, H.imag], dim=1)
        pred = 2.0 * ref
        res = FieldAccuracyMetrics().compute(predicted_fields=pred, reference_fields=ref)
        assert res["relative_error"].value == pytest.approx(1.0, rel=1e-6)
        assert res["mse"].value == pytest.approx(float(torch.mean(ref**2)), rel=1e-6)
        assert res["Ey_correlation"].value == pytest.approx(1.0, abs=1e-6)

    def test_custom_field_names(self):
        a = torch.randn(10, 2, dtype=torch.float64)
        res = FieldAccuracyMetrics().compute(predicted_fields=a, reference_fields=a, field_names=["u", "v"])
        assert "u_mse" in res and "v_correlation" in res


class TestBoundaryAndEnergyMetrics:
    def test_boundary_metrics_zero_for_identical_fields(self, plane_wave_net):
        coords = sample_coords(8).detach()
        coords[:, 2] = 0.0
        eps_mm = torch.eye(3, dtype=torch.complex128).unsqueeze(0).expand(8, -1, -1)
        metric = BoundaryConditionMetrics(interface_normal=(0, 0, 1))
        res = metric.compute(network=plane_wave_net, interface_coords=coords,
                             epsilon_metamaterial=eps_mm, epsilon_dielectric=1.0)
        assert set(res) == {"boundary_total"}
        assert res["boundary_total"].value == pytest.approx(0.0, abs=1e-12)

    def test_energy_metrics_plane_wave(self, plane_wave_net):
        res = EnergyConservationMetrics().compute(network=plane_wave_net, coords=sample_coords(16))
        assert "poynting_divergence" in res and "energy_density" in res
        assert res["poynting_divergence"].value / (K0 / ETA0) < 1e-9
        assert res["energy_density"].value > 0

    def test_training_metrics_run(self):
        metric = TrainingMetrics()
        for epoch, total in enumerate([1.0, 0.5, 0.25, 0.125]):
            res = metric.compute(
                total_loss=torch.tensor(total),
                loss_components={"curl": torch.tensor(total / 2), "div": torch.tensor(total / 2)},
                epoch=epoch,
                learning_rate=1e-3,
            )
        assert isinstance(res, dict) and res
        for r in res.values():
            assert isinstance(r, MetricResult)


class TestMetricsCollector:
    def test_evaluate_all_runs(self, plane_wave_net):
        collector = MetricsCollector({"physics": {"frequency": OMEGA}})
        assert set(collector.metrics) == {"maxwell", "boundary", "spp", "accuracy", "energy", "training"}
        results = collector.evaluate_all(network=plane_wave_net, coords=sample_coords(16), epoch=0)
        assert set(results) == {"maxwell", "spp", "energy"}
        assert results["maxwell"]["curl_E"].value / K0 < 1e-9
        assert 0 in collector.all_results
        report = collector.get_summary_report()
        assert isinstance(report, str) and report

    def test_evaluate_all_warns_instead_of_raising(self):
        class Broken(torch.nn.Module):
            def forward(self, c):
                raise RuntimeError("boom")

        collector = MetricsCollector({})
        with pytest.warns(RuntimeWarning, match="Maxwell metrics failed"):
            results = collector.evaluate_all(network=Broken(), coords=sample_coords(4), epoch=1)
        assert results["maxwell"] == {}
        assert results["spp"] == {}
        assert results["energy"] == {}

    def test_save_metrics(self, plane_wave_net, tmp_path):
        collector = MetricsCollector({"physics": {"frequency": OMEGA}})
        collector.evaluate_all(network=plane_wave_net, coords=sample_coords(8), epoch=0)
        out = tmp_path / "metrics.json"
        collector.save_metrics(str(out))
        assert out.exists() and out.stat().st_size > 0


# =========================================================================== plotting
@pytest.fixture
def field_data(plane_wave_net):
    coords = sample_coords(200, extent=1e-6).detach()
    fields = plane_wave_net(coords).detach().to(torch.float32)
    return coords.to(torch.float32), fields


class TestEMFieldPlotter:
    @pytest.mark.parametrize("plane", ["xy", "xz", "yz"])
    def test_plot_field_2d(self, field_data, plane):
        coords, fields = field_data
        fig = EMFieldPlotter(PlotConfig(figsize=(4, 3), dpi=50)).plot_field_2d(
            coords, fields, field_component="Ey", plane=plane, interface_position=0.0
        )
        assert isinstance(fig, plt.Figure)
        assert fig.axes  # at least the main axes (plus colorbar)

    def test_plot_field_2d_real_layout(self, field_data):
        coords, fields = field_data
        fig = EMFieldPlotter().plot_field_2d(coords, fields[..., 0], field_component="Hz", title="t")
        assert fig.axes[0].get_title() == "t"

    @pytest.mark.parametrize("log_scale", [True, False])
    def test_plot_field_magnitude_2d(self, field_data, log_scale):
        coords, fields = field_data
        fig = EMFieldPlotter().plot_field_magnitude_2d(coords, fields, field_type="E", plane="xz",
                                                       interface_position=0.0, log_scale=log_scale)
        assert isinstance(fig, plt.Figure)

    def test_plot_spp_decay_profile(self, field_data):
        coords, fields = field_data
        fig = EMFieldPlotter().plot_spp_decay_profile(coords, fields, interface_position=0.0, field_component="Ey")
        assert isinstance(fig, plt.Figure)

    def test_save_figure(self, field_data, tmp_path):
        coords, fields = field_data
        plotter = EMFieldPlotter(PlotConfig(dpi=30))
        fig = plotter.plot_field_2d(coords, fields)
        plotter.save_figure(fig, "test_fig", directory=str(tmp_path))
        assert (tmp_path / "test_fig.png").exists()


class TestOtherPlotters:
    def test_complex_field_visualizer(self, field_data):
        coords, fields = field_data
        viz = ComplexFieldVisualizer()
        fig = viz.plot_complex_field(coords, fields, component="Ey")
        assert isinstance(fig, plt.Figure) and len(fig.axes) >= 2
        # Already-complex input path
        fig2 = viz.plot_complex_field(coords, torch.complex(fields[..., 0], fields[..., 1]), component="Ey")
        assert isinstance(fig2, plt.Figure)

    def test_vector_field(self, field_data):
        coords, fields = field_data
        fig = ComplexFieldVisualizer().plot_vector_field(coords, fields)
        assert isinstance(fig, plt.Figure)

    def test_training_plotter(self):
        plotter = TrainingPlotter()
        hist = {"total": list(np.geomspace(1, 1e-3, 20)), "curl": list(np.geomspace(0.5, 1e-3, 20))}
        fig = plotter.plot_training_history(hist)
        assert isinstance(fig, plt.Figure) and len(fig.axes) == 1
        fig2 = plotter.plot_training_history(hist, metrics_history={"residual": list(np.linspace(1, 0, 20))})
        assert len(fig2.axes) == 2
        fig3 = plotter.plot_loss_components_breakdown({"curl": 0.5, "div": 0.25, "bc": 0.25})
        assert isinstance(fig3, plt.Figure)

    def test_training_convergence_analysis(self):
        fig = TrainingPlotter().plot_convergence_analysis(list(np.geomspace(1, 1e-4, 50)))
        assert isinstance(fig, plt.Figure)

    def test_spp_analysis_plotter(self):
        freqs = np.linspace(1e14, 1e15, 10)
        k = (1 + 0.1j) * freqs / 3e8
        plotter = SPPAnalysisPlotter()
        fig = plotter.plot_dispersion_relation(freqs, k, analytical_k=k * 1.01)
        assert isinstance(fig, plt.Figure) and len(fig.axes) == 2
        fig2 = plotter.plot_penetration_depths(freqs, 1e-8 / (freqs / 1e14), 1e-7 / (freqs / 1e14))
        assert isinstance(fig2, plt.Figure)

    def test_dispersion_plotter(self):
        k = np.linspace(0, 2e7, 30)
        freqs = np.stack([k * 3e8, k * 2e8], axis=1)
        fig = DispersionPlotter().plot_band_structure(k, freqs, mode_labels=["light line", "slow"])
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes[0].lines) == 2
