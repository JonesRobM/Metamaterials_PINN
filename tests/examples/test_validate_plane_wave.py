"""Tests for examples/validate_plane_wave.py."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from examples import validate_plane_wave as vpw

DEVICE = torch.device("cpu")

EXPECTED_METRIC_KEYS = {
    "rel_l2_E",
    "rel_l2_H",
    "rel_l2_total",
    "curl_E_residual_rel",
    "curl_H_residual_rel",
    "curl_E_residual_max_rel",
    "curl_H_residual_max_rel",
    "div_E_rel",
    "div_H_rel",
    "E_H_orthogonality",
    "E_k_transversality",
    "poynting_alignment",
    "mean_abs_E",
    "mean_abs_H",
    "impedance_ratio",
    "wavelength_fft",
    "wavelength_fft_rel_error",
    "wavelength_phase_fit",
    "wavelength_phase_fit_rel_error",
}


def test_constants():
    assert vpw.OMEGA == pytest.approx(2 * np.pi * 1e9)
    assert vpw.C > 0 and vpw.MU0 > 0 and vpw.EPS0 > 0
    assert vpw.K0 == pytest.approx(vpw.OMEGA / vpw.C)
    assert vpw.WAVELENGTH == pytest.approx(0.2998, rel=1e-3)
    assert vpw.H0 == pytest.approx(vpw.E0 / vpw.ETA0)
    # Dimensionless frame: k0 * lambda = 2 pi
    assert vpw.K0 * vpw.WAVELENGTH == pytest.approx(vpw.K_HAT)


def test_sample_collocation_points():
    coords = vpw.sample_collocation_points(100, device=DEVICE)
    assert coords.shape == (100, 3)
    assert coords.dtype == torch.float32
    assert coords.requires_grad
    assert torch.all(coords >= 0.0) and torch.all(coords <= vpw.WAVELENGTH)


def test_sample_boundary_points_lie_on_faces():
    pts = vpw.sample_boundary_points(60, device=DEVICE)
    assert pts.shape == (60, 3)
    assert torch.all(pts >= 0.0) and torch.all(pts <= vpw.WAVELENGTH)
    on_face = (pts == 0.0) | (pts == vpw.WAVELENGTH)
    assert torch.all(on_face.any(dim=1))


def test_create_network_output_shape():
    torch.manual_seed(0)
    network = vpw.create_network(hidden_dims=(32, 32), fourier_modes=16, device=DEVICE)
    assert isinstance(network, nn.Module)
    coords = vpw.sample_collocation_points(10, device=DEVICE)
    out = network(coords)
    assert out.shape == (10, 6, 2)
    assert out.dtype == torch.float32
    # dimensionless core also produces [N, 6, 2]
    assert network.core(coords / vpw.WAVELENGTH).shape == (10, 6, 2)


def test_analytical_network_satisfies_maxwell():
    """The analytical field must have ~zero residual under the e^{-iwt} convention."""
    torch.manual_seed(0)
    metrics = vpw.validate(vpw.AnalyticalPlaneWave(), n_points=500, device=DEVICE)
    assert EXPECTED_METRIC_KEYS <= set(metrics)
    for key, value in metrics.items():
        assert isinstance(value, float), key
        assert np.isfinite(value), key
    assert metrics["curl_E_residual_rel"] < 1e-5
    assert metrics["curl_H_residual_rel"] < 1e-5
    assert metrics["rel_l2_E"] < 1e-6
    assert metrics["E_H_orthogonality"] < 1e-5
    assert metrics["E_k_transversality"] < 1e-6
    assert metrics["poynting_alignment"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["impedance_ratio"] == pytest.approx(1.0, rel=1e-5)
    assert metrics["wavelength_phase_fit_rel_error"] < 1e-4
    assert metrics["wavelength_fft_rel_error"] < 2e-2  # limited by zero-padded FFT resolution


def test_validate_on_untrained_network_returns_finite_metrics():
    torch.manual_seed(0)
    network = vpw.create_network(hidden_dims=(32, 32), fourier_modes=16, device=DEVICE)
    metrics = vpw.validate(network, n_points=300, device=DEVICE)
    assert EXPECTED_METRIC_KEYS <= set(metrics)
    assert all(np.isfinite(v) for v in metrics.values())


def test_success_tier_classification():
    good = {
        "curl_E_residual_rel": 1e-7, "curl_H_residual_rel": 1e-7,
        "E_H_orthogonality": 1e-3, "wavelength_phase_fit_rel_error": 1e-3,
    }
    assert vpw.success_tier(good) == "target"
    assert vpw.success_tier({**good, "curl_E_residual_rel": 1e-5}) == "minimum"
    assert vpw.success_tier({**good, "curl_E_residual_rel": 1e-2}) == "not met"


@pytest.mark.slow
def test_short_training_reduces_loss():
    torch.manual_seed(0)
    network = vpw.create_network(hidden_dims=(32, 32), fourier_modes=16, device=DEVICE)
    network, history = vpw.train(
        network, n_epochs=50, n_points=128, device=DEVICE, log_every=1000, lbfgs_steps=1
    )
    assert set(history) == {"epoch", "total", "curl", "div", "boundary", "lr"}
    assert len(history["total"]) == 51  # 50 Adam epochs + 1 L-BFGS step
    assert all(np.isfinite(history["total"]))
    assert min(history["total"][-10:]) < history["total"][0]


def test_main_quick_writes_outputs(tmp_path, monkeypatch):
    """End-to-end smoke test of ``main`` with tiny settings."""
    monkeypatch.setattr(vpw, "QUICK_EPOCHS", 3)
    metrics = vpw.main([
        "--quick", "--device", "cpu", "--seed", "1",
        "--figures-dir", str(tmp_path / "figs"), "--model-out", str(tmp_path / "model.pth"),
    ])
    assert metrics["epochs"] == 3
    assert (tmp_path / "model.pth").exists()
    assert (tmp_path / "figs" / "metrics.json").exists()
    for name in ("training_history", "field_slices", "line_profile", "residual_histogram", "spectrum"):
        assert (tmp_path / "figs" / f"{name}.png").exists()
