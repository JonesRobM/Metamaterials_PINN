"""
Plane-wave PINN tests.

* A smoke test on an untrained ``ComplexPINN`` (finite, correct shape, non-constant,
  finite Maxwell loss).
* A regression test on the checkpoint ``artifacts/models/plane_wave_pinn.pth`` produced by
  ``scripts/train_plane_wave_pinn.py`` (architecture built by that script's
  ``build_plane_wave_pinn``). The checkpoint is trained with the Maxwell curl loss plus a
  boundary anchor on the analytical wave, so it is compared against the analytical wave
  in relative L2 and against an untrained network in terms of the curl residual.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.train_plane_wave_pinn import build_plane_wave_pinn  # noqa: E402
from src.analytical import analytical_plane_wave, complex_to_pinn_format
from src.constants import C0
from src.data.domain_sampler import UniformSampler
from src.models.loss_functions import MaxwellCurlLoss
from src.models.pinn_network import ComplexPINN
from tests.conftest import OMEGA

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO_ROOT / "artifacts" / "models" / "plane_wave_pinn.pth"
# Shipped checkpoint (2000 epochs, Adam 1e-3) measures rel-L2 ~0.19 for E and ~0.16 for H
# on 4000 in-box points; threshold is ~1.5x that.
REL_L2_THRESHOLD = 0.3

# Wave used by the training / visualisation scripts: |k| = k0 along x, E along y.
K0 = OMEGA / C0
K_VEC = torch.tensor([K0, 0.0, 0.0])
E0_POL = torch.tensor([0.0, 1.0, 0.0], dtype=torch.complex64)


def _analytical_pinn_format(coords: torch.Tensor) -> torch.Tensor:
    E, H = analytical_plane_wave(coords, K_VEC, E0_POL, OMEGA)
    return torch.cat([complex_to_pinn_format(E), complex_to_pinn_format(H)], dim=1)


def _trained_architecture():
    """Exactly the network ``scripts/train_plane_wave_pinn.py`` trains and saves."""
    return build_plane_wave_pinn(OMEGA)


def _relative_l2(pred: torch.Tensor, ref: torch.Tensor) -> float:
    return float(torch.norm(pred - ref) / torch.norm(ref))


def _raw_curl_loss(net: torch.nn.Module, coords: torch.Tensor) -> float:
    coords = coords.detach().clone().requires_grad_(True)
    return float(MaxwellCurlLoss(OMEGA)(network=net, coords=coords).detach())


class TestUntrainedPINN:
    def test_untrained_pinn_smoke(self):
        torch.manual_seed(0)
        pinn = ComplexPINN(spatial_dim=3, field_components=6, complex_valued=True, frequency=OMEGA,
                           use_fourier=True, hidden_dims=[64, 64], fourier_modes=64)
        sampler = UniformSampler(domain_bounds=[(-1e-6, 1e-6), (-1e-7, 1e-7), (-1e-7, 1e-7)], seed=0)
        coords = sampler.sample_points(n_points=100)["points"]

        pinn_output = pinn(coords)
        assert pinn_output.shape == (100, 6, 2)
        assert torch.isfinite(pinn_output).all()
        # Non-constant over the domain. The variation is tiny (~1e-7) because raw SI
        # coordinates are fed to a network whose Fourier band is (0.1, 20) rad per input
        # unit -- see TestFourierScaling in test_models.py for the nondimensional fix.
        assert float(pinn_output.detach().std(dim=0).max()) > 0
        assert not torch.allclose(pinn_output[0], pinn_output[-1])

        reference = _analytical_pinn_format(coords)
        assert reference.shape == pinn_output.shape
        assert torch.isfinite(reference).all()

        maxwell_loss = MaxwellCurlLoss(OMEGA)(network=pinn, coords=coords.clone().requires_grad_(True))
        assert torch.isfinite(maxwell_loss)
        assert float(maxwell_loss) > 0
        maxwell_loss.backward()
        assert all(p.grad is not None for p in pinn.parameters())


@pytest.mark.skipif(not CHECKPOINT.exists(), reason=f"trained checkpoint not found at {CHECKPOINT}")
class TestTrainedCheckpoint:
    @pytest.fixture(scope="class")
    def trained(self):
        net = _trained_architecture()
        state = torch.load(CHECKPOINT, map_location="cpu")
        net.load_state_dict(state)  # strict: raises if the architecture drifted
        net.eval()
        return net

    @pytest.fixture(scope="class")
    def coords(self):
        # Same cube of side 2λ that scripts/train_plane_wave_pinn.py trains on
        lam = 2 * np.pi / K0
        sampler = UniformSampler(domain_bounds=[(-lam, lam)] * 3, seed=1234)
        return sampler.sample_points(n_points=512)["points"]

    def test_checkpoint_matches_architecture(self, trained):
        assert trained.core.fourier_encoder.k_vectors.shape == (64, 3)
        assert trained.core.input_projection.in_features == 131  # 128 Fourier + 3 raw coords
        assert trained.length_scale == pytest.approx(2 * np.pi / K0)

    def test_prediction_is_finite(self, trained, coords):
        with torch.no_grad():
            out = trained(coords)
        assert out.shape == (512, 6, 2)
        assert torch.isfinite(out).all()

    def test_curl_loss_far_below_untrained(self, trained, coords):
        """The trained network's curl residual must be orders of magnitude below a fresh
        network of the same architecture (guards against an un-optimised checkpoint)."""
        torch.manual_seed(0)
        untrained = _trained_architecture()
        # Measured ratio ~3e-4 for the shipped checkpoint
        assert _raw_curl_loss(trained, coords) < 1e-2 * _raw_curl_loss(untrained, coords)

    def test_field_amplitude_not_collapsed(self, trained, coords):
        """The zero field minimises the curl loss exactly; the boundary anchor must prevent
        that collapse (an earlier checkpoint had |E| ~ 1e-6)."""
        with torch.no_grad():
            E = trained(coords)[:, :3]
        assert float(E.norm(dim=(1, 2)).mean()) > 0.5

    def test_relative_l2_error_vs_analytical(self, trained, coords):
        """Threshold set at ~1.5x the value measured for the shipped checkpoint (see
        docs/plans/2026-08-29-plane-wave-validation-results.md for the full experiment,
        which reaches far tighter agreement)."""
        with torch.no_grad():
            out = trained(coords)
        rel_E = _relative_l2(out[:, :3], _analytical_pinn_format(coords)[:, :3])
        rel_H = _relative_l2(out[:, 3:], _analytical_pinn_format(coords)[:, 3:])
        assert np.isfinite(rel_E) and np.isfinite(rel_H)
        assert rel_E < REL_L2_THRESHOLD
        assert rel_H < REL_L2_THRESHOLD
