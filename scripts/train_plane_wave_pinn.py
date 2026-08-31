"""Train a ComplexPINN on the free-space plane-wave problem (Maxwell curl loss + boundary anchor).

Usage:
    python scripts/train_plane_wave_pinn.py [--epochs N] [--lr LR] [--points N]
                                            [--model-out PATH] [--figures-dir DIR] [--show]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analytical import analytical_plane_wave, complex_to_pinn_format  # noqa: E402
from src.constants import C0  # noqa: E402
from src.data.domain_sampler import UniformSampler  # noqa: E402
from src.models.loss_functions import MaxwellCurlLoss  # noqa: E402
from src.models.pinn_network import ComplexPINN, NondimensionalPINN  # noqa: E402
from src.utils.plotting import EMFieldPlotter, PlotConfig, TrainingPlotter  # noqa: E402

DEFAULT_MODEL_OUT = REPO_ROOT / "artifacts" / "models" / "plane_wave_pinn.pth"
DEFAULT_FIGURES_DIR = REPO_ROOT / "figures"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--points", type=int, default=1000, help="collocation points per epoch")
    p.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
    p.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    p.add_argument("--show", action="store_true", help="display figures interactively")
    return p.parse_args(argv)


def build_plane_wave_pinn(omega: float) -> NondimensionalPINN:
    """Complex Fourier-feature PINN evaluated in SI units via a nondimensional wrapper.

    The core network sees coordinates in wavelengths (so its default Fourier band
    (0.1, 20) rad/λ brackets the target wavenumber 2π) and outputs O(1) fields; the
    wrapper rescales H by 1/η0 so the physical E and H are both represented.
    """
    wavelength = 2 * np.pi * C0 / omega
    core = ComplexPINN(
        spatial_dim=3,
        field_components=6,
        hidden_dims=[128, 128, 128],
        complex_valued=True,
        frequency=omega,
        use_fourier=True,
        fourier_modes=128,
    )
    return NondimensionalPINN(core, wavelength, NondimensionalPINN.em_field_scale(1.0))


def train_plane_wave_pinn_by_hand(
    num_epochs: int = 500,
    learning_rate: float = 1e-3,
    points_per_epoch: int = 1000,
    model_out: Path = DEFAULT_MODEL_OUT,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
    show: bool = False,
):
    """Train a ComplexPINN for plane waves with a manual loop, then plot the results."""
    print("Starting manual training of PINN for plane waves...")

    omega = 2 * np.pi * 1e15  # 1 PHz
    k0 = omega / C0
    wavelength = 2 * np.pi / k0

    pinn = build_plane_wave_pinn(omega)

    # Demo domain: a cube of side 2λ. (Larger boxes hold many more oscillations and
    # need the schedule/L-BFGS refinement of examples/validate_plane_wave.py.)
    half = wavelength
    sampler = UniformSampler(domain_bounds=[(-half, half)] * 3)
    loss_fn = MaxwellCurlLoss(frequency=omega, weight=1.0)
    optimizer = optim.Adam(pinn.parameters(), lr=learning_rate)

    # The curl residual alone admits the trivial solution E = H = 0, so anchor the
    # network with a soft Dirichlet term on the domain boundary taken from the
    # analytical wave (k along x with |k| = omega / c, E along y).
    k_vec = torch.tensor([k0, 0.0, 0.0], dtype=torch.float32)
    E0_polarization = torch.tensor([0.0, 1.0, 0.0], dtype=torch.complex64)
    inv_scale = 1.0 / pinn.field_scale  # compare E and H in the network's O(1) units

    def anchor_loss(n_points: int) -> torch.Tensor:
        b = sampler.sample_domain_boundary(n_points)
        E_a, H_a = analytical_plane_wave(b, k_vec, E0_polarization, omega)
        target = complex_to_pinn_format(torch.cat([E_a, H_a], dim=1))
        return torch.mean(((pinn(b) - target) * inv_scale) ** 2)

    # Curl residuals are O(k0 |E|) ~ 1e7 per unit field, so scale them to O(1)
    curl_scale = 1.0 / k0**2

    loss_history = []
    print(f"Training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        coords = sampler.sample_points(n_points=points_per_epoch)["points"]
        coords.requires_grad_(True)

        optimizer.zero_grad()
        loss_curl = loss_fn(network=pinn, coords=coords) * curl_scale
        loss_bc = anchor_loss(points_per_epoch // 4)
        loss = loss_curl + loss_bc
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if (epoch + 1) % 50 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4e} "
                f"(curl {loss_curl.item():.3e}, boundary {loss_bc.item():.3e})"
            )

    print("Training finished.")

    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pinn.state_dict(), model_out)
    print(f"Trained model saved to {model_out}")

    print("Generating plots...")
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    training_plotter = TrainingPlotter()
    fig_loss = training_plotter.plot_training_history({"Maxwell Curl Loss": loss_history})
    training_plotter.save_figure(fig_loss, "plane_wave_training_curve", directory=str(figures_dir))

    pinn.eval()
    n_points_x = n_points_y = 200
    x = torch.linspace(-half, half, n_points_x)
    y = torch.linspace(-half, half, n_points_y)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    z = torch.zeros_like(grid_x)
    plot_coords = torch.stack([grid_x.flatten(), grid_y.flatten(), z.flatten()], dim=1)

    with torch.no_grad():
        predicted_fields = pinn(plot_coords)

    field_plotter = EMFieldPlotter(config=PlotConfig(figsize=(10, 6)))
    fig_field = field_plotter.plot_field_2d(
        coords=plot_coords,
        fields=predicted_fields,
        field_component="Ey",
        plane="xy",
        title="Trained PINN - Predicted E_y Field",
    )
    field_plotter.save_figure(fig_field, "trained_pinn_plane_wave_Ey", directory=str(figures_dir))

    if show:
        plt.show()
    plt.close(fig_loss)
    plt.close(fig_field)

    print(f"Plotting complete. Figures written to {figures_dir}")


def main(argv=None):
    args = parse_args(argv)
    train_plane_wave_pinn_by_hand(
        num_epochs=args.epochs,
        learning_rate=args.lr,
        points_per_epoch=args.points,
        model_out=args.model_out,
        figures_dir=args.figures_dir,
        show=args.show,
    )


if __name__ == "__main__":
    main()
