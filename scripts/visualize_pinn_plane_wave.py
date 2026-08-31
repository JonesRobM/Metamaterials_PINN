"""Plot the analytical plane-wave E_y field alongside an untrained PINN's prediction.

Usage:
    python scripts/visualize_pinn_plane_wave.py [--points N] [--figures-dir DIR] [--show]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analytical import analytical_plane_wave, complex_to_pinn_format  # noqa: E402
from src.constants import C0  # noqa: E402
from src.data.domain_sampler import UniformSampler  # noqa: E402
from src.models.pinn_network import ComplexPINN  # noqa: E402
from src.utils.plotting import EMFieldPlotter, PlotConfig  # noqa: E402

DEFAULT_FIGURES_DIR = REPO_ROOT / "figures"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--points", type=int, default=5000, help="number of sample points")
    p.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    p.add_argument("--show", action="store_true", help="display figures interactively")
    return p.parse_args(argv)


def visualize_pinn_plane_wave_comparison(
    n_points: int = 5000, figures_dir: Path = DEFAULT_FIGURES_DIR, show: bool = False
):
    """Visualise the analytical plane wave E-field and the (untrained) PINN's prediction."""
    print("Visualizing PINN plane wave comparison...")

    omega = 2 * np.pi * 1e15  # 1 PHz
    k_vec = torch.tensor([omega / C0, 0, 0], dtype=torch.float32)  # |k| = omega / c, along x
    E0_polarization = torch.tensor([0, 1.0, 0], dtype=torch.complex64)  # E along y

    pinn = ComplexPINN(
        spatial_dim=3,
        field_components=6,
        hidden_dims=[64, 64],
        complex_valued=True,
        frequency=omega,
        use_fourier=True,
        fourier_modes=64,
    )

    wavelength = 2 * np.pi * C0 / omega
    sampler = UniformSampler(domain_bounds=[(-wavelength, wavelength), (-1e-7, 1e-7), (-1e-7, 1e-7)])
    coords = sampler.sample_points(n_points=n_points)["points"]

    E_analytical, _ = analytical_plane_wave(coords, k_vec, E0_polarization, omega)
    pinn_output = pinn(coords)  # [N, 6, 2]

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    plotter = EMFieldPlotter(config=PlotConfig(figsize=(10, 6)))

    fig_analytical = plotter.plot_field_2d(
        coords=coords,
        fields=complex_to_pinn_format(E_analytical),
        field_component="Ey",
        plane="xy",
        title="Analytical Plane Wave E_y Field",
    )
    plotter.save_figure(fig_analytical, "analytical_plane_wave_Ey", directory=str(figures_dir))

    fig_pinn = plotter.plot_field_2d(
        coords=coords,
        fields=pinn_output,
        field_component="Ey",
        plane="xy",
        title="Untrained PINN Predicted E_y Field",
    )
    plotter.save_figure(fig_pinn, "untrained_pinn_plane_wave_Ey", directory=str(figures_dir))

    if show:
        plt.show()
    plt.close(fig_analytical)
    plt.close(fig_pinn)

    print(f"Visualization complete. Figures written to {figures_dir}")


def main(argv=None):
    args = parse_args(argv)
    visualize_pinn_plane_wave_comparison(n_points=args.points, figures_dir=args.figures_dir, show=args.show)


if __name__ == "__main__":
    main()
