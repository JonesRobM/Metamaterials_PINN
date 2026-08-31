"""Visualise the electric field of a point charge from a trained ElectrostaticsPINN.

Usage:
    python scripts/visualize_pinn_field.py [--model PATH] [--figures-dir DIR] [--grid-size N] [--show]
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

from src.analytical import analytical_potential  # noqa: E402
from src.models.electrostatics_pinn import ElectrostaticsPINN  # noqa: E402

CHARGE_Q = 1.0
CHARGE_POS = (0.0, 0.0)
X_RANGE = (-1.0, 1.0)
Y_RANGE = (-1.0, 1.0)

DEFAULT_MODEL = REPO_ROOT / "artifacts" / "models" / "point_charge_pinn.pth"
DEFAULT_FIGURES_DIR = REPO_ROOT / "figures"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    p.add_argument("--grid-size", type=int, default=20)
    p.add_argument("--show", action="store_true", help="display the figure interactively")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ElectrostaticsPINN().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    x = np.linspace(X_RANGE[0], X_RANGE[1], args.grid_size)
    y = np.linspace(Y_RANGE[0], Y_RANGE[1], args.grid_size)
    X, Y = np.meshgrid(x, y)

    coords = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32, device=device)
    coords.requires_grad_(True)

    # The PINN learns the correction to the analytical solution
    pinn_potential = model(coords)
    # Evaluate the analytical part in torch so that autograd differentiates it too
    analytical_V = analytical_potential(coords[:, 0], coords[:, 1], CHARGE_Q, CHARGE_POS).unsqueeze(1)
    total_potential = analytical_V + pinn_potential

    # E = -grad V
    grad_V = torch.autograd.grad(outputs=total_potential.sum(), inputs=coords, create_graph=True)[0]
    Ex = -grad_V[:, 0].detach().cpu().numpy().reshape(X.shape)
    Ey = -grad_V[:, 1].detach().cpu().numpy().reshape(Y.shape)
    E_mag = np.sqrt(Ex**2 + Ey**2)

    fig = plt.figure(figsize=(10, 8))
    c = plt.pcolormesh(X, Y, E_mag, cmap="viridis", shading="gouraud")
    plt.colorbar(c, label="Electric Field Magnitude |E| (V/m)")
    # Direction-only quiver on a coarse grid (magnitude is carried by the colour map)
    st = max(1, args.grid_size // 20)
    Em = np.maximum(E_mag, 1e-30)
    plt.quiver(X[::st, ::st], Y[::st, ::st], (Ex / Em)[::st, ::st], (Ey / Em)[::st, ::st],
               color="white", scale=30, width=0.003, alpha=0.8)
    plt.plot(CHARGE_POS[0], CHARGE_POS[1], "ro", markersize=8, label="Point Charge")
    plt.title("Electric Field of a Point Charge (PINN-computed Heatmap)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.xlim(X_RANGE)
    plt.ylim(Y_RANGE)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend()
    plt.grid(True)

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / "pinn_point_charge_heatmap.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved plot: {out}")

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
