"""Plot the analytical electric field of a point charge (no PINN involved).

Usage:
    python scripts/visualize_point_charge.py [--figures-dir DIR] [--grid-size N] [--show]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analytical import analytical_point_charge_field  # noqa: E402

DEFAULT_FIGURES_DIR = REPO_ROOT / "figures"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--charge", type=float, default=1.0, help="charge q [C]")
    p.add_argument("--grid-size", type=int, default=20)
    p.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    p.add_argument("--show", action="store_true", help="display the figure interactively")
    return p.parse_args(argv)


def plot_point_charge_field(
    charge_q=1.0,
    charge_pos=(0, 0),
    grid_size=20,
    x_range=(-1, 1),
    y_range=(-1, 1),
    figures_dir: Path = DEFAULT_FIGURES_DIR,
    show: bool = False,
):
    """Calculate and plot the electric field of a point charge."""
    x = np.linspace(x_range[0], x_range[1], grid_size)
    y = np.linspace(y_range[0], y_range[1], grid_size)
    X, Y = np.meshgrid(x, y)
    Ex, Ey = analytical_point_charge_field(X, Y, charge_q, charge_pos)

    fig = plt.figure(figsize=(8, 8))
    plt.quiver(X, Y, Ex, Ey, scale=5e10, color="b")
    plt.plot(charge_pos[0], charge_pos[1], "ro", markersize=8)
    plt.title(f"Electric Field of a Point Charge (q={charge_q} C)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True)

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / "point_charge_electric_field.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved plot: {out}")

    if show:
        plt.show()
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    plot_point_charge_field(
        charge_q=args.charge, grid_size=args.grid_size, figures_dir=args.figures_dir, show=args.show
    )


if __name__ == "__main__":
    main()
