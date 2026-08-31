"""Train an ElectrostaticsPINN to learn the correction to a point-charge potential.

Usage:
    python scripts/train_point_charge_pinn.py [--epochs N] [--lr LR] [--model-out PATH]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.electrostatics_pinn import (  # noqa: E402
    ElectrostaticsPINN,
    boundary_loss,
    laplace_residual,
)

# Problem configuration
CHARGE_Q = 1.0
CHARGE_POS = (0.0, 0.0)
X_RANGE = (-1.0, 1.0)
Y_RANGE = (-1.0, 1.0)
NUM_COLLOCATION_POINTS = 10000
NUM_BOUNDARY_POINTS = 400

DEFAULT_MODEL_OUT = REPO_ROOT / "artifacts" / "models" / "point_charge_pinn.pth"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--collocation-points", type=int, default=NUM_COLLOCATION_POINTS)
    p.add_argument("--boundary-points", type=int, default=NUM_BOUNDARY_POINTS)
    p.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Collocation points
    collocation_coords = torch.rand(args.collocation_points, 2, device=device)
    collocation_coords[:, 0] = collocation_coords[:, 0] * (X_RANGE[1] - X_RANGE[0]) + X_RANGE[0]
    collocation_coords[:, 1] = collocation_coords[:, 1] * (Y_RANGE[1] - Y_RANGE[0]) + Y_RANGE[0]

    # Boundary points and values
    n_side = args.boundary_points // 4
    boundary_x = np.linspace(X_RANGE[0], X_RANGE[1], n_side)
    boundary_y = np.linspace(Y_RANGE[0], Y_RANGE[1], n_side)

    def as_t(a):
        return torch.tensor(a, dtype=torch.float32)

    top = as_t(np.stack([boundary_x, np.full_like(boundary_x, Y_RANGE[1])], axis=1))
    bottom = as_t(np.stack([boundary_x, np.full_like(boundary_x, Y_RANGE[0])], axis=1))
    left = as_t(np.stack([np.full_like(boundary_y, X_RANGE[0]), boundary_y], axis=1))
    right = as_t(np.stack([np.full_like(boundary_y, X_RANGE[1]), boundary_y], axis=1))
    boundary_coords = torch.cat([top, bottom, left, right], dim=0).to(device)

    # The PINN learns the correction to the analytical solution, so the BC is 0
    boundary_values = torch.zeros(boundary_coords.shape[0], 1, device=device)

    model = ElectrostaticsPINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        laplace_res = laplace_residual(model, collocation_coords)
        loss_physics = torch.mean(laplace_res**2)
        loss_b = boundary_loss(model, boundary_coords, boundary_values)
        total_loss = loss_physics + loss_b
        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 500 == 0:
            print(
                f"Epoch [{epoch + 1}/{args.epochs}], Total Loss: {total_loss.item():.4e}, "
                f"Laplace Loss: {loss_physics.item():.4e}, Boundary Loss: {loss_b.item():.4e}"
            )

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_out)
    print(f"Training complete and model saved to {model_out}")


if __name__ == "__main__":
    main()
