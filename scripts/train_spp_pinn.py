"""Train the SPPNetwork on a metal/dielectric interface using the composite EM loss.

Usage:
    python scripts/train_spp_pinn.py [--config PATH] [--epochs N] [--lr LR] [--model-out PATH]

Values on the command line override those in the YAML config.
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.loss_functions import (  # noqa: E402
    EM_CompositeLoss,
    MaxwellCurlLoss,
    SPPBoundaryLoss,
    TangentialContinuityLoss,
)
from src.models.pinn_network import SPPNetwork  # noqa: E402
from src.physics.metamaterial import MetamaterialProperties  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "config" / "spp_config.yaml"
DEFAULT_MODEL_OUT = REPO_ROOT / "artifacts" / "models" / "spp_pinn.pth"


def load_config(config_path=DEFAULT_CONFIG):
    """Load the SPP configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--epochs", type=int, default=None, help="override training.num_epochs")
    p.add_argument("--lr", type=float, default=None, help="override training.learning_rate")
    p.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)
    config["frequency"] = float(config["frequency"])
    num_epochs = args.epochs if args.epochs is not None else config["training"]["num_epochs"]
    lr = args.lr if args.lr is not None else config["training"]["learning_rate"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    x_range, y_range, z_range = config["x_range"], config["y_range"], config["z_range"]
    interface_z = config["interface_z"]
    num_collocation = config["training"]["num_collocation_points"]
    num_interface = config["training"]["num_interface_points"]

    def sample_collocation(n: int) -> torch.Tensor:
        c = torch.rand(n, 3, device=device)
        c[:, 0] = c[:, 0] * (x_range[1] - x_range[0]) + x_range[0]
        c[:, 1] = c[:, 1] * (y_range[1] - y_range[0]) + y_range[0]
        c[:, 2] = c[:, 2] * (z_range[1] - z_range[0]) + z_range[0]
        return c

    def sample_interface(n: int) -> torch.Tensor:
        c = torch.rand(n, 3, device=device)
        c[:, 0] = c[:, 0] * (x_range[1] - x_range[0]) + x_range[0]
        c[:, 1] = c[:, 1] * (y_range[1] - y_range[0]) + y_range[0]
        c[:, 2] = interface_z
        return c

    interface_coords = sample_interface(num_interface)

    normal_vectors = torch.zeros_like(interface_coords)
    normal_vectors[:, 2] = 1.0

    model_params = config["model"]
    spp_network = SPPNetwork(
        spatial_dim=model_params["spatial_dim"],
        hidden_dims=model_params["hidden_dims"],
        use_fourier=model_params["use_fourier"],
        fourier_modes=model_params["fourier_modes"],
        activation_type=model_params["activation_type"],
        interface_position=interface_z,
        metal_permittivity=complex(config["metal_permittivity"][0], config["metal_permittivity"][1]),
        dielectric_permittivity=config["dielectric_permittivity"],
        frequency=config["frequency"],
    ).to(device)

    # Physical scales derived from the material system (not hardcoded)
    eps_m = complex(config["metal_permittivity"][0], config["metal_permittivity"][1])
    material = MetamaterialProperties(eps_m, eps_m, "z", omega=config["frequency"])
    delta_d = material.penetration_depth_dielectric(
        eps_dielectric=config["dielectric_permittivity"]
    )
    spp_wavelength = 2 * torch.pi / material.spp_wavevector(
        eps_dielectric=config["dielectric_permittivity"]
    ).real

    loss_fns = {
        "maxwell_curl": MaxwellCurlLoss(frequency=config["frequency"]),
        # Offset must sit well inside the decay length on both sides
        "tangential_continuity": TangentialContinuityLoss(offset=min(delta_d, spp_wavelength) / 50),
        "spp_boundary": SPPBoundaryLoss(
            spp_wavevector=material.spp_wavevector(
                eps_dielectric=config["dielectric_permittivity"]
            ).real,
            decay_length=delta_d,  # dielectric-side |E| decay (metal side is far shorter)
        ),
    }
    composite_loss = EM_CompositeLoss(losses=loss_fns, adaptive_weights=True)
    optimizer = torch.optim.Adam(spp_network.parameters(), lr=lr)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # Fresh collocation points every epoch (a fixed set lets the network overfit it)
        collocation_coords = sample_collocation(num_collocation)
        total_loss, loss_dict = composite_loss.compute(
            network=spp_network,
            coords=collocation_coords,
            interface_coords=interface_coords,
            normal_vectors=normal_vectors,
        )
        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 500 == 0:
            loss_str = ", ".join(f"{name}: {val.item():.4e}" for name, val in loss_dict.items())
            print(f"Epoch [{epoch + 1}/{num_epochs}], Total Loss: {total_loss.item():.4e}, {loss_str}")

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(spp_network.state_dict(), model_out)
    print(f"SPP PINN training complete and model saved to {model_out}")


if __name__ == "__main__":
    main()
