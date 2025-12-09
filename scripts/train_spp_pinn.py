import yaml
import torch
import numpy as np
from src.models.pinn_network import SPPNetwork
from src.models.loss_functions import MaxwellCurlLoss, TangentialContinuityLoss, SPPBoundaryLoss, EM_CompositeLoss

def load_config(config_path='config/spp_config.yaml'):
    """Loads the SPP configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # --- Load Configuration and Setup ---
    config = load_config()
    config['frequency'] = float(config['frequency']) # Ensure frequency is a float
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Generation ---
    x_range = config['x_range']
    y_range = config['y_range']
    z_range = config['z_range']
    interface_z = config['interface_z']
    
    num_collocation = config['training']['num_collocation_points']
    num_interface = config['training']['num_interface_points']
    
    # Collocation points
    collocation_coords = torch.rand(num_collocation, 3, device=device)
    collocation_coords[:, 0] = collocation_coords[:, 0] * (x_range[1] - x_range[0]) + x_range[0]
    collocation_coords[:, 1] = collocation_coords[:, 1] * (y_range[1] - y_range[0]) + y_range[0]
    collocation_coords[:, 2] = collocation_coords[:, 2] * (z_range[1] - z_range[0]) + z_range[0]
    
    # Interface points
    interface_coords = torch.rand(num_interface, 3, device=device)
    interface_coords[:, 0] = interface_coords[:, 0] * (x_range[1] - x_range[0]) + x_range[0]
    interface_coords[:, 1] = interface_coords[:, 1] * (y_range[1] - y_range[0]) + y_range[0]
    interface_coords[:, 2] = interface_z
    
    # Normal vectors for the interface
    normal_vectors = torch.zeros_like(interface_coords)
    normal_vectors[:, 2] = 1.0
    
    # --- Model Instantiation ---
    model_params = config['model']
    spp_network = SPPNetwork(
        spatial_dim=model_params['spatial_dim'],
        hidden_dims=model_params['hidden_dims'],
        use_fourier=model_params['use_fourier'],
        fourier_modes=model_params['fourier_modes'],
        activation_type=model_params['activation_type'],
        interface_position=interface_z,
        metal_permittivity=complex(config['metal_permittivity'][0], config['metal_permittivity'][1]),
        dielectric_permittivity=config['dielectric_permittivity'],
        frequency=float(config['frequency'])
    ).to(device)
    
    # --- Loss Function Setup ---
    loss_fns = {
        'maxwell_curl': MaxwellCurlLoss(frequency=float(config['frequency'])),
        'tangential_continuity': TangentialContinuityLoss(),
        'spp_boundary': SPPBoundaryLoss(
            spp_wavevector=spp_network.k_spp.real.item(),
            decay_length=300e-9 # Approximate decay length for Ag/Air at 633nm
        )
    }
    
    composite_loss = EM_CompositeLoss(
        losses=loss_fns,
        adaptive_weights=True
    )
    
    # --- Optimizer Setup ---
    optimizer = torch.optim.Adam(spp_network.parameters(), lr=config['training']['learning_rate'])
    
    # --- Training Loop ---
    for epoch in range(config['training']['num_epochs']):
        optimizer.zero_grad()
        
        # The SPPNetwork's forward pass expects complex-valued outputs if complex_valued=True
        # However, the loss functions handle the complex numbers internally.
        
        total_loss, loss_dict = composite_loss.compute(
            network=spp_network,
            coords=collocation_coords,
            interface_coords=interface_coords,
            normal_vectors=normal_vectors
        )
        
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 500 == 0:
            loss_str = ", ".join([f"{name}: {val.item():.4e}" for name, val in loss_dict.items()])
            print(f"Epoch [{epoch+1}/{config['training']['num_epochs']}], Total Loss: {total_loss.item():.4e}, " + loss_str)

    # --- Save the model ---
    torch.save(spp_network.state_dict(), "spp_pinn.pth")
    print("SPP PINN training complete and model saved.")

if __name__ == "__main__":
    main()
