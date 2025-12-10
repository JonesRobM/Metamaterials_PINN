import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.pinn_network import ComplexPINN
from src.data.domain_sampler import UniformSampler
from src.models.loss_functions import MaxwellCurlLoss
from src.utils.plotting import EMFieldPlotter, TrainingPlotter, PlotConfig

def train_plane_wave_pinn_by_hand():
    """
    Trains a ComplexPINN for plane waves with a manual training loop,
    then plots the results.
    """
    print("Starting manual training of PINN for plane waves...")

    # 1. Setup Parameters
    omega = 2 * np.pi * 1e15  # 1 PHz
    learning_rate = 1e-3
    num_epochs = 500
    points_per_epoch = 1000

    # 2. Initialization
    pinn = ComplexPINN(
        spatial_dim=3,
        field_components=6,
        hidden_dims=[128, 128, 128],
        complex_valued=True,
        frequency=omega,
        use_fourier=True,
        fourier_modes=128
    )

    sampler = UniformSampler(
        domain_bounds=[
            (-2e-6, 2e-6),
            (-2e-6, 2e-6),
            (-2e-6, 2e-6)
        ]
    )
    
    loss_fn = MaxwellCurlLoss(frequency=omega, weight=1.0)
    optimizer = optim.Adam(pinn.parameters(), lr=learning_rate)

    loss_history = []

    # 3. Training Loop
    print(f"Training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        # Sample new collocation points for each epoch
        coords_dict = sampler.sample_points(n_points=points_per_epoch)
        coords = coords_dict['points']
        coords.requires_grad_(True)

        # Zero the gradients
        optimizer.zero_grad()

        # Calculate loss
        loss = loss_fn(network=pinn, coords=coords)

        # Backpropagate
        loss.backward()

        # Step the optimizer
        optimizer.step()

        # Record and print loss
        loss_history.append(loss.item())
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4e}")

    print("Training finished.")

    # 4. Save the Trained Model
    model_save_path = 'plane_wave_pinn.pth'
    torch.save(pinn.state_dict(), model_save_path)
    print(f"Trained model saved to {model_save_path}")

    # 5. Plotting
    print("Generating plots...")
    
    # Plot Training Curve
    training_plotter = TrainingPlotter()
    fig_loss = training_plotter.plot_training_history({'Maxwell Curl Loss': loss_history})
    training_plotter.save_figure(fig_loss, 'plane_wave_training_curve', directory='figures')
    plt.close(fig_loss)
    
    # Plot Electric Field Distribution
    pinn.eval() # Set model to evaluation mode
    
    # Generate a grid for smooth plotting
    n_points_x = 200
    n_points_y = 200
    x = torch.linspace(-2e-6, 2e-6, n_points_x)
    y = torch.linspace(-2e-6, 2e-6, n_points_y)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    z = torch.zeros_like(grid_x)
    
    plot_coords = torch.stack([grid_x.flatten(), grid_y.flatten(), z.flatten()], dim=1)

    # Get PINN prediction
    with torch.no_grad():
        predicted_fields = pinn(plot_coords)

    # Plot the Ey component
    field_plotter = EMFieldPlotter(config=PlotConfig(figsize=(10, 6)))
    fig_field = field_plotter.plot_field_2d(
        coords=plot_coords,
        fields=predicted_fields,
        field_component='Ey',
        plane='xy', # We are plotting in the z=0 plane
        title='Trained PINN - Predicted E_y Field'
    )
    field_plotter.save_figure(fig_field, 'trained_pinn_plane_wave_Ey', directory='figures')
    plt.close(fig_field)
    
    print("Plotting complete. Check the 'figures' directory.")

if __name__ == "__main__":
    train_plane_wave_pinn_by_hand()
