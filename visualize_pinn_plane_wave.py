import torch
import numpy as np
import matplotlib.pyplot as plt

from src.models.pinn_network import ComplexPINN
from src.physics.maxwell_equations import MaxwellEquations
from src.data.domain_sampler import UniformSampler
from src.utils.plotting import EMFieldPlotter, PlotConfig

def visualize_pinn_plane_wave_comparison():
    """
    Visualizes the analytical plane wave E-field and the PINN's initial prediction.
    """
    print("Visualizing PINN plane wave comparison...")

    # Setup parameters (same as in test_pinn_plane_wave_solution)
    omega = 2 * np.pi * 1e15  # 1 PHz
    k_vec = torch.tensor([1e7, 0, 0], dtype=torch.float32)  # Wavevector in x-direction
    E0_amplitude = 1.0
    E0_polarization = torch.tensor([0, E0_amplitude, 0], dtype=torch.complex64)  # E field in y-direction

    # Initialize PINN (untrained)
    pinn = ComplexPINN(
        spatial_dim=3,
        field_components=6,
        hidden_dims=[64, 64],
        complex_valued=True,
        frequency=omega,
        use_fourier=True,
        fourier_modes=64
    )

    # Generate collocation points for plotting (wider range for better visualization)
    sampler = UniformSampler(
        domain_bounds=[
            (-2e-6, 2e-6),
            (-1e-7, 1e-7),
            (-1e-7, 1e-7)
        ]
    )
    coords = sampler.sample_points(n_points=5000)['points']

    # Calculate analytical plane wave fields at collocation points
    phase = torch.sum(k_vec.unsqueeze(0) * coords, dim=1)
    exp_factor = torch.exp(1j * phase).unsqueeze(1)
    E_analytical = E0_polarization.unsqueeze(0) * exp_factor # [N, 3] complex

    # Evaluate PINN output (untrained)
    pinn_output = pinn(coords) # [N, 6, 2]

    # Plotting setup
    plotter = EMFieldPlotter(config=PlotConfig(figsize=(10, 6)))

    # Plot Analytical E-field (Ey component)
    fig_analytical = plotter.plot_field_2d(
        coords=coords,
        fields=torch.stack([E_analytical.real, E_analytical.imag], dim=-1),
        field_component='Ey',
        plane='xy',
        title='Analytical Plane Wave E_y Field'
    )
    plotter.save_figure(fig_analytical, 'analytical_plane_wave_Ey', directory='figures')
    plt.close(fig_analytical)

    # Plot PINN Predicted E-field (Ey component)
    fig_pinn = plotter.plot_field_2d(
        coords=coords,
        fields=pinn_output, # PINN output is already in [N, 6, 2] format
        field_component='Ey',
        plane='xy',
        title='Untrained PINN Predicted E_y Field'
    )
    plotter.save_figure(fig_pinn, 'untrained_pinn_plane_wave_Ey', directory='figures')
    plt.close(fig_pinn)
    
    print("Visualization complete. Check the 'figures' directory.")

if __name__ == "__main__":
    visualize_pinn_plane_wave_comparison()
