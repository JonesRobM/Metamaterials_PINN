import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.electrostatics_pinn import ElectrostaticsPINN

# --- Configuration ---
CHARGE_Q = 1.0
CHARGE_POS = (0.0, 0.0)
X_RANGE = (-1.0, 1.0)
Y_RANGE = (-1.0, 1.0)
GRID_SIZE = 20
EPS0 = 8.854e-12

def analytical_potential(x, y, q, q_pos):
    """Analytical solution for the potential of a point charge."""
    k = 1 / (4 * np.pi * EPS0)
    r = np.sqrt((x - q_pos[0])**2 + (y - q_pos[1])**2)
    return k * q / (r + 1e-9)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Load the trained model ---
    model = ElectrostaticsPINN().to(device)
    model.load_state_dict(torch.load("point_charge_pinn.pth"))
    model.eval()
    
    # --- Create a grid of points for visualization ---
    x = np.linspace(X_RANGE[0], X_RANGE[1], GRID_SIZE)
    y = np.linspace(Y_RANGE[0], Y_RANGE[1], GRID_SIZE)
    X, Y = np.meshgrid(x, y)
    
    coords = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32, device=device)
    coords.requires_grad_(True)
    
    # --- Predict potential and calculate electric field ---
    # The PINN learns the correction to the analytical solution
    pinn_potential = model(coords)
    
    # The total potential is the sum of the analytical solution and the PINN's correction
    analytical_V = analytical_potential(coords[:, 0].detach().cpu().numpy(), coords[:, 1].detach().cpu().numpy(), CHARGE_Q, CHARGE_POS)
    analytical_V = torch.tensor(analytical_V, dtype=torch.float32, device=device).unsqueeze(1)
    
    total_potential = analytical_V + pinn_potential
    
    # Calculate the electric field E = -∇V
    grad_V = torch.autograd.grad(
        outputs=total_potential.sum(),
        inputs=coords,
        create_graph=True,
        retain_graph=True
    )[0]
    
    Ex = -grad_V[:, 0].detach().cpu().numpy().reshape(X.shape)
    Ey = -grad_V[:, 1].detach().cpu().numpy().reshape(Y.shape)

    # --- Plotting ---
    E_mag = np.sqrt(Ex**2 + Ey**2)
    
    plt.figure(figsize=(10, 8))
    
    # Heatmap of the electric field magnitude
    c = plt.pcolormesh(X, Y, E_mag, cmap='viridis', shading='gouraud')
    plt.colorbar(c, label='Electric Field Magnitude |E| (V/m)')
    
    # Overlay quiver plot
    plt.quiver(X, Y, Ex, Ey, color='white', scale=5e10, alpha=0.8)
    
    plt.plot(CHARGE_POS[0], CHARGE_POS[1], 'ro', markersize=8, label='Point Charge')
    plt.title("Electric Field of a Point Charge (PINN-computed Heatmap)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.xlim(X_RANGE)
    plt.ylim(Y_RANGE)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.grid(True)
    plt.savefig("pinn_point_charge_heatmap.png")
    plt.show()

if __name__ == "__main__":
    main()
