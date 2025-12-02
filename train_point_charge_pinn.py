import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.electrostatics_pinn import ElectrostaticsPINN, laplace_residual, boundary_loss

# --- Configuration ---
CHARGE_Q = 1.0
CHARGE_POS = (0.0, 0.0)
X_RANGE = (-1.0, 1.0)
Y_RANGE = (-1.0, 1.0)
GRID_SIZE = 100
NUM_COLLOCATION_POINTS = 10000
NUM_BOUNDARY_POINTS = 400
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5000
EPS0 = 8.854e-12

def analytical_potential(x, y, q, q_pos):
    """Analytical solution for the potential of a point charge."""
    k = 1 / (4 * np.pi * EPS0)
    r = np.sqrt((x - q_pos[0])**2 + (y - q_pos[1])**2)
    # Add a small epsilon to avoid division by zero if r is very small
    return k * q / (r + 1e-9)




def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Data Generation ---
    # Collocation points
    collocation_coords = torch.rand(NUM_COLLOCATION_POINTS, 2, device=device)
    collocation_coords[:, 0] = collocation_coords[:, 0] * (X_RANGE[1] - X_RANGE[0]) + X_RANGE[0]
    collocation_coords[:, 1] = collocation_coords[:, 1] * (Y_RANGE[1] - Y_RANGE[0]) + Y_RANGE[0]
    
    # Boundary points and values
    boundary_x = np.linspace(X_RANGE[0], X_RANGE[1], NUM_BOUNDARY_POINTS // 4)
    boundary_y = np.linspace(Y_RANGE[0], Y_RANGE[1], NUM_BOUNDARY_POINTS // 4)
    
    top_coords = torch.tensor(np.stack([boundary_x, np.full_like(boundary_x, Y_RANGE[1])], axis=1), dtype=torch.float32)
    bottom_coords = torch.tensor(np.stack([boundary_x, np.full_like(boundary_x, Y_RANGE[0])], axis=1), dtype=torch.float32)
    left_coords = torch.tensor(np.stack([np.full_like(boundary_y, X_RANGE[0]), boundary_y], axis=1), dtype=torch.float32)
    right_coords = torch.tensor(np.stack([np.full_like(boundary_y, X_RANGE[1]), boundary_y], axis=1), dtype=torch.float32)
    
    boundary_coords = torch.cat([top_coords, bottom_coords, left_coords, right_coords], dim=0).to(device)
    
    # For the new formulation, the PINN learns the correction, so the boundary condition is 0
    boundary_values = torch.zeros(boundary_coords.shape[0], 1, device=device)
    
    # --- Model and Training Setup ---
    model = ElectrostaticsPINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- Training Loop ---
    for epoch in range(NUM_EPOCHS):
        optimizer.zero_grad()
        
        # Physics loss (Laplace)
        laplace_res = laplace_residual(model, collocation_coords)
        loss_physics = torch.mean(laplace_res**2)
        
        # Boundary loss
        loss_b = boundary_loss(model, boundary_coords, boundary_values)
        
        total_loss = loss_physics + loss_b # Equal weighting
        
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 500 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Total Loss: {total_loss.item():.4e}, "
                  f"Laplace Loss: {loss_physics.item():.4e}, Boundary Loss: {loss_b.item():.4e}")

    # --- Save the model ---
    torch.save(model.state_dict(), "point_charge_pinn.pth")
    print("Training complete and model saved.")

if __name__ == "__main__":
    main()
