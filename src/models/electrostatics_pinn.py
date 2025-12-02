import torch
import torch.nn as nn

class ElectrostaticsPINN(nn.Module):
    """
    PINN for solving 2D electrostatic problems (Poisson's equation).
    """
    def __init__(self, num_layers=4, hidden_dim=64):
        super().__init__()
        
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def laplace_residual(network, coords):
    """
    Calculates the residual of Laplace's equation: ∇²V = 0
    """
    coords.requires_grad_(True)
    potential = network(coords)
    
    # First derivatives
    grad_V = torch.autograd.grad(
        outputs=potential.sum(),
        inputs=coords,
        create_graph=True,
        retain_graph=True
    )[0]
    
    Vx = grad_V[:, 0]
    Vy = grad_V[:, 1]
    
    # Second derivatives (Laplacian)
    Vxx = torch.autograd.grad(
        outputs=Vx.sum(),
        inputs=coords,
        create_graph=True,
        retain_graph=True
    )[0][:, 0]
    
    Vyy = torch.autograd.grad(
        outputs=Vy.sum(),
        inputs=coords,
        create_graph=True,
        retain_graph=True
    )[0][:, 1]
    
    laplacian_V = Vxx + Vyy
    
    return laplacian_V

def boundary_loss(network, boundary_coords, boundary_values):
    """
    Calculates the loss on the boundary.
    """
    predicted_potential = network(boundary_coords)
    loss = nn.functional.mse_loss(predicted_potential, boundary_values)
    return loss
