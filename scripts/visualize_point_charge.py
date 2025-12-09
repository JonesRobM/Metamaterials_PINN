import numpy as np
import matplotlib.pyplot as plt

def plot_point_charge_field(charge_q=1.0, charge_pos=(0, 0), grid_size=20, x_range=(-1, 1), y_range=(-1, 1)):
    """
    Calculates and plots the electric field of a point charge.
    """
    eps0 = 8.854e-12  # Permittivity of free space
    k = 1 / (4 * np.pi * eps0)

    x = np.linspace(x_range[0], x_range[1], grid_size)
    y = np.linspace(y_range[0], y_range[1], grid_size)
    X, Y = np.meshgrid(x, y)

    Ex = np.zeros_like(X)
    Ey = np.zeros_like(Y)

    for i in range(grid_size):
        for j in range(grid_size):
            px, py = X[i, j], Y[i, j]
            rx = px - charge_pos[0]
            ry = py - charge_pos[1]
            r_squared = rx**2 + ry**2
            
            # Avoid division by zero at the charge's location
            if r_squared < 1e-6:
                continue

            r = np.sqrt(r_squared)
            Ex[i, j] = k * charge_q * rx / (r**3)
            Ey[i, j] = k * charge_q * ry / (r**3)

    # Plotting
    plt.figure(figsize=(8, 8))
    plt.quiver(X, Y, Ex, Ey, scale=5e10, color='b')
    plt.plot(charge_pos[0], charge_pos[1], 'ro', markersize=8)
    plt.title(f"Electric Field of a Point Charge (q={charge_q} C)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.savefig("point_charge_electric_field.png")
    plt.show()

if __name__ == "__main__":
    plot_point_charge_field()
