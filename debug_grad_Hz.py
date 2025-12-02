import torch
import numpy as np

# Sample data
x_val = torch.tensor([-0.0573, 0.7525, 1.2345, -1.9606, 1.1463], requires_grad=True, dtype=torch.float32)
k_val = 1.0

# Calculate cos(k*x)
output_cos = torch.cos(k_val * x_val)

# Compute gradient of cos(k*x) w.r.t x
grad_cos = torch.autograd.grad(outputs=output_cos, inputs=x_val, 
                               grad_outputs=torch.ones_like(output_cos),
                               create_graph=True, retain_graph=True)[0]
print(f"grad_cos (d(cos(kx))/dx): {grad_cos}")
# Expected: -k*sin(k*x) = -1*sin(1*x)

# Calculate sin(k*x)
output_sin = torch.sin(k_val * x_val)

# Compute gradient of sin(k*x) w.r.t x
grad_sin = torch.autograd.grad(outputs=output_sin, inputs=x_val, 
                               grad_outputs=torch.ones_like(output_sin),
                               create_graph=True, retain_graph=True)[0]
print(f"grad_sin (d(sin(kx))/dx): {grad_sin}")
# Expected: k*cos(k*x) = 1*cos(1*x)

print("\n--- Expected values for x = -0.0573 ---")
x_single = torch.tensor([-0.0573], requires_grad=True, dtype=torch.float32)
expected_grad_cos = -k_val * torch.sin(k_val * x_single)
expected_grad_sin = k_val * torch.cos(k_val * x_single)
print(f"Expected d(cos(kx))/dx: {expected_grad_cos.item()}")
print(f"Expected d(sin(kx))/dx: {expected_grad_sin.item()}")
