# Network Output Format Design

**Date:** 2025-12-06
**Status:** Implemented
**Context:** Metamaterials PINN MVP Development

## Problem

The `MaxwellCurlLoss` test was failing with:
```
RuntimeError: Expected both inputs to be Half, Float or Double tensors but got ComplexFloat and ComplexFloat
```

Root cause: The test's `AnalyticalPlaneWaveNetwork` output complex tensors directly `[batch_size, 6]` with dtype `complex64`, but `MaxwellCurlLoss` expected real tensors `[batch_size, 6, 2]` where the last dimension holds `[real, imag]` components.

## Architecture Decision

**All neural networks in the PINN system output real tensors with shape `[batch_size, 6, 2]`**

Where:
- **Dimension 0:** Batch size
- **Dimension 1:** 6 field components (Ex, Ey, Ez, Hx, Hy, Hz)
- **Dimension 2:** 2 values (real part, imaginary part)

## Rationale

This format:
- ✓ Works naturally with PyTorch autograd for computing spatial gradients
- ✓ Avoids complex autograd complications and edge cases
- ✓ Makes the real/imag split explicit for debugging
- ✓ Easier to visualize individual components during development
- ✓ Compatible with loss functions that need to compute curls via automatic differentiation

## Implementation

### Test Fix

Modified `tests/test_loss_functions.py`, the `AnalyticalPlaneWaveNetwork.forward()` method:

```python
def forward(self, coords_input):
    phase = torch.einsum('j,ij->i', self.k_vec, coords_input)
    exp_factor = torch.exp(1j * phase).unsqueeze(1)

    E_field = (self.E_pol.unsqueeze(0) * exp_factor)  # [batch, 3] complex
    H_field = (self.H_pol.unsqueeze(0) * exp_factor)  # [batch, 3] complex

    # Convert to [batch, 6, 2] format: real/imag split
    fields_complex = torch.cat([E_field, H_field], dim=1)  # [batch, 6] complex
    fields_real_imag = torch.stack([fields_complex.real, fields_complex.imag], dim=-1)

    return fields_real_imag  # Returns [batch, 6, 2] float32
```

**Key pattern:**
1. Compute electromagnetic fields using complex arithmetic (cleaner math)
2. Convert to `[batch, 6, 2]` format before returning using `.real` and `.imag` properties
3. Stack along the last dimension with `torch.stack(..., dim=-1)`

### Loss Function Compatibility

The `MaxwellCurlLoss.compute()` method correctly handles this format:
- Splits fields: `E = fields[:, :3, :]` → `[batch, 3, 2]`
- Splits fields: `H = fields[:, 3:, :]` → `[batch, 3, 2]`
- Converts to complex: `torch.complex(E[..., 0], E[..., 1])` → `[batch, 3]` complex

## Verification

All 13 tests pass after the fix:
- ✓ `test_loss_functions.py::TestMaxwellCurlLoss::test_compute_with_known_solution`
- ✓ All other existing tests remain passing

## Future Considerations

When implementing new neural network architectures:
1. Internal layers can use any representation
2. Final output layer must convert to `[batch, 6, 2]` format
3. Use the pattern: compute in complex, convert to real/imag at output
4. Document the output format in network docstrings
