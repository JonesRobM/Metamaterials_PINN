import torch
import numpy as np
import matplotlib.pyplot as plt
from src.physics import MaxwellEquations, MetamaterialProperties, BoundaryConditions


def validate_maxwell_equations():
    """Validate Maxwell equations with plane wave solution."""
    print("Validating Maxwell equations...")
    
    # Setup
    omega = 2 * np.pi * 1e15  # 1 PHz
    maxwell = MaxwellEquations(omega)
    
    # Create plane wave in vacuum
    k = torch.tensor([1e7, 0, 0], dtype=torch.float32)  # Wavevector in x-direction
    E0 = torch.tensor([0, 1, 0], dtype=torch.complex64)  # E field in y-direction
    
    # Spatial coordinates
    x = torch.linspace(-1e-6, 1e-6, 50, requires_grad=True)
    coords = torch.stack([x, torch.zeros_like(x), torch.zeros_like(x)], dim=1)
    
    # Plane wave fields
    phase = torch.sum(k.unsqueeze(0) * coords, dim=1)
    exp_factor = torch.exp(1j * phase).unsqueeze(1)
    E_field = E0.unsqueeze(0) * exp_factor
    
    # H field from Maxwell's equations: H = (k × E)/(ωμ₀)
    k_expanded = k.unsqueeze(0).expand(coords.shape[0], -1)
    k_cross_E = torch.cross(k_expanded, E_field.real) + 1j * torch.cross(k_expanded, E_field.imag)
    H_field = k_cross_E / (omega * maxwell.mu0)
    
    # Vacuum permittivity tensor
    eps_tensor = torch.eye(3, dtype=torch.complex64).unsqueeze(0).expand(coords.shape[0], -1, -1)
    
    # Check Maxwell residuals
    total_residual = maxwell.total_residual(E_field, H_field, coords, eps_tensor)
    max_residual = torch.max(torch.abs(total_residual))
    
    print(f"Maximum Maxwell residual: {max_residual:.2e}")
    
    if max_residual < 1e-3:
        print("✓ Maxwell equations validation passed")
        return True
    else:
        print("✗ Maxwell equations validation failed")
        return False


def validate_metamaterial_properties():
    """Validate metamaterial dispersion relations."""
    print("\nValidating metamaterial properties...")
    
    # Metamaterial parameters (typical values for SPP)
    eps_par = -2.0 + 0.1j
    eps_perp = 4.0 + 0.05j
    metamaterial = MetamaterialProperties(eps_par, eps_perp, optical_axis='z')
    
    # Frequency range
    frequencies = np.linspace(0.5e15, 2e15, 100)  # 0.5-2 PHz
    
    propagation_lengths = []
    spp_supported = []
    
    for freq in frequencies:
        omega = 2 * np.pi * freq
        
        # Check if SPPs are supported
        supported = metamaterial.is_spp_supported(eps_dielectric=1.0)
        spp_supported.append(supported)
        
        if supported:
            # Calculate propagation length
            L_prop = metamaterial.propagation_length(omega)
            propagation_lengths.append(L_prop)
        else:
            propagation_lengths.append(0)
    
    # Validation checks
    num_supported = sum(spp_supported)
    avg_prop_length = np.mean([L for L in propagation_lengths if L > 0])
    
    print(f"SPP support frequency range: {num_supported}/{len(frequencies)} points")
    print(f"Average propagation length: {avg_prop_length:.2e} m")
    
    # Basic validation criteria
    validation_passed = True
    
    if num_supported == 0:
        print("✗ No SPP modes found - check metamaterial parameters")
        validation_passed = False
    
    if avg_prop_length < 1e-9 or avg_prop_length > 1e-3:
        print("✗ Unrealistic propagation lengths")
        validation_passed = False
    
    if validation_passed:
        print("✓ Metamaterial properties validation passed")
        return True
    else:
        print("✗ Metamaterial properties validation failed")
        return False


def validate_boundary_conditions():
    """Validate boundary condition implementations."""
    print("\nValidating boundary conditions...")
    
    bc = BoundaryConditions(interface_normal=(0, 0, 1))
    
    # Test case: identical fields on both sides (should satisfy all BCs)
    batch_size = 10
    E_field = torch.complex(torch.randn(batch_size, 3), torch.randn(batch_size, 3))
    H_field = torch.complex(torch.randn(batch_size, 3), torch.randn(batch_size, 3))
    
    # Permittivity tensors
    eps1 = torch.eye(3, dtype=torch.complex64).unsqueeze(0).expand(batch_size, -1, -1)
    eps2 = 2.0 * torch.eye(3, dtype=torch.complex64).unsqueeze(0).expand(batch_size, -1, -1)
    
    # Test tangential field continuity with identical fields
    tang_E_res = bc.tangential_E_continuity(E_field, E_field)
    tang_H_res = bc.tangential_H_continuity(H_field, H_field)
    
    # Test normal field continuity
    norm_D_res = bc.normal_D_continuity(E_field, E_field, eps1, eps1)  # Same permittivity
    norm_B_res = bc.normal_B_continuity(H_field, H_field)
    
    # Check residuals
    max_tang_E = torch.max(torch.abs(tang_E_res))
    max_tang_H = torch.max(torch.abs(tang_H_res))
    max_norm_D = torch.max(torch.abs(norm_D_res))
    max_norm_B = torch.max(torch.abs(norm_B_res))
    
    print(f"Max tangential E residual: {max_tang_E:.2e}")
    print(f"Max tangential H residual: {max_tang_H:.2e}")
    print(f"Max normal D residual: {max_norm_D:.2e}")
    print(f"Max normal B residual: {max_norm_B:.2e}")
    
    tolerance = 1e-6
    validation_passed = (max_tang_E < tolerance and max_tang_H < tolerance and 
                        max_norm_D < tolerance and max_norm_B < tolerance)
    
    if validation_passed:
        print("✓ Boundary conditions validation passed")
        return True
    else:
        print("✗ Boundary conditions validation failed")
        return False


def demonstrate_spp_physics():
    """Demonstrate complete SPP physics calculation."""
    print("\nDemonstrating SPP physics...")
    
    # Setup metamaterial
    eps_par = -2.0 + 0.1j
    eps_perp = 4.0 + 0.05j
    metamaterial = MetamaterialProperties(eps_par, eps_perp, optical_axis='z')
    
    # Frequency
    omega = 2 * np.pi * 1e15  # 1 PHz
    
    # Check SPP existence
    if not metamaterial.is_spp_supported():
        print("✗ SPPs not supported with these parameters")
        return False
    
    # Calculate SPP properties
    k_real, k_imag = metamaterial.spp_dispersion_relation(omega)
    L_prop = metamaterial.propagation_length(omega)
    depth_meta = metamaterial.penetration_depth_metamaterial(omega)
    depth_diel = metamaterial.penetration_depth_dielectric(omega)
    enhancement = metamaterial.field_enhancement_factor(omega)
    
    print(f"SPP wavevector: k = {k_real:.2e} + {k_imag:.2e}i m⁻¹")
    print(f"Propagation length: {L_prop:.2e} m")
    print(f"Penetration depth (metamaterial): {depth_meta:.2e} m")
    print(f"Penetration depth (dielectric): {depth_diel:.2e} m")
    print(f"Field enhancement factor: {enhancement:.1f}")
    
    # Validate physical reasonableness
    validation_passed = True
    
    if k_real <= 0:
        print("✗ Invalid SPP wavevector")
        validation_passed = False
    
    if L_prop < 1e-9 or L_prop > 1e-3:
        print("✗ Unrealistic propagation length")
        validation_passed = False
    
    if depth_meta < 1e-9 or depth_meta > 1e-6:
        print("✗ Unrealistic metamaterial penetration depth")
        validation_passed = False
    
    if validation_passed:
        print("✓ SPP physics demonstration passed")
        return True
    else:
        print("✗ SPP physics demonstration failed")
        return False


def main():
    """Run all physics validations."""
    print("SPP Metamaterial PINN Physics Validation")
    print("=" * 50)
    
    # Check PyTorch setup
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()
    
    # Run validation tests
    validations = [
        validate_maxwell_equations,
        validate_metamaterial_properties,
        validate_boundary_conditions,
        demonstrate_spp_physics
    ]
    
    results = []
    for validation in validations:
        try:
            result = validation()
            results.append(result)
        except Exception as e:
            print(f"✗ Validation failed with error: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All physics validations passed successfully!")
        print("The implementation is ready for PINN training.")
        return 0
    else:
        print("✗ Some validations failed. Review implementation before proceeding.")
        return 1


if __name__ == "__main__":
    exit(main())
