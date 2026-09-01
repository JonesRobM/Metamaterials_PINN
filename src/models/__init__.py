"""
Electromagnetic Physics-Informed Neural Networks (PINNs) Models Module

This module provides neural network architectures and loss functions
specifically designed for solving Maxwell's equations in metamaterial
systems, particularly for Surface Plasmon Polariton (SPP) modeling.
"""

from .field_format import join_complex, split_complex, to_complex
from .loss_functions import (
    BaseLoss,
    EM_CompositeLoss,
    InterfaceBoundaryLoss,
    MaxwellCurlLoss,
    MaxwellDivergenceLoss,
    PowerFlowLoss,
    RadiationLoss,
    SPPBoundaryLoss,
    TangentialContinuityLoss,
    WaveguideLoss,
)
from .pinn_network import (
    ComplexPINN,
    ElectromagneticPINN,
    FourierEMFeatures,
    MetamaterialDeepONet,
    MultiFrequencyPINN,
    NondimensionalPINN,
    SPPNetwork,
)

__all__ = [
    "ElectromagneticPINN",
    "ComplexPINN",
    "NondimensionalPINN",
    "SPPNetwork",
    "MetamaterialDeepONet",
    "MultiFrequencyPINN",
    "FourierEMFeatures",
    "split_complex",
    "to_complex",
    "join_complex",
    "BaseLoss",
    "MaxwellCurlLoss",
    "MaxwellDivergenceLoss",
    "InterfaceBoundaryLoss",
    "SPPBoundaryLoss",
    "TangentialContinuityLoss",
    "PowerFlowLoss",
    "EM_CompositeLoss",
    "WaveguideLoss",
    "RadiationLoss",
]

__version__ = "0.1.0"
