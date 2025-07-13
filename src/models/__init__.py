"""
Electromagnetic Physics-Informed Neural Networks (PINNs) Models Module

This module provides neural network architectures and loss functions
specifically designed for solving Maxwell's equations in metamaterial
systems, particularly for Surface Plasmon Polariton (SPP) modeling.
"""

from .pinn_network import (
    ElectromagneticPINN,
    ComplexPINN,
    SPPNetwork,
    MetamaterialDeepONet,
    MultiFrequencyPINN
)
from .loss_functions import (
    MaxwellCurlLoss,
    MaxwellDivergenceLoss,
    MetamaterialConstitutiveLoss,
    InterfaceBoundaryLoss,
    SPPBoundaryLoss,
    TangentialContinuityLoss,
    PowerFlowLoss,
    EM_CompositeLoss,
    WaveguideLoss,
    RadiationLoss
)

__all__ = [
    'ElectromagneticPINN',
    'ComplexPINN', 
    'SPPNetwork',
    'MetamaterialDeepONet',
    'MultiFrequencyPINN',
    'MaxwellCurlLoss',
    'MaxwellDivergenceLoss',
    'MetamaterialConstitutiveLoss',
    'InterfaceBoundaryLoss',
    'SPPBoundaryLoss',
    'TangentialContinuityLoss',
    'PowerFlowLoss',
    'EM_CompositeLoss',
    'WaveguideLoss',
    'RadiationLoss'
]

__version__ = '0.1.0'