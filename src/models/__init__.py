"""
Physics-Informed Neural Networks (PINNs) Models Module

This module provides neural network architectures and loss functions
specifically designed for solving partial differential equations using
physics-informed neural networks.
"""

from .pinn_network import PINNNetwork, DeepONet
from .loss_functions import (
    PDELoss,
    BoundaryLoss,
    InitialConditionLoss,
    DataLoss,
    CompositeLoss,
    GradientPenalty,
    CausalityLoss
)

__all__ = [
    'PINNNetwork',
    'DeepONet', 
    'PDELoss',
    'BoundaryLoss',
    'InitialConditionLoss',
    'DataLoss',
    'CompositeLoss',
    'GradientPenalty',
    'CausalityLoss'
]

__version__ = '0.1.0'