"""
Data handling module for SPP metamaterial PINN implementation.

This module provides domain sampling, collocation point generation, and
data management utilities for training Physics-Informed Neural Networks
on electromagnetic problems.
"""

from .domain_sampler import (
    DomainSampler,
    UniformSampler,
    AdaptiveSampler,
    InterfaceSampler,
    SPPDomainSampler,
    StratifiedSampler
)

from .collection_points import (
    CollocationPointGenerator,
    MaxwellPointGenerator,
    BoundaryPointGenerator,
    SPPCollocationGenerator,
    AdaptiveCollocationManager
)

__all__ = [
    'DomainSampler',
    'UniformSampler', 
    'AdaptiveSampler',
    'InterfaceSampler',
    'SPPDomainSampler',
    'StratifiedSampler',
    'CollocationPointGenerator',
    'MaxwellPointGenerator',
    'BoundaryPointGenerator', 
    'SPPCollocationGenerator',
    'AdaptiveCollocationManager'
]