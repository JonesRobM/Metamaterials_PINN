"""
Data handling module for SPP metamaterial PINN implementation.

Provides domain and collocation-point sampling utilities for training
Physics-Informed Neural Networks on electromagnetic problems. All samplers
live in ``domain_sampler``; every ``DomainSampler`` can also sample the outer
domain boundary via ``sample_domain_boundary`` and ``AdaptiveSampler`` supports
residual-driven refinement via ``refine_around_high_residuals``.
"""

from .domain_sampler import (
    AdaptiveSampler,
    DomainSampler,
    InterfaceSampler,
    SamplingRegion,
    SamplingStrategy,
    SPPDomainSampler,
    StratifiedSampler,
    UniformSampler,
)

__all__ = [
    "DomainSampler",
    "UniformSampler",
    "AdaptiveSampler",
    "InterfaceSampler",
    "SPPDomainSampler",
    "StratifiedSampler",
    "SamplingRegion",
    "SamplingStrategy",
]
