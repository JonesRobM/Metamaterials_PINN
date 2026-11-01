"""
Physics module for surface plasmon polariton modelling on metamaterials.

Contents:
- Frequency-domain Maxwell's equations (``exp(-iωt)`` convention)
- Autograd differential operators (gradient, curl, divergence)
- Metamaterial constitutive relations and SPP dispersion
- Boundary conditions at interfaces
"""

from .boundary_conditions import BoundaryConditions
from .differential_ops import (
    curl,
    curl_complex,
    divergence,
    divergence_complex,
    gradient,
    jacobian,
)
from .maxwell_equations import MaxwellEquations
from .metamaterial import MetamaterialProperties

__all__ = [
    "MaxwellEquations",
    "MetamaterialProperties",
    "BoundaryConditions",
    "gradient",
    "jacobian",
    "curl",
    "divergence",
    "curl_complex",
    "divergence_complex",
]
