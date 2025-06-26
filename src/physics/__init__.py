"""
Physics module for surface plasmon polariton modelling on metamaterials.

This module contains the core electromagnetic physics implementations:
- Maxwell's equations in frequency domain
- Metamaterial constitutive relations
- Boundary conditions at interfaces
"""

from .maxwell_equations import MaxwellEquations
from .metamaterial import MetamaterialProperties
from .boundary_conditions import BoundaryConditions

__all__ = ['MaxwellEquations', 'MetamaterialProperties', 'BoundaryConditions']