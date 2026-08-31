"""Physical constants (SI units) shared across the project.

CODATA 2018 values. Import from here rather than redefining locally so every
module agrees on the same numbers.
"""

import math

EPS0: float = 8.8541878128e-12
"""Vacuum permittivity [F/m]."""

MU0: float = 1.25663706212e-6
"""Vacuum permeability [H/m]."""

C0: float = 1.0 / math.sqrt(EPS0 * MU0)
"""Speed of light in vacuum [m/s] (~2.99792458e8)."""

ETA0: float = math.sqrt(MU0 / EPS0)
"""Impedance of free space [Ohm] (~376.73)."""

__all__ = ["EPS0", "MU0", "C0", "ETA0"]
