"""Physical constants (SI units) shared across the project.

Import from here rather than redefining locally so every module agrees on the
same numbers.

Definition order matters. Since the 2019 SI redefinition the speed of light is
*exact* by definition, while the vacuum permeability is a measured quantity
(CODATA 2018) and the permittivity follows from them:

    c   = 299 792 458 m/s        (exact, definition)
    mu0 = 1.256 637 062 12e-6    (CODATA 2018, measured)
    eps0 = 1 / (mu0 c^2)         (derived)

Defining ``EPS0`` and ``MU0`` independently and computing ``C0`` from them
instead — as this module previously did — leaves ``C0`` at 299792458.0000065,
about 2e-14 too high. That is physically irrelevant, but it makes ``k0 = 2*pi/lam``
and ``k0 = omega/C0`` disagree by ~100 ulp, so machine-precision comparisons
between modules are quietly inexact. Deriving ``EPS0`` keeps the set
self-consistent to the last bit.
"""

import math

C0: float = 299792458.0
"""Speed of light in vacuum [m/s] (exact, SI definition)."""

MU0: float = 1.25663706212e-6
"""Vacuum permeability [H/m] (CODATA 2018)."""

EPS0: float = 1.0 / (MU0 * C0 * C0)
"""Vacuum permittivity [F/m] (~8.8541878128e-12), derived from MU0 and C0."""

ETA0: float = math.sqrt(MU0 / EPS0)
"""Impedance of free space [Ohm] (~376.73)."""

__all__ = ["EPS0", "MU0", "C0", "ETA0"]
