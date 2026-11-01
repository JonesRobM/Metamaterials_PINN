r"""
Transfer-matrix solver for TM waves in a planar stack of isotropic layers.

This is the *exact* counterpart to the homogenised description in
:mod:`src.effective_medium`: where that module replaces a metal/dielectric
multilayer by a single uniaxial medium ``(ε_t, ε_n)``, this one keeps every
layer and solves Maxwell's equations in the stack directly. It is what
``examples/emt_validity.py`` uses to measure how good the homogenisation
actually is, and it is validated against the closed forms the repository
already trusts (Fresnel in ``tests/test_benchmark_fresnel.py``, the SPP
dispersion of :class:`src.physics.metamaterial.MetamaterialProperties`).

Only **TM** (p) polarisation is implemented, because surface plasmon
polaritons are TM: an isotropic metal/dielectric interface supports no bound TE
mode.

Convention
----------
Time dependence ``exp(-iωt)`` throughout the project, so lossy media have
``Im ε > 0`` and a wave decaying along its propagation direction has
``Im k_x > 0``. Fields are ``F(x, z) = F(z) exp(i k_x x)`` with

    H = ŷ H_y,     E = (E_x, 0, E_z),

and, from ``i k × H = -i ω ε₀ ε E`` (identical to the derivation in
``tests/test_benchmark_fresnel.py``),

    E_x = +k_z H_y / (ω ε₀ ε)      for a ``+z``-going wave,
    E_x = -k_z H_y / (ω ε₀ ε)      for a ``-z``-going wave.

Geometry
--------
Layers are listed **in order of increasing z**. ``eps_layers[0]`` is the
semi-infinite medium at ``z < z_1``, ``eps_layers[-1]`` the semi-infinite
medium at ``z > z_L``, and ``thicknesses`` gives the (finite) thicknesses of
the ``len(eps_layers) - 2`` interior layers. A bare interface is
``eps_layers = [ε_below, ε_above]`` with ``thicknesses = []``.

Branch of the normal wavevector (derivation)
--------------------------------------------
In each layer ``k_z² = ε k₀² − k_x²``, which fixes ``k_z`` only up to sign. The
two solutions are ``H_y ∝ exp(±i k_z z)``. Write ``κ ≡ −i k_z``, so that
``exp(+i k_z z) = exp(−κ z)``. In the **upper** semi-infinite medium a bound
mode may contain only the ``+z`` solution, and it must decay as ``z → +∞``:

    |exp(−κ z)| → 0   ⟺   Re κ > 0   ⟺   Re(−i k_z) > 0   ⟺   **Im k_z > 0**,

because ``Re(−i k_z) = Re(−i(a + ib)) = b = Im k_z`` identically — the two
statements quoted in the literature are the *same* statement. Symmetrically,
the lower semi-infinite medium may contain only ``exp(−i k_z z) = exp(+κ z)``,
which decays as ``z → −∞`` under the same condition. So a single branch choice,

    **Im k_z ≥ 0** (and ``Re k_z > 0`` when ``k_z`` is real),

serves both half-spaces: it makes the ``A`` (``+z``) amplitude the
decaying-upward solution and the ``B`` (``−z``) amplitude the
decaying-downward one, and for a propagating wave it selects the outgoing
(radiating) direction. It is the branch used by ``_kz`` in
``tests/test_benchmark_fresnel.py`` and by ``_propagating_root`` in
:mod:`src.physics.metamaterial`; :func:`normal_wavevector` is the vectorised
version.

Matrices
--------
Amplitudes in layer *j* are referenced at that layer's **lower** boundary
``z_j``:

    H_y(z) = A_j exp(+i k_j (z − z_j)) + B_j exp(−i k_j (z − z_j)).

Continuity of ``H_y`` and ``E_x`` at ``z_{j+1}`` gives, with the TM ratio
``p ≡ k_z/ε`` and ``η ≡ p_{j+1}/p_j``,

    D(j → j+1) = ½ [[1 + η, 1 − η], [1 − η, 1 + η]],

and crossing layer *j* of thickness ``d_j`` back to its lower boundary gives
``P_j^{-1} = diag(exp(−i k_j d_j), exp(+i k_j d_j))``, so that

    (A_0, B_0)ᵀ = M · (A_L, B_L)ᵀ,
    M = D(0→1) · Π_{j=1}^{L-1} [ P_j^{-1} D(j → j+1) ].

``P_j^{-1}`` contains a *growing* exponential for an evanescent layer, which is
the classic overflow mode of transfer matrices. This module therefore
accumulates the **rescaled** product

    M̃ ≡ M · Π_j exp(+i k_j d_j),      i.e.   P̃_j = diag(1, exp(2 i k_j d_j)),

whose second entry has modulus ``≤ 1`` on the ``Im k_z ≥ 0`` branch. The
rescaling factor ``exp(Σ_j i k_j d_j)`` is analytic and never zero, so it
changes neither the reflection coefficient (a ratio of matrix elements) nor the
location of any mode; :class:`StackMatrix` carries ``log_scale = Σ_j i k_j d_j``
so the unscaled ``M = M̃ exp(−log_scale)`` and the transmission amplitude can be
recovered exactly.

Reflection and the mode condition
---------------------------------
With a unit ``+z``-going wave incident from medium 0 and nothing incident from
above (``B_L = 0``),

    r = M₁₀ / M₀₀ = M̃₁₀ / M̃₀₀,      t = 1 / M₀₀ = exp(log_scale) / M̃₀₀,

both in the **H-amplitude** convention of ``tests/test_benchmark_fresnel.py``
(``r_p = (ε₂ k_{1z} − ε₁ k_{2z}) / (ε₂ k_{1z} + ε₁ k_{2z})``, so ``r_p > 0`` at
normal incidence into a denser medium).

A **guided or bound mode** is a solution with no wave incident from either
side: ``A_0 = 0`` (nothing coming up from below) *and* ``B_L = 0`` (nothing
coming down from above), with the fields not identically zero. Putting
``B_L = 0`` into ``(A_0, B_0)ᵀ = M (A_L, 0)ᵀ`` gives ``A_0 = M₀₀ A_L``, so a
non-trivial solution exists precisely when

    **M₀₀(k_x) = 0.**

That is the element :func:`mode_dispersion_function` returns. Equivalently it
is the pole of ``r`` and of ``t`` — the resonance of the stack — which is the
usual way the condition is stated. For a single interface it reduces to
``½(1 + η) = 0``, i.e. ``k_{z0}/ε_0 + k_{z1}/ε_1 = 0``; writing ``k_z = iκ``
this is exactly the unsquared matching condition ``κ_d/ε_d + κ_m/ε_t = 0`` of
:mod:`src.physics.metamaterial`, which is why :func:`find_mode` reproduces
``MetamaterialProperties.spp_wavevector`` to machine precision.

Root finding
------------
``M₀₀`` is analytic in ``k_x`` away from the branch cuts of the ``k_z``, so its
zeros are found with **Muller's method** (:func:`muller`) — a quadratic secant
that needs no derivative and moves happily into the complex plane.
:func:`find_mode` polishes one guess, :func:`find_modes` polishes several and
de-duplicates, and :func:`scan_modes` seeds the search from the local minima of
``|M₀₀|`` over a rectangle of the complex ``k_x`` plane. Every candidate is
accepted only after a *contrast* test: ``|M₀₀|`` at the root must be smaller
than its median on a small circle around the root by a large factor, which
rejects the flat regions and near-cancellations that a bare residual threshold
would wave through.

Field profiles
--------------
:func:`find_mode` returns only ``k_x``. :func:`mode_field_profile` turns that
number back into the mode's actual ``z``-profile — ``H_y(z)``, ``E_x(z)``,
``E_z(z)`` and ``D_z(z)`` through the whole stack — by running the amplitude
recursion explicitly (see its docstring for the direction and its stability).
That is what an external solver has to be compared against point by point;
:func:`permittivity_profile` and :func:`layer_boundaries` give the piecewise
``ε(z)`` and the interface positions that go with it.

Everything is float64/complex128 and vectorised over ``k_x`` (and over any
broadcastable layer permittivities).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.constants import C0, EPS0

__all__ = [
    "ArrayLike",
    "StackMatrix",
    "ModeFieldProfile",
    "normal_wavevector",
    "tm_ratio",
    "interface_matrix",
    "propagation_matrix",
    "stack_matrix",
    "reflection_coefficient",
    "reflection_transmission",
    "power_coefficients",
    "mode_dispersion_function",
    "muller",
    "find_mode",
    "find_modes",
    "scan_modes",
    "layer_boundaries",
    "layer_index_at",
    "permittivity_profile",
    "mode_field_profile",
]

ArrayLike = Union[float, complex, np.ndarray]

# Rescale the running product if it ever threatens to overflow. With the
# analytic rescaling above this is essentially unreachable for physical stacks
# (the running product is a sub-stack transfer matrix, bounded by the mode's own
# evanescent growth), but a runaway guess from a root finder must not produce
# inf/nan instead of a large finite number.
_RENORM_THRESHOLD = 1e100


# --------------------------------------------------------------------- layers
def normal_wavevector(eps: ArrayLike, k_x: ArrayLike, k0: float) -> np.ndarray:
    r"""
    Normal wavevector ``k_z = sqrt(ε k₀² − k_x²)`` on the bound/outgoing branch.

    The branch is ``Im k_z ≥ 0`` (with ``Re k_z > 0`` when ``k_z`` is real):
    with ``exp(-iωt)`` and ``H_y ∝ exp(i k_z z)``, writing ``κ = −i k_z`` gives
    ``exp(−κ z)``, so decay as ``z → +∞`` needs ``Re κ = Im k_z > 0``. See the
    module docstring for the full derivation. This is the same branch as
    ``_propagating_root`` in :mod:`src.physics.metamaterial`.

    Args:
        eps: Relative permittivity (complex, ``Im ε > 0`` for loss).
        k_x: In-plane wavevector (1/m), may be complex.
        k0: Free-space wavenumber ω/c (1/m).

    Returns:
        Complex ``k_z`` broadcast over ``eps`` and ``k_x``.
    """
    eps_c = np.asarray(eps, dtype=complex)
    kx_c = np.asarray(k_x, dtype=complex)
    with np.errstate(over="ignore", invalid="ignore"):
        kz = np.sqrt(eps_c * complex(k0) ** 2 - kx_c**2)
    flip = (kz.imag < 0.0) | ((kz.imag == 0.0) & (kz.real < 0.0))
    return np.where(flip, -kz, kz)


def tm_ratio(k_z: ArrayLike, eps: ArrayLike) -> np.ndarray:
    r"""
    The TM interface ratio ``p = k_z / ε``.

    ``E_x = ± p H_y / (ω ε₀)``, so continuity of ``E_x`` across an interface is
    continuity of ``p (A − B)``. It is the only combination of ``k_z`` and ``ε``
    that enters the TM matrices, and the common factor ``ω ε₀`` cancels
    everywhere (including in ``r`` and ``t``).
    """
    return np.asarray(k_z, dtype=complex) / np.asarray(eps, dtype=complex)


def _two_by_two(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Assemble ``[[a, b], [c, d]]`` with the 2×2 block as the *last* two axes."""
    a, b, c, d = np.broadcast_arrays(*(np.asarray(v, dtype=complex) for v in (a, b, c, d)))
    top = np.stack([a, b], axis=-1)
    bottom = np.stack([c, d], axis=-1)
    return np.stack([top, bottom], axis=-2)


def interface_matrix(p_from: ArrayLike, p_to: ArrayLike) -> np.ndarray:
    r"""
    TM interface matrix ``D`` mapping the amplitudes *after* the interface to
    those *before* it: ``(A_j, B_j)ᵀ = D · (A_{j+1}, B_{j+1})ᵀ``.

    From continuity of ``H_y`` (``A_j + B_j = A_{j+1} + B_{j+1}``) and of
    ``E_x`` (``p_j (A_j − B_j) = p_{j+1} (A_{j+1} − B_{j+1})``), with
    ``η = p_{j+1}/p_j``,

        D = ½ [[1 + η, 1 − η], [1 − η, 1 + η]].

    In the ``σ_x`` eigenbasis ``D = diag(1, η)``, so interface matrices compose
    exactly (``D(a→b) D(b→c) = D(a→c)``) — the algebraic reason a zero-thickness
    layer is the identity insertion.

    Args:
        p_from, p_to: ``k_z/ε`` on the incoming and outgoing side (see
            :func:`tm_ratio`); broadcastable.

    Returns:
        Complex array of shape ``(..., 2, 2)``.
    """
    eta = np.asarray(p_to, dtype=complex) / np.asarray(p_from, dtype=complex)
    plus = 0.5 * (1.0 + eta)
    minus = 0.5 * (1.0 - eta)
    return _two_by_two(plus, minus, minus, plus)


def propagation_matrix(k_z: ArrayLike, thickness: float, scaled: bool = True) -> np.ndarray:
    r"""
    TM propagation matrix across a layer, referencing amplitudes back to the
    layer's lower boundary.

    The unscaled matrix is ``P⁻¹ = diag(exp(−i k_z d), exp(+i k_z d))``. With
    ``scaled=True`` (the default) the analytic, never-zero factor
    ``exp(+i k_z d)`` is divided out, giving

        P̃ = diag(1, exp(2 i k_z d)),

    whose entries have modulus ``≤ 1`` on the ``Im k_z ≥ 0`` branch. The
    discarded factor is accumulated in :attr:`StackMatrix.log_scale`.

    Args:
        k_z: Normal wavevector in the layer (1/m).
        thickness: Layer thickness (m), ``≥ 0``.
        scaled: Whether to return the rescaled form.

    Returns:
        Complex array of shape ``(..., 2, 2)``.
    """
    d = float(thickness)
    if d < 0.0:
        raise ValueError("layer thickness must be non-negative")
    kz = np.asarray(k_z, dtype=complex)
    zero = np.zeros_like(kz)
    if scaled:
        return _two_by_two(np.ones_like(kz), zero, zero, np.exp(2j * kz * d))
    return _two_by_two(np.exp(-1j * kz * d), zero, zero, np.exp(1j * kz * d))


# ---------------------------------------------------------------- stack matrix
@dataclass(frozen=True)
class StackMatrix:
    """
    The accumulated TM transfer matrix of a stack.

    Attributes:
        matrix: Rescaled matrix ``M̃`` of shape ``(..., 2, 2)``. The physical
            transfer matrix is ``M = M̃ · exp(−log_scale)``.
        log_scale: ``Σ_j i k_{z,j} d_j`` over the interior layers (plus any
            overflow renormalisation actually applied), shape ``(...)``.
        k_z: Normal wavevectors of every medium, shape ``(L, ...)`` with ``L =
            len(eps_layers)``.
    """

    matrix: np.ndarray
    log_scale: np.ndarray
    k_z: np.ndarray

    @property
    def unscaled(self) -> np.ndarray:
        """The physical transfer matrix ``M`` (may overflow for thick stacks)."""
        return self.matrix * np.exp(-self.log_scale)[..., None, None]


def _validate_stack(
    eps_layers: Sequence[ArrayLike], thicknesses: Sequence[float]
) -> Tuple[List[np.ndarray], List[float]]:
    eps_list = [np.asarray(e, dtype=complex) for e in eps_layers]
    if len(eps_list) < 2:
        raise ValueError("eps_layers needs at least the two semi-infinite media")
    d_list = [float(d) for d in thicknesses]
    if len(d_list) != len(eps_list) - 2:
        raise ValueError(
            f"expected {len(eps_list) - 2} thicknesses for {len(eps_list)} media "
            f"(the two outer media are semi-infinite), got {len(d_list)}"
        )
    if any(d < 0.0 for d in d_list):
        raise ValueError("layer thicknesses must be non-negative")
    return eps_list, d_list


def stack_matrix(
    k_x: ArrayLike,
    k0: float,
    eps_layers: Sequence[ArrayLike],
    thicknesses: Sequence[float] = (),
) -> StackMatrix:
    r"""
    Accumulate the TM transfer matrix of a planar stack.

    ``M = D(0→1) · Π_{j=1}^{L-1} [P_j^{-1} D(j→j+1)]`` in the rescaled form
    described in the module docstring, so that ``(A_0, B_0)ᵀ = M (A_L, B_L)ᵀ``
    with ``A`` the ``+z``-going and ``B`` the ``−z``-going amplitude.

    Args:
        k_x: In-plane wavevector (1/m), scalar or array; complex allowed.
        k0: Free-space wavenumber ω/c (1/m).
        eps_layers: Relative permittivities ordered by increasing ``z``; the
            first and last are the semi-infinite media.
        thicknesses: Thicknesses (m) of the ``len(eps_layers) - 2`` interior
            layers, in the same order.

    Returns:
        :class:`StackMatrix`.

    Raises:
        ValueError: on a malformed stack (too few media, wrong number of
            thicknesses, negative thickness).
    """
    eps_list, d_list = _validate_stack(eps_layers, thicknesses)
    kx = np.asarray(k_x, dtype=complex)
    k0f = float(k0)

    k_z = [normal_wavevector(e, kx, k0f) for e in eps_list]
    shape = np.broadcast_shapes(*(kz.shape for kz in k_z))
    k_z = [np.broadcast_to(kz, shape).astype(complex) for kz in k_z]

    # A root finder may probe wild guesses; those must yield inf/nan quietly
    # rather than a warning storm, and are filtered out by the caller's checks.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        p = [tm_ratio(kz, e) for kz, e in zip(k_z, eps_list, strict=True)]
        matrix = interface_matrix(p[0], p[1])
        log_scale = np.zeros(shape, dtype=complex)
        for j, d in enumerate(d_list, start=1):
            matrix = matrix @ propagation_matrix(k_z[j], d) @ interface_matrix(p[j], p[j + 1])
            log_scale = log_scale + 1j * k_z[j] * d
            peak = np.abs(matrix).max(axis=(-2, -1))
            if np.any(peak > _RENORM_THRESHOLD):
                # Analytic rescaling was not enough (pathological guess): fold
                # the magnitude into log_scale rather than overflow to inf.
                factor = np.where(peak > _RENORM_THRESHOLD, peak, 1.0)
                matrix = matrix / factor[..., None, None]
                log_scale = log_scale + np.log(factor.astype(complex))

    return StackMatrix(matrix=matrix, log_scale=log_scale, k_z=np.stack(k_z, axis=0))


# ------------------------------------------------------------ Fresnel / R, T
def reflection_transmission(
    k_x: ArrayLike,
    k0: float,
    eps_layers: Sequence[ArrayLike],
    thicknesses: Sequence[float] = (),
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    TM amplitude reflection and transmission of the stack, for a unit
    ``+z``-going wave incident from ``eps_layers[0]``.

    ``r = M₁₀/M₀₀`` and ``t = 1/M₀₀``, both as **H_y amplitude** ratios (the
    convention of ``tests/test_benchmark_fresnel.py``); for a single interface
    ``r = (ε₂ k_{1z} − ε₁ k_{2z}) / (ε₂ k_{1z} + ε₁ k_{2z})`` and ``t = 1 + r``.

    Args:
        k_x: In-plane wavevector (1/m); for a plane wave at angle ``θ`` in
            medium 0, ``k_x = sqrt(ε₀) k₀ sin θ``.
        k0: Free-space wavenumber (1/m).
        eps_layers, thicknesses: See :func:`stack_matrix`.

    Returns:
        ``(r, t)``, complex arrays broadcast over ``k_x``.
    """
    stack = stack_matrix(k_x, k0, eps_layers, thicknesses)
    m00 = stack.matrix[..., 0, 0]
    m10 = stack.matrix[..., 1, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = m10 / m00
        t = np.exp(stack.log_scale) / m00
    return r, t


def reflection_coefficient(
    k_x: ArrayLike,
    k0: float,
    eps_layers: Sequence[ArrayLike],
    thicknesses: Sequence[float] = (),
) -> np.ndarray:
    """TM amplitude reflection coefficient; see :func:`reflection_transmission`."""
    return reflection_transmission(k_x, k0, eps_layers, thicknesses)[0]


def power_coefficients(
    k_x: ArrayLike,
    k0: float,
    eps_layers: Sequence[ArrayLike],
    thicknesses: Sequence[float] = (),
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Power reflectance and transmittance of the stack for TM incidence.

    In the H-amplitude convention the time-averaged ``z``-component of the
    Poynting vector of a wave is ``½ Re(k_z/ε) |H_y|² / (ω ε₀)``, so

        R = |r|²,     T = Re(k_{zL}/ε_L) / Re(k_{z0}/ε_0) · |t|².

    ``R + T = 1`` exactly for a lossless stack illuminated below its critical
    angle (checked in ``tests/test_transfer_matrix.py``); beyond it ``T = 0``
    and ``R = 1``.

    Returns:
        ``(R, T)`` real arrays.
    """
    stack = stack_matrix(k_x, k0, eps_layers, thicknesses)
    eps_list, _ = _validate_stack(eps_layers, thicknesses)
    m00 = stack.matrix[..., 0, 0]
    m10 = stack.matrix[..., 1, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = m10 / m00
        t = np.exp(stack.log_scale) / m00
    p_in = tm_ratio(stack.k_z[0], eps_list[0]).real
    p_out = tm_ratio(stack.k_z[-1], eps_list[-1]).real
    with np.errstate(divide="ignore", invalid="ignore"):
        transmittance = np.where(p_in != 0.0, p_out / p_in * np.abs(t) ** 2, np.nan)
    return np.abs(r) ** 2, transmittance


# ------------------------------------------------------------- mode condition
def mode_dispersion_function(
    k_x: ArrayLike,
    k0: float,
    eps_layers: Sequence[ArrayLike],
    thicknesses: Sequence[float] = (),
) -> np.ndarray:
    r"""
    The mode-dispersion function ``M₀₀(k_x)``, whose complex zeros are the
    guided/bound modes of the stack.

    **Why this element.** A bound mode carries no incoming wave: nothing
    travelling ``+z`` in the bottom half-space (``A_0 = 0``) and nothing
    travelling ``−z`` in the top one (``B_L = 0``). Since
    ``(A_0, B_0)ᵀ = M (A_L, B_L)ᵀ``, imposing ``B_L = 0`` gives
    ``A_0 = M₀₀ A_L``; a non-trivial field (``A_L ≠ 0``) with ``A_0 = 0``
    therefore exists **iff ``M₀₀ = 0``**. Because ``r = M₁₀/M₀₀`` and
    ``t = 1/M₀₀``, the zeros of ``M₀₀`` are exactly the poles of the scattering
    problem — the stack resonating with no drive. On the ``Im k_z ≥ 0`` branch
    of :func:`normal_wavevector` the ``A`` solution is the one that decays
    upwards and ``B`` the one that decays downwards, so the same condition also
    enforces "decaying only" on both sides whenever ``k_x`` is beyond the light
    line of the outer media.

    The value returned is the **rescaled** ``M̃₀₀`` (module docstring): it
    differs from ``M₀₀`` by the analytic, nowhere-zero factor
    ``exp(Σ_j i k_{z,j} d_j)``, so it has exactly the same zeros while staying
    numerically bounded for thick evanescent stacks.

    Args:
        k_x: In-plane wavevector (1/m), scalar or array, complex allowed.
        k0: Free-space wavenumber (1/m).
        eps_layers, thicknesses: See :func:`stack_matrix`.

    Returns:
        Complex array broadcast over ``k_x``.
    """
    return stack_matrix(k_x, k0, eps_layers, thicknesses).matrix[..., 0, 0]


# ------------------------------------------------------------- root finding
def muller(
    func: Callable[[complex], complex],
    x0: complex,
    x1: Optional[complex] = None,
    x2: Optional[complex] = None,
    tol: float = 1e-13,
    max_iter: int = 100,
) -> Tuple[complex, bool, int]:
    r"""
    Muller's method for a complex root of an analytic function.

    Muller fits a *parabola* through the last three iterates and jumps to its
    root, which (unlike Newton or the secant method) can leave the real axis
    from real data and needs no derivative — the two properties that matter for
    ``M₀₀``, whose zeros are complex and whose derivative is only available
    numerically. Convergence is superlinear (order ≈ 1.84).

    Args:
        func: Analytic function of one complex variable.
        x0: Initial guess. When ``x1``/``x2`` are omitted a starting triplet
            ``x0(1 ∓ 10⁻³), x0`` is generated around it.
        x1, x2: Optional additional starting points.
        tol: Relative step tolerance ``|Δx| ≤ tol·max(|x|, 1)``.
        max_iter: Iteration cap.

    Returns:
        ``(root, converged, iterations)``. ``converged`` reports only that the
        *step* collapsed; callers should verify the residual (see
        :func:`find_mode`).
    """
    x0 = complex(x0)
    if x1 is None:
        x1 = x0 * (1.0 - 1e-3) if x0 != 0 else -1e-3
    if x2 is None:
        x2 = x0 * (1.0 + 1e-3) if x0 != 0 else 1e-3
    a, b, c = complex(x1), complex(x2), x0
    fa, fb, fc = complex(func(a)), complex(func(b)), complex(func(c))

    for iteration in range(1, max_iter + 1):
        if fc == 0.0:
            return c, True, iteration
        if b == a or c == b:
            return c, False, iteration
        q = (c - b) / (b - a)
        big_a = q * fc - q * (1.0 + q) * fb + q * q * fa
        big_b = (2.0 * q + 1.0) * fc - (1.0 + q) ** 2 * fb + q * q * fa
        big_c = (1.0 + q) * fc

        disc = np.sqrt(complex(big_b * big_b - 4.0 * big_a * big_c))
        den_plus, den_minus = big_b + disc, big_b - disc
        den = den_plus if abs(den_plus) >= abs(den_minus) else den_minus
        if den == 0.0 or not np.isfinite(den):
            # Degenerate parabola: fall back to a secant step.
            if fc == fb or not np.isfinite(fc - fb):
                return c, False, iteration
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                step = -fc * (c - b) / (fc - fb)
        else:
            step = -(c - b) * 2.0 * big_c / den
        if not np.isfinite(step):
            return c, False, iteration

        x_new = c + step
        a, b, c = b, c, x_new
        fa, fb, fc = fb, fc, complex(func(c))
        if abs(step) <= tol * max(abs(c), 1.0):
            return c, True, iteration
    return c, False, max_iter


def _contrast(
    func: Callable[[np.ndarray], np.ndarray],
    x: complex,
    rel_radius: float = 1e-2,
    n_probe: int = 8,
) -> float:
    """
    Ratio of ``|f|`` on a small circle around ``x`` to ``|f(x)|``.

    A genuine simple zero sits in a deep, isolated well: the median modulus on a
    circle of radius ``rel_radius·|x|`` exceeds the centre value by many orders
    of magnitude. A flat region or a numerical near-cancellation does not. This
    is scale-free, so it needs no absolute residual threshold.
    """
    radius = rel_radius * max(abs(x), 1.0)
    ring = x + radius * np.exp(2j * np.pi * np.arange(n_probe) / n_probe)
    centre = abs(complex(np.asarray(func(np.asarray(x, dtype=complex))).reshape(())))
    on_ring = np.abs(np.asarray(func(ring), dtype=complex))
    reference = float(np.median(on_ring))
    if not np.isfinite(reference) or reference == 0.0:
        return 0.0
    if centre == 0.0:
        return np.inf
    return reference / centre


def find_mode(
    k_x_guess: complex,
    k0: float,
    eps_layers: Sequence[ArrayLike],
    thicknesses: Sequence[float] = (),
    *,
    tol: float = 1e-13,
    max_iter: int = 100,
    min_contrast: float = 1e6,
    require_bound: bool = True,
    bound_tol: float = 1e-3,
) -> Optional[complex]:
    r"""
    Polish one initial guess into a complex zero of :func:`mode_dispersion_function`.

    The search runs in the dimensionless effective index ``n = k_x/k₀`` (which
    is ``O(1)`` for every mode of interest and keeps the step tolerance
    meaningful), then converts back.

    Args:
        k_x_guess: Starting in-plane wavevector (1/m); the closed-form
            ``MetamaterialProperties.spp_wavevector`` is an excellent seed.
        k0: Free-space wavenumber (1/m).
        eps_layers, thicknesses: See :func:`stack_matrix`.
        tol: Relative step tolerance handed to :func:`muller`.
        max_iter: Iteration cap.
        min_contrast: Minimum ratio of ``|M₀₀|`` on a small circle around the
            candidate to its value at the candidate (see :func:`_contrast`);
            candidates below it are rejected as spurious.
        require_bound: Reject roots that are not bound in **both** semi-infinite
            media, i.e. that fail ``Im k_z > bound_tol·|k_z|`` there. Set
            ``False`` to keep leaky/radiative roots.
        bound_tol: Tolerance of that boundedness test. The default ``1e-3``
            matches ``MetamaterialProperties.is_spp_supported``: with loss,
            ``Im k_z`` is small but *positive* for a wave that is really
            oscillating in the outer medium, so a machine-epsilon threshold
            would pass leaky modes as bound.

    Returns:
        The complex ``k_x`` of the mode, or ``None`` if the iteration failed the
        step, contrast or boundedness tests.
    """
    k0f = float(k0)
    eps_list, d_list = _validate_stack(eps_layers, thicknesses)

    def f_index(n: ArrayLike) -> np.ndarray:
        return mode_dispersion_function(np.asarray(n, dtype=complex) * k0f, k0f, eps_list, d_list)

    def f_scalar(n: complex) -> complex:
        return complex(np.asarray(f_index(np.asarray(n, dtype=complex))).reshape(()))

    n_guess = complex(k_x_guess) / k0f
    n_root, converged, _ = muller(f_scalar, n_guess, tol=tol, max_iter=max_iter)
    if not converged or not np.isfinite(n_root):
        return None
    if _contrast(f_index, n_root) < min_contrast:
        return None

    k_x = n_root * k0f
    if require_bound:
        for eps_outer in (eps_list[0], eps_list[-1]):
            kz = complex(normal_wavevector(eps_outer, k_x, k0f))
            if kz.imag <= bound_tol * abs(kz):
                return None
    return complex(k_x)


def _dedupe(roots: Sequence[complex], rel_tol: float) -> List[complex]:
    """Keep one representative per cluster of roots agreeing to ``rel_tol``."""
    kept: List[complex] = []
    for root in roots:
        if not any(abs(root - other) <= rel_tol * max(abs(root), abs(other), 1.0) for other in kept):
            kept.append(root)
    return kept


def find_modes(
    k_x_guesses: Union[complex, Sequence[complex], np.ndarray],
    k0: float,
    eps_layers: Sequence[ArrayLike],
    thicknesses: Sequence[float] = (),
    *,
    rel_tol: float = 1e-7,
    sort: bool = True,
    **kwargs: object,
) -> np.ndarray:
    """
    Polish a set of guesses and return the **distinct** modes found.

    Args:
        k_x_guesses: One guess or a sequence of them (1/m).
        k0: Free-space wavenumber (1/m).
        eps_layers, thicknesses: See :func:`stack_matrix`.
        rel_tol: Two roots closer than ``rel_tol·|k_x|`` count as one.
        sort: Sort the result by increasing ``Re k_x``.
        **kwargs: Forwarded to :func:`find_mode` (``tol``, ``max_iter``,
            ``min_contrast``, ``require_bound``, ``bound_tol``).

    Returns:
        Complex128 array of distinct roots (possibly empty).
    """
    guesses = np.atleast_1d(np.asarray(k_x_guesses, dtype=complex)).ravel()
    found = [
        root
        for root in (
            find_mode(complex(g), k0, eps_layers, thicknesses, **kwargs)  # type: ignore[arg-type]
            for g in guesses
        )
        if root is not None
    ]
    distinct = _dedupe(found, rel_tol)
    out = np.asarray(distinct, dtype=complex)
    if sort and out.size:
        out = out[np.argsort(out.real)]
    return out


def scan_modes(
    k0: float,
    eps_layers: Sequence[ArrayLike],
    thicknesses: Sequence[float] = (),
    *,
    k_x_re_range: Optional[Tuple[float, float]] = None,
    k_x_im_range: Tuple[float, float] = (0.0, 0.0),
    n_re: int = 200,
    n_im: int = 1,
    rel_tol: float = 1e-7,
    **kwargs: object,
) -> np.ndarray:
    r"""
    Scan a rectangle of the complex ``k_x`` plane and return the distinct modes.

    ``|M₀₀|`` is evaluated on the grid (one vectorised call), the grid's local
    minima — points no larger than all eight neighbours — are taken as seeds,
    and each is polished with :func:`find_mode`. Seeding from minima rather than
    from every node makes the scan cheap enough to use a fine grid, which is
    what actually decides whether closely spaced branches (such as the
    long-range/short-range pair of a thin metal film) are both found.

    Args:
        k0: Free-space wavenumber (1/m).
        eps_layers, thicknesses: See :func:`stack_matrix`.
        k_x_re_range: ``(min, max)`` of ``Re k_x`` (1/m). Defaults to
            ``(1.001, 20) × k₀·sqrt(max|ε|)``-ish: ``(1.001 k₀, 20 k₀ √ε_max)``,
            which spans the light line to the deeply bound modes.
        k_x_im_range: ``(min, max)`` of ``Im k_x`` (1/m); a single row at 0 by
            default, which is enough for low-loss stacks because Muller leaves
            the real axis on its own.
        n_re, n_im: Grid resolution.
        rel_tol: De-duplication tolerance.
        **kwargs: Forwarded to :func:`find_mode`.

    Returns:
        Complex128 array of distinct roots, sorted by ``Re k_x``.
    """
    eps_list, d_list = _validate_stack(eps_layers, thicknesses)
    k0f = float(k0)
    if k_x_re_range is None:
        n_max = float(np.sqrt(max(abs(complex(np.asarray(e).reshape(()))) for e in eps_list)))
        k_x_re_range = (1.001 * k0f, 20.0 * k0f * max(n_max, 1.0))

    re = np.linspace(float(k_x_re_range[0]), float(k_x_re_range[1]), int(n_re))
    im = np.linspace(float(k_x_im_range[0]), float(k_x_im_range[1]), int(n_im))
    grid = re[None, :] + 1j * im[:, None]
    with np.errstate(over="ignore", invalid="ignore"):
        magnitude = np.abs(mode_dispersion_function(grid, k0f, eps_list, d_list))
    magnitude = np.where(np.isfinite(magnitude), magnitude, np.inf)

    padded = np.pad(magnitude, 1, mode="constant", constant_values=np.inf)
    is_min = np.ones_like(magnitude, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            shifted = padded[1 + di : 1 + di + magnitude.shape[0],
                             1 + dj : 1 + dj + magnitude.shape[1]]
            is_min &= magnitude <= shifted

    seeds = grid[is_min]
    return find_modes(seeds, k0f, eps_list, d_list, rel_tol=rel_tol, **kwargs)


# ------------------------------------------------------------- field profiles
def layer_boundaries(thicknesses: Sequence[float], z0: float = 0.0) -> np.ndarray:
    r"""
    Positions of the ``L - 1`` interfaces of an ``L``-medium stack.

    With the geometry of :func:`stack_matrix` (media ordered by increasing
    ``z``), the lowest interface — between the semi-infinite medium 0 and the
    first interior layer — sits at ``z0``, and each subsequent one is a layer
    thickness further up.

    Args:
        thicknesses: The ``L - 2`` interior thicknesses (m), in order.
        z0: Position of the **lowest** interface (m). To put the *topmost*
            interface at the origin instead, pass ``z0 = -sum(thicknesses)``.

    Returns:
        Float array of shape ``(len(thicknesses) + 1,)``, strictly increasing
        wherever the thicknesses are positive.
    """
    d = np.asarray([float(t) for t in thicknesses], dtype=float)
    return float(z0) + np.concatenate([[0.0], np.cumsum(d)])


def layer_index_at(
    z: ArrayLike, thicknesses: Sequence[float], z0: float = 0.0
) -> np.ndarray:
    r"""
    Index of the medium containing each ``z``, in the ordering of ``eps_layers``.

    ``0`` is the semi-infinite medium below the stack, ``len(thicknesses) + 1``
    the semi-infinite medium above it.

    **A point exactly on an interface is assigned to the medium above it**
    (the profile is treated as right-continuous, ``ε(z_j) = ε_j``). That is the
    same convention the PINN experiments use when they classify collocation
    points by ``z < 0``, and it is the only choice that keeps ``ε(z)``
    single-valued; the physical fields ``H_y``, ``E_x`` and ``D_z`` are
    continuous there anyway, so only ``E_z`` is affected by which side is
    picked.

    Args:
        z: Coordinate(s) (m).
        thicknesses: Interior layer thicknesses (m).
        z0: Position of the lowest interface (see :func:`layer_boundaries`).

    Returns:
        Integer array shaped like ``z``.
    """
    bounds = layer_boundaries(thicknesses, z0)
    return np.searchsorted(bounds, np.asarray(z, dtype=float), side="right")


def permittivity_profile(
    z: ArrayLike,
    eps_layers: Sequence[ArrayLike],
    thicknesses: Sequence[float] = (),
    z0: float = 0.0,
) -> np.ndarray:
    r"""
    The piecewise-constant ``ε(z)`` of a stack, sampled at ``z``.

    This is the profile the transfer matrix describes, written out explicitly —
    the thing a differential solver (a PINN, a finite-difference code) needs in
    order to be solving the *same* problem. On an interface the medium above is
    returned; see :func:`layer_index_at`.

    Args:
        z: Coordinate(s) (m).
        eps_layers, thicknesses: See :func:`stack_matrix`.
        z0: Position of the lowest interface (see :func:`layer_boundaries`).

    Returns:
        Complex array shaped like ``z``.

    Raises:
        ValueError: on a malformed stack.
    """
    eps_list, d_list = _validate_stack(eps_layers, thicknesses)
    values = np.asarray([complex(np.asarray(e).reshape(())) for e in eps_list], dtype=complex)
    return values[layer_index_at(z, d_list, z0)]


@dataclass(frozen=True)
class ModeFieldProfile:
    r"""
    The ``z``-profile of one TM mode of a stack, in SI units.

    The full fields are ``F(x, z) = F(z) exp(i k_x x)`` with ``H = ŷ H_y`` and
    ``E = (E_x, 0, E_z)``.

    Attributes:
        z: Sample coordinates (m), as given.
        H_y, E_x, E_z: Field profiles (A/m, V/m, V/m).
        D_z: Normal displacement ``ε₀ ε(z) E_z`` (C/m²) — **continuous across
            every interface**, unlike ``E_z``.
        eps: ``ε(z)`` at the sample points (see :func:`permittivity_profile`).
        layer_index: Medium index at each sample point.
        boundaries: The ``L - 1`` interface positions (m).
        amplitudes: ``(L, 2)`` complex ``(A_j, B_j)``, each referenced at its
            medium's lower boundary (the top medium at the topmost interface),
            after the same normalisation as the fields.
        k_z: ``(L,)`` normal wavevectors, on the ``Im k_z ≥ 0`` branch.
        k_x, k0: The mode's in-plane wavevector and the free-space wavenumber.
        leakage: ``|B_top| / max|A_j, B_j|`` — the ``−z``-going amplitude left
            in the top half-space. It vanishes exactly at a mode, so it is a
            direct measure of how well ``k_x`` solves ``M₀₀ = 0``; a value that
            is not ~1e-10 or below means the profile is not a bound mode.
    """

    z: np.ndarray
    H_y: np.ndarray
    E_x: np.ndarray
    E_z: np.ndarray
    D_z: np.ndarray
    eps: np.ndarray
    layer_index: np.ndarray
    boundaries: np.ndarray
    amplitudes: np.ndarray
    k_z: np.ndarray
    k_x: complex
    k0: float
    leakage: float


def _mode_amplitudes(
    k_x: complex,
    k0: float,
    eps_list: Sequence[np.ndarray],
    d_list: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Amplitudes ``(A_j, B_j)`` of the bound mode in every medium.

    **The recursion runs upward**, from ``(A_0, B_0) = (0, 1)`` in the bottom
    half-space — the boundary condition ``A_0 = 0`` imposed exactly — through
    each interface in turn:

        (Ã_j, B̃_j) = (A_j e^{+i k_j d_j}, B_j e^{−i k_j d_j})      (to the top
                                                                    of layer j)
        (A_{j+1}, B_{j+1}) = D(j → j+1)^{-1} (Ã_j, B̃_j)
                           = interface_matrix(p_{j+1}, p_j) (Ã_j, B̃_j),

    using ``D(a→b)^{-1} = D(b→a)`` (both are ``diag(1, η^{±1})`` in the ``σ_x``
    eigenbasis).

    *Why upward.* A bound surface mode peaks at the top interface and decays
    both ways from it, so below that interface the field **grows** with ``z``.
    Integrating in the direction in which the wanted solution is the dominant
    one is the stable choice: the unwanted solution (the one with ``A_0 ≠ 0``)
    is being suppressed rather than amplified. Going the other way would
    amplify roundoff by the stack's whole evanescent factor. Nothing is
    propagated *inside* the top half-space — its amplitudes come from a single
    interface step — so the growing solution there never gets a chance to act;
    what is left in ``B_top`` is reported as ``leakage``.

    Returns:
        ``(amplitudes (L, 2), k_z (L,), p (L,))``.
    """
    k0f = float(k0)
    kx = complex(k_x)
    k_z = np.asarray(
        [complex(normal_wavevector(e, kx, k0f)) for e in eps_list], dtype=complex
    )
    eps_values = np.asarray(
        [complex(np.asarray(e).reshape(())) for e in eps_list], dtype=complex
    )
    p = k_z / eps_values

    n_media = len(eps_values)
    amps = np.zeros((n_media, 2), dtype=complex)
    amps[0] = (0.0 + 0.0j, 1.0 + 0.0j)
    for j in range(n_media - 1):
        if j == 0:
            tilde = amps[0]
        else:
            d = float(d_list[j - 1])
            tilde = np.array(
                [amps[j, 0] * np.exp(1j * k_z[j] * d), amps[j, 1] * np.exp(-1j * k_z[j] * d)],
                dtype=complex,
            )
        amps[j + 1] = interface_matrix(p[j + 1], p[j]) @ tilde
        peak = float(np.abs(amps[j + 1]).max())
        if peak > _RENORM_THRESHOLD:
            # A pathologically thick evanescent stack: rescale everything
            # computed so far, which leaves every ratio (and hence the
            # normalised profile) untouched.
            amps[: j + 2] /= peak
    return amps, k_z, p


def mode_field_profile(
    k_x: complex,
    k0: float,
    eps_layers: Sequence[ArrayLike],
    thicknesses: Sequence[float] = (),
    z: Optional[ArrayLike] = None,
    *,
    z0: float = 0.0,
    H0: complex = 1.0,
    h0_at: Optional[float] = None,
    omega: Optional[float] = None,
    eps0: float = EPS0,
) -> ModeFieldProfile:
    r"""
    Reconstruct the field profile of the mode at ``k_x`` through the stack.

    :func:`find_mode` locates a mode as a complex ``k_x``; this turns that
    number back into fields. In medium *j*, with ``ζ = z − z_ref(j)`` measured
    from that medium's reference plane (its lower boundary; the topmost
    interface for the upper half-space),

        H_y(z) = A_j e^{+i k_j ζ} + B_j e^{−i k_j ζ},
        E_x(z) = (k_j / ε_j) (A_j e^{+i k_j ζ} − B_j e^{−i k_j ζ}) / (ω ε₀),
        E_z(z) = −k_x H_y(z) / (ω ε₀ ε_j),

    the last two following from ``∇ × H = −i ω ε₀ ε E`` with
    ``∂_x → i k_x``, ``∂_y → 0``. The amplitudes come from :func:`_mode_amplitudes`.

    Two consequences are worth stating because they are exactly what makes a
    layered stack no harder than a single interface for a solver that knows
    them: ``H_y`` and ``E_x`` are continuous at **every** interface (that is the
    content of ``D(j → j+1)``), and since ``D_z = ε₀ ε E_z = −k_x H_y / ω``
    carries no ``ε`` at all, **``D_z`` is continuous at every interface too**,
    with the whole of the ``E_z`` discontinuity coming from dividing that
    continuous quantity by the local ``ε``.

    Args:
        k_x: The mode's in-plane wavevector (1/m), typically from
            :func:`find_mode`. Any value is accepted; a value that is not a
            root simply produces a large :attr:`ModeFieldProfile.leakage`.
        k0: Free-space wavenumber ω/c (1/m).
        eps_layers, thicknesses: See :func:`stack_matrix`.
        z: Coordinates to evaluate at (m). Defaults to 512 points spanning the
            stack plus one stack-thickness (or one free-space wavelength, if
            the stack has no interior) of each half-space.
        z0: Position of the lowest interface (see :func:`layer_boundaries`).
        H0: Value of ``H_y`` at ``h0_at`` after normalisation (A/m).
        h0_at: Where that normalisation is applied (m). Defaults to the
            **topmost** interface — for a single interface that is the
            interface itself, which is exactly the normalisation of
            :func:`src.analytical.analytical_spp_fields`.
        omega: Angular frequency (rad/s). Defaults to ``k0 * C0``, which is the
            definition of ``k0``; pass it only to match another module's
            rounding.
        eps0: Vacuum permittivity, for the ``E``-field scale.

    Returns:
        :class:`ModeFieldProfile`.

    Raises:
        ValueError: on a malformed stack, or if ``H_y`` is zero or non-finite
            at ``h0_at`` (which would make the normalisation singular).
    """
    eps_list, d_list = _validate_stack(eps_layers, thicknesses)
    k0f = float(k0)
    omega_f = k0f * C0 if omega is None else float(omega)
    bounds = layer_boundaries(d_list, z0)
    amps, k_z, p = _mode_amplitudes(k_x, k0f, eps_list, d_list)
    eps_values = np.asarray(
        [complex(np.asarray(e).reshape(())) for e in eps_list], dtype=complex
    )
    # Reference plane of medium j: its lower boundary; medium 0 and medium 1
    # share the lowest interface, and the top medium uses the topmost one.
    reference = bounds[np.clip(np.arange(len(eps_values)) - 1, 0, None)]

    if z is None:
        pad = float(np.sum(d_list)) if d_list else 2.0 * np.pi / k0f
        z = np.linspace(bounds[0] - pad, bounds[-1] + pad, 512)
    z_arr = np.asarray(z, dtype=float)

    def evaluate(points: np.ndarray) -> Tuple[np.ndarray, ...]:
        idx = layer_index_at(points, d_list, z0)
        zeta = points - reference[idx]
        # Far outside the half-spaces the excluded exponential overflows; that
        # is a caller error, caught by the finiteness check below, not a warning.
        with np.errstate(over="ignore", invalid="ignore"):
            up = amps[idx, 0] * np.exp(1j * k_z[idx] * zeta)
            down = amps[idx, 1] * np.exp(-1j * k_z[idx] * zeta)
        eps_at = eps_values[idx]
        h_y = up + down
        e_x = p[idx] * (up - down) / (omega_f * eps0)
        e_z = -complex(k_x) * h_y / (omega_f * eps0 * eps_at)
        d_z = eps0 * eps_at * e_z
        return h_y, e_x, e_z, d_z, eps_at, idx

    anchor = bounds[-1] if h0_at is None else float(h0_at)
    h_anchor = complex(evaluate(np.asarray([anchor], dtype=float))[0][0])
    if h_anchor == 0.0 or not np.isfinite(h_anchor):
        raise ValueError(
            f"H_y is {h_anchor} at h0_at = {anchor!r}; cannot normalise there"
        )
    scale = complex(H0) / h_anchor

    h_y, e_x, e_z, d_z, eps_at, idx = evaluate(z_arr)
    peak = float(np.abs(amps).max())
    return ModeFieldProfile(
        z=z_arr,
        H_y=scale * h_y,
        E_x=scale * e_x,
        E_z=scale * e_z,
        D_z=scale * d_z,
        eps=eps_at,
        layer_index=idx,
        boundaries=bounds,
        amplitudes=scale * amps,
        k_z=k_z,
        k_x=complex(k_x),
        k0=k0f,
        leakage=float(abs(amps[-1, 1]) / peak) if peak > 0.0 else float("inf"),
    )
