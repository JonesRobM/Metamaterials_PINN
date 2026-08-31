"""
Loss functions for electromagnetic Physics-Informed Neural Networks (PINNs).

Every loss wraps a network that maps coordinates ``(N, D)`` to fields in the
real ``(N, 6, 2)`` format (``[Ex, Ey, Ez, Hx, Hy, Hz]`` × ``[Re, Im]``). The
fields are converted to complex ``(N, 3)`` tensors with
:mod:`src.models.field_format` and all differential operators come from
:mod:`src.physics.differential_ops` via :class:`src.physics.MaxwellEquations`,
so the physics and loss layers share one implementation.

Sign convention: time dependence ``exp(-iωt)``, i.e. ``∇×E = iωμ₀μᵣH`` and
``∇×H = -iωε₀εᵣE``; lossy media have ``Im(ε) > 0``.

Keyword arguments understood by :meth:`EM_CompositeLoss.compute` are forwarded
to each sub-loss only if its ``compute`` signature accepts them; see the
individual ``compute`` docstrings for the required names.
"""

from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from ..constants import EPS0, MU0
from ..physics.boundary_conditions import BoundaryConditions
from ..physics.differential_ops import divergence, divergence_complex, gradient
from ..physics.maxwell_equations import MaxwellEquations
from ..physics.metamaterial import MetamaterialProperties
from .field_format import to_complex

logger = logging.getLogger(__name__)

__all__ = [
    "BaseLoss",
    "MaxwellCurlLoss",
    "MaxwellDivergenceLoss",
    "InterfaceBoundaryLoss",
    "SPPBoundaryLoss",
    "TangentialContinuityLoss",
    "PowerFlowLoss",
    "WaveguideLoss",
    "RadiationLoss",
    "EM_CompositeLoss",
]

PermittivitySpec = Union[None, complex, torch.Tensor, MetamaterialProperties]


def _identity_eps(coords: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(3, device=coords.device, dtype=ref.dtype)
    return eye.unsqueeze(0).expand(coords.shape[0], -1, -1)


def _resolve_permittivity(spec: PermittivitySpec, coords: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Turn a permittivity specification into an ``(N, 3, 3)`` complex tensor.

    ``spec`` may be ``None`` (vacuum), a scalar, a ``(3,)`` diagonal, a ``(3, 3)``
    tensor, an ``(N, 3, 3)`` tensor or a :class:`MetamaterialProperties`.
    """
    if spec is None:
        return _identity_eps(coords, ref)
    if isinstance(spec, MetamaterialProperties):
        return spec.permittivity_tensor(coords).to(ref.dtype)
    if isinstance(spec, (int, float, complex)):
        return _identity_eps(coords, ref) * complex(spec)
    t = torch.as_tensor(spec, device=coords.device).to(ref.dtype)
    if t.dim() == 1 and t.shape[0] == 3:
        t = torch.diag_embed(t)
    if t.dim() == 2:
        t = t.unsqueeze(0).expand(coords.shape[0], -1, -1)
    if t.shape != (coords.shape[0], 3, 3):
        raise ValueError(f"Unsupported permittivity shape {tuple(t.shape)}")
    return t


def _evaluate(network: nn.Module, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the network and return complex ``(E, H)``."""
    return to_complex(network(coords))


class BaseLoss(ABC):
    """Abstract base class for electromagnetic PINN loss components."""

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    @abstractmethod
    def compute(self, *args, **kwargs) -> torch.Tensor:
        """Compute the (unweighted) loss value."""

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.weight * self.compute(*args, **kwargs)


class MaxwellCurlLoss(BaseLoss):
    """
    Curl-equation residuals ``∇×E − iωμ₀μᵣH`` and ``∇×H + iωε₀εᵣE``.

    Args:
        frequency: Angular frequency ω (rad/s).
        mu0: Vacuum permeability.
        eps0: Vacuum permittivity.
        weight: Loss weight.
    """

    def __init__(self, frequency: float, mu0: float = MU0, eps0: float = EPS0, weight: float = 1.0):
        super().__init__(weight)
        self.omega = frequency
        self.mu0 = mu0
        self.eps0 = eps0
        self.maxwell_solver = MaxwellEquations(frequency, mu0=mu0, eps0=eps0)

    def compute(
        self,
        network: nn.Module,
        coords: torch.Tensor,
        material_props: Optional[torch.Tensor] = None,
        epsilon: PermittivitySpec = None,
        mu_r: Union[complex, torch.Tensor] = 1.0,
    ) -> torch.Tensor:
        """
        Args:
            network: Maps ``coords`` to fields ``(N, 6, 2)``.
            coords: Collocation points ``(N, D)``; gradients are enabled in place.
            material_props: Legacy ``(N, k, 2)`` tensor: row 0 is ``μᵣ`` (re, im) and,
                if ``k >= 2``, row 1 is a scalar ``εᵣ`` (re, im). Ignored when
                ``epsilon`` is given.
            epsilon: Relative permittivity (``None`` = vacuum, scalar, ``(3,)``,
                ``(3, 3)``, ``(N, 3, 3)`` or :class:`MetamaterialProperties`).
            mu_r: Relative permeability (scalar or ``(N,)``/``(N, 1)`` tensor).

        Returns:
            Scalar loss: mean squared magnitude of both residual vectors.
        """
        coords.requires_grad_(True)
        E, H = _evaluate(network, coords)

        mu = mu_r
        if material_props is not None:
            mu = torch.complex(material_props[:, 0, 0], material_props[:, 0, 1]).unsqueeze(1)
            if epsilon is None and material_props.shape[1] >= 2:
                epsilon = torch.complex(material_props[:, 1, 0], material_props[:, 1, 1])
                epsilon = epsilon.unsqueeze(1).unsqueeze(2) * torch.eye(3, device=E.device, dtype=E.dtype)
        if isinstance(mu, torch.Tensor) and mu.dim() == 1:
            mu = mu.unsqueeze(1)

        eps_tensor = _resolve_permittivity(epsilon, coords, E)

        curl_E = self.maxwell_solver.curl_operator(E, coords)
        curl_H = self.maxwell_solver.curl_operator(H, coords)
        eps_E = torch.einsum("nij,nj->ni", eps_tensor, E)

        residual_E = curl_E - 1j * self.omega * self.mu0 * mu * H
        residual_H = curl_H + 1j * self.omega * self.eps0 * eps_E

        return torch.mean(residual_E.abs() ** 2) + torch.mean(residual_H.abs() ** 2)


class MaxwellDivergenceLoss(BaseLoss):
    """
    Divergence constraints ``∇·(εᵣE) = ρ/ε₀`` and ``∇·H = 0``.

    Args:
        weight: Loss weight.
    """

    def compute(
        self,
        network: nn.Module,
        coords: torch.Tensor,
        epsilon: PermittivitySpec = None,
        charge_density: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            network, coords: As for :class:`MaxwellCurlLoss`.
            epsilon: Relative permittivity specification (``None`` = vacuum).
            charge_density: Optional ``ρ/ε₀`` (complex or real, shape ``(N,)``).
        """
        coords.requires_grad_(True)
        E, H = _evaluate(network, coords)
        eps_tensor = _resolve_permittivity(epsilon, coords, E)

        div_D = divergence_complex(torch.einsum("nij,nj->ni", eps_tensor, E), coords)
        div_H = divergence_complex(H, coords)

        if charge_density is not None:
            div_D = div_D - charge_density.reshape(-1).to(div_D.dtype)

        return torch.mean(div_D.abs() ** 2) + torch.mean(div_H.abs() ** 2)


class MetamaterialConstitutiveLoss(BaseLoss):
    """
    Placeholder for a constitutive-relation loss ``D = ε·E``, ``B = μ·H``.

    The current networks output only ``E`` and ``H``, so ``D`` and ``B`` are
    defined *by* the constitutive relations and there is nothing independent to
    penalise. This class is retained for API stability but is not exported
    and raises when used.
    """

    def __init__(self, metamaterial_solver: Optional[MetamaterialProperties] = None, weight: float = 1.0):
        super().__init__(weight)
        self.metamaterial = metamaterial_solver

    def compute(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError(
            "MetamaterialConstitutiveLoss requires a network that predicts D and B "
            "independently of E and H; no such architecture exists in this repository. "
            "Material response is enforced through MaxwellCurlLoss/MaxwellDivergenceLoss "
            "via the epsilon argument."
        )


class InterfaceBoundaryLoss(BaseLoss):
    """
    Continuity of tangential ``E``, tangential ``H``, normal ``εᵣE`` and normal
    ``μᵣH`` across a planar interface.

    The network (or a pair of networks) is evaluated at ``interface_coords ± δ n``
    where ``n`` is the normal of ``boundary_solver`` (pointing from medium 1 into
    medium 2). Normal-component residuals are computed in relative units so all
    four terms have comparable magnitude.

    Args:
        boundary_solver: :class:`BoundaryConditions` defining the normal.
        eps_medium_1, eps_medium_2: Permittivity of each side (see
            :class:`MaxwellCurlLoss` for accepted forms).
        offset: Evaluation offset δ (m) on either side of the interface.
        interface_coords: Optional fixed interface points used when none are
            passed to :meth:`compute`.
        weight: Loss weight.
    """

    def __init__(
        self,
        boundary_solver: Optional[BoundaryConditions] = None,
        eps_medium_1: PermittivitySpec = None,
        eps_medium_2: PermittivitySpec = None,
        offset: float = 1e-9,
        interface_coords: Optional[torch.Tensor] = None,
        weight: float = 1.0,
    ):
        super().__init__(weight)
        self.boundary_solver = boundary_solver or BoundaryConditions()
        self.eps_1 = eps_medium_1
        self.eps_2 = eps_medium_2
        self.offset = offset
        self.interface_coords = interface_coords

    def compute(
        self,
        network: nn.Module,
        interface_coords: Optional[torch.Tensor] = None,
        network_2: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        Args:
            network: Network for medium 1 (and medium 2 if ``network_2`` is None).
            interface_coords: Points on the interface ``(N, D)``; defaults to the
                constructor value.
            network_2: Optional separate network for medium 2.
        """
        pts = interface_coords if interface_coords is not None else self.interface_coords
        if pts is None:
            raise ValueError("InterfaceBoundaryLoss needs interface_coords")

        n = self.boundary_solver.interface_normal.to(pts.device, pts.dtype)[: pts.shape[1]]
        coords_1 = pts - self.offset * n
        coords_2 = pts + self.offset * n

        E1, H1 = _evaluate(network, coords_1)
        E2, H2 = _evaluate(network_2 or network, coords_2)
        eps1 = _resolve_permittivity(self.eps_1, coords_1, E1)
        eps2 = _resolve_permittivity(self.eps_2, coords_2, E2)

        bc = self.boundary_solver
        residuals = torch.cat(
            [
                bc.tangential_E_continuity(E1, E2),
                bc.tangential_H_continuity(H1, H2),
                bc.normal_D_continuity(E1, E2, eps1, eps2, relative=True),
                bc.normal_B_continuity(H1, H2, relative=True),
            ],
            dim=1,
        )
        return torch.mean(residuals**2)


class SPPBoundaryLoss(BaseLoss):
    """
    Soft constraint that ``|E|`` decays as ``exp(−|z| / decay_length)`` away
    from the interface at ``z = 0``.

    Args:
        spp_wavevector: Expected SPP wavevector (stored for reference).
        decay_length: Expected field decay length (m).
        weight: Loss weight.
    """

    def __init__(self, spp_wavevector: float, decay_length: float = 1e-6, weight: float = 1.0):
        super().__init__(weight)
        self.k_spp = spp_wavevector
        self.decay_length = decay_length

    def compute(self, network: nn.Module, coords: torch.Tensor) -> torch.Tensor:
        E, _ = _evaluate(network, coords)
        z_coords = coords[:, 2] if coords.shape[1] > 2 else coords[:, -1]

        field_magnitude = torch.linalg.vector_norm(E, dim=1)
        max_field = torch.max(field_magnitude).clamp_min(torch.finfo(field_magnitude.dtype).tiny)
        expected_decay = torch.exp(-torch.abs(z_coords) / self.decay_length)
        decay_residual = field_magnitude / max_field - expected_decay
        return torch.mean(decay_residual**2)


class TangentialContinuityLoss(BaseLoss):
    """
    Tangential continuity ``n × (E₂ − E₁) = 0`` and ``n × (H₂ − H₁) = 0`` for a
    single network evaluated on both sides of an interface with per-point normals.

    Args:
        offset: Evaluation offset on either side (m).
        weight: Loss weight.
    """

    def __init__(self, offset: float = 1e-6, weight: float = 1.0):
        super().__init__(weight)
        self.offset = offset

    def compute(
        self, network: nn.Module, interface_coords: torch.Tensor, normal_vectors: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            interface_coords: Interface points ``(N, 3)``.
            normal_vectors: Unit normals ``(N, 3)``.
        """
        E_plus, H_plus = _evaluate(network, interface_coords + self.offset * normal_vectors)
        E_minus, H_minus = _evaluate(network, interface_coords - self.offset * normal_vectors)

        n = normal_vectors.to(E_plus.dtype)
        n_cross_E = torch.linalg.cross(n, E_plus - E_minus, dim=1)
        n_cross_H = torch.linalg.cross(n, H_plus - H_minus, dim=1)
        return torch.mean(n_cross_E.abs() ** 2) + torch.mean(n_cross_H.abs() ** 2)


class PowerFlowLoss(BaseLoss):
    """
    Power conservation ``∇·S = 0`` for the time-averaged Poynting vector
    ``S = ½ Re(E × H*)``. Exact in lossless, source-free regions; in lossy
    media ``∇·S = −½ ωε₀ Im(εᵣ)|E|²`` and this loss should be down-weighted.
    """

    def compute(self, network: nn.Module, coords: torch.Tensor) -> torch.Tensor:
        coords.requires_grad_(True)
        E, H = _evaluate(network, coords)
        S = MaxwellEquations.poynting_vector(E, H)  # real (N, 3)
        div_S = divergence(S, coords)
        return torch.mean(div_S**2)


class WaveguideLoss(BaseLoss):
    """
    Guided-mode constraint: the phase of ``E_x`` advances as ``exp(iβx)`` along
    the propagation direction, i.e. ``∂ arg(E_x)/∂x = β``.

    Args:
        propagation_direction: Coordinate index of the propagation axis.
        weight: Loss weight.
    """

    def __init__(self, propagation_direction: int = 0, weight: float = 1.0):
        super().__init__(weight)
        self.prop_dir = propagation_direction

    def compute(self, network: nn.Module, coords: torch.Tensor, beta: float) -> torch.Tensor:
        coords.requires_grad_(True)
        E, _ = _evaluate(network, coords)
        Ex = E[:, 0]
        # d(arg E)/dx = Im(conj(E) dE/dx) / |E|^2, avoiding the branch cut of angle()
        dEx = torch.complex(gradient(Ex.real, coords), gradient(Ex.imag, coords))[:, self.prop_dir]
        mag_sq = Ex.abs() ** 2 + 1e-30
        phase_grad = (Ex.conj() * dEx).imag / mag_sq
        return torch.mean((phase_grad - beta) ** 2)


class RadiationLoss(BaseLoss):
    """
    First-order Sommerfeld radiation condition ``∂E/∂r − ik₀E = 0`` on an
    outer boundary, applied component-wise to the complex field.
    """

    def compute(self, network: nn.Module, boundary_coords: torch.Tensor, k0: float) -> torch.Tensor:
        boundary_coords.requires_grad_(True)
        E, _ = _evaluate(network, boundary_coords)

        r = torch.linalg.vector_norm(boundary_coords, dim=1, keepdim=True)
        radial_dir = torch.where(r > 0, boundary_coords / r.clamp_min(torch.finfo(r.dtype).tiny), 0.0)
        d = boundary_coords.shape[1]

        residuals = []
        for i in range(3):
            grad_c = torch.complex(gradient(E[:, i].real, boundary_coords), gradient(E[:, i].imag, boundary_coords))
            dE_dr = torch.sum(grad_c[:, :d] * radial_dir.to(grad_c.dtype), dim=1)
            residuals.append(dE_dr - 1j * k0 * E[:, i])
        residual = torch.stack(residuals, dim=1)
        return torch.mean(residual.abs() ** 2)


class EM_CompositeLoss:
    """
    Weighted sum of electromagnetic loss components with optional adaptive
    re-weighting.

    ``compute(**kwargs)`` forwards to each sub-loss only the keyword arguments
    its ``compute`` signature declares, so heterogeneous losses can share one
    call. Typical keys: ``network``, ``coords``, ``interface_coords``,
    ``normal_vectors``, ``epsilon``, ``beta``, ``boundary_coords``, ``k0``.

    When ``adaptive_weights`` is True, every ``update_interval`` steps each
    component's ``weight`` is rescaled toward the mean running loss so that no
    single term dominates; Maxwell/curl terms receive a 1.5× and boundary terms
    a 1.2× priority factor. The rescaled weights take effect from the next step.

    Args:
        losses: Mapping name → loss component.
        adaptive_weights: Enable adaptive re-weighting.
        update_interval: Steps between weight updates.
    """

    def __init__(
        self,
        losses: Dict[str, BaseLoss],
        adaptive_weights: bool = True,
        update_interval: int = 50,
        frequency_dependent: bool = False,
    ):
        self.losses = losses
        self.adaptive_weights = adaptive_weights
        self.update_interval = update_interval
        self.frequency_dependent = frequency_dependent
        self.step_count = 0
        self.loss_history: Dict[str, list] = {name: [] for name in losses}
        self.alpha = 0.9
        self.running_means = {name: 1.0 for name in losses}
        self._signatures = {name: inspect.signature(fn.compute) for name, fn in losses.items()}

    def _select_kwargs(self, name: str, kwargs: dict) -> dict:
        sig = self._signatures[name]
        params = sig.parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return kwargs
        selected = {k: v for k, v in kwargs.items() if k in params}
        missing = [
            k
            for k, p in params.items()
            if p.default is inspect.Parameter.empty
            and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            and k not in selected
        ]
        if missing:
            raise TypeError(f"Loss '{name}' requires keyword argument(s) {missing} not supplied to compute()")
        return selected

    def compute(self, **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return ``(total_loss, {name: weighted component loss})``."""
        loss_dict: Dict[str, torch.Tensor] = {}
        for name, loss_fn in self.losses.items():
            component_loss = loss_fn(**self._select_kwargs(name, kwargs))
            loss_dict[name] = component_loss
            self.loss_history[name].append(float(component_loss.detach()))
            logger.debug("loss %s = %.4e", name, self.loss_history[name][-1])

        total_loss = sum(loss_dict.values())

        if self.adaptive_weights and self.step_count % self.update_interval == 0:
            self._update_adaptive_weights()

        self.step_count += 1
        return total_loss, loss_dict

    def _update_adaptive_weights(self) -> None:
        for name in self.losses:
            if self.loss_history[name]:
                recent = float(np.mean(self.loss_history[name][-10:]))
                self.running_means[name] = self.alpha * self.running_means[name] + (1 - self.alpha) * recent

        mean_loss = float(np.mean(list(self.running_means.values())))
        for name, loss_fn in self.losses.items():
            if self.running_means[name] > 0:
                weight = mean_loss / self.running_means[name]
                lname = name.lower()
                if "maxwell" in lname or "curl" in lname:
                    weight *= 1.5
                elif "boundary" in lname:
                    weight *= 1.2
                loss_fn.weight = weight

    def get_physics_residuals(self) -> Dict[str, float]:
        """Most recent value of each component."""
        return {name: hist[-1] for name, hist in self.loss_history.items() if hist}
