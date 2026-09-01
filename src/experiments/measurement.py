r"""
Two measurements every SPP experiment makes on its trained network.

Both are *diagnostics on a field*, not physics: they take a network that maps
metres to SI fields, probe it on a line or a surface the experiment chooses, and
return numbers for ``metrics.json``. Which line, which surface and which
analytical values to compare against is the experiment's business and stays in
its own file; the arithmetic is here.

:func:`continuity_residuals`
    How badly the tangential fields jump where they should be continuous.
    Reported *relative to the field RMS*, because an absolute residual means
    nothing without a scale — and because it then compares directly across
    experiments whose fields differ by orders of magnitude.

:func:`fit_decay_constants`
    The mode's decay rates, recovered from the slope of ``ln|H_y|`` on each side
    of the interface. This is what says the network found a genuinely *bound*
    mode rather than something that merely fits the anchor: the fitted κ must
    come out **positive** on both sides, and close to the analytical value.

Neither routine knows what a metamaterial is, and neither takes gradients — both
run under ``torch.no_grad``.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from src.models import to_complex

__all__ = ["continuity_residuals", "fit_decay_constants"]

_CPU = torch.device("cpu")


def continuity_residuals(
    net3: nn.Module,
    coords: torch.Tensor,
    normals: torch.Tensor,
    offset: float,
) -> Dict[str, float]:
    r"""
    RMS tangential-continuity residual across an interface, relative to the field RMS.

    The fields are evaluated at ``coords ± offset·n̂`` and compared through
    ``n̂ × ΔE`` and ``n̂ × ΔH``, which vanish for the exact mode. The result is
    divided by the RMS field over the same two sheets, so it reads as a fraction
    rather than a number in V/m.

    An honest caveat, and the reason the offset is a parameter rather than a
    constant: the exact solution does **not** give zero here. ``H_y`` varies by
    ``≈ 2κδ`` across the gap, so every measurement has to be read against that
    floor — the layered experiment's exact TMM mode scores ~0.03, not 0.

    Args:
        net3: ``coords_m (N, 3) -> [N, 6, 2]`` SI fields.
        coords: ``(N, 3)`` points *on* the interface, in metres.
        normals: ``(N, 3)`` unit normals at those points.
        offset: Evaluation half-gap in metres.

    Returns:
        ``continuity_E_rel`` and ``continuity_H_rel``.
    """
    with torch.no_grad():
        E_p, H_p = to_complex(net3(coords + offset * normals))
        E_m, H_m = to_complex(net3(coords - offset * normals))
        n = normals.to(E_p.dtype)
        res_E = torch.linalg.vector_norm(torch.linalg.cross(n, E_p - E_m, dim=1), dim=1)
        res_H = torch.linalg.vector_norm(torch.linalg.cross(n, H_p - H_m, dim=1), dim=1)
        E_rms = torch.sqrt(
            torch.mean(torch.sum(E_p.abs() ** 2 + E_m.abs() ** 2, dim=1) / 2)
        ).clamp_min(1e-30)
        H_rms = torch.sqrt(
            torch.mean(torch.sum(H_p.abs() ** 2 + H_m.abs() ** 2, dim=1) / 2)
        ).clamp_min(1e-30)
    return {
        "continuity_E_rel": (torch.sqrt(torch.mean(res_E**2)) / E_rms).item(),
        "continuity_H_rel": (torch.sqrt(torch.mean(res_H**2)) / H_rms).item(),
    }


def fit_decay_constants(
    net3: nn.Module,
    kappa_d: float,
    kappa_m: float,
    *,
    x: float,
    y: float,
    z_min: float,
    z_max: float,
    guard: float,
    n_line: int = 200,
    device: torch.device = _CPU,
) -> Dict[str, float]:
    r"""
    κ fits from ``ln|H_y|`` vs ``z`` on each side of the interface, at fixed ``(x, y)``.

    Above the interface ``|H_y| ∝ e^{−κ_d z}``, so the slope is ``−κ_d``; below
    it ``|H_y| ∝ e^{+κ_m z}`` and the slope is ``+κ_m``. Both fitted values are
    therefore reported with the sign that makes a **bound** mode positive, which
    is what ``decay_sign_correct_*`` records: a network that has drifted onto
    the radiative branch gets a negative κ, and no amount of small relative
    error would otherwise reveal that.

    The fit spans ``[guard, 0.9 z_max]`` above and ``[0.95 z_min, −guard]``
    below. The guard band excludes the interface itself, where ε jumps; the 0.9
    and 0.95 keep the fit off the domain faces, where the boundary anchor rather
    than the physics sets the field.

    Args:
        net3: ``coords_m (N, 3) -> [N, 6, 2]`` SI fields.
        kappa_d: Analytical air-side decay constant (1/m), for the error.
        kappa_m: Analytical metal-side decay constant (1/m).
        x, y: The line's fixed transverse position, in metres.
        z_min, z_max: The domain's extent in ``z``, in metres.
        guard: Half-width of the excluded band around the interface, in metres.
        n_line: Points per fit.
        device: Where to build the probe line.

    Returns:
        For each of ``kappa_d`` / ``kappa_m``: the fit, its relative error and
        the analytical value; plus ``decay_sign_correct_air`` / ``_metal``.
    """
    out: Dict[str, float] = {}
    for side, z_lo, z_hi, kappa_ref, sign, name in [
        ("air", guard, 0.9 * z_max, kappa_d, -1.0, "kappa_d"),
        ("metal", 0.95 * z_min, -guard, kappa_m, 1.0, "kappa_m"),
    ]:
        z = torch.linspace(z_lo, z_hi, n_line, device=device)
        coords = torch.stack(
            [torch.full_like(z, x), torch.full_like(z, y), z], dim=1
        )
        with torch.no_grad():
            _, H = to_complex(net3(coords))
        log_hy = np.log(np.abs(H[:, 1].cpu().numpy().astype(np.complex128)) + 1e-30)
        slope = float(np.polyfit(z.cpu().numpy().astype(np.float64), log_hy, 1)[0])
        kappa_fit = sign * slope
        out[f"{name}_fit"] = kappa_fit
        out[f"{name}_fit_rel_error"] = float(abs(kappa_fit - kappa_ref) / kappa_ref)
        out[f"{name}_analytical"] = float(kappa_ref)
        out[f"decay_sign_correct_{side}"] = float(kappa_fit > 0)
    return out
