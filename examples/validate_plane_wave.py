"""
Plane Wave Validation for PINN Fundamentals

Trains a PINN to reproduce a free-space plane wave at 1 GHz on the cube
[0, λ]³ and checks it against the closed-form solution. The interior is
constrained by Maxwell's curl and divergence equations only
(:class:`src.models.MaxwellCurlLoss`, :class:`src.models.MaxwellDivergenceLoss`);
the solution is made unique by a *soft Dirichlet* term that pins the fields
on the six faces of the cube to the analytical plane wave (the design doc's
"boundary/data term"). No interior data are used.

Sign convention (as in ``src.models.loss_functions``): time dependence
``exp(-iωt)``, so ``∇×E = iωμ₀H`` and ``∇×H = -iωε₀E``.

Non-dimensionalisation
----------------------
In SI units the two curl residuals differ in scale by the free-space impedance
``η₀ ≈ 377`` (``|∇×E| ~ k₀E₀`` versus ``|∇×H| ~ k₀E₀/η₀``), so the squared
``H`` residual is ~1.4e5 times smaller than the ``E`` residual and Adam
effectively ignores it. The network is therefore trained in a dimensionless
frame::

    x̂ = x / λ,     Ê = E / E₀,     Ĥ = η₀ H / E₀

in which Maxwell's equations read ``∇̂×Ê = i·2π·Ĥ`` and ``∇̂×Ĥ = -i·2π·Ê``,
i.e. exactly ``MaxwellCurlLoss(frequency=2π, mu0=1, eps0=1)``. Both residuals
are then O(1) and equally weighted. :class:`PlaneWavePINN` wraps the trained
core so that it accepts coordinates in metres and returns SI fields
all
validation metrics are computed in SI units with the physical constants from
:mod:`src.constants`.

Usage::

    python examples/validate_plane_wave.py [--epochs 10000] [--n-points 2048]
                                           [--lr 1e-3] [--seed 0] [--device cpu]
                                           [--quick]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analytical import analytical_plane_wave, complex_to_pinn_format  # noqa: E402
from src.constants import C0, EPS0, ETA0, MU0  # noqa: E402
from src.data.domain_sampler import UniformSampler  # noqa: E402
from src.models import (  # noqa: E402
    ElectromagneticPINN,
    MaxwellCurlLoss,
    MaxwellDivergenceLoss,
    to_complex,
)
from src.physics import MaxwellEquations  # noqa: E402

logger = logging.getLogger("validate_plane_wave")

# --------------------------------------------------------------------------- physics
FREQUENCY = 1e9  # Hz
OMEGA = 2 * np.pi * FREQUENCY  # rad/s
C = C0  # m/s
K0 = OMEGA / C  # rad/m
WAVELENGTH = 2 * np.pi / K0  # m (~0.3 m)
E0 = 1.0  # V/m, amplitude of E_y
H0 = E0 / ETA0  # A/m, amplitude of H_z

K_VEC = torch.tensor([K0, 0.0, 0.0], dtype=torch.float32)
E0_POLARIZATION = torch.tensor([0.0, E0, 0.0], dtype=torch.complex64)

# Dimensionless frame: x̂ = x/λ  ->  k̂ = k₀λ = 2π, μ̂₀ = ε̂₀ = 1
K_HAT = 2 * np.pi
K_VEC_HAT = torch.tensor([K_HAT, 0.0, 0.0], dtype=torch.float32)
E0_HAT = torch.tensor([0.0, 1.0, 0.0], dtype=torch.complex64)

# --------------------------------------------------------------------------- defaults
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2048
N_EPOCHS = 10000
LEARNING_RATE = 1e-3
BOUNDARY_WEIGHT = 10.0
DIVERGENCE_WEIGHT = 1.0
QUICK_EPOCHS = 200
LBFGS_STEPS = 25  # outer L-BFGS steps (max_iter=20 evaluations each) after Adam
LBFGS_POINTS_FACTOR = 4  # fixed L-BFGS collocation set = factor * n_points

FIGURES_DIR = REPO_ROOT / "figures" / "plane_wave_validation"
MODEL_PATH = REPO_ROOT / "artifacts" / "models" / "plane_wave_validation.pth"


# --------------------------------------------------------------------------- analytical
def analytical_fields_pinn_format(
    coords: torch.Tensor,
    k_vec: torch.Tensor = K_VEC,
    polarization: torch.Tensor = E0_POLARIZATION,
    omega: float = OMEGA,
    mu0: float = MU0,
) -> torch.Tensor:
    """Analytical ``(E, H)`` at ``coords`` in the network's ``[N, 6, 2]`` layout."""
    E, H = analytical_plane_wave(
        coords, k_vec.to(coords.device), polarization.to(coords.device), omega, mu0=mu0
    )
    return torch.cat([complex_to_pinn_format(E), complex_to_pinn_format(H)], dim=1)


class AnalyticalPlaneWave(nn.Module):
    """Exact SI plane wave as an ``nn.Module`` (coords in metres -> ``[N, 6, 2]``)."""

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return analytical_fields_pinn_format(coords)


# --------------------------------------------------------------------------- network
class PlaneWavePINN(nn.Module):
    """
    SI-unit wrapper around a dimensionless :class:`ElectromagneticPINN` core.

    ``forward(coords_m)`` accepts coordinates in metres and returns SI fields
    ``[N, 6, 2]``
    ``core(coords_m / wavelength)`` returns the dimensionless
    fields ``(Ê, Ĥ)`` used for training.
    """

    def __init__(self, core: nn.Module, wavelength: float = WAVELENGTH, e0: float = E0, h0: float = H0):
        super().__init__()
        self.core = core
        self.wavelength = float(wavelength)
        scale = torch.tensor([e0, e0, e0, h0, h0, h0], dtype=torch.float32).view(1, 6, 1)
        self.register_buffer("field_scale", scale)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.core(coords / self.wavelength) * self.field_scale


def create_network(
    hidden_dims: Tuple[int, ...] = (128, 128, 128, 128),
    fourier_modes: int = 128,
    device: torch.device = DEVICE,
) -> PlaneWavePINN:
    """
    Build the plane-wave PINN (complex-valued MLP with Fourier features).

    The Fourier encoder of :class:`ElectromagneticPINN` samples wavevectors in
    ``[0.1, 20]`` per unit length
    in the dimensionless frame the target
    wavenumber is ``2π ≈ 6.3``, comfortably inside that band.
    """
    core = ElectromagneticPINN(
        spatial_dim=3,
        field_components=6,
        hidden_dims=list(hidden_dims),
        complex_valued=True,
        frequency=OMEGA,  # time-harmonic input layout (3 coordinates)
        use_fourier=True,
        fourier_modes=fourier_modes,
        activation_type="complex_tanh",
    )
    return PlaneWavePINN(core).to(device)


# --------------------------------------------------------------------------- sampling
def sample_collocation_points(n_points: int, device: torch.device = DEVICE) -> torch.Tensor:
    """Uniform interior points in ``[0, λ]³`` (metres), shape ``(n_points, 3)``, ``requires_grad``."""
    sampler = UniformSampler(domain_bounds=[(0.0, WAVELENGTH)] * 3, device=str(device))
    coords = sampler.sample_points(n_points=n_points)["points"]
    coords.requires_grad_(True)
    return coords


def sample_boundary_points(n_points: int, device: torch.device = DEVICE) -> torch.Tensor:
    """Points on the six faces of ``[0, λ]³`` (metres), ``n_points // 6`` per face."""
    per_face = max(1, n_points // 6)
    faces = []
    for axis in range(3):
        for value in (0.0, WAVELENGTH):
            pts = torch.rand(per_face, 3, device=device) * WAVELENGTH
            pts[:, axis] = value
            faces.append(pts)
    return torch.cat(faces, dim=0)


# --------------------------------------------------------------------------- training
def train(
    network: PlaneWavePINN,
    n_epochs: int = N_EPOCHS,
    n_points: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    boundary_weight: float = BOUNDARY_WEIGHT,
    divergence_weight: float = DIVERGENCE_WEIGHT,
    device: torch.device = DEVICE,
    log_every: int = 100,
    lbfgs_steps: int = 0,
) -> Tuple[PlaneWavePINN, Dict[str, list]]:
    """
    Train the dimensionless core with Maxwell curl + divergence residuals in
    the interior and a soft Dirichlet (analytical) term on the boundary.

    Phase 1: ``n_epochs`` of Adam (cosine-annealed LR) on freshly sampled points.
    Phase 2 (if ``lbfgs_steps > 0``): L-BFGS refinement on a fixed set of
    ``LBFGS_POINTS_FACTOR * n_points`` interior points and their boundary set;
    each outer step runs up to 20 function evaluations with strong-Wolfe line
    search. Adam alone stalls around 1e-2 relative residual on this problem.

    Returns the network (weights restored to the lowest-loss iterate) and a
    history dict with ``epoch``, ``total``, ``curl``, ``div``, ``boundary``,
    ``lr`` (one entry per Adam epoch and per L-BFGS outer step).
    """
    core = network.core
    curl_loss = MaxwellCurlLoss(frequency=K_HAT, mu0=1.0, eps0=1.0)
    div_loss = MaxwellDivergenceLoss()

    optimizer = torch.optim.Adam(core.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, n_epochs), eta_min=learning_rate * 1e-2
    )

    history: Dict[str, list] = {"epoch": [], "total": [], "curl": [], "div": [], "boundary": [], "lr": []}
    best_loss = float("inf")
    best_state = {k: v.detach().clone() for k, v in core.state_dict().items()}

    n_boundary = max(6, n_points // 2)
    core.train()
    t0 = time.perf_counter()
    for epoch in range(n_epochs):
        coords_hat = sample_collocation_points(n_points, device=device) / WAVELENGTH
        coords_hat = coords_hat.detach().requires_grad_(True)
        boundary_hat = sample_boundary_points(n_boundary, device=device) / WAVELENGTH
        target_hat = analytical_fields_pinn_format(boundary_hat, K_VEC_HAT, E0_HAT, K_HAT, mu0=1.0)

        optimizer.zero_grad(set_to_none=True)
        l_curl = curl_loss.compute(network=core, coords=coords_hat)
        l_div = div_loss.compute(network=core, coords=coords_hat)
        l_bc = torch.mean((core(boundary_hat) - target_hat) ** 2)
        loss = l_curl + divergence_weight * l_div + boundary_weight * l_bc
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        history["epoch"].append(epoch)
        history["total"].append(loss_val)
        history["curl"].append(l_curl.item())
        history["div"].append(l_div.item())
        history["boundary"].append(l_bc.item())
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if loss_val < best_loss and math.isfinite(loss_val):
            best_loss = loss_val
            best_state = {k: v.detach().clone() for k, v in core.state_dict().items()}

        if epoch % log_every == 0 or epoch == n_epochs - 1:
            logger.info(
                "epoch %5d | total %.3e | curl %.3e | div %.3e | bc %.3e | lr %.2e | %.0fs",
                epoch, loss_val, l_curl.item(), l_div.item(), l_bc.item(),
                optimizer.param_groups[0]["lr"], time.perf_counter() - t0,
            )

    if lbfgs_steps > 0:
        core.load_state_dict(best_state)
        coords_hat = (sample_collocation_points(LBFGS_POINTS_FACTOR * n_points, device=device) / WAVELENGTH)
        coords_hat = coords_hat.detach().requires_grad_(True)
        boundary_hat = sample_boundary_points(LBFGS_POINTS_FACTOR * n_boundary, device=device) / WAVELENGTH
        target_hat = analytical_fields_pinn_format(boundary_hat, K_VEC_HAT, E0_HAT, K_HAT, mu0=1.0)
        lbfgs = torch.optim.LBFGS(
            core.parameters(), lr=1.0, max_iter=20, history_size=50,
            tolerance_grad=1e-12, tolerance_change=1e-14, line_search_fn="strong_wolfe",
        )
        parts: Dict[str, float] = {}

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            l_curl = curl_loss.compute(network=core, coords=coords_hat)
            l_div = div_loss.compute(network=core, coords=coords_hat)
            l_bc = torch.mean((core(boundary_hat) - target_hat) ** 2)
            loss = l_curl + divergence_weight * l_div + boundary_weight * l_bc
            loss.backward()
            parts.update(curl=l_curl.item(), div=l_div.item(), bc=l_bc.item())
            return loss

        for step in range(lbfgs_steps):
            loss_val = float(lbfgs.step(closure).detach())
            epoch = n_epochs + step
            history["epoch"].append(epoch)
            history["total"].append(loss_val)
            history["curl"].append(parts["curl"])
            history["div"].append(parts["div"])
            history["boundary"].append(parts["bc"])
            history["lr"].append(float("nan"))
            if loss_val < best_loss and math.isfinite(loss_val):
                best_loss = loss_val
                best_state = {k: v.detach().clone() for k, v in core.state_dict().items()}
            logger.info(
                "lbfgs %3d | total %.3e | curl %.3e | div %.3e | bc %.3e | %.0fs",
                step, loss_val, parts["curl"], parts["div"], parts["bc"], time.perf_counter() - t0,
            )
            if not math.isfinite(loss_val):
                logger.warning("L-BFGS produced a non-finite loss; stopping refinement")
                break

    core.load_state_dict(best_state)
    logger.info("restored best weights (loss %.3e)", best_loss)
    return network, history


# --------------------------------------------------------------------------- validation
def _relative_l2(pred: torch.Tensor, ref: torch.Tensor) -> float:
    return (torch.linalg.vector_norm(pred - ref) / torch.linalg.vector_norm(ref).clamp_min(1e-30)).item()


def estimate_wavelength(network: nn.Module, n_line: int = 512, pad_factor: int = 64,
                        device: torch.device = DEVICE) -> Dict[str, float]:
    """
    Learned wavelength along x from a line profile of complex ``E_y`` at the
    cube centre: (a) peak of the zero-padded FFT and (b) a linear fit to the
    unwrapped phase (``dφ/dx = k``).
    """
    x = torch.linspace(0.0, WAVELENGTH, n_line + 1, device=device)[:-1]  # one period, periodic grid
    coords = torch.stack([x, torch.full_like(x, WAVELENGTH / 2), torch.full_like(x, WAVELENGTH / 2)], dim=1)
    with torch.no_grad():
        E, _ = to_complex(network(coords))
    ey = E[:, 1].cpu().numpy().astype(np.complex128)
    dx = WAVELENGTH / n_line

    spectrum = np.abs(np.fft.fft(ey, n=n_line * pad_factor))
    freqs = np.fft.fftfreq(n_line * pad_factor, d=dx)
    peak = int(np.argmax(spectrum))
    k_fft = 2 * np.pi * abs(freqs[peak])
    wavelength_fft = 2 * np.pi / k_fft if k_fft > 0 else float("inf")

    phase = np.unwrap(np.angle(ey))
    k_fit = float(np.polyfit(x.cpu().numpy(), phase, 1)[0])
    wavelength_fit = 2 * np.pi / abs(k_fit) if k_fit != 0 else float("inf")

    return {
        "wavelength_fft": float(wavelength_fft),
        "wavelength_fft_rel_error": float(abs(wavelength_fft - WAVELENGTH) / WAVELENGTH),
        "wavelength_phase_fit": float(wavelength_fit),
        "wavelength_phase_fit_rel_error": float(abs(wavelength_fit - WAVELENGTH) / WAVELENGTH),
        "k_fft": float(k_fft),
        "k_phase_fit": float(abs(k_fit)),
        "propagation_direction_sign": float(np.sign(k_fit)),
    }


def validate(network: nn.Module, n_points: int = 20000, device: torch.device = DEVICE) -> Dict[str, float]:
    """
    Compute SI-unit validation metrics on ``n_points`` random interior points.

    Keys: ``rel_l2_E``, ``rel_l2_H``, ``rel_l2_total``, ``curl_E_residual_rel``,
    ``curl_H_residual_rel`` (RMS residual / (k₀ · RMS field)),
    ``curl_E_residual_max_rel``, ``curl_H_residual_max_rel``, ``div_E_rel``,
    ``div_H_rel``, ``E_H_orthogonality``, ``E_k_transversality``,
    ``poynting_alignment``, ``mean_abs_E``, ``mean_abs_H``, ``impedance_ratio``
    and the wavelength keys of :func:`estimate_wavelength`.
    """
    network.eval()
    coords = sample_collocation_points(n_points, device=device)
    fields = network(coords)
    E, H = to_complex(fields)

    maxwell = MaxwellEquations(OMEGA, mu0=MU0, eps0=EPS0)
    curl_E = maxwell.curl_operator(E, coords)
    curl_H = maxwell.curl_operator(H, coords)
    res_E = curl_E - 1j * OMEGA * MU0 * H
    res_H = curl_H + 1j * OMEGA * EPS0 * E
    div_E = maxwell.divergence_operator(E, coords)
    div_H = maxwell.divergence_operator(H, coords)

    with torch.no_grad():
        E_ref, H_ref = analytical_plane_wave(coords, K_VEC.to(device), E0_POLARIZATION.to(device), OMEGA)
        E_rms = torch.sqrt(torch.mean(torch.sum(E.abs() ** 2, dim=1))).clamp_min(1e-30)
        H_rms = torch.sqrt(torch.mean(torch.sum(H.abs() ** 2, dim=1))).clamp_min(1e-30)
        rE = torch.linalg.vector_norm(res_E, dim=1)
        rH = torch.linalg.vector_norm(res_H, dim=1)

        E_mag = torch.linalg.vector_norm(E, dim=1)
        H_mag = torch.linalg.vector_norm(H, dim=1)
        e_dot_h = torch.sum(E * H.conj(), dim=1).abs() / (E_mag * H_mag).clamp_min(1e-30)
        e_dot_k = E[:, 0].abs() / E_mag.clamp_min(1e-30)
        S = MaxwellEquations.poynting_vector(E, H)  # real (N, 3)
        S_norm = torch.linalg.vector_norm(S, dim=1).clamp_min(1e-30)

        metrics = {
            "rel_l2_E": _relative_l2(E, E_ref),
            "rel_l2_H": _relative_l2(H, H_ref),
            "rel_l2_total": _relative_l2(fields, torch.cat([complex_to_pinn_format(E_ref), complex_to_pinn_format(H_ref)], 1)),
            "curl_E_residual_rel": (torch.sqrt(torch.mean(rE**2)) / (K0 * E_rms)).item(),
            "curl_H_residual_rel": (torch.sqrt(torch.mean(rH**2)) / (K0 * H_rms)).item(),
            "curl_E_residual_max_rel": (rE.max() / (K0 * E_rms)).item(),
            "curl_H_residual_max_rel": (rH.max() / (K0 * H_rms)).item(),
            "curl_E_residual_rms_si": torch.sqrt(torch.mean(rE**2)).item(),
            "curl_H_residual_rms_si": torch.sqrt(torch.mean(rH**2)).item(),
            "div_E_rel": (torch.sqrt(torch.mean(div_E.abs() ** 2)) / (K0 * E_rms)).item(),
            "div_H_rel": (torch.sqrt(torch.mean(div_H.abs() ** 2)) / (K0 * H_rms)).item(),
            "E_H_orthogonality": e_dot_h.mean().item(),
            "E_k_transversality": e_dot_k.mean().item(),
            "poynting_alignment": (S[:, 0] / S_norm).mean().item(),
            "mean_abs_E": E_mag.mean().item(),
            "mean_abs_H": H_mag.mean().item(),
            "impedance_ratio": ((E_rms / H_rms) / ETA0).item(),
        }
    metrics.update(estimate_wavelength(network, device=device))
    return metrics


# --------------------------------------------------------------------------- plots
def _grid_slice(n: int, z: float, device: torch.device) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    x = torch.linspace(0.0, WAVELENGTH, n, device=device)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    coords = torch.stack([X.flatten(), Y.flatten(), torch.full_like(X.flatten(), z)], dim=1)
    return coords, X.cpu().numpy(), Y.cpu().numpy()


def visualize(network: nn.Module, history: Optional[Dict[str, list]], metrics: Dict[str, float],
              out_dir: Path = FIGURES_DIR, device: torch.device = DEVICE) -> Dict[str, str]:
    """Write training-curve, field-slice, line-profile, residual and spectrum figures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    network.eval()

    if history is not None and history["epoch"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for key in ("total", "curl", "div", "boundary"):
            ax.semilogy(history["epoch"], history[key], label=key, linewidth=1)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss (dimensionless)")
        ax.set_title("Plane-wave PINN training")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        p = out_dir / "training_history.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths["training_history"] = str(p)

    # Field slice at z = λ/2: predicted, analytical, error for Re(Ey), Re(Hz)
    n = 80
    coords, X, Y = _grid_slice(n, WAVELENGTH / 2, device)
    with torch.no_grad():
        pred = network(coords).cpu().numpy()
        ref = analytical_fields_pinn_format(coords).cpu().numpy()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for row, (idx, name, scale) in enumerate([(1, "Re(Ey) [V/m]", 1.0), (5, "Re(Hz) [A/m]", 1.0)]):
        p_ = pred[:, idx, 0].reshape(n, n) * scale
        r_ = ref[:, idx, 0].reshape(n, n) * scale
        vmax = np.abs(r_).max()
        for col, (data, title, cmap, lim) in enumerate([
            (p_, f"PINN {name}", "RdBu_r", vmax), (r_, f"Analytical {name}", "RdBu_r", vmax),
            (p_ - r_, "Error", "PuOr", np.abs(p_ - r_).max() + 1e-30),
        ]):
            im = axes[row, col].pcolormesh(X, Y, data, cmap=cmap, vmin=-lim, vmax=lim, shading="auto")
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel("x [m]")
            axes[row, col].set_ylabel("y [m]")
            axes[row, col].set_aspect("equal")
            fig.colorbar(im, ax=axes[row, col])
    fig.suptitle("Fields at z = λ/2")
    fig.tight_layout()
    p = out_dir / "field_slices.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["field_slices"] = str(p)

    # Line profile along x at the cube centre
    x = torch.linspace(0.0, WAVELENGTH, 400, device=device)
    line = torch.stack([x, torch.full_like(x, WAVELENGTH / 2), torch.full_like(x, WAVELENGTH / 2)], dim=1)
    with torch.no_grad():
        pl = network(line).cpu().numpy()
        rl = analytical_fields_pinn_format(line).cpu().numpy()
    xs = x.cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(xs, rl[:, 1, 0], "k-", label="Re Ey analytical")
    axes[0].plot(xs, pl[:, 1, 0], "r--", label="Re Ey PINN")
    axes[0].plot(xs, rl[:, 1, 1], "k:", label="Im Ey analytical")
    axes[0].plot(xs, pl[:, 1, 1], "b--", label="Im Ey PINN")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("E_y [V/m]")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].plot(xs, rl[:, 5, 0], "k-", label="Re Hz analytical")
    axes[1].plot(xs, pl[:, 5, 0], "r--", label="Re Hz PINN")
    axes[1].plot(xs, rl[:, 5, 1], "k:", label="Im Hz analytical")
    axes[1].plot(xs, pl[:, 5, 1], "b--", label="Im Hz PINN")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("H_z [A/m]")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.suptitle(f"Line profile (y = z = λ/2); λ_fit/λ = {metrics.get('wavelength_phase_fit', float('nan')) / WAVELENGTH:.4f}")
    fig.tight_layout()
    p = out_dir / "line_profile.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["line_profile"] = str(p)

    # Residual histogram (relative to k0 * field RMS)
    coords = sample_collocation_points(5000, device=device)
    E, H = to_complex(network(coords))
    maxwell = MaxwellEquations(OMEGA, mu0=MU0, eps0=EPS0)
    rE = torch.linalg.vector_norm(maxwell.curl_operator(E, coords) - 1j * OMEGA * MU0 * H, dim=1)
    rH = torch.linalg.vector_norm(maxwell.curl_operator(H, coords) + 1j * OMEGA * EPS0 * E, dim=1)
    with torch.no_grad():
        E_rms = torch.sqrt(torch.mean(torch.sum(E.abs() ** 2, 1)))
        H_rms = torch.sqrt(torch.mean(torch.sum(H.abs() ** 2, 1)))
        rE = (rE / (K0 * E_rms)).cpu().numpy()
        rH = (rH / (K0 * H_rms)).cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.logspace(np.log10(max(min(rE.min(), rH.min()), 1e-12)), np.log10(max(rE.max(), rH.max()) + 1e-12), 50)
    ax.hist(rE, bins=bins, alpha=0.6, label="|∇×E − iωμ₀H| / (k₀|E|rms)")
    ax.hist(rH, bins=bins, alpha=0.6, label="|∇×H + iωε₀E| / (k₀|H|rms)")
    ax.set_xscale("log")
    ax.set_xlabel("relative residual")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("Maxwell curl residuals on 5000 random points")
    fig.tight_layout()
    p = out_dir / "residual_histogram.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["residual_histogram"] = str(p)

    # Spatial spectrum of Ey along x
    n_line, pad = 512, 64
    xl = torch.linspace(0.0, WAVELENGTH, n_line + 1, device=device)[:-1]
    lc = torch.stack([xl, torch.full_like(xl, WAVELENGTH / 2), torch.full_like(xl, WAVELENGTH / 2)], 1)
    with torch.no_grad():
        ey = to_complex(network(lc))[0][:, 1].cpu().numpy()
    spec = np.abs(np.fft.fft(ey, n=n_line * pad))
    k = 2 * np.pi * np.fft.fftfreq(n_line * pad, d=WAVELENGTH / n_line)
    order = np.argsort(k)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(k[order] / K0, spec[order] / spec.max() + 1e-12)
    ax.axvline(1.0, color="k", ls="--", lw=1, label="k₀")
    ax.set_xlim(-4, 4)
    ax.set_xlabel("k_x / k₀")
    ax.set_ylabel("|FFT(Ey)| (normalised)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title(f"Spectrum along x: peak at k/k₀ = {metrics.get('k_fft', float('nan')) / K0:.4f}")
    fig.tight_layout()
    p = out_dir / "spectrum.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["spectrum"] = str(p)
    return paths


# --------------------------------------------------------------------------- main
def success_tier(m: Dict[str, float]) -> str:
    """Classify against the design doc's minimum / target / stretch criteria."""
    res = max(m["curl_E_residual_rel"], m["curl_H_residual_rel"])
    wavelike = m["wavelength_phase_fit_rel_error"] < 0.1
    if res < 1e-8 and m["E_H_orthogonality"] < 1e-3 and m["wavelength_phase_fit_rel_error"] < 1e-3:
        return "stretch"
    if res < 1e-6 and m["E_H_orthogonality"] < 1e-2 and m["wavelength_phase_fit_rel_error"] < 1e-2:
        return "target"
    if res < 1e-4 and wavelike:
        return "minimum"
    return "not met"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=N_EPOCHS)
    p.add_argument("--n-points", type=int, default=BATCH_SIZE, help="interior collocation points per epoch")
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=str(DEVICE))
    p.add_argument("--lbfgs-steps", type=int, default=LBFGS_STEPS, help="L-BFGS outer steps after Adam (0 disables)")
    p.add_argument("--quick", action="store_true", help=f"smoke run: {QUICK_EPOCHS} epochs, 512 points")
    p.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    p.add_argument("--model-out", type=Path, default=MODEL_PATH)
    return p.parse_args(argv)


def main(argv=None) -> Dict[str, float]:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    n_epochs, n_points, lbfgs_steps = (
        (QUICK_EPOCHS, 512, min(args.lbfgs_steps, 2)) if args.quick else (args.epochs, args.n_points, args.lbfgs_steps)
    )

    logger.info("f = %.3e Hz, k0 = %.4f rad/m, λ = %.4f m, E0 = %.1f V/m, H0 = %.4e A/m", FREQUENCY, K0, WAVELENGTH, E0, H0)
    logger.info("device=%s epochs=%d n_points=%d lr=%.1e lbfgs_steps=%d seed=%d",
                device, n_epochs, n_points, args.lr, lbfgs_steps, args.seed)

    # Convention check: the analytical solution must satisfy the loss's Maxwell equations.
    ref_metrics = validate(AnalyticalPlaneWave().to(device), n_points=2000, device=device)
    logger.info("analytical-field residuals (convention check): curl_E %.2e, curl_H %.2e",
                ref_metrics["curl_E_residual_rel"], ref_metrics["curl_H_residual_rel"])

    network = create_network(device=device)
    logger.info("network parameters: %d", sum(p.numel() for p in network.parameters()))

    t0 = time.perf_counter()
    network, history = train(network, n_epochs=n_epochs, n_points=n_points, learning_rate=args.lr,
                             device=device, lbfgs_steps=lbfgs_steps)
    train_time = time.perf_counter() - t0
    logger.info("training time %.1f s", train_time)

    metrics = validate(network, device=device)
    metrics["train_time_s"] = train_time
    metrics["epochs"] = n_epochs
    metrics["n_points"] = n_points
    metrics["lbfgs_steps"] = lbfgs_steps
    metrics["lr"] = args.lr
    metrics["seed"] = args.seed
    metrics["final_loss"] = history["total"][-1]
    metrics["best_loss"] = min(history["total"])
    metrics["success_tier"] = success_tier(metrics)
    for k, v in metrics.items():
        logger.info("%-32s %s", k, f"{v:.4e}" if isinstance(v, float) else v)

    figures = visualize(network, history, metrics, out_dir=args.figures_dir, device=device)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": network.state_dict(), "config": {"hidden_dims": [128] * 4, "fourier_modes": 128,
                "wavelength": WAVELENGTH, "E0": E0, "H0": H0}, "metrics": metrics}, args.model_out)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    with open(args.figures_dir / "metrics.json", "w") as fh:
        json.dump({"metrics": metrics, "analytical_reference": ref_metrics, "figures": figures}, fh, indent=2)
    logger.info("saved model to %s, figures + metrics.json to %s", args.model_out, args.figures_dir)
    logger.info("success tier: %s", metrics["success_tier"])
    return metrics


if __name__ == "__main__":
    main()
