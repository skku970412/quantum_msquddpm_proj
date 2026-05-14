from __future__ import annotations

import torch

from .utils import COMPLEX_DTYPE, normalize_statevectors, statevectors_to_density


def linear_beta_schedule(
    steps: int,
    beta_start: float,
    beta_end: float,
    device: torch.device,
) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, steps, dtype=torch.float32, device=device)


def alpha_bar_schedule(betas: torch.Tensor) -> torch.Tensor:
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)


def effective_depolarizing_beta(
    betas: torch.Tensor,
    t: torch.Tensor,
    mode: str = "single_beta",
) -> torch.Tensor:
    t = t.long()
    if mode == "single_beta":
        return betas[t]
    if mode == "cumulative":
        return 1.0 - alpha_bar_schedule(betas)[t]
    raise ValueError(f"Unknown depolarizing mode: {mode}")


def depolarizing_noise(rho: torch.Tensor, beta: torch.Tensor | float) -> torch.Tensor:
    batch_size, dim, _ = rho.shape
    beta = torch.as_tensor(beta, dtype=torch.float32, device=rho.device)
    if beta.ndim == 0:
        beta = beta.expand(batch_size)
    beta = beta.reshape(batch_size, 1, 1)
    eye = torch.eye(dim, dtype=COMPLEX_DTYPE, device=rho.device).unsqueeze(0)
    return ((1.0 - beta) * rho + beta * eye / dim).to(COMPLEX_DTYPE)


def depolarizing_forward_from_states(
    states: torch.Tensor,
    t: torch.Tensor,
    betas: torch.Tensor,
    mode: str = "single_beta",
) -> torch.Tensor:
    clean_rho = statevectors_to_density(states)
    beta_t = effective_depolarizing_beta(betas, t=t, mode=mode)
    return depolarizing_noise(clean_rho, beta_t)


def statevector_depolarizing_proxy(
    states: torch.Tensor,
    t: torch.Tensor,
    betas: torch.Tensor,
    mode: str = "single_beta",
) -> torch.Tensor:
    beta_t = effective_depolarizing_beta(betas, t=t, mode=mode).reshape(-1, 1).to(states.real.dtype)
    noise = torch.randn_like(states.real) + 1j * torch.randn_like(states.real)
    noise = normalize_statevectors(noise.to(COMPLEX_DTYPE))
    mixed = torch.sqrt(1.0 - beta_t) * states + torch.sqrt(beta_t) * noise
    return normalize_statevectors(mixed)


def expected_fidelity_under_depolarizing(
    betas: torch.Tensor,
    dim: int,
    mode: str = "single_beta",
) -> torch.Tensor:
    if mode == "single_beta":
        return 1.0 - betas + betas / dim
    if mode == "cumulative":
        alpha_bar = alpha_bar_schedule(betas)
        return alpha_bar + (1.0 - alpha_bar) / dim
    raise ValueError(f"Unknown depolarizing mode: {mode}")


def fidelity_under_depolarizing(
    states: torch.Tensor,
    betas: torch.Tensor,
    mode: str = "single_beta",
) -> torch.Tensor:
    return expected_fidelity_under_depolarizing(betas=betas, dim=states.shape[-1], mode=mode)


def measure_target_noisy_fidelity(
    states: torch.Tensor,
    noisy_rhos_or_states: torch.Tensor,
    input_mode: str = "density",
) -> tuple[float, float]:
    states = normalize_statevectors(states)
    if input_mode == "density":
        values = torch.einsum(
            "bi,bij,bj->b",
            states.conj(),
            noisy_rhos_or_states,
            states,
        ).real.clamp_min(0.0)
    elif input_mode == "statevector":
        noisy = normalize_statevectors(noisy_rhos_or_states)
        overlap = torch.sum(states.conj() * noisy, dim=-1)
        values = torch.abs(overlap).pow(2).real
    else:
        raise ValueError(f"Unknown input_mode: {input_mode}")
    return float(values.mean().detach().cpu()), float(values.std(unbiased=False).detach().cpu())


def calibrate_depolarizing_beta_for_target_fidelity(
    target_fidelity: float,
    dim: int,
    mode: str = "single_beta",
) -> tuple[float, float]:
    if mode != "single_beta":
        raise ValueError("calibrate_depolarizing_beta_for_target_fidelity only supports single_beta")
    denom = 1.0 - (1.0 / max(dim, 1))
    if denom <= 0.0:
        return 0.0, 1.0
    beta = (1.0 - float(target_fidelity)) / denom
    beta = float(max(0.0, min(1.0, beta)))
    expected = 1.0 - beta + beta / dim
    return beta, float(expected)


def calibrate_cumulative_beta_end_for_target_fidelity(
    target_fidelity: float,
    noise_steps: int,
    beta_start: float,
    dim: int,
) -> tuple[float, float, float]:
    target = float(max(1.0 / max(dim, 1), min(1.0, target_fidelity)))
    low = max(0.0, beta_start)
    high = 0.999
    best_beta_end = high
    best_alpha_bar = 0.0
    best_fidelity = 1.0 / max(dim, 1)
    device = torch.device("cpu")
    for _ in range(60):
        beta_end = 0.5 * (low + high)
        betas = linear_beta_schedule(noise_steps, beta_start, beta_end, device=device)
        alpha_bar_final = float(alpha_bar_schedule(betas)[-1].item())
        fidelity = float(alpha_bar_final + (1.0 - alpha_bar_final) / dim)
        if fidelity > target:
            low = beta_end
        else:
            high = beta_end
        best_beta_end = beta_end
        best_alpha_bar = alpha_bar_final
        best_fidelity = fidelity
    return float(best_beta_end), float(best_alpha_bar), float(best_fidelity)
