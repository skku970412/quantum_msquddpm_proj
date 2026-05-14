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
