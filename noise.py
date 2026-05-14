from __future__ import annotations

import torch

from utils import COMPLEX_DTYPE, normalize_statevectors, statevectors_to_density


def linear_beta_schedule(
    steps: int,
    beta_start: float,
    beta_end: float,
    device: torch.device,
) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, steps, dtype=torch.float32, device=device)


def depolarizing_noise(rho: torch.Tensor, beta: torch.Tensor | float) -> torch.Tensor:
    batch_size, dim, _ = rho.shape
    beta = torch.as_tensor(beta, dtype=torch.float32, device=rho.device)
    if beta.ndim == 0:
        beta = beta.expand(batch_size)
    beta = beta.reshape(batch_size, 1, 1)
    eye = torch.eye(dim, dtype=COMPLEX_DTYPE, device=rho.device).unsqueeze(0)
    return ((1.0 - beta) * rho + beta * eye / dim).to(COMPLEX_DTYPE)


def depolarizing_forward_from_states(
    states: torch.Tensor, t: torch.Tensor, betas: torch.Tensor
) -> torch.Tensor:
    clean_rho = statevectors_to_density(states)
    beta_t = betas[t.long()]
    return depolarizing_noise(clean_rho, beta_t)


def statevector_depolarizing_proxy(
    states: torch.Tensor, t: torch.Tensor, betas: torch.Tensor
) -> torch.Tensor:
    beta_t = betas[t.long()].reshape(-1, 1).to(states.real.dtype)
    noise = torch.randn_like(states.real) + 1j * torch.randn_like(states.real)
    noise = normalize_statevectors(noise.to(COMPLEX_DTYPE))
    mixed = torch.sqrt(1.0 - beta_t) * states + torch.sqrt(beta_t) * noise
    return normalize_statevectors(mixed)


def fidelity_under_depolarizing(states: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
    dim = states.shape[-1]
    # For pure rho, <psi|((1-beta)rho + beta I/dim)|psi> = 1-beta + beta/dim.
    return 1.0 - betas + betas / dim
