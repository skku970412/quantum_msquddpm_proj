from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch

from .config import ExperimentConfig
from .datasets import build_dataset
from .metrics import (
    differentiable_mmd_fidelity_loss,
    mmd_fidelity,
    mean_nearest_fidelity,
    nearest_fidelity_loss,
    pure_density_fidelity,
    pure_state_fidelity,
    resource_metrics,
    wasserstein_fidelity_distance,
)
from .models import build_model
from .noise import (
    alpha_bar_schedule,
    depolarizing_forward_from_states,
    depolarizing_noise,
    linear_beta_schedule,
    statevector_depolarizing_proxy,
)
from .random_unitary import random_unitary_forward_from_states, random_unitary_scramble_states
from .utils import (
    COMPLEX_DTYPE,
    normalize_statevectors,
    resolve_device,
    set_seed,
    statevectors_to_density,
    timer,
)


@dataclass
class TrainingOutput:
    metrics: dict
    losses: pd.DataFrame


def _batch_indices(num_items: int, batch_size: int, device: torch.device) -> list[torch.Tensor]:
    perm = torch.randperm(num_items, device=device)
    return [perm[start : start + batch_size] for start in range(0, num_items, batch_size)]


def _forward_process(
    clean_states: torch.Tensor,
    t: torch.Tensor,
    model_name: str,
    betas: torch.Tensor,
    noise_steps: int,
    depth: int,
    qubits: int,
    input_mode: str,
    depolarizing_mode: str,
) -> torch.Tensor:
    if model_name == "quddpm_baseline":
        if input_mode == "statevector":
            return random_unitary_scramble_states(
                clean_states, t=t, noise_steps=noise_steps, depth=depth, qubits=qubits
            )
        return random_unitary_forward_from_states(
            clean_states, t=t, noise_steps=noise_steps, depth=depth, qubits=qubits
        )
    if input_mode == "statevector":
        return statevector_depolarizing_proxy(
            clean_states,
            t=t,
            betas=betas,
            mode=depolarizing_mode,
        )
    return depolarizing_forward_from_states(
        clean_states,
        t=t,
        betas=betas,
        mode=depolarizing_mode,
    )


def sample_prior_states(
    batch_size: int,
    qubits: int,
    device: torch.device,
    mode: str,
    depolarizing_beta: float = 0.85,
) -> torch.Tensor:
    dim = 2**qubits
    real = torch.randn(batch_size, dim, device=device)
    imag = torch.randn(batch_size, dim, device=device)
    states = normalize_statevectors((real + 1j * imag).to(COMPLEX_DTYPE))
    if mode in {"random_pure", "maximally_mixed_jitter"}:
        return states
    if mode == "depolarized_random":
        noise = normalize_statevectors(
            (torch.randn(batch_size, dim, device=device) + 1j * torch.randn(batch_size, dim, device=device)).to(
                COMPLEX_DTYPE
            )
        )
        beta = depolarizing_beta
        mixed = torch.sqrt(torch.tensor(1.0 - beta, device=device)) * states
        mixed = mixed + torch.sqrt(torch.tensor(beta, device=device)) * noise
        return normalize_statevectors(mixed)
    raise ValueError(f"Unknown prior mode: {mode}")


def sample_prior_density(
    batch_size: int,
    qubits: int,
    device: torch.device,
    mode: str,
    mixed_epsilon: float = 0.05,
    depolarizing_beta: float = 0.85,
) -> torch.Tensor:
    pure_states = sample_prior_states(
        batch_size=batch_size,
        qubits=qubits,
        device=device,
        mode="random_pure",
    )
    pure_rho = statevectors_to_density(pure_states)
    if mode == "random_pure":
        return pure_rho

    dim = 2**qubits
    eye = torch.eye(dim, dtype=COMPLEX_DTYPE, device=device).unsqueeze(0)
    maximally_mixed = eye / dim
    if mode == "maximally_mixed_jitter":
        eps = torch.as_tensor(mixed_epsilon, dtype=torch.float32, device=device)
        return (eps * pure_rho + (1.0 - eps) * maximally_mixed).to(COMPLEX_DTYPE)
    if mode == "depolarized_random":
        return depolarizing_noise(pure_rho, depolarizing_beta)
    raise ValueError(f"Unknown prior mode: {mode}")


def iterative_reverse_sample(
    model: torch.nn.Module,
    prior_input: torch.Tensor,
    noise_steps: int,
    qubits: int,
    input_mode: str,
) -> torch.Tensor:
    del qubits
    current = prior_input
    generated_states = None
    for step in reversed(range(noise_steps)):
        t = torch.full((current.shape[0],), step, dtype=torch.long, device=current.device)
        generated_states = model(current, t)
        current = generated_states if input_mode == "statevector" else statevectors_to_density(generated_states)
    if generated_states is None:
        raise RuntimeError("iterative_reverse_sample requires noise_steps >= 1")
    return generated_states


@torch.no_grad()
def evaluate_reconstruction(
    model: torch.nn.Module,
    test_states: torch.Tensor,
    model_name: str,
    betas: torch.Tensor,
    noise_steps: int,
    depth: int,
    qubits: int,
    config: ExperimentConfig,
) -> dict[str, float]:
    if model_name == "cnr":
        return {
            "reconstruction_fidelity": float("nan"),
            "reconstruction_pure_state_fidelity": float("nan"),
            "reconstruction_mmd": float("nan"),
            "reconstruction_wasserstein": float("nan"),
        }

    model.eval()
    subset = min(config.eval_subset, test_states.shape[0])
    target = test_states[:subset]
    t = torch.full((target.shape[0],), noise_steps - 1, dtype=torch.long, device=target.device)
    noisy_input = _forward_process(
        target,
        t=t,
        model_name=model_name,
        betas=betas,
        noise_steps=noise_steps,
        depth=depth,
        qubits=qubits,
        input_mode=config.input_mode,
        depolarizing_mode=config.depolarizing_mode,
    )
    generated_states = model(noisy_input, t)
    generated_rho = statevectors_to_density(generated_states)
    fidelity = pure_density_fidelity(target, generated_rho).mean().item()
    pure_fidelity = pure_state_fidelity(target, generated_states).mean().item()
    return {
        "reconstruction_fidelity": float(fidelity),
        "reconstruction_pure_state_fidelity": float(pure_fidelity),
        "reconstruction_mmd": mmd_fidelity(target, generated_states, subset=config.mmd_subset),
        "reconstruction_wasserstein": wasserstein_fidelity_distance(
            target,
            generated_states,
            subset=config.wasserstein_subset,
        ),
    }


@torch.no_grad()
def evaluate_generation(
    model: torch.nn.Module,
    test_states: torch.Tensor,
    model_name: str,
    betas: torch.Tensor,
    qubits: int,
    config: ExperimentConfig,
) -> dict[str, float]:
    model.eval()
    subset = min(config.eval_subset, test_states.shape[0])
    target = test_states[:subset]

    if model_name == "cnr":
        latent_dim = getattr(model, "latent_dim", max(4, qubits * 2))
        latent = torch.randn(subset, latent_dim, device=target.device)
        generated_states = model(latent)
        prior_mode = "latent_gaussian"
        sampling_mode = "one_step"
    else:
        if config.input_mode == "statevector":
            prior_input = sample_prior_states(
                batch_size=subset,
                qubits=qubits,
                device=target.device,
                mode=config.prior_mode,
                depolarizing_beta=config.prior_depolarizing_beta,
            )
        else:
            prior_input = sample_prior_density(
                batch_size=subset,
                qubits=qubits,
                device=target.device,
                mode=config.prior_mode,
                mixed_epsilon=config.prior_mixed_epsilon,
                depolarizing_beta=config.prior_depolarizing_beta,
            )
        if config.generation_sampling_mode == "iterative":
            generated_states = iterative_reverse_sample(
                model=model,
                prior_input=prior_input,
                noise_steps=betas.shape[0],
                qubits=qubits,
                input_mode=config.input_mode,
            )
        else:
            t = torch.full((subset,), betas.shape[0] - 1, dtype=torch.long, device=target.device)
            generated_states = model(prior_input, t)
        prior_mode = config.prior_mode
        sampling_mode = config.generation_sampling_mode

    return {
        "generation_mmd": mmd_fidelity(target, generated_states, subset=config.mmd_subset),
        "generation_wasserstein": wasserstein_fidelity_distance(
            target, generated_states, subset=config.wasserstein_subset
        ),
        "generation_nearest_fidelity_mean": float(
            mean_nearest_fidelity(target, generated_states).detach().cpu()
        ),
        "generation_prior_mode": prior_mode,
        "generation_sampling_mode": sampling_mode,
    }


evaluate_model = evaluate_reconstruction


def _compatibility_metrics(
    model_name: str,
    reconstruction_metrics: dict[str, float],
    generation_metrics: dict[str, float],
) -> dict[str, float]:
    if model_name == "cnr":
        return {
            "fidelity": generation_metrics["generation_nearest_fidelity_mean"],
            "pure_state_fidelity": float("nan"),
            "mmd": generation_metrics["generation_mmd"],
            "wasserstein": generation_metrics["generation_wasserstein"],
        }
    return {
        "fidelity": reconstruction_metrics["reconstruction_fidelity"],
        "pure_state_fidelity": reconstruction_metrics["reconstruction_pure_state_fidelity"],
        "mmd": reconstruction_metrics["reconstruction_mmd"],
        "wasserstein": reconstruction_metrics["reconstruction_wasserstein"],
    }


@torch.no_grad()
def _forward_corruption_metrics(
    reference_states: torch.Tensor,
    model_name: str,
    betas: torch.Tensor,
    noise_steps: int,
    depth: int,
    qubits: int,
    config: ExperimentConfig,
) -> dict[str, float | str]:
    subset = min(config.corruption_match_subset, reference_states.shape[0])
    target = reference_states[:subset]
    alpha_bar = alpha_bar_schedule(betas)
    if model_name == "cnr":
        return {
            "forward_final_target_noisy_fidelity_mean": float("nan"),
            "forward_final_target_noisy_fidelity_std": float("nan"),
            "forward_process_type": "cnr_none",
            "depolarizing_mode": "none",
            "beta_end": float(betas[-1].detach().cpu()),
            "alpha_bar_final": float(alpha_bar[-1].detach().cpu()),
        }

    t = torch.full((target.shape[0],), noise_steps - 1, dtype=torch.long, device=target.device)
    if model_name == "quddpm_baseline":
        noisy_states = random_unitary_scramble_states(
            target,
            t=t,
            noise_steps=noise_steps,
            depth=depth,
            qubits=qubits,
        )
        fidelity = pure_state_fidelity(target, noisy_states)
        process_type = "random_unitary"
        depolarizing_mode = "none"
    else:
        noisy_rho = depolarizing_forward_from_states(
            target,
            t=t,
            betas=betas,
            mode=config.depolarizing_mode,
        )
        fidelity = pure_density_fidelity(target, noisy_rho)
        process_type = f"depolarizing_{config.depolarizing_mode}"
        depolarizing_mode = config.depolarizing_mode

    return {
        "forward_final_target_noisy_fidelity_mean": float(fidelity.mean().detach().cpu()),
        "forward_final_target_noisy_fidelity_std": float(fidelity.std(unbiased=False).detach().cpu()),
        "forward_process_type": process_type,
        "depolarizing_mode": depolarizing_mode,
        "beta_end": float(betas[-1].detach().cpu()),
        "alpha_bar_final": float(alpha_bar[-1].detach().cpu()),
    }


def train_one(
    config: ExperimentConfig,
    model_name: str,
    seed: int,
    noise_steps: int,
    depth: int,
) -> TrainingOutput:
    set_seed(seed)
    device = resolve_device(config.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    dataset = build_dataset(
        size=config.dataset_size,
        qubits=config.qubits,
        device=device,
        train_fraction=config.train_fraction,
        kind=config.dataset_kind,
    )
    betas = linear_beta_schedule(
        steps=noise_steps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        device=device,
    )
    model = build_model(
        model_name=model_name,
        qubits=config.qubits,
        noise_steps=noise_steps,
        depth=depth,
        hidden_dim=config.hidden_dim,
        time_embedding_dim=config.time_embedding_dim,
        input_mode=config.input_mode,
        cnr_latent_dim=config.cnr_latent_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    losses: list[dict] = []
    cnr_nearest_weight = 0.1

    with timer() as elapsed:
        for epoch in range(config.epochs):
            model.train()
            epoch_losses = []
            for idx in _batch_indices(dataset.train.shape[0], config.batch_size, device):
                clean_states = dataset.train[idx]
                if model_name == "cnr":
                    latent_dim = getattr(model, "latent_dim", max(4, config.qubits * 2))
                    latent = torch.randn(clean_states.shape[0], latent_dim, device=device)
                    generated_states = model(latent)
                    loss = differentiable_mmd_fidelity_loss(clean_states, generated_states)
                    loss = loss + cnr_nearest_weight * nearest_fidelity_loss(
                        clean_states, generated_states
                    )
                else:
                    t = torch.randint(0, noise_steps, (clean_states.shape[0],), device=device)
                    with torch.no_grad():
                        noisy_input = _forward_process(
                            clean_states,
                            t=t,
                            model_name=model_name,
                            betas=betas,
                            noise_steps=noise_steps,
                            depth=depth,
                            qubits=config.qubits,
                            input_mode=config.input_mode,
                            depolarizing_mode=config.depolarizing_mode,
                        )
                    generated_states = model(noisy_input, t)
                    fidelity = pure_state_fidelity(clean_states, generated_states)
                    loss = (1.0 - fidelity).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            losses.append(
                {
                    "epoch": epoch + 1,
                    "loss": sum(epoch_losses) / max(len(epoch_losses), 1),
                    "model": model_name,
                    "seed": seed,
                    "noise_steps": noise_steps,
                    "depth": depth,
                }
            )
    reconstruction_metrics = evaluate_reconstruction(
        model=model,
        test_states=dataset.test,
        model_name=model_name,
        betas=betas,
        noise_steps=noise_steps,
        depth=depth,
        qubits=config.qubits,
        config=config,
    )
    generation_metrics = evaluate_generation(
        model=model,
        test_states=dataset.test,
        model_name=model_name,
        betas=betas,
        qubits=config.qubits,
        config=config,
    )
    forward_metrics = _forward_corruption_metrics(
        reference_states=dataset.all_states,
        model_name=model_name,
        betas=betas,
        noise_steps=noise_steps,
        depth=depth,
        qubits=config.qubits,
        config=config,
    )
    row = {
        "model": model_name,
        "seed": seed,
        "qubits": config.qubits,
        "dataset_size": config.dataset_size,
        "dataset_kind": config.dataset_kind,
        "input_mode": config.input_mode,
        "noise_steps": noise_steps,
        "depth": depth,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "device": str(device),
        **reconstruction_metrics,
        **generation_metrics,
        **_compatibility_metrics(model_name, reconstruction_metrics, generation_metrics),
        **forward_metrics,
        **resource_metrics(
            model=model,
            qubits=config.qubits,
            depth=depth,
            noise_steps=noise_steps,
            runtime_sec=elapsed["elapsed"],
            device=device,
            model_name=model_name,
            generation_sampling_mode=(
                "one_step" if model_name == "cnr" else config.generation_sampling_mode
            ),
        ),
    }
    return TrainingOutput(metrics=row, losses=pd.DataFrame(losses))
