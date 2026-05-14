from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch

from config import ExperimentConfig
from datasets import build_dataset
from metrics import (
    mmd_fidelity,
    pure_density_fidelity,
    pure_state_fidelity,
    resource_metrics,
    wasserstein_fidelity_distance,
)
from models import build_model
from noise import depolarizing_forward_from_states, linear_beta_schedule, statevector_depolarizing_proxy
from random_unitary import random_unitary_forward_from_states, random_unitary_scramble_states
from utils import resolve_device, set_seed, statevectors_to_density, timer


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
        return statevector_depolarizing_proxy(clean_states, t=t, betas=betas)
    return depolarizing_forward_from_states(clean_states, t=t, betas=betas)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    test_states: torch.Tensor,
    model_name: str,
    betas: torch.Tensor,
    noise_steps: int,
    depth: int,
    qubits: int,
    config: ExperimentConfig,
) -> dict[str, float]:
    model.eval()
    subset = min(config.eval_subset, test_states.shape[0])
    target = test_states[:subset]
    t = torch.full((target.shape[0],), noise_steps - 1, dtype=torch.long, device=target.device)
    noisy_rho = _forward_process(
        target,
        t=t,
        model_name=model_name,
        betas=betas,
        noise_steps=noise_steps,
        depth=depth,
        qubits=qubits,
        input_mode=config.input_mode,
    )
    generated_states = model(noisy_rho, t)
    generated_rho = statevectors_to_density(generated_states)
    fidelity = pure_density_fidelity(target, generated_rho).mean().item()
    pure_fidelity = pure_state_fidelity(target, generated_states).mean().item()
    return {
        "fidelity": float(fidelity),
        "pure_state_fidelity": float(pure_fidelity),
        "mmd": mmd_fidelity(target, generated_states, subset=config.mmd_subset),
        "wasserstein": wasserstein_fidelity_distance(
            target, generated_states, subset=config.wasserstein_subset
        ),
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
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    losses: list[dict] = []

    with timer() as elapsed:
        for epoch in range(config.epochs):
            model.train()
            epoch_losses = []
            for idx in _batch_indices(dataset.train.shape[0], config.batch_size, device):
                clean_states = dataset.train[idx]
                t = torch.randint(0, noise_steps, (clean_states.shape[0],), device=device)
                with torch.no_grad():
                    noisy_rho = _forward_process(
                        clean_states,
                        t=t,
                        model_name=model_name,
                        betas=betas,
                        noise_steps=noise_steps,
                        depth=depth,
                        qubits=config.qubits,
                        input_mode=config.input_mode,
                    )
                generated_states = model(noisy_rho, t)
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
    eval_metrics = evaluate_model(
        model=model,
        test_states=dataset.test,
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
        **eval_metrics,
        **resource_metrics(
            model=model,
            qubits=config.qubits,
            depth=depth,
            noise_steps=noise_steps,
            runtime_sec=elapsed["elapsed"],
            device=device,
            model_name=model_name,
        ),
    }
    return TrainingOutput(metrics=row, losses=pd.DataFrame(losses))
