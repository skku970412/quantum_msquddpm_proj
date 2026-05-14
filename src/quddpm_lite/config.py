from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path


@dataclass
class ExperimentConfig:
    project_name: str = "GPU-accelerated Resource-Efficient QuDDPM-lite Benchmark"
    results_dir: Path = Path("results")

    qubits: int = 6
    optional_qubits: int = 8
    dataset_size: int = 512
    large_dataset_size: int = 1024
    train_fraction: float = 0.8
    dataset_kind: str = "mixed"
    input_mode: str = "density"

    noise_steps: int = 10
    noise_steps_grid: list[int] = field(default_factory=lambda: [10, 20])
    beta_start: float = 0.01
    beta_end: float = 0.15
    depolarizing_mode: str = "single_beta"

    epochs: int = 1000
    max_epochs: int = 3000
    batch_size: int = 64
    learning_rate: float = 1e-3
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    depth: int = 2
    depth_grid: list[int] = field(default_factory=lambda: [2, 4])
    hidden_dim: int = 128
    time_embedding_dim: int = 32

    models: list[str] = field(
        default_factory=lambda: ["msquddpm", "quddpm_baseline", "t_msquddpm"]
    )
    cnr_latent_dim: int = 0
    mmd_subset: int = 256
    wasserstein_subset: int = 256
    eval_subset: int = 256
    prior_mode: str = "depolarized_random"
    generation_sampling_mode: str = "one_step"
    prior_mixed_epsilon: float = 0.05
    prior_depolarizing_beta: float = 0.85
    match_corruption: bool = False
    corruption_match_subset: int = 64
    ancilla_success_penalty: float = 0.01
    device: str = "auto"
    include_8qubit: bool = False


def preset_config(preset: str) -> ExperimentConfig:
    """Return a runnable preset while keeping ExperimentConfig defaults research-sized."""
    base = ExperimentConfig()
    if preset == "research":
        return base
    if preset == "smoke":
        return replace(
            base,
            qubits=2,
            dataset_size=16,
            train_fraction=0.8,
            dataset_kind="cluster1q",
            noise_steps=2,
            noise_steps_grid=[2],
            epochs=1,
            batch_size=8,
            seeds=[0],
            depth=1,
            depth_grid=[1],
            hidden_dim=32,
            time_embedding_dim=16,
            models=["msquddpm"],
            cnr_latent_dim=4,
            mmd_subset=16,
            wasserstein_subset=16,
            eval_subset=16,
        )
    if preset == "mini":
        return replace(
            base,
            qubits=4,
            dataset_size=64,
            noise_steps=3,
            noise_steps_grid=[3],
            epochs=10,
            batch_size=16,
            seeds=[0],
            depth=1,
            depth_grid=[1],
            hidden_dim=48,
            time_embedding_dim=16,
            models=["msquddpm", "t_msquddpm", "cnr"],
            cnr_latent_dim=8,
            mmd_subset=32,
            wasserstein_subset=32,
            eval_subset=32,
        )
    if preset == "tenminute":
        return replace(
            base,
            dataset_size=256,
            epochs=60,
            batch_size=64,
            seeds=[0],
            noise_steps=6,
            noise_steps_grid=[6],
            depth=2,
            depth_grid=[2],
            hidden_dim=96,
            mmd_subset=64,
            wasserstein_subset=64,
            eval_subset=64,
        )
    if preset == "twohour_readme":
        return replace(
            base,
            dataset_size=512,
            epochs=400,
            batch_size=128,
            hidden_dim=96,
            noise_steps=10,
            noise_steps_grid=[10],
            depth=2,
            depth_grid=[2],
            seeds=[0, 1, 2],
            models=[
                "quddpm_baseline",
                "msquddpm",
                "t_msquddpm",
                "cnr",
                "independent_step_quddpm",
            ],
            cnr_latent_dim=12,
            match_corruption=True,
            mmd_subset=128,
            wasserstein_subset=128,
            eval_subset=128,
        )
    if preset == "twohour":
        return replace(
            base,
            dataset_size=512,
            epochs=400,
            batch_size=128,
            hidden_dim=96,
            mmd_subset=128,
            wasserstein_subset=128,
            eval_subset=128,
        )
    if preset == "full":
        return replace(base, epochs=base.max_epochs, dataset_size=base.large_dataset_size)
    raise ValueError(f"Unknown preset: {preset}")
