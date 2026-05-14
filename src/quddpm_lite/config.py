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
    mmd_subset: int = 256
    wasserstein_subset: int = 256
    eval_subset: int = 256
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
            dataset_size=48,
            epochs=2,
            batch_size=16,
            hidden_dim=48,
            mmd_subset=32,
            wasserstein_subset=32,
            eval_subset=32,
        )
    if preset == "mini":
        return replace(
            base,
            dataset_size=128,
            epochs=20,
            batch_size=32,
            hidden_dim=64,
            mmd_subset=64,
            wasserstein_subset=64,
            eval_subset=64,
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
