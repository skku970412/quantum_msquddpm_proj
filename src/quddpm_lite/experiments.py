from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import torch

from .config import ExperimentConfig
from .noise import alpha_bar_schedule, expected_fidelity_under_depolarizing, linear_beta_schedule
from .train import train_one
from .utils import ensure_dir, resolve_device, save_json
from .visualize import write_all_plots


def _write_summaries(metrics: pd.DataFrame, results_dir: Path) -> None:
    summary = (
        metrics.groupby(["model", "noise_steps", "depth"])
        .agg(
            fidelity_mean=("fidelity", "mean"),
            fidelity_std=("fidelity", "std"),
            reconstruction_fidelity_mean=("reconstruction_fidelity", "mean"),
            reconstruction_pure_state_fidelity_mean=(
                "reconstruction_pure_state_fidelity",
                "mean",
            ),
            reconstruction_mmd_mean=("reconstruction_mmd", "mean"),
            reconstruction_wasserstein_mean=("reconstruction_wasserstein", "mean"),
            generation_mmd_mean=("generation_mmd", "mean"),
            generation_wasserstein_mean=("generation_wasserstein", "mean"),
            generation_nearest_fidelity_mean=("generation_nearest_fidelity_mean", "mean"),
            mmd_mean=("mmd", "mean"),
            mmd_std=("mmd", "std"),
            wasserstein_mean=("wasserstein", "mean"),
            wasserstein_std=("wasserstein", "std"),
            runtime_sec_mean=("runtime_sec", "mean"),
            trainable_parameters_mean=("trainable_parameters", "mean"),
            estimated_two_qubit_gate_count_mean=("estimated_two_qubit_gate_count", "mean"),
            total_estimated_depth_mean=("total_estimated_depth", "mean"),
            total_estimated_two_qubit_gate_count_mean=(
                "total_estimated_two_qubit_gate_count",
                "mean",
            ),
            total_reverse_depth_mean=("total_reverse_depth", "mean"),
            channel_application_count_mean=("channel_application_count", "mean"),
            forward_final_target_noisy_fidelity_mean=(
                "forward_final_target_noisy_fidelity_mean",
                "mean",
            ),
            generation_prior_mode=("generation_prior_mode", "first"),
            generation_sampling_mode=("generation_sampling_mode", "first"),
            forward_process_type=("forward_process_type", "first"),
            depolarizing_mode=("depolarizing_mode", "first"),
            resource_notes=("resource_notes", "first"),
            gpu_memory_mb_max=("gpu_memory_mb", "max"),
        )
        .reset_index()
    )
    summary.to_csv(results_dir / "summary_table.csv", index=False)

    seed_summary = (
        metrics.groupby(["model", "seed"])
        .agg(
            fidelity_mean=("fidelity", "mean"),
            reconstruction_fidelity_mean=("reconstruction_fidelity", "mean"),
            generation_mmd_mean=("generation_mmd", "mean"),
            generation_wasserstein_mean=("generation_wasserstein", "mean"),
            generation_nearest_fidelity_mean=("generation_nearest_fidelity_mean", "mean"),
            mmd_mean=("mmd", "mean"),
            wasserstein_mean=("wasserstein", "mean"),
            runtime_sec_mean=("runtime_sec", "mean"),
            total_estimated_depth_mean=("total_estimated_depth", "mean"),
        )
        .reset_index()
    )
    seed_summary.to_csv(results_dir / "seed_summary.csv", index=False)


def _noise_curve(config: ExperimentConfig) -> pd.DataFrame:
    device = resolve_device(config.device)
    steps = max(config.noise_steps_grid)
    betas = linear_beta_schedule(steps, config.beta_start, config.beta_end, device=device).cpu()
    alphas = 1.0 - betas
    alpha_bar = alpha_bar_schedule(betas)
    dim = 2**config.qubits
    expected_single = expected_fidelity_under_depolarizing(
        betas=betas,
        dim=dim,
        mode="single_beta",
    )
    expected_cumulative = expected_fidelity_under_depolarizing(
        betas=betas,
        dim=dim,
        mode="cumulative",
    )
    return pd.DataFrame(
        {
            "step": list(range(steps)),
            "beta": betas.numpy(),
            "alpha": alphas.numpy(),
            "alpha_bar": alpha_bar.numpy(),
            "expected_fidelity_single_beta": expected_single.numpy(),
            "expected_fidelity_cumulative": expected_cumulative.numpy(),
        }
    )


def _serializable_config(config: ExperimentConfig) -> dict:
    data = asdict(config)
    data["results_dir"] = str(data["results_dir"])
    return data


def run_experiments(config: ExperimentConfig) -> dict[str, Path]:
    results_dir = ensure_dir(config.results_dir)
    save_json(results_dir / "run_config.json", _serializable_config(config))

    metric_rows: list[dict] = []
    loss_frames: list[pd.DataFrame] = []
    total = len(config.models) * len(config.seeds) * len(config.noise_steps_grid) * len(config.depth_grid)
    run_no = 0
    for model_name in config.models:
        for noise_steps in config.noise_steps_grid:
            for depth in config.depth_grid:
                for seed in config.seeds:
                    run_no += 1
                    print(
                        f"[{run_no}/{total}] model={model_name} seed={seed} "
                        f"T={noise_steps} depth={depth}",
                        flush=True,
                    )
                    output = train_one(
                        config=config,
                        model_name=model_name,
                        seed=seed,
                        noise_steps=noise_steps,
                        depth=depth,
                    )
                    metric_rows.append(output.metrics)
                    loss_frames.append(output.losses)
                    pd.DataFrame(metric_rows).to_csv(results_dir / "metrics.csv", index=False)

    metrics = pd.DataFrame(metric_rows)
    losses = pd.concat(loss_frames, ignore_index=True)
    noise_curve = _noise_curve(config)

    metrics.to_csv(results_dir / "metrics.csv", index=False)
    losses.to_csv(results_dir / "loss_history.csv", index=False)
    noise_curve.to_csv(results_dir / "noise_curve.csv", index=False)
    _write_summaries(metrics, results_dir)
    write_all_plots(metrics, losses, noise_curve, results_dir)

    if config.include_8qubit:
        optional = replace(
            config,
            qubits=config.optional_qubits,
            dataset_size=min(config.dataset_size, 64),
            epochs=min(config.epochs, 5),
            models=["msquddpm"],
            noise_steps_grid=[10],
            depth_grid=[2],
            eval_subset=min(config.eval_subset, 32),
            input_mode="statevector",
        )
        optional_rows = []
        for seed in optional.seeds:
            output = train_one(optional, "msquddpm", seed=seed, noise_steps=10, depth=2)
            optional_rows.append(output.metrics)
        pd.DataFrame(optional_rows).to_csv(results_dir / "optional_8qubit_msquddpm.csv", index=False)

    return {
        "results_dir": results_dir,
        "metrics": results_dir / "metrics.csv",
        "summary_table": results_dir / "summary_table.csv",
        "seed_summary": results_dir / "seed_summary.csv",
    }
