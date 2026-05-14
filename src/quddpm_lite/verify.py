from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from .metrics import pairwise_fidelity_matrix


CORE_RESULT_FILES = [
    "metrics.csv",
    "summary_table.csv",
    "seed_summary.csv",
    "loss_history.csv",
    "noise_curve.csv",
]

CORE_PLOT_FILES = [
    "fidelity_comparison.png",
    "mmd_wasserstein_comparison.png",
    "resource_tradeoff.png",
    "fidelity_vs_noise.png",
    "generation_quality_comparison.png",
    "total_resource_tradeoff.png",
]

MODEL_LOSS_PLOTS = {
    "msquddpm": "loss_curve_msquddpm.png",
    "quddpm_baseline": "loss_curve_baseline.png",
    "t_msquddpm": "loss_curve_t_msquddpm.png",
    "cnr": "loss_curve_cnr.png",
}

REQUIRED_COLUMNS = {
    "model",
    "seed",
    "qubits",
    "dataset_size",
    "noise_steps",
    "depth",
    "fidelity",
    "reconstruction_fidelity",
    "generation_mmd",
    "generation_wasserstein",
    "generation_nearest_fidelity_mean",
    "generation_prior_mode",
    "generation_sampling_mode",
    "mmd",
    "wasserstein",
    "forward_process_type",
    "forward_final_target_noisy_fidelity_mean",
    "trainable_parameters",
    "estimated_circuit_depth",
    "estimated_two_qubit_gate_count",
    "total_estimated_depth",
    "total_estimated_two_qubit_gate_count",
    "runtime_sec",
}


def _load_run_config(results_dir: Path) -> dict:
    path = results_dir / "run_config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_rows(config: dict) -> int | None:
    keys = ["models", "seeds", "noise_steps_grid", "depth_grid"]
    if not all(key in config for key in keys):
        return None
    return (
        len(config["models"])
        * len(config["seeds"])
        * len(config["noise_steps_grid"])
        * len(config["depth_grid"])
    )


def verify(results_dir: Path) -> None:
    config = _load_run_config(results_dir)
    required_files = list(CORE_RESULT_FILES) + list(CORE_PLOT_FILES)
    for model_name in config.get("models", []):
        plot_name = MODEL_LOSS_PLOTS.get(model_name)
        if plot_name is not None:
            required_files.append(plot_name)

    missing = [name for name in required_files if not (results_dir / name).exists()]
    if missing:
        raise SystemExit(f"Missing result files: {missing}")

    metrics = pd.read_csv(results_dir / "metrics.csv")
    missing_columns = REQUIRED_COLUMNS.difference(metrics.columns)
    if missing_columns:
        raise SystemExit(f"metrics.csv missing columns: {sorted(missing_columns)}")

    expected_rows = _expected_rows(config)
    if expected_rows is not None and len(metrics) != expected_rows:
        raise SystemExit(
            f"metrics.csv row count mismatch: expected {expected_rows}, found {len(metrics)}"
        )

    if config.get("models") and set(metrics["model"]) != set(config["models"]):
        raise SystemExit("metrics.csv model set does not match run_config.json")
    if config.get("seeds") and set(metrics["seed"]) != set(config["seeds"]):
        raise SystemExit("metrics.csv seed set does not match run_config.json")
    if config.get("noise_steps_grid") and set(metrics["noise_steps"]) != set(config["noise_steps_grid"]):
        raise SystemExit("metrics.csv noise_steps set does not match run_config.json")
    if config.get("depth_grid") and set(metrics["depth"]) != set(config["depth_grid"]):
        raise SystemExit("metrics.csv depth set does not match run_config.json")
    if config.get("qubits") is not None and not (metrics["qubits"] == config["qubits"]).all():
        raise SystemExit("metrics.csv qubit count does not match run_config.json")

    fidelity = metrics["fidelity"].dropna()
    if not fidelity.between(0.0, 1.0).all():
        raise SystemExit("fidelity values must be in [0, 1]")
    if not (metrics["trainable_parameters"] > 0).all():
        raise SystemExit("parameter counts must be positive")
    if not (metrics["estimated_two_qubit_gate_count"] >= 0).all():
        raise SystemExit("two-qubit gate counts must be non-negative")
    if not (metrics["total_estimated_depth"] >= 0).all():
        raise SystemExit("total_estimated_depth must be non-negative")
    if not (metrics["total_estimated_two_qubit_gate_count"] >= 0).all():
        raise SystemExit("total_estimated_two_qubit_gate_count must be non-negative")
    cnr_rows = metrics[metrics["model"] == "cnr"]
    if not cnr_rows.empty and cnr_rows["generation_mmd"].isna().any():
        raise SystemExit("CNR rows must record generation_mmd")
    if not cnr_rows.empty and not cnr_rows["reconstruction_fidelity"].isna().all():
        raise SystemExit("CNR reconstruction_fidelity should be recorded as NaN")

    x = torch.eye(4, dtype=torch.complex64)[:2]
    f = pairwise_fidelity_matrix(x, x)
    if f.dtype != torch.float32 or f.shape != (2, 2):
        raise SystemExit("pairwise fidelity matrix smoke check failed")

    print(f"Verified {len(metrics)} benchmark rows in {results_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    verify(Path(args.results_dir))


if __name__ == "__main__":
    main()
