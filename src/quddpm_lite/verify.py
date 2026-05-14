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
    "parameter_efficiency_table.csv",
]

CORE_PLOT_FILES = [
    "fidelity_comparison.png",
    "mmd_wasserstein_comparison.png",
    "resource_tradeoff.png",
    "fidelity_vs_noise.png",
    "generation_quality_comparison.png",
    "total_resource_tradeoff.png",
    "parameter_efficiency.png",
    "quality_vs_params.png",
    "parameter_reduction_vs_quality.png",
]

MODEL_LOSS_PLOTS = {
    "msquddpm": "loss_curve_msquddpm.png",
    "quddpm_baseline": "loss_curve_baseline.png",
    "t_msquddpm": "loss_curve_t_msquddpm.png",
    "cnr": "loss_curve_cnr.png",
    "independent_step_quddpm": "loss_curve_independent_step_quddpm.png",
    "ancilla_toy": "loss_curve_ancilla_toy.png",
}

REQUIRED_COLUMNS = {
    "model",
    "seed",
    "qubits",
    "dataset_size",
    "hidden_dim",
    "time_embedding_dim",
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
    "success_probability_mean",
    "success_probability_std",
    "forward_process_type",
    "forward_final_target_noisy_fidelity_mean",
    "match_corruption_enabled",
    "target_corruption_fidelity",
    "actual_forward_fidelity_mean",
    "corruption_match_error_abs",
    "trainable_parameters",
    "estimated_circuit_depth",
    "estimated_two_qubit_gate_count",
    "total_estimated_depth",
    "total_estimated_two_qubit_gate_count",
    "step_parameterization",
    "effective_denoiser_count",
    "ancilla_qubits",
    "post_selection_required",
    "runtime_sec",
}

PARAMETER_TABLE_REQUIRED_COLUMNS = {
    "condition_id",
    "qubits",
    "noise_steps",
    "depth",
    "hidden_dim",
    "model",
    "step_parameterization",
    "trainable_parameters",
    "parameter_ratio_vs_independent",
    "parameter_reduction_percent_vs_independent",
    "reconstruction_fidelity",
    "generation_wasserstein",
    "generation_mmd",
    "runtime_sec",
    "quality_drop_vs_independent",
    "efficiency_notes",
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
    parameter_table = pd.read_csv(results_dir / "parameter_efficiency_table.csv")
    missing_parameter_columns = PARAMETER_TABLE_REQUIRED_COLUMNS.difference(parameter_table.columns)
    if missing_parameter_columns:
        raise SystemExit(
            f"parameter_efficiency_table.csv missing columns: {sorted(missing_parameter_columns)}"
        )

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
    independent_rows = metrics[metrics["model"] == "independent_step_quddpm"]
    if not independent_rows.empty and not independent_rows["is_step_independent_model"].all():
        raise SystemExit("independent_step_quddpm rows must set is_step_independent_model=True")
    ancilla_rows = metrics[metrics["model"] == "ancilla_toy"]
    if not ancilla_rows.empty and ancilla_rows["success_probability_mean"].isna().any():
        raise SystemExit("ancilla_toy rows must record success_probability_mean")
    if not ancilla_rows.empty and not ancilla_rows["post_selection_required"].all():
        raise SystemExit("ancilla_toy rows must set post_selection_required=True")
    if config.get("match_corruption"):
        non_cnr_rows = metrics[metrics["model"] != "cnr"]
        if non_cnr_rows["target_corruption_fidelity"].isna().any():
            raise SystemExit("match_corruption run must record target_corruption_fidelity")
        if non_cnr_rows["actual_forward_fidelity_mean"].isna().any():
            raise SystemExit("match_corruption run must record actual_forward_fidelity_mean")

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
