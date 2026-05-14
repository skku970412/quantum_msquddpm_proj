from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from metrics import pairwise_fidelity_matrix


REQUIRED_FILES = [
    "metrics.csv",
    "summary_table.csv",
    "seed_summary.csv",
    "loss_curve_msquddpm.png",
    "loss_curve_baseline.png",
    "fidelity_comparison.png",
    "mmd_wasserstein_comparison.png",
    "resource_tradeoff.png",
    "fidelity_vs_noise.png",
]


REQUIRED_COLUMNS = {
    "model",
    "seed",
    "qubits",
    "dataset_size",
    "noise_steps",
    "depth",
    "fidelity",
    "mmd",
    "wasserstein",
    "trainable_parameters",
    "estimated_circuit_depth",
    "estimated_two_qubit_gate_count",
    "runtime_sec",
}


def verify(results_dir: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (results_dir / name).exists()]
    if missing:
        raise SystemExit(f"Missing result files: {missing}")

    metrics = pd.read_csv(results_dir / "metrics.csv")
    missing_columns = REQUIRED_COLUMNS.difference(metrics.columns)
    if missing_columns:
        raise SystemExit(f"metrics.csv missing columns: {sorted(missing_columns)}")

    expected_models = {"msquddpm", "quddpm_baseline", "t_msquddpm"}
    if not expected_models.issubset(set(metrics["model"])):
        raise SystemExit("metrics.csv does not contain all three model families")
    if set(metrics["seed"]) != {0, 1, 2}:
        raise SystemExit("metrics.csv does not contain the requested seed set [0, 1, 2]")
    if not {10, 20}.issubset(set(metrics["noise_steps"])):
        raise SystemExit("metrics.csv does not contain T=10 and T=20 ablation rows")
    if not {2, 4}.issubset(set(metrics["depth"])):
        raise SystemExit("metrics.csv does not contain depth=2 and depth=4 ablation rows")
    if not (metrics["qubits"] == 6).all():
        raise SystemExit("the verified benchmark rows must be 6-qubit rows")
    if not metrics["fidelity"].between(0.0, 1.0).all():
        raise SystemExit("fidelity values must be in [0, 1]")
    if not (metrics["trainable_parameters"] > 0).all():
        raise SystemExit("parameter counts must be positive")
    if not (metrics["estimated_two_qubit_gate_count"] > 0).all():
        raise SystemExit("two-qubit gate counts must be positive")

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

