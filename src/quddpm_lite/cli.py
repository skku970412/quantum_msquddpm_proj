from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import preset_config
from .experiments import run_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the QuDDPM-lite benchmark.")
    parser.add_argument(
        "--preset",
        choices=["smoke", "mini", "twohour", "research", "full"],
        default="smoke",
        help="smoke is for quick verification; twohour targets a practical 512-sample run; research keeps 512/1000 defaults.",
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset-size", type=int)
    parser.add_argument("--qubits", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--time-embedding-dim", type=int)
    parser.add_argument("--cnr-latent-dim", type=int)
    parser.add_argument(
        "--dataset-kind",
        choices=["mixed", "product", "cluster1q", "entangled", "bell"],
    )
    parser.add_argument("--input-mode", choices=["density", "statevector"])
    parser.add_argument("--prior-mode", choices=["random_pure", "maximally_mixed_jitter", "depolarized_random"])
    parser.add_argument("--depolarizing-mode", choices=["single_beta", "cumulative"])
    parser.add_argument("--generation-sampling-mode", choices=["one_step", "iterative"])
    parser.add_argument("--match-corruption", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--noise-steps-grid", type=int, nargs="+")
    parser.add_argument("--depth-grid", type=int, nargs="+")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[
            "msquddpm",
            "quddpm_baseline",
            "t_msquddpm",
            "cnr",
            "independent_step_quddpm",
            "ancilla_toy",
        ],
    )
    parser.add_argument("--include-8qubit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = preset_config(args.preset)
    project_dir = Path(__file__).resolve().parents[2]
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = project_dir / results_dir

    updates = {
        "results_dir": results_dir,
        "device": args.device,
        "include_8qubit": args.include_8qubit,
        "match_corruption": args.match_corruption,
    }
    for key in [
        "dataset_size",
        "qubits",
        "epochs",
        "batch_size",
        "hidden_dim",
        "time_embedding_dim",
        "cnr_latent_dim",
        "dataset_kind",
        "input_mode",
        "prior_mode",
        "depolarizing_mode",
        "generation_sampling_mode",
    ]:
        value = getattr(args, key)
        if value is not None:
            updates[key] = value
    if args.seeds is not None:
        updates["seeds"] = args.seeds
    if args.noise_steps_grid is not None:
        updates["noise_steps_grid"] = args.noise_steps_grid
    if args.depth_grid is not None:
        updates["depth_grid"] = args.depth_grid
    if args.models is not None:
        updates["models"] = args.models

    config = replace(config, **updates)
    outputs = run_experiments(config)
    print("Completed benchmark.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
