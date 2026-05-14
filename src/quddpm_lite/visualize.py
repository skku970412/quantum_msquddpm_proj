from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_loss_curves(losses: pd.DataFrame, results_dir: Path) -> None:
    name_map = {
        "msquddpm": "loss_curve_msquddpm.png",
        "quddpm_baseline": "loss_curve_baseline.png",
        "t_msquddpm": "loss_curve_t_msquddpm.png",
        "cnr": "loss_curve_cnr.png",
    }
    for model, filename in name_map.items():
        subset = losses[losses["model"] == model]
        if subset.empty:
            continue
        grouped = subset.groupby("epoch", as_index=False)["loss"].mean()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(grouped["epoch"], grouped["loss"], linewidth=2)
        ax.set_title(f"{model} training loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training loss")
        ax.grid(True, alpha=0.25)
        _save(fig, results_dir / filename)


def plot_fidelity_comparison(metrics: pd.DataFrame, results_dir: Path) -> None:
    grouped = metrics.groupby("model")["fidelity"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(grouped["model"], grouped["mean"], yerr=grouped["std"].fillna(0.0), capsize=4)
    ax.set_ylim(0, 1.0)
    ax.set_title("Fidelity comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Average fidelity")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, axis="y", alpha=0.25)
    _save(fig, results_dir / "fidelity_comparison.png")


def plot_mmd_wasserstein_comparison(metrics: pd.DataFrame, results_dir: Path) -> None:
    grouped = metrics.groupby("model")[["mmd", "wasserstein"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(grouped))
    width = 0.36
    ax.bar([i - width / 2 for i in x], grouped["mmd"], width=width, label="MMD")
    ax.bar([i + width / 2 for i in x], grouped["wasserstein"], width=width, label="Wasserstein")
    ax.set_xticks(list(x), grouped["model"], rotation=15)
    ax.set_title("Distribution distance comparison")
    ax.set_ylabel("Lower is better")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    _save(fig, results_dir / "mmd_wasserstein_comparison.png")


def plot_resource_tradeoff(metrics: pd.DataFrame, results_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for model, subset in metrics.groupby("model"):
        ax.scatter(
            subset["trainable_parameters"],
            subset["fidelity"],
            s=35 + 20 * subset["runtime_sec"].clip(lower=0.0),
            alpha=0.75,
            label=model,
        )
    ax.set_xscale("log")
    ax.set_ylim(0, 1.0)
    ax.set_title("Resource tradeoff")
    ax.set_xlabel("Trainable parameters, log scale")
    ax.set_ylabel("Average fidelity")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save(fig, results_dir / "resource_tradeoff.png")


def plot_fidelity_vs_noise(curve: pd.DataFrame, results_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        curve["step"],
        curve["expected_fidelity_single_beta"],
        marker="o",
        linewidth=2,
        label="single_beta",
    )
    ax.plot(
        curve["step"],
        curve["expected_fidelity_cumulative"],
        marker="s",
        linewidth=2,
        label="cumulative",
    )
    ax.set_title("Expected fidelity under depolarizing schedules")
    ax.set_xlabel("Noise step")
    ax.set_ylabel("Expected fidelity")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save(fig, results_dir / "fidelity_vs_noise.png")


def plot_generation_quality_comparison(metrics: pd.DataFrame, results_dir: Path) -> None:
    grouped = (
        metrics.groupby("model")["generation_wasserstein"].agg(["mean", "std"]).reset_index()
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        grouped["model"],
        grouped["mean"],
        yerr=grouped["std"].fillna(0.0),
        capsize=4,
    )
    ax.set_title("Generation quality comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Generation Wasserstein")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, axis="y", alpha=0.25)
    _save(fig, results_dir / "generation_quality_comparison.png")


def plot_total_resource_tradeoff(metrics: pd.DataFrame, results_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for model, subset in metrics.groupby("model"):
        y = subset["generation_wasserstein"]
        if y.isna().all():
            y = subset["fidelity"]
        ax.scatter(
            subset["total_estimated_depth"],
            y,
            s=35 + 20 * subset["runtime_sec"].clip(lower=0.0),
            alpha=0.75,
            label=model,
        )
    ax.set_title("Total resource tradeoff")
    ax.set_xlabel("Total estimated depth")
    ax.set_ylabel("Generation Wasserstein")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save(fig, results_dir / "total_resource_tradeoff.png")


def write_all_plots(
    metrics: pd.DataFrame,
    losses: pd.DataFrame,
    noise_curve: pd.DataFrame,
    results_dir: Path,
) -> None:
    plot_loss_curves(losses, results_dir)
    plot_fidelity_comparison(metrics, results_dir)
    plot_mmd_wasserstein_comparison(metrics, results_dir)
    plot_resource_tradeoff(metrics, results_dir)
    plot_fidelity_vs_noise(noise_curve, results_dir)
    plot_generation_quality_comparison(metrics, results_dir)
    plot_total_resource_tradeoff(metrics, results_dir)
