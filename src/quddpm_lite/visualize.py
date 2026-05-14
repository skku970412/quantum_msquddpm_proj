from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "#fbfbfd",
        "axes.edgecolor": "#d4d8e2",
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.color": "#6d7280",
        "font.size": 11,
        "axes.titlesize": 16,
        "axes.titleweight": "semibold",
        "axes.labelsize": 12,
        "legend.frameon": False,
    }
)

MODEL_ORDER = [
    "quddpm_baseline",
    "msquddpm",
    "t_msquddpm",
    "independent_step_quddpm",
    "cnr",
    "ancilla_toy",
]

MODEL_LABELS = {
    "quddpm_baseline": "Baseline",
    "msquddpm": "MSQuDDPM",
    "t_msquddpm": "T-MSQuDDPM",
    "independent_step_quddpm": "Independent",
    "cnr": "CNR",
    "ancilla_toy": "Ancilla Toy",
}

MODEL_COLORS = {
    "quddpm_baseline": "#c44e52",
    "msquddpm": "#4c72b0",
    "t_msquddpm": "#55a868",
    "independent_step_quddpm": "#8172b2",
    "cnr": "#dd8452",
    "ancilla_toy": "#937860",
}


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _placeholder_plot(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    _save(fig, path)


def _styled_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d4d8e2")
    ax.spines["bottom"].set_color("#d4d8e2")


def _sort_frame(frame: pd.DataFrame) -> pd.DataFrame:
    order = {name: idx for idx, name in enumerate(MODEL_ORDER)}
    sorted_frame = frame.copy()
    sorted_frame["_sort"] = sorted_frame["model"].map(order).fillna(len(order))
    sorted_frame["model_label"] = sorted_frame["model"].map(MODEL_LABELS).fillna(sorted_frame["model"])
    sorted_frame["color"] = sorted_frame["model"].map(MODEL_COLORS).fillna("#4c72b0")
    return sorted_frame.sort_values("_sort").reset_index(drop=True)


def _annotate_bar_values(ax: plt.Axes, labels: list[str], values: list[float], horizontal: bool = False) -> None:
    for label, value in zip(labels, values):
        if pd.isna(value):
            continue
        if horizontal:
            ax.text(value, label, f"  {value:.3f}", va="center", ha="left", fontsize=9, color="#2f3440")
        else:
            ax.text(label, value, f"{value:.3f}", va="bottom", ha="center", fontsize=9, color="#2f3440")


def plot_loss_curves(losses: pd.DataFrame, results_dir: Path) -> None:
    name_map = {
        "msquddpm": "loss_curve_msquddpm.png",
        "quddpm_baseline": "loss_curve_baseline.png",
        "t_msquddpm": "loss_curve_t_msquddpm.png",
        "cnr": "loss_curve_cnr.png",
        "independent_step_quddpm": "loss_curve_independent_step_quddpm.png",
        "ancilla_toy": "loss_curve_ancilla_toy.png",
    }
    for model, filename in name_map.items():
        subset = losses[losses["model"] == model]
        if subset.empty:
            continue
        grouped = subset.groupby("epoch", as_index=False)["loss"].mean()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(
            grouped["epoch"],
            grouped["loss"],
            linewidth=2.4,
            color=MODEL_COLORS.get(model, "#4c72b0"),
        )
        ax.set_title(f"{MODEL_LABELS.get(model, model)} training loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training loss")
        _styled_axes(ax)
        _save(fig, results_dir / filename)


def plot_fidelity_comparison(metrics: pd.DataFrame, results_dir: Path) -> None:
    grouped = metrics.groupby("model")["fidelity"].agg(["mean", "std"]).reset_index()
    grouped = _sort_frame(grouped)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.barh(
        grouped["model_label"],
        grouped["mean"],
        xerr=grouped["std"].fillna(0.0),
        color=grouped["color"],
        capsize=4,
    )
    ax.set_ylim(0, 1.0)
    ax.set_title("Fidelity comparison")
    ax.set_xlabel("Average fidelity")
    ax.set_ylabel("")
    _styled_axes(ax)
    _annotate_bar_values(ax, grouped["model_label"].tolist(), grouped["mean"].tolist(), horizontal=True)
    _save(fig, results_dir / "fidelity_comparison.png")


def plot_mmd_wasserstein_comparison(metrics: pd.DataFrame, results_dir: Path) -> None:
    grouped = metrics.groupby("model")[["mmd", "wasserstein"]].mean().reset_index()
    grouped = _sort_frame(grouped)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = range(len(grouped))
    width = 0.36
    ax.bar([i - width / 2 for i in x], grouped["mmd"], width=width, label="MMD", color="#4c72b0")
    ax.bar(
        [i + width / 2 for i in x],
        grouped["wasserstein"],
        width=width,
        label="Wasserstein",
        color="#55a868",
    )
    ax.set_xticks(list(x), grouped["model_label"], rotation=12)
    ax.set_title("Distribution distance comparison")
    ax.set_ylabel("Lower is better")
    _styled_axes(ax)
    ax.legend()
    _save(fig, results_dir / "mmd_wasserstein_comparison.png")


def plot_resource_tradeoff(metrics: pd.DataFrame, results_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    grouped = (
        metrics.groupby("model", as_index=False)
        .agg(
            trainable_parameters=("trainable_parameters", "mean"),
            fidelity=("fidelity", "mean"),
            runtime_sec=("runtime_sec", "mean"),
        )
        .pipe(_sort_frame)
    )
    for _, row in grouped.iterrows():
        model = row["model"]
        ax.scatter(
            row["trainable_parameters"],
            row["fidelity"],
            s=80 + 10 * max(float(row["runtime_sec"]), 0.0) ** 0.5,
            alpha=0.85,
            label=MODEL_LABELS.get(model, model),
            color=MODEL_COLORS.get(model, "#4c72b0"),
        )
        ax.annotate(
            MODEL_LABELS.get(model, model),
            (
                float(row["trainable_parameters"]),
                float(row["fidelity"]),
            ),
            fontsize=9,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xscale("log")
    ax.set_ylim(0, 1.0)
    ax.set_title("Resource tradeoff")
    ax.set_xlabel("Trainable parameters, log scale")
    ax.set_ylabel("Average fidelity")
    _styled_axes(ax)
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
    _styled_axes(ax)
    ax.legend()
    _save(fig, results_dir / "fidelity_vs_noise.png")


def plot_generation_quality_comparison(metrics: pd.DataFrame, results_dir: Path) -> None:
    grouped = (
        metrics.groupby("model")["generation_wasserstein"].agg(["mean", "std"]).reset_index()
    )
    grouped = _sort_frame(grouped)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.barh(
        grouped["model_label"],
        grouped["mean"].fillna(0.0),
        xerr=grouped["std"].fillna(0.0),
        color=grouped["color"],
        capsize=4,
    )
    ax.set_title("Generation quality comparison")
    ax.set_xlabel("Generation Wasserstein")
    ax.set_ylabel("")
    _styled_axes(ax)
    _annotate_bar_values(ax, grouped["model_label"].tolist(), grouped["mean"].fillna(0.0).tolist(), horizontal=True)
    _save(fig, results_dir / "generation_quality_comparison.png")


def plot_total_resource_tradeoff(metrics: pd.DataFrame, results_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    grouped = (
        metrics.groupby("model", as_index=False)
        .agg(
            total_estimated_depth=("total_estimated_depth", "mean"),
            generation_wasserstein=("generation_wasserstein", "mean"),
            fidelity=("fidelity", "mean"),
            runtime_sec=("runtime_sec", "mean"),
        )
        .pipe(_sort_frame)
    )
    for _, row in grouped.iterrows():
        model = row["model"]
        y = row["generation_wasserstein"]
        if pd.isna(y):
            y = row["fidelity"]
        ax.scatter(
            row["total_estimated_depth"],
            y,
            s=80 + 10 * max(float(row["runtime_sec"]), 0.0) ** 0.5,
            alpha=0.85,
            label=MODEL_LABELS.get(model, model),
            color=MODEL_COLORS.get(model, "#4c72b0"),
        )
        ax.annotate(
            MODEL_LABELS.get(model, model),
            (float(row["total_estimated_depth"]), float(y)),
            fontsize=9,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_title("Total resource tradeoff")
    ax.set_xlabel("Total estimated depth")
    ax.set_ylabel("Generation Wasserstein")
    _styled_axes(ax)
    ax.legend()
    _save(fig, results_dir / "total_resource_tradeoff.png")


def plot_parameter_efficiency(parameter_table: pd.DataFrame, results_dir: Path) -> None:
    if parameter_table.empty or parameter_table["trainable_parameters"].dropna().empty:
        return _placeholder_plot(
            results_dir / "parameter_efficiency.png",
            "Parameter efficiency",
            "No parameter-efficiency data available.",
        )
    grouped = parameter_table.groupby("model")["trainable_parameters"].mean().reset_index()
    grouped = _sort_frame(grouped)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.barh(grouped["model_label"], grouped["trainable_parameters"], color=grouped["color"])
    ax.set_title("Parameter efficiency")
    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel("")
    _styled_axes(ax)
    for label, value in zip(grouped["model_label"], grouped["trainable_parameters"]):
        ax.text(value, label, f"  {value:,.0f}", va="center", ha="left", fontsize=9, color="#2f3440")
    _save(fig, results_dir / "parameter_efficiency.png")


def plot_quality_vs_params(parameter_table: pd.DataFrame, results_dir: Path) -> None:
    if parameter_table.empty:
        return _placeholder_plot(
            results_dir / "quality_vs_params.png",
            "Quality vs params",
            "No parameter-efficiency data available.",
        )
    fig, ax = plt.subplots(figsize=(8, 4.8))
    plotted = False
    for _, row in parameter_table.iterrows():
        quality = row["reconstruction_fidelity"]
        if pd.isna(quality):
            continue
        plotted = True
        model = row["model"]
        ax.scatter(
            row["trainable_parameters"],
            quality,
            alpha=0.85,
            s=70,
            color=MODEL_COLORS.get(model, "#4c72b0"),
        )
        ax.annotate(
            MODEL_LABELS.get(model, model),
            (row["trainable_parameters"], quality),
            fontsize=9,
            xytext=(4, 2),
            textcoords="offset points",
        )
    if not plotted:
        return _placeholder_plot(
            results_dir / "quality_vs_params.png",
            "Quality vs params",
            "No reconstruction-quality values available.",
        )
    ax.set_xscale("log")
    ax.set_title("Quality vs params")
    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel("Reconstruction fidelity")
    _styled_axes(ax)
    _save(fig, results_dir / "quality_vs_params.png")


def plot_parameter_reduction_vs_quality(parameter_table: pd.DataFrame, results_dir: Path) -> None:
    valid = parameter_table.dropna(
        subset=["parameter_reduction_percent_vs_independent", "quality_drop_vs_independent"]
    )
    if valid.empty:
        return _placeholder_plot(
            results_dir / "parameter_reduction_vs_quality.png",
            "Parameter reduction vs quality",
            "Independent-step reference not available in this result directory.",
        )
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for _, row in valid.iterrows():
        model = row["model"]
        ax.scatter(
            row["parameter_reduction_percent_vs_independent"],
            row["quality_drop_vs_independent"],
            alpha=0.8,
            s=70,
            color=MODEL_COLORS.get(model, "#4c72b0"),
        )
        ax.annotate(
            MODEL_LABELS.get(model, model),
            (
                row["parameter_reduction_percent_vs_independent"],
                row["quality_drop_vs_independent"],
            ),
            fontsize=9,
            xytext=(4, 2),
            textcoords="offset points",
        )
    ax.set_title("Parameter reduction vs quality")
    ax.set_xlabel("Parameter reduction percent vs independent")
    ax.set_ylabel("Quality drop vs independent")
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color="#8c91a1", alpha=0.7)
    _styled_axes(ax)
    _save(fig, results_dir / "parameter_reduction_vs_quality.png")


def write_all_plots(
    metrics: pd.DataFrame,
    losses: pd.DataFrame,
    noise_curve: pd.DataFrame,
    parameter_table: pd.DataFrame,
    results_dir: Path,
) -> None:
    plot_loss_curves(losses, results_dir)
    plot_fidelity_comparison(metrics, results_dir)
    plot_mmd_wasserstein_comparison(metrics, results_dir)
    plot_resource_tradeoff(metrics, results_dir)
    plot_fidelity_vs_noise(noise_curve, results_dir)
    plot_generation_quality_comparison(metrics, results_dir)
    plot_total_resource_tradeoff(metrics, results_dir)
    plot_parameter_efficiency(parameter_table, results_dir)
    plot_quality_vs_params(parameter_table, results_dir)
    plot_parameter_reduction_vs_quality(parameter_table, results_dir)
