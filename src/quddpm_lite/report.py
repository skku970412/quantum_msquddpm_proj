from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PNG_ORDER = [
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


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "not available"
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return "not available"
    except TypeError:
        pass
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _table(headers: list[str], rows: list[list[object]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(_fmt(cell) for cell in row) + " |")
    return "\n".join(out)


def _model_role(model: str) -> str:
    mapping = {
        "quddpm_baseline": "baseline denoiser with random-unitary forward",
        "msquddpm": "proposed depolarizing denoiser",
        "t_msquddpm": "temporal-sharing denoiser",
        "cnr": "one-step generation comparator",
        "independent_step_quddpm": "naive step-wise independent denoiser baseline",
        "ancilla_toy": "ancilla measurement/post-selection concept demo",
    }
    return mapping.get(model, "not available")


def _safe_mean(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns or df.empty:
        return None
    value = df[column].mean()
    return None if pd.isna(value) else float(value)


def build_report(results_dir: Path) -> str:
    metrics = _read_csv(results_dir / "metrics.csv")
    summary = _read_csv(results_dir / "summary_table.csv")
    parameter_table = _read_csv(results_dir / "parameter_efficiency_table.csv")
    noise_curve = _read_csv(results_dir / "noise_curve.csv")
    config = _read_json(results_dir / "run_config.json")

    models = config.get("models") or sorted(
        metrics.get("model", pd.Series(dtype=str)).dropna().unique().tolist()
    )
    sections: list[str] = []

    sections.append("# Project Report")
    sections.append("")
    sections.append("## Project Scope")
    sections.append(
        "이 프로젝트는 QuDDPM/MSQuDDPM 논문 완전 재현이 아니라, **QuDDPM/MSQuDDPM 아이디어를 바탕으로 한 본선 대비용 lightweight sandbox**입니다."
    )
    sections.append("")

    sections.append("## Run Configuration")
    sections.append(
        _table(
            ["Field", "Value"],
            [
                ["qubits", config.get("qubits")],
                ["dataset_size", config.get("dataset_size")],
                ["models", " ".join(models) if models else "not available"],
                ["noise_steps_grid", config.get("noise_steps_grid")],
                ["depth_grid", config.get("depth_grid")],
                ["epochs", config.get("epochs")],
                ["prior_mode", config.get("prior_mode")],
                ["generation_sampling_mode", config.get("generation_sampling_mode")],
                ["depolarizing_mode", config.get("depolarizing_mode")],
                ["match_corruption", config.get("match_corruption")],
            ],
        )
    )
    sections.append("")

    sections.append("## Models Compared")
    sections.append(_table(["Model", "Role"], [[model, _model_role(model)] for model in models]))
    sections.append("")

    sections.append("## Reconstruction vs Generation")
    sections.append("- reconstruction은 clean state에서 noisy input을 만든 뒤 복원 성능을 봅니다.")
    sections.append("- generation은 target input 없이 prior에서 ensemble을 생성하고 test ensemble과 비교합니다.")
    sections.append("- `cnr`는 QuDDPM 대체가 아니라 generation-only one-step comparator입니다.")
    sections.append("")

    sections.append("## Fairness Checks")
    fairness_rows = []
    if not summary.empty:
        for _, row in summary.iterrows():
            fairness_rows.append(
                [
                    row.get("model"),
                    row.get("forward_process_type"),
                    row.get("forward_final_target_noisy_fidelity_mean"),
                    row.get("target_corruption_fidelity_mean"),
                    row.get("actual_forward_fidelity_mean"),
                    row.get("corruption_match_error_abs_mean"),
                ]
            )
    sections.append(
        _table(
            [
                "Model",
                "Forward",
                "Forward Fidelity",
                "Target Corruption Fidelity",
                "Actual Forward Fidelity",
                "Match Error",
            ],
            fairness_rows or [["not available"] * 6],
        )
    )
    sections.append(
        "match_corruption은 physical equivalence가 아니라, finite benchmark에서 random-unitary와 depolarizing forward의 severity를 operational fidelity 기준으로 맞추는 calibration입니다."
    )
    sections.append("")

    sections.append("## Resource Analysis")
    resource_rows = []
    if not summary.empty:
        for _, row in summary.iterrows():
            resource_rows.append(
                [
                    row.get("model"),
                    row.get("trainable_parameters_mean"),
                    row.get("total_estimated_depth_mean"),
                    row.get("total_reverse_depth_mean"),
                    row.get("total_estimated_two_qubit_gate_count_mean"),
                    row.get("channel_application_count_mean"),
                ]
            )
    sections.append(
        _table(
            [
                "Model",
                "Trainable Params",
                "Total Estimated Depth",
                "Total Reverse Depth",
                "Total Estimated 2Q Gates",
                "Channel Applications",
            ],
            resource_rows or [["not available"] * 6],
        )
    )
    sections.append("")

    sections.append("## Parameter Efficiency")
    if not parameter_table.empty and (parameter_table["model"] == "independent_step_quddpm").any():
        sections.append(
            _table(
                [
                    "Condition",
                    "Model",
                    "Trainable Params",
                    "Param Ratio vs Independent",
                    "Param Reduction %",
                    "Quality Drop",
                ],
                parameter_table[
                    [
                        "condition_id",
                        "model",
                        "trainable_parameters",
                        "parameter_ratio_vs_independent",
                        "parameter_reduction_percent_vs_independent",
                        "quality_drop_vs_independent",
                    ]
                ].values.tolist(),
            )
        )
    else:
        sections.append("independent_step_quddpm not run in this result directory.")
    sections.append("")

    sections.append("## Ancilla / Post-selection")
    ancilla_rows = metrics[metrics["model"] == "ancilla_toy"] if not metrics.empty else pd.DataFrame()
    if not ancilla_rows.empty:
        sections.append(
            _table(
                ["Metric", "Value"],
                [
                    ["success_probability_mean", ancilla_rows["success_probability_mean"].mean()],
                    ["success_probability_std", ancilla_rows["success_probability_std"].mean()],
                    ["post_selection_required", bool(ancilla_rows["post_selection_required"].all())],
                    ["ancilla_qubits", ancilla_rows["ancilla_qubits"].max()],
                ],
            )
        )
    else:
        sections.append("ancilla_toy not run in this result directory.")
    sections.append("")

    sections.append("## Key Findings")
    findings: list[str] = []
    if not summary.empty and "reconstruction_fidelity_mean" in summary.columns:
        best_recon = summary.sort_values("reconstruction_fidelity_mean", ascending=False).iloc[0]
        findings.append(
            f"- `{best_recon['model']}` achieved the highest mean reconstruction fidelity under this result directory ({_fmt(best_recon['reconstruction_fidelity_mean'])})."
        )
    if not summary.empty and "generation_wasserstein_mean" in summary.columns:
        generation_rows = summary.dropna(subset=["generation_wasserstein_mean"])
        if not generation_rows.empty:
            best_gen = generation_rows.sort_values("generation_wasserstein_mean", ascending=True).iloc[0]
            findings.append(
                f"- `{best_gen['model']}` achieved the lowest mean generation Wasserstein in this lightweight setting ({_fmt(best_gen['generation_wasserstein_mean'])})."
            )
    if "cnr" in models:
        findings.append("- `cnr` is reported as a one-step generation comparator, not as a QuDDPM replacement.")
    if _safe_mean(summary, "target_corruption_fidelity_mean") is not None:
        findings.append("- Forward corruption severity is logged to avoid unfair comparisons between random-unitary and depolarizing forwards.")
    if not findings:
        findings.append("- No summary findings available for this result directory.")
    sections.extend(findings[:5])
    sections.append("")

    sections.append("## Limitations")
    sections.append("- full paper reproduction 아님")
    sections.append("- resource metrics are heuristic, not hardware-transpiled counts")
    sections.append("- small-scale benchmark")
    sections.append("- match_corruption은 operational fidelity matching이지 physical equivalence가 아님")
    sections.append("- ancilla_toy는 conceptual module일 뿐 full measurement-based QuDDPM 아님")
    sections.append("")

    sections.append("## Application Note")
    sections.append(
        "본 프로젝트는 본선 세부 문제의 정답을 미리 구현한 것이 아니라, Quantum DDPM 계열 문제에 필요한 forward diffusion, denoising, generation evaluation, metric calculation, resource comparison을 팀 단위로 사전 연습하기 위한 PyTorch 기반 sandbox입니다."
    )
    sections.append(
        f"현재 결과 디렉터리는 qubits={_fmt(config.get('qubits'))}, dataset_size={_fmt(config.get('dataset_size'))}, noise_steps={_fmt(config.get('noise_steps_grid'))}, depth={_fmt(config.get('depth_grid'))} 조건을 중심으로 생성됐습니다."
    )
    sections.append("reconstruction metric과 generation metric을 분리해 기록함으로써 denoising benchmark와 generative benchmark를 혼동하지 않도록 했습니다.")
    sections.append("CNR comparator, independent-step baseline, ancilla toy module을 통해 comparator, parameter efficiency, 공식 구조 정합성 측면을 각각 분리해 검토할 수 있게 했습니다.")
    sections.append("공정 비교를 위해 forward corruption severity와 match_corruption calibration 결과를 함께 기록하며, 과장 없이 한계와 다음 개선 우선순위를 드러내는 연구형 sandbox를 지향합니다.")
    sections.append("")

    if not noise_curve.empty:
        sections.append("## Noise Curve")
        sections.append(
            _table(
                [
                    "step",
                    "beta",
                    "alpha",
                    "alpha_bar",
                    "expected_fidelity_single_beta",
                    "expected_fidelity_cumulative",
                ],
                noise_curve.head(8).values.tolist(),
            )
        )
        sections.append("")

    sections.append("## Figures")
    for name in PNG_ORDER:
        path = results_dir / name
        if path.exists():
            sections.append(f"![{name}]({name})")
    sections.append("")
    return "\n".join(sections).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out", default="report.md")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_path = Path(args.out)
    if not out_path.is_absolute() and out_path.parent == Path("."):
        out_path = results_dir / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_report(results_dir), encoding="utf-8")
    print(f"Wrote report to {out_path}")


if __name__ == "__main__":
    main()
