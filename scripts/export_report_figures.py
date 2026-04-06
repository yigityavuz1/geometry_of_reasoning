# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from plotly import graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_demo_case import prepare_demo_case

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export report-ready figures from an ablation results directory.")
    parser.add_argument("--results", required=True, help="Ablation results directory")
    parser.add_argument("--out-dir", default="report/figures", help="Figure output directory")
    parser.add_argument("--case-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--case-index", type=int, default=-1, help="Dataset index for the seismograph case")
    parser.add_argument("--case-primary-k", type=int, default=-1, help="Optional primary k override")
    parser.add_argument("--case-analysis-layer", default="", help="Optional explicit case layer override")
    parser.add_argument(
        "--case-layer-selection",
        choices=["fixed", "classification_best", "early_warning_best"],
        default="fixed",
        help="How to choose the demo layer when --case-analysis-layer is not set",
    )
    parser.add_argument(
        "--html-only",
        action="store_true",
        help="Write interactive HTML only and skip PNG export",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_plot_outputs(
    fig: go.Figure,
    out_dir: Path,
    stem: str,
    *,
    html_only: bool,
) -> dict[str, Any]:
    html_path = out_dir / f"{stem}.html"
    fig.write_html(str(html_path))
    payload: dict[str, Any] = {"html": str(html_path), "png": None}
    if html_only:
        return payload

    png_path = out_dir / f"{stem}.png"
    try:
        fig.write_image(str(png_path), width=1200, height=700, scale=2)
        payload["png"] = str(png_path)
    except Exception as exc:  # pragma: no cover - environment specific
        LOGGER.warning("Static PNG export failed for %s: %s", stem, exc)
        payload["png_error"] = str(exc)
    return payload


def _metric_bar_figure(df: pd.DataFrame, *, title: str, x_labels: list[str]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x_labels, y=df["auroc"].tolist(), name="AUROC"))
    fig.add_trace(go.Bar(x=x_labels, y=df["auprc"].tolist(), name="AUPRC"))
    fig.update_layout(title=title, barmode="group", template="plotly_white")
    return fig


def _write_report_summary(
    out_dir: Path,
    ablation_summary: dict[str, Any],
    case_summary: dict[str, Any],
) -> dict[str, str]:
    experiment_a = ablation_summary["experiment_a"]["summary"]
    experiment_b = ablation_summary["experiment_b"]
    experiment_c = ablation_summary["experiment_c"]["model_outputs"]
    payload = {
        "results_dir": ablation_summary["output_layout"]["models_root"].rsplit("/models", 1)[0],
        "primary_analysis_layer": str(experiment_a["analysis_layer"]),
        "primary_k": int(experiment_a["k_used"]),
        "experiment_a_overall": {
            "auroc": float(experiment_a["logistic"]["auroc"]),
            "auprc": float(experiment_a["logistic"]["auprc"]),
            "brier": float(experiment_a["logistic"]["brier"]),
            "ece": float(experiment_a["logistic"]["ece"]),
            "lead_time_mean": float(experiment_a["lead_time"]["lead_time_mean"]),
            "alarm_before_error_rate": float(experiment_a["lead_time"]["alarm_before_error_rate"]),
            "false_alarm_before_any_error_rate": float(
                experiment_a["lead_time"]["false_alarm_before_any_error_rate"]
            ),
            "selected_alarm_policy": experiment_a["selected_alarm_policy"],
            "layer_recommendation_classification": experiment_a["layer_recommendation_classification"],
            "layer_recommendation_early_warning": experiment_a["layer_recommendation_early_warning"],
        },
        "experiment_b_common_index": experiment_b["common_index"]["rows"],
        "experiment_b_best_layer": experiment_b["best_layer"]["rows"],
        "experiment_c_stability": [
            {
                "model_name": str(row["model_name"]),
                "auroc_cv_over_k": float(row["auroc_cv_over_k"]),
                "kendall_tau_mean": float(row["kendall_tau_mean"]),
            }
            for row in experiment_c
        ],
        "case_summary": case_summary,
    }
    summary_json = out_dir / "report_summary.json"
    summary_md = out_dir / "report_summary.md"
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    common_rows = experiment_b["common_index"]["rows"]
    lines = [
        "# Report Summary",
        "",
        f"- Primary analysis layer: `{experiment_a['analysis_layer']}`",
        f"- Primary k: `{experiment_a['k_used']}`",
        f"- Experiment A AUROC/AUPRC: `{experiment_a['logistic']['auroc']:.3f}` / "
        f"`{experiment_a['logistic']['auprc']:.3f}`",
        f"- Experiment A lead time mean: `{experiment_a['lead_time']['lead_time_mean']:.3f}`",
        f"- Experiment A alarm-before-error rate: "
        f"`{experiment_a['lead_time']['alarm_before_error_rate']:.3f}`",
        "",
        "## Experiment B Common Index",
        "",
    ]
    for row in common_rows:
        lines.append(
            f"- `{row['model_name']}`: AUROC `{row['auroc']:.3f}`, "
            f"AUPRC `{row['auprc']:.3f}`, signal gap `{row['signal_gap_mean']:.3f}`"
        )
    lines.extend(
        [
            "",
            "## Demo Case",
            "",
            f"- Model: `{case_summary['model_name']}`",
            f"- Layer: `{case_summary['analysis_layer']}`",
            f"- Dataset index: `{case_summary['dataset_index']}`",
            f"- Lead time: `{case_summary['lead_time']}`",
            f"- Selected policy: `{case_summary['selected_policy']['policy_name']}`",
            "",
        ]
    )
    summary_md.write_text("\n".join(lines), encoding="utf-8")
    return {"json": str(summary_json), "md": str(summary_md)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()
    results_dir = Path(args.results)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ablation_summary = _load_json(results_dir / "ablation_summary.json")
    figure_outputs: dict[str, Any] = {}

    comparison_df = pd.read_csv(results_dir / "experiment_b_model_comparison_common_index.csv")
    comparison_fig = _metric_bar_figure(
        comparison_df,
        title="Experiment B - Common Index Model Comparison",
        x_labels=comparison_df["model_name"].astype(str).tolist(),
    )
    figure_outputs["model_comparison_common_index"] = _write_plot_outputs(
        comparison_fig,
        out_dir,
        "model_comparison_common_index",
        html_only=bool(args.html_only),
    )

    best_layer_path = results_dir / "experiment_b_model_comparison_best_layer.csv"
    if best_layer_path.exists():
        best_layer_df = pd.read_csv(best_layer_path)
        best_labels = [
            f"{row.model_name}<br>({row.analysis_layer})"
            for row in best_layer_df[["model_name", "analysis_layer"]].itertuples(index=False)
        ]
        best_layer_fig = _metric_bar_figure(
            best_layer_df,
            title="Experiment B - Per-Model Best Layer Comparison",
            x_labels=best_labels,
        )
        figure_outputs["model_comparison_best_layer"] = _write_plot_outputs(
            best_layer_fig,
            out_dir,
            "model_comparison_best_layer",
            html_only=bool(args.html_only),
        )

    sensitivity_df = pd.read_csv(results_dir / "experiment_c_k_sensitivity.csv")
    sensitivity_fig = go.Figure()
    for model_name in sensitivity_df["model_name"].dropna().unique().tolist():
        subset = sensitivity_df.loc[sensitivity_df["model_name"] == model_name].copy()
        sensitivity_fig.add_trace(
            go.Scatter(
                x=subset["k_requested"].tolist(),
                y=subset["auroc"].tolist(),
                mode="lines+markers",
                name=str(model_name),
            )
        )
    sensitivity_fig.update_layout(
        title="Experiment C - k Sensitivity",
        xaxis_title="k",
        yaxis_title="AUROC",
        template="plotly_white",
    )
    figure_outputs["k_sensitivity"] = _write_plot_outputs(
        sensitivity_fig,
        out_dir,
        "k_sensitivity",
        html_only=bool(args.html_only),
    )

    sweep_df = pd.read_csv(results_dir / "experiment_a_threshold_sweep.csv")
    sweep_fig = go.Figure()
    for column, label in [
        ("alarm_before_error_rate", "Alarm Before Error Rate"),
        ("false_alarm_before_any_error_rate", "False Alarm Rate"),
        ("late_alarm_rate", "Late Alarm Rate"),
    ]:
        if column in sweep_df.columns:
            sweep_fig.add_trace(
                go.Scatter(
                    x=sweep_df["threshold"].tolist(),
                    y=sweep_df[column].tolist(),
                    mode="lines+markers",
                    name=label,
                )
            )
    sweep_fig.update_layout(
        title="Experiment A - Threshold Sweep",
        xaxis_title="Threshold",
        yaxis_title="Rate",
        template="plotly_white",
    )
    figure_outputs["early_warning_threshold_sweep"] = _write_plot_outputs(
        sweep_fig,
        out_dir,
        "early_warning_threshold_sweep",
        html_only=bool(args.html_only),
    )

    warning_path = results_dir / "experiment_a_warning_trajectory.csv"
    if warning_path.exists():
        warning_df = pd.read_csv(warning_path)
        warning_fig = go.Figure()
        for outcome in warning_df["trace_outcome"].dropna().unique().tolist():
            subset = warning_df.loc[warning_df["trace_outcome"] == outcome].copy()
            warning_fig.add_trace(
                go.Scatter(
                    x=subset["step_index"].tolist(),
                    y=subset["warning_score_mean"].tolist(),
                    mode="lines+markers",
                    name=f"{outcome} warning",
                )
            )
            if "trajectory_warning_score_mean" in subset.columns:
                warning_fig.add_trace(
                    go.Scatter(
                        x=subset["step_index"].tolist(),
                        y=subset["trajectory_warning_score_mean"].tolist(),
                        mode="lines+markers",
                        line=dict(dash="dash"),
                        name=f"{outcome} trajectory",
                    )
                )
        warning_fig.update_layout(
            title="Experiment A - Warning Trajectory",
            xaxis_title="Step Index",
            yaxis_title="Mean Score",
            template="plotly_white",
        )
        figure_outputs["warning_trajectory"] = _write_plot_outputs(
            warning_fig,
            out_dir,
            "warning_trajectory",
            html_only=bool(args.html_only),
        )

    case_df, case_fig, case_summary = prepare_demo_case(
        results_dir,
        str(args.case_model),
        dataset_index=int(args.case_index),
        primary_k=int(args.case_primary_k),
        analysis_layer=str(args.case_analysis_layer),
        layer_selection=str(args.case_layer_selection),
    )
    figure_outputs["case_seismograph"] = _write_plot_outputs(
        case_fig,
        out_dir,
        "case_seismograph",
        html_only=bool(args.html_only),
    )
    case_steps_path = out_dir / "case_steps.csv"
    case_summary_path = out_dir / "case_summary.json"
    case_df.to_csv(case_steps_path, index=False)
    case_summary_path.write_text(json.dumps(case_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_outputs = _write_report_summary(out_dir, ablation_summary, case_summary)
    manifest = {
        "results_dir": str(results_dir),
        "out_dir": str(out_dir),
        "html_only": bool(args.html_only),
        "case_model": str(args.case_model),
        "case_index": int(case_summary["dataset_index"]),
        "case_layer_selection": str(args.case_layer_selection),
        "case_analysis_layer": str(case_summary["analysis_layer"]),
        "case_primary_k": int(case_summary["primary_k"]),
        "report_summary": summary_outputs,
        "case_summary_path": str(case_summary_path),
        "case_steps_path": str(case_steps_path),
        "figures": figure_outputs,
    }
    (out_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
