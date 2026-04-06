# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_ablation import _build_experiment_b_rows, _common_dataset_indices_for_primary_k, _load_input_table, _run_experiment_a_core


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a seed/slice stability suite on a precomputed step table.")
    parser.add_argument("--input", required=True, help="Combined step table (.csv or .jsonl)")
    parser.add_argument("--out", default="results/stability_suite", help="Output directory")
    parser.add_argument("--primary-k", type=int, default=20)
    parser.add_argument("--early-n", type=int, default=2)
    parser.add_argument("--bootstrap-iters", type=int, default=100)
    parser.add_argument("--bootstrap-alpha", type=float, default=0.05)
    parser.add_argument("--seeds", default="7,42,123", help="Comma-separated seed list")
    parser.add_argument("--slice-size", type=int, default=400)
    parser.add_argument("--slice-starts", default="0,400,800", help="Comma-separated slice starts")
    return parser.parse_args()


def _parse_csv_int(raw: str) -> list[int]:
    values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _metric_row(
    *,
    slice_label: str,
    seed: int,
    experiment: str,
    scope: str,
    model_name: str,
    metric_name: str,
    metric_value: float,
) -> dict[str, object]:
    return {
        "slice_label": slice_label,
        "seed": int(seed),
        "experiment": experiment,
        "scope": scope,
        "model_name": model_name,
        "metric_name": metric_name,
        "metric_value": float(metric_value),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    step_df = _load_input_table(Path(args.input))
    seeds = _parse_csv_int(args.seeds)
    slice_starts = _parse_csv_int(args.slice_starts)
    metric_rows: list[dict[str, object]] = []
    run_summaries: list[dict[str, object]] = []

    for slice_start in slice_starts:
        slice_end = int(slice_start) + int(args.slice_size)
        slice_df = step_df.loc[
            (step_df["dataset_index"].astype(int) >= int(slice_start))
            & (step_df["dataset_index"].astype(int) < int(slice_end))
        ].copy()
        if slice_df.empty:
            continue
        slice_label = f"{int(slice_start)}_{int(slice_end)}"

        for seed in seeds:
            combined_metrics, _, _, _ = _run_experiment_a_core(
                df=slice_df,
                requested_k=args.primary_k,
                early_n=args.early_n,
                seed=seed,
                bootstrap_iters=args.bootstrap_iters,
                bootstrap_alpha=args.bootstrap_alpha,
            )
            run_summaries.append(
                {
                    "slice_label": slice_label,
                    "seed": int(seed),
                    "combined_n_samples": int(combined_metrics["n_samples"]),
                }
            )
            for metric_name in ("auroc", "auprc", "brier", "ece"):
                metric_rows.append(
                    _metric_row(
                        slice_label=slice_label,
                        seed=seed,
                        experiment="experiment_a",
                        scope="combined",
                        model_name="combined",
                        metric_name=metric_name,
                        metric_value=float(combined_metrics["logistic"][metric_name]),
                    )
                )
            for metric_name in ("alarm_before_error_rate", "lead_time_mean", "lead_time_median"):
                metric_rows.append(
                    _metric_row(
                        slice_label=slice_label,
                        seed=seed,
                        experiment="experiment_a",
                        scope="combined",
                        model_name="combined",
                        metric_name=metric_name,
                        metric_value=float(combined_metrics["lead_time"][metric_name]),
                    )
                )

            for model_name, model_slice_df in slice_df.groupby("model_name", sort=False):
                model_metrics, _, _, _ = _run_experiment_a_core(
                    df=model_slice_df,
                    requested_k=args.primary_k,
                    early_n=args.early_n,
                    seed=seed,
                    bootstrap_iters=args.bootstrap_iters,
                    bootstrap_alpha=args.bootstrap_alpha,
                )
                for metric_name in ("auroc", "auprc", "brier", "ece"):
                    metric_rows.append(
                        _metric_row(
                            slice_label=slice_label,
                            seed=seed,
                            experiment="experiment_a",
                            scope="per_model",
                            model_name=str(model_name),
                            metric_name=metric_name,
                            metric_value=float(model_metrics["logistic"][metric_name]),
                        )
                    )
                for metric_name in ("alarm_before_error_rate", "lead_time_mean", "lead_time_median"):
                    metric_rows.append(
                        _metric_row(
                            slice_label=slice_label,
                            seed=seed,
                            experiment="experiment_a",
                            scope="per_model",
                            model_name=str(model_name),
                            metric_name=metric_name,
                            metric_value=float(model_metrics["lead_time"][metric_name]),
                        )
                    )

            comparison_args = argparse.Namespace(
                primary_k=args.primary_k,
                early_n=args.early_n,
                seed=seed,
                bootstrap_iters=args.bootstrap_iters,
                bootstrap_alpha=args.bootstrap_alpha,
            )
            comparison_meta = _common_dataset_indices_for_primary_k(slice_df, requested_k=args.primary_k)
            common_idx = set(int(v) for v in comparison_meta["common_dataset_indices"])
            if common_idx:
                comparison_df = _build_experiment_b_rows(
                    slice_df.loc[slice_df["dataset_index"].astype(int).isin(common_idx)].copy(),
                    comparison_args,
                    scope_label="common_index",
                )
                metric_rows.append(
                    _metric_row(
                        slice_label=slice_label,
                        seed=seed,
                        experiment="experiment_b",
                        scope="common_index",
                        model_name="all_models",
                        metric_name="common_index_count",
                        metric_value=float(comparison_meta["common_index_count"]),
                    )
                )
                for _, row in comparison_df.iterrows():
                    for metric_name in ("auroc", "auprc", "brier", "ece"):
                        metric_rows.append(
                            _metric_row(
                                slice_label=slice_label,
                                seed=seed,
                                experiment="experiment_b",
                                scope="common_index",
                                model_name=str(row["model_name"]),
                                metric_name=metric_name,
                                metric_value=float(row[metric_name]),
                            )
                        )

    metrics_df = pd.DataFrame(metric_rows)
    if metrics_df.empty:
        raise RuntimeError("Stability suite produced no metric rows.")

    summary_df = (
        metrics_df.groupby(["experiment", "scope", "model_name", "metric_name"], dropna=False)["metric_value"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )

    metrics_path = out_dir / "stability_runs_long.csv"
    summary_csv = out_dir / "stability_summary.csv"
    summary_json = out_dir / "stability_summary.json"
    run_summary_json = out_dir / "stability_run_summary.json"

    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_csv, index=False)
    summary_json.write_text(
        json.dumps(summary_df.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    run_summary_json.write_text(json.dumps(run_summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote stability metrics: {metrics_path}")
    print(f"Wrote stability summary: {summary_csv}")
    print(f"Wrote stability JSON:    {summary_json}")


if __name__ == "__main__":
    main()
