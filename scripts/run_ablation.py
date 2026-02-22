from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sympy as sp
from datasets import load_dataset
from plotly import graph_objects as go
from scipy.stats import kendalltau, mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.step_parser import extract_numeric_tokens, normalize_math_text
from src.evaluation.sympy_judge import (
    StepJudgement,
    build_task_reference,
    judge_step_task_correctness,
    summarize_judgement_records,
)
from src.generation.runner import GenerationConfig, generate_reasoning_trace, load_model_and_tokenizer
from src.metrics.global_dim import participation_ratio
from src.metrics.lid_estimators import abid_local_batch, coefficient_of_variation, lid_mle_batch, twonn_global_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run phase-4 A/B/C ablation experiments.")
    parser.add_argument(
        "--experiment",
        choices=["A", "B", "C", "ALL"],
        default="ALL",
        help="A: early warning, B: model comparison, C: estimator sensitivity, ALL: run all",
    )
    parser.add_argument(
        "--input",
        default="",
        help="Optional path to precomputed step-signal table (.csv or .jsonl).",
    )
    parser.add_argument("--out", default="results/ablation", help="Output directory")
    parser.add_argument(
        "--models",
        default="Qwen/Qwen2.5-0.5B-Instruct,sshleifer/tiny-gpt2",
        help="Comma-separated model ids for generation when --input is not provided.",
    )
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--early-n", type=int, default=2, help="First N steps for early-warning features")
    parser.add_argument("--primary-k", type=int, default=10, help="Primary k for A/B experiments")
    parser.add_argument("--k-values", default="5,10,20,40", help="Comma-separated k values")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _parse_csv_str(raw: str) -> list[str]:
    values = [token.strip() for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("Expected at least one non-empty value.")
    return values


def _parse_csv_int(raw: str) -> list[int]:
    values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _zscore(series: pd.Series) -> pd.Series:
    arr = series.astype(float)
    std = float(arr.std(ddof=0))
    if std < 1e-12:
        return pd.Series(np.zeros(len(arr), dtype=np.float64), index=arr.index)
    return (arr - float(arr.mean())) / std


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _ece_score(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(len(y_true))
    if total == 0:
        return float("nan")
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= left) & (y_prob < right)
        if not np.any(mask):
            continue
        confidence = float(np.mean(y_prob[mask]))
        accuracy = float(np.mean(y_true[mask]))
        ece += abs(confidence - accuracy) * float(np.sum(mask)) / total
    return float(ece)


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    classes = np.unique(y_true)
    if len(classes) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "brier": float("nan"), "ece": float("nan")}
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": _ece_score(y_true, y_prob),
    }


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    pooled = math.sqrt(max(((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2), 1e-12))
    return float((np.mean(a) - np.mean(b)) / pooled)


def _sympy_values_from_text(text: str) -> list[sp.Expr]:
    values: list[sp.Expr] = []
    for token in extract_numeric_tokens(text):
        try:
            values.append(sp.sympify(normalize_math_text(token), evaluate=True))
        except Exception:
            continue
    return values


def _expr_equal(lhs: sp.Expr, rhs: sp.Expr) -> bool:
    try:
        return bool(sp.simplify(lhs - rhs) == 0)
    except Exception:
        try:
            return bool(lhs.equals(rhs))
        except Exception:
            return False


def _final_answer_match(generated_text: str, final_answer: sp.Expr | None) -> bool:
    if final_answer is None:
        return False
    generated_values = _sympy_values_from_text(generated_text)
    return any(_expr_equal(value, final_answer) for value in generated_values)


def _collect_judged_steps(step_texts: list[str], question: str, gold_answer: str) -> list[dict[str, Any]]:
    reference = build_task_reference(question, gold_answer)
    judged: list[dict[str, Any]] = []
    for idx, step_text in enumerate(step_texts, start=1):
        judgement = judge_step_task_correctness(step_text, reference)
        judged.append(
            {
                "step_index": idx,
                "text": step_text,
                "is_correct": bool(judgement.is_correct),
                "parse_fail": bool(judgement.parse_fail),
                "reason": judgement.reason,
                "matched_values": judgement.matched_values or [],
            }
        )
    return judged


def _derive_final_correct(generated_text: str, judged_steps: list[dict[str, Any]], question: str, gold_answer: str) -> bool:
    reference = build_task_reference(question, gold_answer)
    if _final_answer_match(generated_text, reference.final_answer):
        return True
    summary = summarize_judgement_records(judged_steps)
    return bool(
        summary["total_steps"] >= 2
        and summary["correct_rate"] >= 0.999
        and summary["parse_fail_count"] == 0
    )


def _collect_step_signal_table(args: argparse.Namespace) -> pd.DataFrame:
    models = _parse_csv_str(args.models)
    k_values = _parse_csv_int(args.k_values)
    dataset = load_dataset("openai/gsm8k", "main", split=args.split)
    max_index = args.start_index + args.num_samples
    if max_index > len(dataset):
        raise ValueError(
            f"Requested range [{args.start_index}, {max_index}) exceeds split size {len(dataset)}."
        )

    rows: list[dict[str, Any]] = []
    for model_name in models:
        model, tokenizer = load_model_and_tokenizer(model_name)
        cfg = GenerationConfig(
            model_name=model_name,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            collect_token_embeddings=True,
            collect_step_signals=True,
        )
        for dataset_index in range(args.start_index, max_index):
            sample = dataset[int(dataset_index)]
            question = str(sample["question"])
            gold_answer = str(sample["answer"])
            trace = generate_reasoning_trace(question, cfg, model=model, tokenizer=tokenizer)
            step_texts = [str(text) for text in trace.get("step_texts", [])]
            judged_steps = _collect_judged_steps(step_texts, question, gold_answer)
            judged_by_idx = {int(row["step_index"]): row for row in judged_steps}
            final_correct = _derive_final_correct(trace.get("generated_text", ""), judged_steps, question, gold_answer)

            token_embeddings = np.asarray(trace.get("token_embeddings", []), dtype=np.float64)
            step_signal_rows = trace.get("step_signal_rows", [])
            if token_embeddings.ndim != 2:
                continue

            for signal in step_signal_rows:
                step_index = int(signal["step_index"])
                start_token = int(signal["start_token"])
                end_token = int(signal["end_token"])
                if end_token <= start_token:
                    continue
                if start_token < 0 or end_token > token_embeddings.shape[0]:
                    continue

                step_tokens = token_embeddings[start_token:end_token]
                if step_tokens.shape[0] < 3:
                    continue

                judgement = judged_by_idx.get(
                    step_index,
                    {
                        "is_correct": False,
                        "parse_fail": True,
                        "reason": "missing_step_judgement",
                    },
                )
                twonn_value = float(twonn_global_id(step_tokens))
                pr_value = float(participation_ratio(step_tokens))

                for k_requested in k_values:
                    k_effective = max(2, min(int(k_requested), int(step_tokens.shape[0]) - 1))
                    lid_value = float(np.mean(lid_mle_batch(step_tokens, k=k_effective)))
                    abid_value = float(np.mean(abid_local_batch(step_tokens, k=k_effective)))
                    rows.append(
                        {
                            "model_name": model_name,
                            "dataset_split": args.split,
                            "dataset_index": int(dataset_index),
                            "sample_id": f"{model_name}::{dataset_index}",
                            "step_index": step_index,
                            "k_requested": int(k_requested),
                            "k_effective": int(k_effective),
                            "step_token_count": int(step_tokens.shape[0]),
                            "lid": lid_value,
                            "abid": abid_value,
                            "twonn": twonn_value,
                            "pr": pr_value,
                            "entropy": float(signal["entropy_mean"]),
                            "is_correct": bool(judgement["is_correct"]),
                            "parse_fail": bool(judgement["parse_fail"]),
                            "reason": str(judgement["reason"]),
                            "final_correct": bool(final_correct),
                        }
                    )

        del model
        del tokenizer

    if not rows:
        raise RuntimeError("No step-signal rows collected. Increase num-samples or max-new-tokens.")
    return pd.DataFrame(rows)


def _load_input_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        df = pd.DataFrame(rows)
    else:
        raise ValueError("Unsupported input format. Use .csv or .jsonl.")

    required = {
        "model_name",
        "sample_id",
        "step_index",
        "k_requested",
        "lid",
        "abid",
        "entropy",
        "is_correct",
        "parse_fail",
        "final_correct",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input table missing required columns: {missing}")

    bool_cols = ["is_correct", "parse_fail", "final_correct"]
    for col in bool_cols:
        if df[col].dtype != bool:
            df[col] = df[col].astype(str).str.lower().map({"true": True, "false": False}).fillna(False)

    int_cols = ["step_index", "k_requested"]
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def _nearest_k(df: pd.DataFrame, k: int) -> int:
    available = sorted(set(int(v) for v in df["k_requested"].tolist()))
    if not available:
        raise ValueError("No k values available in step table.")
    return min(available, key=lambda v: abs(v - int(k)))


def _build_sample_features(df_steps: pd.DataFrame, early_n: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sample_id, group in df_steps.groupby("sample_id", sort=False):
        g = group.sort_values("step_index")
        prefix = g.head(max(1, early_n))
        if prefix.empty:
            continue
        rows.append(
            {
                "sample_id": sample_id,
                "model_name": str(g["model_name"].iloc[0]),
                "k_requested": int(g["k_requested"].iloc[0]),
                "final_failure": int(not bool(g["final_correct"].iloc[0])),
                "lid_prefix_mean": float(prefix["lid"].mean()),
                "abid_prefix_mean": float(prefix["abid"].mean()),
                "entropy_prefix_mean": float(prefix["entropy"].mean()),
                "parse_fail_prefix_rate": float(prefix["parse_fail"].astype(int).mean()),
                "step_count": int(g["step_index"].max()),
            }
        )
    return pd.DataFrame(rows)


def _cross_validated_probs(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return np.full_like(y, fill_value=float(np.mean(y)), dtype=np.float64)
    min_class = int(counts.min())
    n_splits = min(5, min_class)
    if n_splits < 2:
        return np.full_like(y, fill_value=float(np.mean(y)), dtype=np.float64)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    probs = cross_val_predict(clf, x, y, cv=cv, method="predict_proba")[:, 1]
    return probs.astype(np.float64)


def _compute_lead_time(step_df: pd.DataFrame, threshold: float) -> dict[str, float]:
    rows: list[float] = []
    for _, group in step_df.groupby("sample_id", sort=False):
        g = group.sort_values("step_index")
        if bool(g["final_correct"].iloc[0]):
            continue
        wrong = g.loc[(~g["is_correct"]) | (g["parse_fail"]), "step_index"]
        if wrong.empty:
            continue
        alarms = g.loc[g["warning_score"] >= threshold, "step_index"]
        if alarms.empty:
            continue
        first_wrong = int(wrong.iloc[0])
        first_alarm = int(alarms.iloc[0])
        rows.append(float(first_wrong - first_alarm))
    if not rows:
        return {
            "n_failed_with_alarm": 0.0,
            "lead_time_mean": float("nan"),
            "lead_time_median": float("nan"),
            "alarm_before_error_rate": float("nan"),
        }
    arr = np.asarray(rows, dtype=np.float64)
    return {
        "n_failed_with_alarm": float(len(arr)),
        "lead_time_mean": float(np.mean(arr)),
        "lead_time_median": float(np.median(arr)),
        "alarm_before_error_rate": float(np.mean(arr > 0)),
    }


def _run_experiment_a_core(
    df: pd.DataFrame,
    requested_k: int,
    early_n: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    k_use = _nearest_k(df, requested_k)
    step_df = df.loc[df["k_requested"] == k_use].copy()
    step_df["warning_score"] = _zscore(step_df["lid"]) + _zscore(step_df["entropy"])
    alarm_threshold = float(step_df["warning_score"].quantile(0.8))
    sample_df = _build_sample_features(step_df, early_n=early_n)
    if sample_df.empty:
        raise RuntimeError("No sample-level features could be built for experiment A.")

    feature_cols = [
        "lid_prefix_mean",
        "abid_prefix_mean",
        "entropy_prefix_mean",
        "parse_fail_prefix_rate",
    ]
    x = sample_df[feature_cols].to_numpy(dtype=np.float64)
    y = sample_df["final_failure"].to_numpy(dtype=np.int32)
    probs = _cross_validated_probs(x, y, seed=seed)

    warning_prefix = _zscore(sample_df["lid_prefix_mean"]) + _zscore(sample_df["entropy_prefix_mean"])
    baseline_probs = _sigmoid(warning_prefix.to_numpy(dtype=np.float64))

    metrics = {
        "k_used": int(k_use),
        "n_samples": int(len(sample_df)),
        "failure_rate": float(np.mean(y)),
        "logistic": _binary_metrics(y, probs),
        "baseline_threshold": _binary_metrics(y, baseline_probs),
        "lead_time": _compute_lead_time(step_df, threshold=alarm_threshold),
        "alarm_threshold": alarm_threshold,
    }
    sample_df["failure_prob_logistic"] = probs
    sample_df["failure_prob_baseline"] = baseline_probs
    return metrics, sample_df, step_df


def _run_experiment_a(df: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    metrics, sample_df, step_df = _run_experiment_a_core(
        df=df,
        requested_k=args.primary_k,
        early_n=args.early_n,
        seed=args.seed,
    )
    sample_path = out_dir / "experiment_a_sample_predictions.csv"
    step_path = out_dir / "experiment_a_step_scores.csv"
    summary_path = out_dir / "experiment_a_summary.json"
    sample_df.to_csv(sample_path, index=False)
    step_df.to_csv(step_path, index=False)
    summary_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"summary": metrics, "sample_path": str(sample_path), "step_path": str(step_path)}


def _run_experiment_b(df: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for model_name, model_df in df.groupby("model_name", sort=False):
        metrics, _, step_df = _run_experiment_a_core(
            df=model_df,
            requested_k=args.primary_k,
            early_n=args.early_n,
            seed=args.seed,
        )
        correct_mask = step_df["is_correct"] & (~step_df["parse_fail"])
        wrong_mask = (~step_df["is_correct"]) | (step_df["parse_fail"])
        correct_scores = step_df.loc[correct_mask, "warning_score"].to_numpy(dtype=np.float64)
        wrong_scores = step_df.loc[wrong_mask, "warning_score"].to_numpy(dtype=np.float64)
        mwu_p = float("nan")
        if len(correct_scores) > 0 and len(wrong_scores) > 0:
            try:
                mwu_p = float(mannwhitneyu(wrong_scores, correct_scores, alternative="two-sided").pvalue)
            except Exception:
                mwu_p = float("nan")
        rows.append(
            {
                "model_name": model_name,
                "k_used": int(metrics["k_used"]),
                "n_samples": int(metrics["n_samples"]),
                "failure_rate": float(metrics["failure_rate"]),
                "auroc": float(metrics["logistic"]["auroc"]),
                "auprc": float(metrics["logistic"]["auprc"]),
                "brier": float(metrics["logistic"]["brier"]),
                "ece": float(metrics["logistic"]["ece"]),
                "signal_gap_mean": float(np.mean(wrong_scores) - np.mean(correct_scores))
                if len(correct_scores) > 0 and len(wrong_scores) > 0
                else float("nan"),
                "cohen_d_wrong_vs_correct": _cohen_d(wrong_scores, correct_scores),
                "mann_whitney_pvalue": mwu_p,
            }
        )

    comparison_df = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    csv_path = out_dir / "experiment_b_model_comparison.csv"
    json_path = out_dir / "experiment_b_model_comparison.json"
    comparison_df.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(comparison_df.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not comparison_df.empty:
        fig = go.Figure(
            data=[
                go.Bar(
                    x=comparison_df["model_name"].tolist(),
                    y=comparison_df["auroc"].tolist(),
                    name="AUROC",
                )
            ]
        )
        fig.update_layout(title="Experiment B - Model Comparison (AUROC)", template="plotly_white")
        fig.write_html(str(out_dir / "experiment_b_model_comparison.html"))

    return {"rows": comparison_df.to_dict(orient="records"), "csv_path": str(csv_path)}


def _run_experiment_c(df: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    model_outputs: list[dict[str, Any]] = []
    for model_name, model_df in df.groupby("model_name", sort=False):
        available_k = sorted(set(int(v) for v in model_df["k_requested"].tolist()))
        k_rows: list[dict[str, Any]] = []
        preds_by_k: dict[int, pd.DataFrame] = {}
        for k in available_k:
            metrics, sample_df, _ = _run_experiment_a_core(
                df=model_df,
                requested_k=k,
                early_n=args.early_n,
                seed=args.seed,
            )
            k_rows.append(
                {
                    "model_name": model_name,
                    "k_requested": int(k),
                    "k_used": int(metrics["k_used"]),
                    "n_samples": int(metrics["n_samples"]),
                    "auroc": float(metrics["logistic"]["auroc"]),
                    "auprc": float(metrics["logistic"]["auprc"]),
                    "brier": float(metrics["logistic"]["brier"]),
                    "ece": float(metrics["logistic"]["ece"]),
                }
            )
            preds_by_k[int(k)] = sample_df[["sample_id", "failure_prob_logistic"]].copy()

        metrics_df = pd.DataFrame(k_rows)
        auroc_values = metrics_df["auroc"].dropna().to_numpy(dtype=np.float64)
        cv_auroc = coefficient_of_variation(auroc_values) if len(auroc_values) > 0 else float("nan")

        tau_rows: list[dict[str, float]] = []
        if available_k:
            anchor_k = available_k[0]
            anchor_preds = preds_by_k[anchor_k]
            for k in available_k[1:]:
                merged = anchor_preds.merge(preds_by_k[k], on="sample_id", suffixes=("_anchor", "_other"))
                if len(merged) >= 2:
                    tau = kendalltau(
                        merged["failure_prob_logistic_anchor"],
                        merged["failure_prob_logistic_other"],
                    ).correlation
                    tau_rows.append({"anchor_k": float(anchor_k), "other_k": float(k), "kendall_tau": float(tau)})

        model_outputs.append(
            {
                "model_name": model_name,
                "metrics_by_k": metrics_df.to_dict(orient="records"),
                "auroc_cv_over_k": float(cv_auroc),
                "kendall_tau_pairs": tau_rows,
                "kendall_tau_mean": float(np.mean([row["kendall_tau"] for row in tau_rows]))
                if tau_rows
                else float("nan"),
            }
        )

        if not metrics_df.empty:
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=metrics_df["k_requested"].tolist(),
                        y=metrics_df["auroc"].tolist(),
                        mode="lines+markers",
                        name=model_name,
                    )
                ]
            )
            fig.update_layout(
                title=f"Experiment C - k Sensitivity ({model_name})",
                xaxis_title="k",
                yaxis_title="AUROC",
                template="plotly_white",
            )
            safe_name = model_name.replace("/", "_")
            fig.write_html(str(out_dir / f"experiment_c_k_sensitivity_{safe_name}.html"))

    flat_rows: list[dict[str, Any]] = []
    for model_output in model_outputs:
        for row in model_output["metrics_by_k"]:
            flat_rows.append(row)
    metrics_csv = out_dir / "experiment_c_k_sensitivity.csv"
    pd.DataFrame(flat_rows).to_csv(metrics_csv, index=False)

    summary_json = out_dir / "experiment_c_k_sensitivity.json"
    summary_json.write_text(json.dumps(model_outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"model_outputs": model_outputs, "csv_path": str(metrics_csv)}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        step_df = _load_input_table(Path(args.input))
    else:
        step_df = _collect_step_signal_table(args)
        step_df.to_csv(out_dir / "step_signal_table.csv", index=False)
        with (out_dir / "step_signal_table.jsonl").open("w", encoding="utf-8") as fp:
            for row in step_df.to_dict(orient="records"):
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    output: dict[str, Any] = {
        "rows": int(len(step_df)),
        "models": sorted(set(step_df["model_name"].astype(str).tolist())),
        "k_values": sorted(set(int(v) for v in step_df["k_requested"].tolist())),
    }
    if args.experiment in {"A", "ALL"}:
        output["experiment_a"] = _run_experiment_a(step_df, args, out_dir)
    if args.experiment in {"B", "ALL"}:
        output["experiment_b"] = _run_experiment_b(step_df, args, out_dir)
    if args.experiment in {"C", "ALL"}:
        output["experiment_c"] = _run_experiment_c(step_df, args, out_dir)

    summary_path = out_dir / "ablation_summary.json"
    summary_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote ablation outputs to {out_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
