from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

REASON_WARNING_WEIGHTS = {
    "equation_mismatch": 1.20,
    "sympy_parse_error": 0.70,
    "unsupported_symbolic_form": 0.60,
    "boundary_detected_but_empty_math": 0.45,
    "trace_signal_fail": 0.90,
    "trace_format_fail": 1.05,
    "empty_completion": 1.20,
    "missing_step_judgement": 0.50,
    "equation_consistent_unanchored": 0.25,
    "numeric_off_reference": 0.15,
    "no_math_signal": 0.00,
    "equation_matches_reference": -0.25,
    "equation_consistent_supported": -0.15,
    "non_equational_numeric_reasoning": -0.05,
}

TRACE_FAILURE_REASONS = {
    "trace_signal_fail",
    "trace_format_fail",
    "empty_completion",
}


@dataclass(frozen=True)
class AlarmPolicySpec:
    name: str
    score_col: str
    threshold_kind: str
    threshold_value: float
    consecutive_n: int = 1
    rolling_k: int = 0
    rolling_m: int = 0
    base_floor_quantile: float | None = None


def _zscore(series: pd.Series) -> pd.Series:
    arr = series.astype(float)
    std = float(arr.std(ddof=0))
    if std < 1e-12:
        return pd.Series(np.zeros(len(arr), dtype=np.float64), index=arr.index)
    return (arr - float(arr.mean())) / std


def _reason_weight_series(reason_series: pd.Series) -> pd.Series:
    normalized = reason_series.fillna("").astype(str)
    return normalized.map(REASON_WARNING_WEIGHTS).fillna(0.0).astype(np.float64)


def _impute_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype(np.float64)
    finite = numeric[np.isfinite(numeric)]
    if len(finite) == 0:
        fill_value = float(default)
    else:
        fill_value = float(np.median(finite))
    return numeric.fillna(fill_value)


def prepare_warning_features(step_df: pd.DataFrame) -> pd.DataFrame:
    out = step_df.sort_values(["sample_id", "step_index"], kind="mergesort").copy()
    out["parse_fail_int"] = out["parse_fail"].astype(int)
    if "reason" not in out.columns:
        out["reason"] = ""
    out["reason_weight"] = _reason_weight_series(out["reason"])
    lid_signal = _impute_numeric(out["lid"])
    entropy_signal = _impute_numeric(out["entropy"])
    out["trace_failure_int"] = out["reason"].isin(TRACE_FAILURE_REASONS).astype(int)
    out["warning_score"] = (
        _zscore(lid_signal) + _zscore(entropy_signal) + out["reason_weight"] + 0.20 * out["parse_fail_int"]
    )

    out["warning_score_delta"] = (
        out.groupby("sample_id", sort=False)["warning_score"].diff().fillna(0.0).astype(np.float64)
    )
    out["entropy_delta"] = entropy_signal.groupby(out["sample_id"], sort=False).diff().fillna(0.0).astype(np.float64)

    positive_delta = out["warning_score_delta"].clip(lower=0.0)
    positive_entropy_delta = out["entropy_delta"].clip(lower=0.0)
    out["hybrid_warning_score"] = (
        out["warning_score"] + 0.75 * _zscore(positive_delta) + 0.35 * _zscore(positive_entropy_delta)
    )
    running_warning_mean = (
        out.groupby("sample_id", sort=False)["warning_score"]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
        .astype(np.float64)
    )
    running_warning_peak = out.groupby("sample_id", sort=False)["warning_score"].cummax().astype(np.float64)
    out["trajectory_warning_score"] = (
        out["warning_score"]
        + 0.45 * _zscore(running_warning_mean)
        + 0.35 * _zscore(running_warning_peak)
        + 0.40 * _zscore(positive_delta)
        + 0.20 * _zscore(positive_entropy_delta)
    )
    return out


def default_alarm_policies() -> list[AlarmPolicySpec]:
    return [
        AlarmPolicySpec("static_q70", "warning_score", "quantile", 0.70),
        AlarmPolicySpec("static_q75", "warning_score", "quantile", 0.75),
        AlarmPolicySpec("static_q80", "warning_score", "quantile", 0.80),
        AlarmPolicySpec("zscore_1p00", "warning_score", "zscore", 1.00),
        AlarmPolicySpec("consecutive2_q65", "warning_score", "quantile", 0.65, consecutive_n=2),
        AlarmPolicySpec("consecutive2_q70", "warning_score", "quantile", 0.70, consecutive_n=2),
        AlarmPolicySpec("hybrid_q75", "hybrid_warning_score", "quantile", 0.75),
        AlarmPolicySpec("hybrid_consecutive2_q68", "hybrid_warning_score", "quantile", 0.68, consecutive_n=2),
        AlarmPolicySpec("hybrid_2of3_q70", "hybrid_warning_score", "quantile", 0.70, rolling_k=2, rolling_m=3),
        AlarmPolicySpec("trajectory_q68", "trajectory_warning_score", "quantile", 0.68),
        AlarmPolicySpec("trajectory_2of3_q65", "trajectory_warning_score", "quantile", 0.65, rolling_k=2, rolling_m=3),
        AlarmPolicySpec(
            "delta_q80_floor_q60",
            "warning_score_delta",
            "quantile",
            0.80,
            base_floor_quantile=0.60,
        ),
        AlarmPolicySpec(
            "delta_q75_floor_q55",
            "warning_score_delta",
            "quantile",
            0.75,
            base_floor_quantile=0.55,
        ),
    ]


def _fit_threshold(train_df: pd.DataFrame, spec: AlarmPolicySpec) -> dict[str, Any]:
    series = train_df[spec.score_col].astype(float)
    if spec.threshold_kind == "quantile":
        threshold = float(series.quantile(spec.threshold_value))
    elif spec.threshold_kind == "zscore":
        threshold = float(series.mean() + spec.threshold_value * series.std(ddof=0))
    else:
        raise ValueError(f"Unsupported threshold kind: {spec.threshold_kind}")

    fitted: dict[str, Any] = {"threshold": threshold}
    if spec.base_floor_quantile is not None:
        fitted["base_floor"] = float(train_df["warning_score"].astype(float).quantile(spec.base_floor_quantile))
    return fitted


def _apply_base_alarm(df: pd.DataFrame, spec: AlarmPolicySpec, fitted: dict[str, Any]) -> pd.Series:
    base = df[spec.score_col].astype(float) >= float(fitted["threshold"])
    if spec.base_floor_quantile is not None:
        base &= df["warning_score"].astype(float) >= float(fitted["base_floor"])
    return base.astype(bool)


def _apply_persistence(df: pd.DataFrame, base_alarm: pd.Series, spec: AlarmPolicySpec) -> pd.Series:
    out = pd.Series(False, index=df.index)
    for _, group in df.groupby("sample_id", sort=False):
        mask = base_alarm.loc[group.index].astype(int)
        if spec.consecutive_n > 1:
            persisted = mask.rolling(spec.consecutive_n, min_periods=spec.consecutive_n).sum() >= spec.consecutive_n
        elif spec.rolling_k > 0 and spec.rolling_m > 0:
            persisted = mask.rolling(spec.rolling_m, min_periods=spec.rolling_m).sum() >= spec.rolling_k
        else:
            persisted = mask.astype(bool)
        out.loc[group.index] = persisted.astype(bool).to_numpy()
    return out


def apply_alarm_policy(step_df: pd.DataFrame, spec: AlarmPolicySpec, fitted: dict[str, Any]) -> pd.DataFrame:
    out = step_df.sort_values(["sample_id", "step_index"], kind="mergesort").copy()
    out["alarm_base"] = _apply_base_alarm(out, spec, fitted)
    out["alarm_selected"] = _apply_persistence(out, out["alarm_base"], spec)
    return out


def build_sample_alarm_table(step_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sample_id, group in step_df.groupby("sample_id", sort=False):
        g = group.sort_values("step_index", kind="mergesort")
        failure = int(not bool(g["final_correct"].iloc[0]))
        error_steps = g.loc[(~g["is_correct"]) | (g["parse_fail"]), "step_index"]
        alarm_steps = g.loc[g["alarm_selected"], "step_index"]

        first_error = float(error_steps.iloc[0]) if not error_steps.empty else float("nan")
        first_alarm = float(alarm_steps.iloc[0]) if not alarm_steps.empty else float("nan")
        lead_time = float(first_error - first_alarm) if np.isfinite(first_error) and np.isfinite(first_alarm) else float("nan")

        rows.append(
            {
                "sample_id": sample_id,
                "model_name": str(g["model_name"].iloc[0]),
                "dataset_index": int(g["dataset_index"].iloc[0]),
                "question_id": str(g["question_id"].iloc[0]),
                "final_failure": failure,
                "step_count": int(g["step_index"].max()),
                "first_alarm_step": first_alarm,
                "first_error_step": first_error,
                "lead_time": lead_time,
                "alarm_before_error": bool(
                    failure and np.isfinite(first_alarm) and np.isfinite(first_error) and first_alarm < first_error
                ),
                "late_alarm": bool(
                    failure and np.isfinite(first_alarm) and np.isfinite(first_error) and first_alarm >= first_error
                ),
                "missed_alarm": bool(failure and not np.isfinite(first_alarm)),
                "false_alarm_before_any_error": bool((1 - failure) and np.isfinite(first_alarm)),
            }
        )
    return pd.DataFrame(rows)


def summarize_alarm_metrics(sample_alarm_df: pd.DataFrame) -> dict[str, float]:
    if sample_alarm_df.empty:
        return {
            "n_samples": 0.0,
            "n_failed_samples": 0.0,
            "n_success_samples": 0.0,
            "n_failed_with_alarm": 0.0,
            "lead_time_mean": float("nan"),
            "lead_time_median": float("nan"),
            "alarm_before_error_rate": float("nan"),
            "false_alarm_before_any_error_rate": float("nan"),
            "late_alarm_rate": float("nan"),
            "missed_alarm_rate": float("nan"),
            "first_alarm_step_mean": float("nan"),
            "first_error_step_mean": float("nan"),
            "any_alarm_rate": float("nan"),
        }

    failed = sample_alarm_df.loc[sample_alarm_df["final_failure"] == 1].copy()
    success = sample_alarm_df.loc[sample_alarm_df["final_failure"] == 0].copy()
    failed_with_alarm = failed.loc[failed["first_alarm_step"].notna()].copy()
    lead = failed_with_alarm["lead_time"].dropna().to_numpy(dtype=np.float64)

    return {
        "n_samples": float(len(sample_alarm_df)),
        "n_failed_samples": float(len(failed)),
        "n_success_samples": float(len(success)),
        "n_failed_with_alarm": float(len(failed_with_alarm)),
        "lead_time_mean": float(np.mean(lead)) if len(lead) > 0 else float("nan"),
        "lead_time_median": float(np.median(lead)) if len(lead) > 0 else float("nan"),
        "alarm_before_error_rate": float(failed["alarm_before_error"].astype(float).mean()) if len(failed) > 0 else float("nan"),
        "false_alarm_before_any_error_rate": float(success["false_alarm_before_any_error"].astype(float).mean())
        if len(success) > 0
        else float("nan"),
        "late_alarm_rate": float(failed["late_alarm"].astype(float).mean()) if len(failed) > 0 else float("nan"),
        "missed_alarm_rate": float(failed["missed_alarm"].astype(float).mean()) if len(failed) > 0 else float("nan"),
        "first_alarm_step_mean": float(sample_alarm_df["first_alarm_step"].dropna().mean())
        if sample_alarm_df["first_alarm_step"].notna().any()
        else float("nan"),
        "first_error_step_mean": float(failed["first_error_step"].dropna().mean())
        if failed["first_error_step"].notna().any()
        else float("nan"),
        "any_alarm_rate": float(sample_alarm_df["first_alarm_step"].notna().astype(float).mean()),
    }


def compute_early_objective(metrics: dict[str, float]) -> float:
    alarm_before = float(metrics.get("alarm_before_error_rate", float("nan")))
    false_alarm = float(metrics.get("false_alarm_before_any_error_rate", float("nan")))
    late_alarm = float(metrics.get("late_alarm_rate", float("nan")))
    missed_alarm = float(metrics.get("missed_alarm_rate", float("nan")))
    lead_time = float(metrics.get("lead_time_mean", float("nan")))

    if not np.isfinite(alarm_before):
        return float("-inf")

    normalized_lead = float(np.clip(lead_time, -3.0, 3.0) / 3.0) if np.isfinite(lead_time) else -1.0
    return float(1.8 * alarm_before - 0.9 * false_alarm - 0.7 * late_alarm - 0.5 * missed_alarm + 0.35 * normalized_lead)


def _sample_targets(step_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sample_id, group in step_df.groupby("sample_id", sort=False):
        rows.append(
            {
                "sample_id": sample_id,
                "final_failure": int(not bool(group["final_correct"].iloc[0])),
            }
        )
    return pd.DataFrame(rows)


def _evaluate_policy_cv(
    step_df: pd.DataFrame,
    spec: AlarmPolicySpec,
    *,
    seed: int,
) -> tuple[dict[str, float], pd.DataFrame, list[dict[str, Any]]]:
    sample_targets = _sample_targets(step_df)
    y = sample_targets["final_failure"].to_numpy(dtype=np.int32)
    classes, counts = np.unique(y, return_counts=True)
    n_splits = min(5, int(counts.min())) if len(classes) >= 2 else 0

    sample_details: list[pd.DataFrame] = []
    fold_details: list[dict[str, Any]] = []

    if len(classes) < 2 or n_splits < 2:
        fitted = _fit_threshold(step_df, spec)
        applied = apply_alarm_policy(step_df, spec, fitted)
        samples = build_sample_alarm_table(applied)
        metrics = summarize_alarm_metrics(samples)
        metrics["cv_folds"] = 1.0
        return metrics, samples, [{"fold_index": 0, **fitted}]

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_index, (train_idx, test_idx) in enumerate(cv.split(sample_targets["sample_id"], y)):
        train_ids = set(sample_targets.iloc[train_idx]["sample_id"].tolist())
        test_ids = set(sample_targets.iloc[test_idx]["sample_id"].tolist())
        train_df = step_df.loc[step_df["sample_id"].isin(train_ids)].copy()
        test_df = step_df.loc[step_df["sample_id"].isin(test_ids)].copy()
        fitted = _fit_threshold(train_df, spec)
        applied = apply_alarm_policy(test_df, spec, fitted)
        samples = build_sample_alarm_table(applied)
        samples["fold_index"] = fold_index
        sample_details.append(samples)
        fold_details.append({"fold_index": fold_index, **fitted})

    sample_alarm_df = pd.concat(sample_details, ignore_index=True) if sample_details else pd.DataFrame()
    metrics = summarize_alarm_metrics(sample_alarm_df)
    metrics["cv_folds"] = float(n_splits)
    return metrics, sample_alarm_df, fold_details


def build_threshold_sweep(step_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for quantile in np.linspace(0.55, 0.90, 8):
        spec = AlarmPolicySpec(
            name=f"threshold_q{int(round(float(quantile) * 100)):02d}",
            score_col="warning_score",
            threshold_kind="quantile",
            threshold_value=float(quantile),
        )
        fitted = _fit_threshold(step_df, spec)
        applied = apply_alarm_policy(step_df, spec, fitted)
        metrics = summarize_alarm_metrics(build_sample_alarm_table(applied))
        rows.append(
            {
                "policy_name": spec.name,
                "quantile": float(quantile),
                "threshold": float(fitted["threshold"]),
                "early_objective": compute_early_objective(metrics),
                "alarm_before_error_rate": float(metrics["alarm_before_error_rate"]),
                "false_alarm_before_any_error_rate": float(metrics["false_alarm_before_any_error_rate"]),
                "late_alarm_rate": float(metrics["late_alarm_rate"]),
                "lead_time_mean": float(metrics["lead_time_mean"]),
            }
        )
    return pd.DataFrame(rows)


def build_warning_trajectory(step_df: pd.DataFrame) -> pd.DataFrame:
    out = step_df.copy()
    out["trace_outcome"] = np.where(out["final_correct"].astype(bool), "success", "failure")
    grouped = (
        out.groupby(["trace_outcome", "step_index"], dropna=False)
        .agg(
            warning_score_mean=("warning_score", "mean"),
            hybrid_warning_score_mean=("hybrid_warning_score", "mean"),
            trajectory_warning_score_mean=("trajectory_warning_score", "mean"),
            count=("sample_id", "size"),
        )
        .reset_index()
    )
    return grouped


def evaluate_alarm_policies(step_df: pd.DataFrame, *, seed: int) -> dict[str, Any]:
    scored = prepare_warning_features(step_df)
    comparison_rows: list[dict[str, Any]] = []
    policy_samples: dict[str, pd.DataFrame] = {}
    policy_folds: dict[str, list[dict[str, Any]]] = {}

    for spec in default_alarm_policies():
        metrics, samples, folds = _evaluate_policy_cv(scored, spec, seed=seed)
        policy_samples[spec.name] = samples
        policy_folds[spec.name] = folds
        comparison_rows.append(
            {
                "policy_name": spec.name,
                "score_col": spec.score_col,
                "threshold_kind": spec.threshold_kind,
                "threshold_value": float(spec.threshold_value),
                "consecutive_n": int(spec.consecutive_n),
                "rolling_k": int(spec.rolling_k),
                "rolling_m": int(spec.rolling_m),
                "base_floor_quantile": float(spec.base_floor_quantile)
                if spec.base_floor_quantile is not None
                else float("nan"),
                "early_objective": compute_early_objective(metrics),
                **metrics,
            }
        )

    comparison_df = pd.DataFrame(comparison_rows).sort_values("early_objective", ascending=False).reset_index(drop=True)
    if comparison_df.empty:
        raise RuntimeError("No alarm policy candidates were evaluated.")

    selected_name = str(comparison_df.iloc[0]["policy_name"])
    selected_spec = next(spec for spec in default_alarm_policies() if spec.name == selected_name)
    fitted_full = _fit_threshold(scored, selected_spec)
    selected_step_df = apply_alarm_policy(scored, selected_spec, fitted_full)
    selected_sample_df = build_sample_alarm_table(selected_step_df)
    selected_metrics = summarize_alarm_metrics(selected_sample_df)
    selected_metrics["early_objective"] = compute_early_objective(selected_metrics)

    return {
        "selected_policy": {
            "policy_name": selected_spec.name,
            "score_col": selected_spec.score_col,
            "threshold_kind": selected_spec.threshold_kind,
            "threshold_value": float(selected_spec.threshold_value),
            "consecutive_n": int(selected_spec.consecutive_n),
            "rolling_k": int(selected_spec.rolling_k),
            "rolling_m": int(selected_spec.rolling_m),
            "base_floor_quantile": float(selected_spec.base_floor_quantile)
            if selected_spec.base_floor_quantile is not None
            else float("nan"),
            **fitted_full,
        },
        "selected_metrics": selected_metrics,
        "policy_comparison": comparison_df,
        "selected_step_df": selected_step_df,
        "selected_sample_df": selected_sample_df,
        "selected_policy_fold_fits": policy_folds.get(selected_name, []),
        "threshold_sweep": build_threshold_sweep(scored),
        "warning_trajectory": build_warning_trajectory(scored),
    }
