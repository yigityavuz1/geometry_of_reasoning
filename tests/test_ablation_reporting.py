from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_ablation import (
    _bootstrap_binary_metrics,
    _build_sample_features,
    _common_dataset_indices_for_layer_map,
    _common_dataset_indices_for_primary_k,
    _layer_recommendation_from_df,
)


def test_bootstrap_binary_metrics_returns_confidence_ranges() -> None:
    y_true = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
    y_prob = np.asarray([0.12, 0.92, 0.24, 0.82, 0.31, 0.73, 0.41, 0.61], dtype=np.float64)
    ci = _bootstrap_binary_metrics(
        y_true,
        y_prob,
        n_bootstrap=80,
        alpha=0.1,
        seed=7,
    )

    assert set(ci.keys()) == {"auroc", "auprc", "brier", "ece"}
    for metric in ("auroc", "auprc", "brier", "ece"):
        bounds = ci[metric]
        assert float(bounds["valid_bootstrap_samples"]) > 0
        assert float(bounds["low"]) <= float(bounds["high"])


def test_common_dataset_indices_for_primary_k_intersection() -> None:
    df = pd.DataFrame(
        [
            {"model_name": "model-a", "k_requested": 20, "dataset_index": 0},
            {"model_name": "model-a", "k_requested": 20, "dataset_index": 1},
            {"model_name": "model-a", "k_requested": 40, "dataset_index": 5},
            {"model_name": "model-b", "k_requested": 20, "dataset_index": 1},
            {"model_name": "model-b", "k_requested": 20, "dataset_index": 2},
        ]
    )

    common = _common_dataset_indices_for_primary_k(df, requested_k=20)
    assert common["requested_primary_k"] == 20
    assert common["common_index_count"] == 1
    assert common["common_dataset_indices"] == [1]
    assert common["k_used_by_model"] == {"model-a": 20, "model-b": 20}


def test_common_dataset_indices_respects_analysis_layer() -> None:
    df = pd.DataFrame(
        [
            {"model_name": "model-a", "layer_name": "early", "k_requested": 20, "dataset_index": 0},
            {"model_name": "model-a", "layer_name": "late", "k_requested": 20, "dataset_index": 1},
            {"model_name": "model-b", "layer_name": "early", "k_requested": 20, "dataset_index": 0},
            {"model_name": "model-b", "layer_name": "late", "k_requested": 20, "dataset_index": 2},
        ]
    )

    common = _common_dataset_indices_for_primary_k(df, requested_k=20, analysis_layer="early")
    assert common["analysis_layer"] == "early"
    assert common["common_index_count"] == 1
    assert common["common_dataset_indices"] == [0]


def test_common_dataset_indices_supports_per_model_best_layer_map() -> None:
    df = pd.DataFrame(
        [
            {"model_name": "model-a", "layer_name": "early", "k_requested": 20, "dataset_index": 0},
            {"model_name": "model-a", "layer_name": "early", "k_requested": 20, "dataset_index": 1},
            {"model_name": "model-a", "layer_name": "late", "k_requested": 20, "dataset_index": 2},
            {"model_name": "model-b", "layer_name": "early", "k_requested": 20, "dataset_index": 1},
            {"model_name": "model-b", "layer_name": "late", "k_requested": 20, "dataset_index": 0},
            {"model_name": "model-b", "layer_name": "late", "k_requested": 20, "dataset_index": 1},
        ]
    )

    common = _common_dataset_indices_for_layer_map(
        df,
        requested_k=20,
        analysis_layer_by_model={"model-a": "early", "model-b": "late"},
    )
    assert common["analysis_layer_by_model"] == {"model-a": "early", "model-b": "late"}
    assert common["common_dataset_indices"] == [0, 1]


def test_layer_recommendation_prefers_auroc_then_timing() -> None:
    layer_df = pd.DataFrame(
        [
            {
                "analysis_layer": "late",
                "analysis_layer_index": 28,
                "auroc": 0.82,
                "auprc": 0.60,
                "brier": 0.18,
                "ece": 0.11,
                "lead_time_mean": 0.4,
                "alarm_before_error_rate": 0.20,
            },
            {
                "analysis_layer": "middle",
                "analysis_layer_index": 14,
                "auroc": 0.82,
                "auprc": 0.58,
                "brier": 0.19,
                "ece": 0.12,
                "lead_time_mean": 0.9,
                "alarm_before_error_rate": 0.35,
            },
        ]
    )

    recommendation = _layer_recommendation_from_df(layer_df)
    assert recommendation["analysis_layer"] == "middle"
    assert recommendation["analysis_layer_index"] == 14


def test_layer_recommendation_can_optimize_early_warning_objective() -> None:
    layer_df = pd.DataFrame(
        [
            {
                "analysis_layer": "late",
                "analysis_layer_index": 28,
                "auroc": 0.84,
                "auprc": 0.62,
                "brier": 0.17,
                "ece": 0.10,
                "lead_time_mean": -0.2,
                "alarm_before_error_rate": 0.18,
                "early_objective": 0.05,
                "selected_policy_name": "static_q75",
            },
            {
                "analysis_layer": "middle",
                "analysis_layer_index": 14,
                "auroc": 0.79,
                "auprc": 0.59,
                "brier": 0.18,
                "ece": 0.11,
                "lead_time_mean": 0.8,
                "alarm_before_error_rate": 0.36,
                "early_objective": 0.41,
                "selected_policy_name": "trajectory_q68",
            },
        ]
    )

    recommendation = _layer_recommendation_from_df(layer_df, objective="early_warning")
    assert recommendation["analysis_layer"] == "middle"
    assert recommendation["analysis_layer_index"] == 14
    assert recommendation["selection_metric"] == "early_objective_then_alarm_before_error_rate_then_lead_time_mean_then_auroc"
    assert recommendation["selected_policy_name"] == "trajectory_q68"


def test_build_sample_features_keeps_trace_failure_sample() -> None:
    df = pd.DataFrame(
        [
            {
                "sample_id": "sample-a",
                "model_name": "model-a",
                "dataset_index": 10,
                "question_id": "q10",
                "k_requested": 20,
                "step_index": 0,
                "lid": np.nan,
                "abid": np.nan,
                "twonn": np.nan,
                "pr": np.nan,
                "entropy": np.nan,
                "parse_fail": True,
                "reason": "trace_format_fail",
                "reason_weight": 1.05,
                "warning_score": 1.4,
                "warning_score_delta": 0.0,
                "hybrid_warning_score": 1.6,
                "entropy_delta": 0.0,
                "final_correct": False,
            },
            {
                "sample_id": "sample-b",
                "model_name": "model-a",
                "dataset_index": 11,
                "question_id": "q11",
                "k_requested": 20,
                "step_index": 1,
                "lid": 1.0,
                "abid": 1.0,
                "twonn": 1.0,
                "pr": 1.0,
                "entropy": 1.0,
                "parse_fail": False,
                "reason": "equation_matches_reference",
                "reason_weight": -0.25,
                "warning_score": -0.3,
                "warning_score_delta": 0.1,
                "hybrid_warning_score": -0.2,
                "entropy_delta": 0.1,
                "final_correct": True,
            },
        ]
    )

    sample_df = _build_sample_features(df, early_n=2)
    assert sorted(sample_df["sample_id"].tolist()) == ["sample-a", "sample-b"]
    row = sample_df.loc[sample_df["sample_id"] == "sample-a"].iloc[0]
    assert int(row["step_count"]) == 0
    assert int(row["trace_failure_prefix_any"]) == 1
