from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.early_warning import evaluate_alarm_policies, prepare_warning_features


def test_prepare_warning_features_uses_reason_weights() -> None:
    rows = [
        {
            "sample_id": "s1",
            "model_name": "model-a",
            "dataset_index": 0,
            "question_id": "q0",
            "step_index": 1,
            "lid": 1.0,
            "entropy": 1.0,
            "parse_fail": False,
            "reason": "equation_matches_reference",
            "is_correct": True,
            "final_correct": True,
        },
        {
            "sample_id": "s2",
            "model_name": "model-a",
            "dataset_index": 1,
            "question_id": "q1",
            "step_index": 1,
            "lid": 1.0,
            "entropy": 1.0,
            "parse_fail": False,
            "reason": "equation_mismatch",
            "is_correct": False,
            "final_correct": False,
        },
    ]
    scored = prepare_warning_features(pd.DataFrame(rows))
    good_score = float(scored.loc[scored["sample_id"] == "s1", "warning_score"].iloc[0])
    bad_score = float(scored.loc[scored["sample_id"] == "s2", "warning_score"].iloc[0])
    assert bad_score > good_score


def test_prepare_warning_features_handles_trace_failure_with_missing_geometry() -> None:
    rows = [
        {
            "sample_id": "s0",
            "model_name": "model-a",
            "dataset_index": 0,
            "question_id": "q0",
            "step_index": 0,
            "lid": np.nan,
            "entropy": np.nan,
            "parse_fail": True,
            "reason": "trace_format_fail",
            "is_correct": False,
            "final_correct": False,
        },
        {
            "sample_id": "s1",
            "model_name": "model-a",
            "dataset_index": 1,
            "question_id": "q1",
            "step_index": 1,
            "lid": 1.0,
            "entropy": 1.0,
            "parse_fail": False,
            "reason": "equation_matches_reference",
            "is_correct": True,
            "final_correct": True,
        },
    ]
    scored = prepare_warning_features(pd.DataFrame(rows))
    assert scored["warning_score"].notna().all()
    assert "trajectory_warning_score" in scored.columns
    trace_score = float(scored.loc[scored["sample_id"] == "s0", "warning_score"].iloc[0])
    normal_score = float(scored.loc[scored["sample_id"] == "s1", "warning_score"].iloc[0])
    assert trace_score > normal_score


def test_evaluate_alarm_policies_returns_selected_policy_and_scores() -> None:
    rows = [
        {
            "sample_id": "s1",
            "model_name": "model-a",
            "dataset_index": 0,
            "question_id": "q0",
            "step_index": 1,
            "lid": 0.3,
            "entropy": 0.2,
            "parse_fail": False,
            "is_correct": True,
            "final_correct": True,
        },
        {
            "sample_id": "s1",
            "model_name": "model-a",
            "dataset_index": 0,
            "question_id": "q0",
            "step_index": 2,
            "lid": 0.4,
            "entropy": 0.3,
            "parse_fail": False,
            "is_correct": True,
            "final_correct": True,
        },
        {
            "sample_id": "s2",
            "model_name": "model-a",
            "dataset_index": 1,
            "question_id": "q1",
            "step_index": 1,
            "lid": 2.0,
            "entropy": 1.8,
            "parse_fail": False,
            "is_correct": True,
            "final_correct": False,
        },
        {
            "sample_id": "s2",
            "model_name": "model-a",
            "dataset_index": 1,
            "question_id": "q1",
            "step_index": 2,
            "lid": 2.4,
            "entropy": 2.1,
            "parse_fail": True,
            "is_correct": False,
            "final_correct": False,
        },
        {
            "sample_id": "s3",
            "model_name": "model-a",
            "dataset_index": 2,
            "question_id": "q2",
            "step_index": 1,
            "lid": 0.2,
            "entropy": 0.1,
            "parse_fail": False,
            "is_correct": True,
            "final_correct": True,
        },
        {
            "sample_id": "s3",
            "model_name": "model-a",
            "dataset_index": 2,
            "question_id": "q2",
            "step_index": 2,
            "lid": 0.3,
            "entropy": 0.2,
            "parse_fail": False,
            "is_correct": True,
            "final_correct": True,
        },
        {
            "sample_id": "s4",
            "model_name": "model-a",
            "dataset_index": 3,
            "question_id": "q3",
            "step_index": 1,
            "lid": 1.8,
            "entropy": 1.7,
            "parse_fail": False,
            "is_correct": True,
            "final_correct": False,
        },
        {
            "sample_id": "s4",
            "model_name": "model-a",
            "dataset_index": 3,
            "question_id": "q3",
            "step_index": 2,
            "lid": 2.1,
            "entropy": 2.2,
            "parse_fail": True,
            "is_correct": False,
            "final_correct": False,
        },
    ]
    payload = evaluate_alarm_policies(pd.DataFrame(rows), seed=7)

    assert payload["selected_policy"]["policy_name"]
    assert "alarm_selected" in payload["selected_step_df"].columns
    assert not payload["policy_comparison"].empty
    assert "alarm_before_error_rate" in payload["selected_metrics"]
    assert "early_objective" in payload["selected_metrics"]
