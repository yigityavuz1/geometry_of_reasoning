from __future__ import annotations

import numpy as np

from src.evaluation.calibration import apply_calibration_artifact, evaluate_calibrators


def test_evaluate_calibrators_returns_artifacts_and_selection() -> None:
    y_true = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
    raw_probs = np.asarray([0.08, 0.86, 0.22, 0.79, 0.31, 0.68, 0.41, 0.73, 0.18, 0.92], dtype=np.float64)

    payload = evaluate_calibrators(raw_probs, y_true, seed=11)

    assert payload["selected_method"] in {"raw", "platt", "isotonic"}
    assert set(payload["artifacts"].keys()) == {"raw", "platt", "isotonic"}
    assert not payload["comparison"].empty
    assert not payload["reliability"].empty

    selected = payload["artifacts"][payload["selected_method"]]
    calibrated = apply_calibration_artifact(raw_probs, selected)
    assert np.all(calibrated >= 0.0)
    assert np.all(calibrated <= 1.0)


def test_evaluate_calibrators_falls_back_to_raw_for_single_class() -> None:
    y_true = np.asarray([1, 1, 1, 1], dtype=np.int32)
    raw_probs = np.asarray([0.6, 0.7, 0.8, 0.9], dtype=np.float64)

    payload = evaluate_calibrators(raw_probs, y_true, seed=11)

    assert payload["selected_method"] == "raw"
    assert set(payload["artifacts"].keys()) == {"raw", "platt", "isotonic"}
    assert payload["artifacts"]["platt"]["method"] == "raw"
    calibrated = apply_calibration_artifact(raw_probs, payload["artifacts"]["platt"])
    assert np.allclose(calibrated, raw_probs)
