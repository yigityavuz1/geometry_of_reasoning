from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


def _clip_probs(values: np.ndarray) -> np.ndarray:
    return np.clip(values.astype(np.float64), 1e-6, 1.0 - 1e-6)


def safe_logit(values: np.ndarray) -> np.ndarray:
    clipped = _clip_probs(values)
    return np.log(clipped / (1.0 - clipped))


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(len(y_true))
    if total == 0:
        return float("nan")
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= left) & (y_prob < right)
        if right >= 1.0:
            mask = (y_prob >= left) & (y_prob <= right)
        if not np.any(mask):
            continue
        confidence = float(np.mean(y_prob[mask]))
        accuracy = float(np.mean(y_true[mask]))
        ece += abs(confidence - accuracy) * float(np.sum(mask)) / total
    return float(ece)


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    classes = np.unique(y_true)
    if len(classes) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "brier": float("nan"), "ece": float("nan")}
    clipped = _clip_probs(y_prob)
    return {
        "auroc": float(roc_auc_score(y_true, clipped)),
        "auprc": float(average_precision_score(y_true, clipped)),
        "brier": float(brier_score_loss(y_true, clipped)),
        "ece": float(ece_score(y_true, clipped)),
    }


def fit_platt_artifact(raw_probs: np.ndarray, y_true: np.ndarray, *, seed: int) -> dict[str, Any]:
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    clf.fit(safe_logit(raw_probs).reshape(-1, 1), y_true.astype(np.int32))
    return {
        "method": "platt",
        "coef": float(clf.coef_[0][0]),
        "intercept": float(clf.intercept_[0]),
    }


def fit_isotonic_artifact(raw_probs: np.ndarray, y_true: np.ndarray) -> dict[str, Any]:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(_clip_probs(raw_probs), y_true.astype(np.float64))
    return {
        "method": "isotonic",
        "x_thresholds": [float(v) for v in iso.X_thresholds_],
        "y_thresholds": [float(v) for v in iso.y_thresholds_],
    }


def apply_calibration_artifact(raw_probs: np.ndarray, artifact: dict[str, Any]) -> np.ndarray:
    method = str(artifact.get("method", "raw"))
    clipped = _clip_probs(raw_probs)
    if method == "raw":
        return clipped
    if method == "platt":
        coef = float(artifact["coef"])
        intercept = float(artifact["intercept"])
        logits = coef * safe_logit(clipped) + intercept
        return 1.0 / (1.0 + np.exp(-logits))
    if method == "isotonic":
        xp = np.asarray(artifact["x_thresholds"], dtype=np.float64)
        fp = np.asarray(artifact["y_thresholds"], dtype=np.float64)
        return np.interp(clipped, xp, fp)
    raise ValueError(f"Unsupported calibration method: {method}")


def reliability_curve_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    label: str,
    bins: int = 10,
) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, float | str]] = []
    clipped = _clip_probs(y_prob)
    for bin_index, (left, right) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        mask = (clipped >= left) & (clipped < right)
        if right >= 1.0:
            mask = (clipped >= left) & (clipped <= right)
        if not np.any(mask):
            continue
        rows.append(
            {
                "label": label,
                "bin_index": float(bin_index),
                "bin_left": float(left),
                "bin_right": float(right),
                "count": float(np.sum(mask)),
                "confidence_mean": float(np.mean(clipped[mask])),
                "accuracy_mean": float(np.mean(y_true[mask])),
            }
        )
    return pd.DataFrame(rows)


def _oof_calibrated_probs(
    raw_probs: np.ndarray,
    y_true: np.ndarray,
    *,
    method: str,
    seed: int,
) -> np.ndarray:
    raw_probs = _clip_probs(raw_probs)
    y_true = y_true.astype(np.int32)
    classes, counts = np.unique(y_true, return_counts=True)
    n_splits = min(5, int(counts.min())) if len(classes) >= 2 else 0

    if len(classes) < 2 or n_splits < 2:
        if method == "platt":
            artifact = fit_platt_artifact(raw_probs, y_true, seed=seed)
        elif method == "isotonic":
            artifact = fit_isotonic_artifact(raw_probs, y_true)
        else:
            raise ValueError(f"Unsupported method: {method}")
        return apply_calibration_artifact(raw_probs, artifact)

    out = np.zeros(len(raw_probs), dtype=np.float64)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in cv.split(raw_probs.reshape(-1, 1), y_true):
        x_train = raw_probs[train_idx]
        y_train = y_true[train_idx]
        if method == "platt":
            artifact = fit_platt_artifact(x_train, y_train, seed=seed)
        elif method == "isotonic":
            artifact = fit_isotonic_artifact(x_train, y_train)
        else:
            raise ValueError(f"Unsupported method: {method}")
        out[test_idx] = apply_calibration_artifact(raw_probs[test_idx], artifact)
    return out


def evaluate_calibrators(raw_probs: np.ndarray, y_true: np.ndarray, *, seed: int) -> dict[str, Any]:
    raw_probs = _clip_probs(np.asarray(raw_probs, dtype=np.float64))
    y_true = np.asarray(y_true, dtype=np.int32)
    unique_classes = np.unique(y_true)

    if len(unique_classes) < 2:
        fallback_artifact = {"method": "raw", "fallback_reason": "single_class_target"}
        comparison_df = pd.DataFrame(
            [
                {
                    "method": "raw",
                    **binary_metrics(y_true, raw_probs),
                    "ece_delta_vs_raw": 0.0,
                    "brier_delta_vs_raw": 0.0,
                    "auroc_delta_vs_raw": 0.0,
                    "selected": True,
                },
                {
                    "method": "platt",
                    **binary_metrics(y_true, raw_probs),
                    "ece_delta_vs_raw": 0.0,
                    "brier_delta_vs_raw": 0.0,
                    "auroc_delta_vs_raw": 0.0,
                    "selected": False,
                },
                {
                    "method": "isotonic",
                    **binary_metrics(y_true, raw_probs),
                    "ece_delta_vs_raw": 0.0,
                    "brier_delta_vs_raw": 0.0,
                    "auroc_delta_vs_raw": 0.0,
                    "selected": False,
                },
            ]
        )
        reliability_df = reliability_curve_table(y_true, raw_probs, label="raw")
        return {
            "selected_method": "raw",
            "comparison": comparison_df,
            "reliability": reliability_df,
            "artifacts": {
                "raw": {"method": "raw"},
                "platt": dict(fallback_artifact),
                "isotonic": dict(fallback_artifact),
            },
        }

    method_artifacts = {
        "raw": {"method": "raw"},
        "platt": fit_platt_artifact(raw_probs, y_true, seed=seed),
        "isotonic": fit_isotonic_artifact(raw_probs, y_true),
    }
    method_probs = {
        "raw": raw_probs,
        "platt": _oof_calibrated_probs(raw_probs, y_true, method="platt", seed=seed),
        "isotonic": _oof_calibrated_probs(raw_probs, y_true, method="isotonic", seed=seed),
    }

    comparison_rows: list[dict[str, Any]] = []
    reliability_rows: list[pd.DataFrame] = []
    raw_metrics = binary_metrics(y_true, method_probs["raw"])
    selected_method = "raw"
    selected_ece = float(raw_metrics["ece"])

    for method_name, probs in method_probs.items():
        metrics = binary_metrics(y_true, probs)
        comparison_rows.append(
            {
                "method": method_name,
                **metrics,
                "ece_delta_vs_raw": float(metrics["ece"] - raw_metrics["ece"]),
                "brier_delta_vs_raw": float(metrics["brier"] - raw_metrics["brier"]),
                "auroc_delta_vs_raw": float(metrics["auroc"] - raw_metrics["auroc"]),
            }
        )
        reliability_rows.append(reliability_curve_table(y_true, probs, label=method_name))

        improves_ece = np.isfinite(metrics["ece"]) and float(metrics["ece"]) < selected_ece
        preserves_brier = (not np.isfinite(raw_metrics["brier"])) or float(metrics["brier"]) <= float(raw_metrics["brier"]) + 1e-9
        preserves_auroc = (not np.isfinite(raw_metrics["auroc"])) or float(metrics["auroc"]) >= float(raw_metrics["auroc"]) - 0.01
        if method_name != "raw" and improves_ece and preserves_brier and preserves_auroc:
            selected_method = method_name
            selected_ece = float(metrics["ece"])

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df["selected"] = comparison_df["method"] == selected_method
    reliability_df = (
        pd.concat(reliability_rows, ignore_index=True) if reliability_rows else pd.DataFrame()
    )

    return {
        "selected_method": selected_method,
        "comparison": comparison_df,
        "reliability": reliability_df,
        "artifacts": method_artifacts,
    }
