# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_ablation import _load_input_table
from src.experiments.early_warning import apply_alarm_policy, default_alarm_policies, prepare_warning_features
from src.visualization.seismograph import build_seismograph

VALID_ANALYSIS_LAYERS = {"early", "middle", "late"}
VALID_LAYER_SELECTIONS = {"fixed", "classification_best", "early_warning_best"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a seismograph demo case from an ablation run.")
    parser.add_argument("--results", required=True, help="Path to an ablation results directory")
    parser.add_argument("--model", required=True, help="Model name as stored in the results")
    parser.add_argument(
        "--dataset-index",
        type=int,
        default=-1,
        help="Dataset index to render; auto-select when < 0",
    )
    parser.add_argument("--out", default="", help="Optional output directory")
    parser.add_argument(
        "--primary-k",
        type=int,
        default=-1,
        help="Override primary k; defaults to the model summary's k_used",
    )
    parser.add_argument(
        "--analysis-layer",
        default="",
        help="Optional explicit layer override: early, middle, or late",
    )
    parser.add_argument(
        "--layer-selection",
        choices=sorted(VALID_LAYER_SELECTIONS),
        default="fixed",
        help="How to choose the layer when --analysis-layer is not set",
    )
    return parser.parse_args()


def _safe_model_dir_name(model_name: str) -> str:
    return model_name.strip().replace("/", "__").replace(":", "_").replace(" ", "_")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _model_experiments_dir(results_dir: Path, model_name: str) -> Path:
    return results_dir / "models" / _safe_model_dir_name(model_name) / "experiments"


def _load_step_table_from_results(results_dir: Path) -> pd.DataFrame:
    candidates = [
        results_dir / "step_signal_table.parquet",
        results_dir / "step_signal_table.csv",
        results_dir / "combined" / "step_signal_table.parquet",
        results_dir / "combined" / "step_signal_table.csv",
    ]
    for path in candidates:
        if path.exists():
            return _load_input_table(path)
    raise FileNotFoundError(f"Could not find a step table under {results_dir}")


def _resolve_analysis_layer(
    summary: dict[str, Any],
    *,
    analysis_layer: str,
    layer_selection: str,
) -> str:
    explicit = str(analysis_layer or "").strip().lower()
    if explicit:
        if explicit not in VALID_ANALYSIS_LAYERS:
            raise ValueError(f"Unsupported analysis layer: {analysis_layer}")
        return explicit

    selection = str(layer_selection or "fixed").strip().lower()
    if selection not in VALID_LAYER_SELECTIONS:
        raise ValueError(f"Unsupported layer selection mode: {layer_selection}")

    key_map = {
        "fixed": "primary_fixed_layer",
        "classification_best": "layer_recommendation_classification",
        "early_warning_best": "layer_recommendation_early_warning",
    }
    payload = summary.get(key_map[selection], {}) or {}
    resolved = str(payload.get("analysis_layer") or summary.get("analysis_layer") or "").strip().lower()
    if resolved not in VALID_ANALYSIS_LAYERS:
        raise ValueError(f"Could not resolve a valid analysis layer from summary using mode={selection}")
    return resolved


def _resolve_primary_k(summary: dict[str, Any], primary_k: int) -> int:
    if int(primary_k) > 0:
        return int(primary_k)
    return int(summary.get("k_used") or summary.get("primary_k_policy", {}).get("default_primary_k", 20))


def _load_layer_policy_row(
    model_exp_dir: Path,
    summary: dict[str, Any],
    *,
    analysis_layer: str,
    primary_k: int,
) -> dict[str, Any]:
    layer_path = model_exp_dir / "experiment_a_layer_comparison.csv"
    if layer_path.exists():
        layer_df = pd.read_csv(layer_path)
        if "k_used" in layer_df.columns:
            k_mask = pd.to_numeric(layer_df["k_used"], errors="coerce").fillna(-1).astype(int) == int(primary_k)
            layer_df = layer_df.loc[k_mask].copy()
        layer_df["analysis_layer"] = layer_df["analysis_layer"].astype(str).str.lower()
        match = layer_df.loc[layer_df["analysis_layer"] == str(analysis_layer).lower()]
        if not match.empty:
            return match.iloc[0].to_dict()

    fixed_layer = str(summary.get("analysis_layer", "")).strip().lower()
    if analysis_layer != fixed_layer:
        raise FileNotFoundError(
            f"Missing layer comparison metadata for analysis_layer={analysis_layer} in {model_exp_dir}"
        )

    selected_policy = summary.get("selected_alarm_policy", {}) or {}
    return {
        "analysis_layer": fixed_layer,
        "analysis_layer_index": int(summary.get("analysis_layer_index", -1)),
        "selected_policy_name": str(selected_policy.get("policy_name", "")),
        "selected_policy_score_col": str(selected_policy.get("score_col", "")),
        "selected_policy_threshold": float(selected_policy.get("threshold", summary.get("alarm_threshold", float("nan")))),
    }


def _resolve_policy_spec(policy_name: str):
    for spec in default_alarm_policies():
        if spec.name == str(policy_name):
            return spec
    raise KeyError(f"Unknown alarm policy: {policy_name}")


def _build_hover_texts(step_df: pd.DataFrame) -> list[str]:
    raw = step_df.get("step_text", pd.Series("", index=step_df.index)).fillna("").astype(str).str.strip()
    normalized = (
        step_df.get("normalized_step_text", pd.Series("", index=step_df.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    return raw.where(raw.str.len().gt(0), normalized).tolist()


def select_demo_case_index(step_df: pd.DataFrame) -> int:
    scored = step_df.sort_values(["dataset_index", "step_index"], kind="mergesort").copy()
    candidates: list[tuple[float, float, int]] = []
    for dataset_index, group in scored.groupby("dataset_index", sort=False):
        if bool(group["final_correct"].iloc[0]):
            continue
        wrong = group.loc[(~group["is_correct"].astype(bool)) | (group["parse_fail"].astype(bool)), "step_index"]
        alarm = group.loc[group["alarm_selected"].astype(bool), "step_index"]
        if wrong.empty or alarm.empty:
            continue
        lead_time = float(wrong.iloc[0] - alarm.iloc[0])
        peak_warning = float(group["warning_score"].astype(float).max())
        candidates.append((lead_time, peak_warning, int(dataset_index)))

    if candidates:
        candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
        return candidates[0][2]

    failed = scored.loc[~scored["final_correct"].astype(bool), "dataset_index"]
    if not failed.empty:
        return int(failed.iloc[0])
    return int(scored["dataset_index"].iloc[0])


def load_demo_steps_for_model(
    results_dir: Path,
    model_name: str,
    *,
    primary_k: int = -1,
    analysis_layer: str = "",
    layer_selection: str = "fixed",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    model_exp_dir = _model_experiments_dir(results_dir, model_name)
    summary_path = model_exp_dir / "experiment_a_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")

    summary = _load_json(summary_path)
    resolved_k = _resolve_primary_k(summary, primary_k)
    resolved_layer = _resolve_analysis_layer(
        summary,
        analysis_layer=analysis_layer,
        layer_selection=layer_selection,
    )
    layer_policy = _load_layer_policy_row(
        model_exp_dir,
        summary,
        analysis_layer=resolved_layer,
        primary_k=resolved_k,
    )
    step_df = _load_step_table_from_results(results_dir)

    k_col = "k_requested" if "k_requested" in step_df.columns else "k_used"
    layer_col = "layer_name" if "layer_name" in step_df.columns else None
    if layer_col is None and "analysis_layer" in step_df.columns:
        layer_col = "analysis_layer"

    mask = step_df["model_name"].astype(str) == str(model_name)
    mask &= pd.to_numeric(step_df[k_col], errors="coerce").fillna(-1).astype(int) == int(resolved_k)
    if layer_col is not None:
        mask &= step_df[layer_col].astype(str).str.lower() == str(resolved_layer).lower()
    elif resolved_layer != str(summary.get("analysis_layer", "")).strip().lower():
        raise RuntimeError("This run does not contain multi-layer step rows for a non-fixed demo selection.")

    filtered = step_df.loc[mask].copy()
    if filtered.empty:
        raise RuntimeError(
            "No step rows found for the requested model / k / analysis layer combination."
        )

    scored = prepare_warning_features(filtered)
    selected_policy_name = str(layer_policy.get("selected_policy_name", "")).strip()
    policy_spec = _resolve_policy_spec(selected_policy_name)
    fitted = {"threshold": float(layer_policy.get("selected_policy_threshold", float("nan")))}
    if policy_spec.base_floor_quantile is not None:
        fitted["base_floor"] = float(
            scored["warning_score"].astype(float).quantile(policy_spec.base_floor_quantile)
        )
    scored = apply_alarm_policy(scored, policy_spec, fitted)

    selected_policy = {
        "policy_name": policy_spec.name,
        "score_col": str(layer_policy.get("selected_policy_score_col", policy_spec.score_col) or policy_spec.score_col),
        "threshold_kind": policy_spec.threshold_kind,
        "threshold_value": float(policy_spec.threshold_value),
        "consecutive_n": int(policy_spec.consecutive_n),
        "rolling_k": int(policy_spec.rolling_k),
        "rolling_m": int(policy_spec.rolling_m),
        "base_floor_quantile": (
            float(policy_spec.base_floor_quantile)
            if policy_spec.base_floor_quantile is not None
            else None
        ),
        "threshold": float(fitted["threshold"]),
    }
    context = {
        "results_dir": str(results_dir),
        "model_name": str(model_name),
        "primary_k": int(resolved_k),
        "analysis_layer": str(resolved_layer),
        "analysis_layer_index": int(layer_policy.get("analysis_layer_index", -1)),
        "layer_selection": str(layer_selection),
        "selected_policy": selected_policy,
    }
    return scored, context


def prepare_demo_case(
    results_dir: Path,
    model_name: str,
    *,
    dataset_index: int = -1,
    primary_k: int = -1,
    analysis_layer: str = "",
    layer_selection: str = "fixed",
) -> tuple[pd.DataFrame, Any, dict[str, Any]]:
    scored, context = load_demo_steps_for_model(
        results_dir,
        model_name,
        primary_k=primary_k,
        analysis_layer=analysis_layer,
        layer_selection=layer_selection,
    )
    resolved_dataset_index = int(dataset_index)
    if resolved_dataset_index < 0:
        resolved_dataset_index = select_demo_case_index(scored)

    case_df = scored.loc[
        pd.to_numeric(scored["dataset_index"], errors="coerce").fillna(-1).astype(int) == resolved_dataset_index
    ].copy()
    if case_df.empty:
        raise RuntimeError("No step rows found for the requested model/dataset_index.")

    case_df = case_df.sort_values("step_index", kind="mergesort").reset_index(drop=True)
    if "step_text" not in case_df.columns:
        case_df["step_text"] = ""
    if "normalized_step_text" not in case_df.columns:
        case_df["normalized_step_text"] = ""

    first_alarm = case_df.loc[case_df["alarm_selected"].astype(bool), "step_index"]
    wrong_mask = (~case_df["is_correct"].astype(bool)) | (case_df["parse_fail"].astype(bool))
    first_error = case_df.loc[wrong_mask, "step_index"]
    first_alarm_step = int(first_alarm.iloc[0]) if not first_alarm.empty else None
    first_error_step = int(first_error.iloc[0]) if not first_error.empty else None
    lead_time = (
        float(first_error_step - first_alarm_step)
        if first_alarm_step is not None and first_error_step is not None
        else None
    )

    fig = build_seismograph(
        step_indices=case_df["step_index"].astype(int).tolist(),
        lid_values=case_df["lid"].astype(float).tolist(),
        pr_values=case_df["pr"].astype(float).tolist(),
        entropy_values=case_df["entropy"].astype(float).tolist(),
        warning_scores=case_df["warning_score"].astype(float).tolist(),
        warning_threshold=float(context["selected_policy"]["threshold"]),
        warning_steps=case_df.loc[case_df["alarm_selected"].astype(bool), "step_index"].astype(int).tolist(),
        alarm_step=first_alarm_step,
        incorrect_steps=case_df.loc[~case_df["is_correct"].astype(bool), "step_index"].astype(int).tolist(),
        parse_fail_steps=case_df.loc[case_df["parse_fail"].astype(bool), "step_index"].astype(int).tolist(),
        step_texts=_build_hover_texts(case_df),
        final_correct=bool(case_df["final_correct"].iloc[0]),
        title=(
            f"{model_name} | layer={context['analysis_layer']} | "
            f"dataset_index={resolved_dataset_index}"
        ),
    )

    case_summary = {
        **context,
        "dataset_index": int(resolved_dataset_index),
        "n_steps": int(len(case_df)),
        "final_correct": bool(case_df["final_correct"].iloc[0]),
        "first_alarm_step": first_alarm_step,
        "first_error_step": first_error_step,
        "lead_time": lead_time,
        "step_text_available": bool(any(text for text in _build_hover_texts(case_df))),
    }
    return case_df, fig, case_summary


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results)
    case_df, fig, case_summary = prepare_demo_case(
        results_dir,
        str(args.model),
        dataset_index=int(args.dataset_index),
        primary_k=int(args.primary_k),
        analysis_layer=str(args.analysis_layer),
        layer_selection=str(args.layer_selection),
    )

    model_safe = _safe_model_dir_name(str(args.model))
    default_out = results_dir / "demo_cases" / model_safe / f"dataset_{int(case_summary['dataset_index'])}"
    out_dir = Path(args.out) if args.out else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    html_path = out_dir / "seismograph.html"
    fig.write_html(str(html_path))

    case_steps_path = out_dir / "case_steps.csv"
    summary_out_path = out_dir / "case_summary.json"
    case_df.to_csv(case_steps_path, index=False)

    case_summary = {
        **case_summary,
        "html_path": str(html_path),
        "case_steps_path": str(case_steps_path),
    }
    summary_out_path.write_text(json.dumps(case_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote seismograph HTML: {html_path}")
    print(f"Wrote case steps CSV:  {case_steps_path}")
    print(f"Wrote case summary:    {summary_out_path}")


if __name__ == "__main__":
    main()
