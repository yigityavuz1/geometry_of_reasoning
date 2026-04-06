# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
import warnings
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
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.step_parser import extract_numeric_tokens, normalize_math_text, normalize_step_text
from src.evaluation.sympy_judge import build_task_reference, judge_step_task_correctness, summarize_judgement_records
from src.evaluation.calibration import apply_calibration_artifact, evaluate_calibrators
from src.experiments.early_warning import REASON_WARNING_WEIGHTS, TRACE_FAILURE_REASONS, evaluate_alarm_policies
from src.generation.runner import (
    GenerationConfig,
    collect_model_metadata,
    generate_reasoning_trace,
    load_model_and_tokenizer,
)
from src.metrics.global_dim import participation_ratio
from src.metrics.lid_estimators import abid_local_batch, coefficient_of_variation, lid_mle_batch, twonn_global_id

LOGGER = logging.getLogger("gor.ablation")
DATASET_NAME = "openai/gsm8k"
DATASET_CONFIG = "main"
STEP_TABLE_COLUMNS = [
    "model_name",
    "dataset_name",
    "dataset_config",
    "dataset_split",
    "dataset_index",
    "question_id",
    "question_text",
    "sample_id",
    "layer_name",
    "layer_index",
    "step_index",
    "step_text",
    "normalized_step_text",
    "matched_values",
    "k_requested",
    "k_effective",
    "step_token_count",
    "lid",
    "abid",
    "twonn",
    "pr",
    "entropy",
    "is_correct",
    "parse_fail",
    "reason",
    "final_correct",
]
STEP_DEDUPE_KEYS = ["model_name", "dataset_split", "dataset_index", "layer_name", "step_index", "k_requested"]
PRIMARY_K_DEFAULT = 20
PRIMARY_K_RATIONALE = (
    "Default primary k is fixed to 20 as the project standard. "
    "This keeps neighborhood estimates more stable than very small k while avoiding the "
    "extra smoothing/cost of k=40 for routine A/B runs."
)
DEFAULT_ANALYSIS_LAYER = "late"


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
        help="Optional path to precomputed step-signal table (.csv, .jsonl, or .parquet).",
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
    parser.add_argument(
        "--analysis-layer",
        default=DEFAULT_ANALYSIS_LAYER,
        help=f"Which captured layer to use for Experiment A/B/C scoring (default: {DEFAULT_ANALYSIS_LAYER}).",
    )
    parser.add_argument(
        "--primary-k",
        type=int,
        default=PRIMARY_K_DEFAULT,
        help=f"Primary k for A/B experiments (default: {PRIMARY_K_DEFAULT}).",
    )
    parser.add_argument("--k-values", default="5,10,20,40", help="Comma-separated k values")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=400,
        help="Bootstrap resamples for AUROC/AUPRC/Brier/ECE confidence intervals (0 disables).",
    )
    parser.add_argument(
        "--bootstrap-alpha",
        type=float,
        default=0.05,
        help="Two-sided alpha for bootstrap confidence intervals (default: 0.05).",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=0,
        help="Set GOR_NUM_THREADS (0 keeps current runtime default).",
    )
    parser.add_argument(
        "--torch-interop-threads",
        type=int,
        default=0,
        help="Set GOR_NUM_INTEROP_THREADS (0 keeps current runtime default).",
    )
    parser.add_argument(
        "--cpu-int8",
        action="store_true",
        help="Enable dynamic int8 quantization on CPU (GOR_CPU_INT8=1).",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Enable torch.compile for model forward/generation.",
    )
    parser.add_argument(
        "--torch-compile-mode",
        default="reduce-overhead",
        help="Compile mode passed to torch.compile (default: reduce-overhead).",
    )
    parser.add_argument(
        "--quantization",
        default="none",
        choices=["none", "4bit", "8bit", "int4", "int8"],
        help="Optional bitsandbytes quantization mode (CUDA only).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable detailed runtime logs.")
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Log every N processed samples per model when verbose mode is enabled.",
    )
    parser.add_argument(
        "--quiet-progress",
        action="store_true",
        help="Disable tqdm progress bars and keep only structured logs.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Persist per-model checkpoints every N processed samples.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing checkpoints and start this run from scratch.",
    )
    return parser.parse_args()


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    LOGGER.setLevel(logging.INFO if verbose else logging.WARNING)
    logging.getLogger("gor").setLevel(logging.INFO if verbose else logging.WARNING)
    for noisy_logger in [
        "datasets",
        "huggingface_hub",
        "httpx",
        "httpcore",
        "urllib3",
        "transformers",
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.ERROR)
    warnings.filterwarnings(
        "ignore",
        message="Warning: You are sending unauthenticated requests to the HF Hub.*",
    )
    warnings.filterwarnings(
        "ignore",
        message="The tied weights mapping and config.*",
    )
    try:
        from datasets.utils.logging import disable_progress_bar as disable_datasets_progress

        disable_datasets_progress()
    except Exception:
        pass
    try:
        from transformers.utils.logging import disable_progress_bar as disable_transformers_progress

        disable_transformers_progress()
    except Exception:
        pass


def _apply_runtime_env(args: argparse.Namespace) -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.torch_threads > 0:
        os.environ["GOR_NUM_THREADS"] = str(args.torch_threads)
    if args.torch_interop_threads > 0:
        os.environ["GOR_NUM_INTEROP_THREADS"] = str(args.torch_interop_threads)
    if args.cpu_int8:
        os.environ["GOR_CPU_INT8"] = "1"
    if args.torch_compile:
        os.environ["GOR_TORCH_COMPILE"] = "1"
        os.environ["GOR_TORCH_COMPILE_MODE"] = str(args.torch_compile_mode)
    if args.quantization != "none":
        os.environ["GOR_QUANTIZATION"] = args.quantization


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


def _safe_model_dir_name(model_name: str) -> str:
    return (
        model_name.strip()
        .replace("/", "__")
        .replace(":", "_")
        .replace(" ", "_")
    )


def _model_output_dir(out_dir: Path, model_name: str) -> Path:
    return out_dir / "models" / _safe_model_dir_name(model_name)


def _build_question_id(split: str, dataset_index: int) -> str:
    return f"{DATASET_NAME}::{DATASET_CONFIG}::{split}::{int(dataset_index)}"


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            LOGGER.warning("Skipping malformed JSONL line %d in %s", line_no, path)
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _append_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_step_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=STEP_TABLE_COLUMNS)

    out = df.copy()
    if "dataset_name" not in out.columns:
        out["dataset_name"] = DATASET_NAME
    if "dataset_config" not in out.columns:
        out["dataset_config"] = DATASET_CONFIG
    if "dataset_split" not in out.columns:
        out["dataset_split"] = ""
    if "dataset_index" not in out.columns:
        out["dataset_index"] = -1
    out["dataset_index"] = pd.to_numeric(out["dataset_index"], errors="coerce").fillna(-1).astype(int)
    out = out.loc[out["dataset_index"] >= 0].copy()
    if out.empty:
        return pd.DataFrame(columns=STEP_TABLE_COLUMNS)

    if "question_id" not in out.columns:
        out["question_id"] = out.apply(
            lambda row: _build_question_id(str(row["dataset_split"]), int(row["dataset_index"])),
            axis=1,
        )
    else:
        out["question_id"] = out["question_id"].astype(str)
        missing = out["question_id"].str.len() == 0
        if missing.any():
            out.loc[missing, "question_id"] = out.loc[missing].apply(
                lambda row: _build_question_id(str(row["dataset_split"]), int(row["dataset_index"])),
                axis=1,
            )

    if "sample_id" not in out.columns:
        out["sample_id"] = out["model_name"].astype(str) + "::" + out["question_id"].astype(str)
    else:
        out["sample_id"] = out["sample_id"].astype(str)
        missing = out["sample_id"].str.len() == 0
        if missing.any():
            out.loc[missing, "sample_id"] = (
                out.loc[missing, "model_name"].astype(str) + "::" + out.loc[missing, "question_id"].astype(str)
            )

    bool_cols = ["is_correct", "parse_fail", "final_correct"]
    for col in bool_cols:
        if col not in out.columns:
            out[col] = False
        if out[col].dtype != bool:
            out[col] = (
                out[col]
                .astype(str)
                .str.lower()
                .map({"true": True, "false": False, "1": True, "0": False})
                .fillna(False)
            )

    int_cols = ["step_index", "k_requested", "k_effective", "step_token_count"]
    for col in int_cols:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    string_cols = ["question_text", "step_text", "normalized_step_text", "matched_values"]
    for col in string_cols:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)
    if "layer_name" not in out.columns:
        out["layer_name"] = DEFAULT_ANALYSIS_LAYER
    out["layer_name"] = out["layer_name"].fillna(DEFAULT_ANALYSIS_LAYER).astype(str)
    out.loc[out["layer_name"].str.len() == 0, "layer_name"] = DEFAULT_ANALYSIS_LAYER
    if "layer_index" not in out.columns:
        out["layer_index"] = -1
    out["layer_index"] = pd.to_numeric(out["layer_index"], errors="coerce").fillna(-1).astype(int)
    if "reason" not in out.columns:
        out["reason"] = "unknown"
    out["reason"] = out["reason"].astype(str)
    if "model_name" not in out.columns:
        out["model_name"] = ""
    out["model_name"] = out["model_name"].astype(str)

    numeric_cols = ["lid", "abid", "twonn", "pr", "entropy"]
    for col in numeric_cols:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.drop_duplicates(subset=STEP_DEDUPE_KEYS, keep="last")
    out = out.sort_values(
        by=["model_name", "dataset_split", "dataset_index", "layer_name", "step_index", "k_requested"],
        kind="mergesort",
    ).reset_index(drop=True)

    for col in STEP_TABLE_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    return out[STEP_TABLE_COLUMNS]


def _write_step_table(df: pd.DataFrame, *, jsonl_path: Path, csv_path: Path) -> pd.DataFrame:
    normalized = _normalize_step_table(df)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as fp:
        for row in normalized.to_dict(orient="records"):
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_table_with_parquet(normalized, csv_path)
    return normalized


def _write_table_with_parquet(df: pd.DataFrame, csv_path: Path) -> Path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    parquet_path = csv_path.with_suffix(".parquet")
    df.to_parquet(parquet_path, index=False)
    return parquet_path


def _stable_payload_hash(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _trace_layer_payloads(trace: dict[str, Any]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    raw_by_layer = trace.get("token_embeddings_by_layer", {})
    if isinstance(raw_by_layer, dict) and raw_by_layer:
        captured_layers = trace.get("captured_layers", [])
        layer_order = [str(item.get("layer_name", "")) for item in captured_layers if str(item.get("layer_name", ""))]
        if not layer_order:
            layer_order = [str(name) for name in raw_by_layer.keys()]
        for layer_name in layer_order:
            payload = raw_by_layer.get(layer_name, {})
            embeddings = np.asarray(payload.get("embeddings", []), dtype=np.float64)
            payloads.append(
                {
                    "layer_name": str(layer_name),
                    "layer_index": int(payload.get("layer_index", -1)),
                    "token_embeddings": embeddings,
                }
            )
    else:
        payloads.append(
            {
                "layer_name": DEFAULT_ANALYSIS_LAYER,
                "layer_index": -1,
                "token_embeddings": np.asarray(trace.get("token_embeddings", []), dtype=np.float64),
            }
        )
    return payloads


def _available_layers(df: pd.DataFrame) -> list[str]:
    if "layer_name" not in df.columns:
        return [DEFAULT_ANALYSIS_LAYER]
    values = [str(value) for value in df["layer_name"].dropna().astype(str).tolist() if str(value)]
    if not values:
        return [DEFAULT_ANALYSIS_LAYER]
    return list(dict.fromkeys(values))


def _filter_analysis_layer(df: pd.DataFrame, analysis_layer: str) -> pd.DataFrame:
    available_layers = _available_layers(df)
    selected_layer = str(analysis_layer or DEFAULT_ANALYSIS_LAYER)
    if selected_layer not in available_layers:
        raise ValueError(
            f"Requested analysis layer '{selected_layer}' is not available. "
            f"Available layers: {available_layers}"
        )
    if "layer_name" not in df.columns:
        return df.copy()
    return df.loc[df["layer_name"].astype(str) == selected_layer].copy()


def _expected_resume_metadata(
    *,
    model_name: str,
    split: str,
    start_index: int,
    max_index: int,
    k_values: list[int],
    do_sample: bool,
    max_new_tokens: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "version": 1,
        "model_name": model_name,
        "dataset_name": DATASET_NAME,
        "dataset_config": DATASET_CONFIG,
        "dataset_split": split,
        "start_index": int(start_index),
        "end_index_exclusive": int(max_index),
        "k_values": [int(v) for v in k_values],
        "do_sample": bool(do_sample),
        "max_new_tokens": int(max_new_tokens),
        "seed": int(seed),
    }


def _load_processed_indices(progress_path: Path, expected_metadata: dict[str, Any]) -> set[int]:
    if not progress_path.exists():
        return set()
    try:
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
    except Exception:
        LOGGER.warning("Failed to parse progress file, starting from scratch: %s", progress_path)
        return set()
    metadata = payload.get("metadata", {})
    if metadata != expected_metadata:
        LOGGER.warning("Existing progress metadata mismatch in %s; ignoring resume state.", progress_path)
        return set()
    out: set[int] = set()
    for value in payload.get("processed_indices", []):
        try:
            out.add(int(value))
        except Exception:
            continue
    return out


def _write_progress(
    progress_path: Path,
    *,
    metadata: dict[str, Any],
    processed_indices: set[int],
    total_target: int,
) -> None:
    payload = {
        "metadata": metadata,
        "processed_indices": sorted(int(v) for v in processed_indices),
        "processed_count": int(len(processed_indices)),
        "total_target": int(total_target),
        "updated_unix_time": float(time.time()),
    }
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _reset_model_checkpoint_files(model_dir: Path) -> None:
    for filename in ("step_signal_table.jsonl", "step_signal_table.csv", "step_signal_table.parquet", "progress.json"):
        path = model_dir / filename
        if path.exists():
            path.unlink()


def _run_config_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "experiment": str(args.experiment),
        "input": str(args.input or ""),
        "out": str(args.out),
        "models": _parse_csv_str(args.models) if not args.input else [],
        "split": str(args.split),
        "start_index": int(args.start_index),
        "num_samples": int(args.num_samples),
        "max_new_tokens": int(args.max_new_tokens),
        "do_sample": bool(args.do_sample),
        "early_n": int(args.early_n),
        "analysis_layer": str(args.analysis_layer),
        "primary_k": int(args.primary_k),
        "k_values": _parse_csv_int(args.k_values),
        "seed": int(args.seed),
        "bootstrap_iters": int(args.bootstrap_iters),
        "bootstrap_alpha": float(args.bootstrap_alpha),
        "torch_threads": int(args.torch_threads),
        "torch_interop_threads": int(args.torch_interop_threads),
        "cpu_int8": bool(args.cpu_int8),
        "torch_compile": bool(args.torch_compile),
        "torch_compile_mode": str(args.torch_compile_mode),
        "quantization": str(args.quantization),
        "checkpoint_every": int(args.checkpoint_every),
        "no_resume": bool(args.no_resume),
    }


def _load_model_metadata_map(out_dir: Path, model_names: list[str]) -> dict[str, dict[str, Any]]:
    metadata_by_model: dict[str, dict[str, Any]] = {}
    for model_name in model_names:
        metadata_path = _model_output_dir(out_dir, model_name) / "model_metadata.json"
        if not metadata_path.exists():
            continue
        try:
            metadata_by_model[str(model_name)] = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
    return metadata_by_model


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


def _bootstrap_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bootstrap: int,
    alpha: float,
    seed: int,
) -> dict[str, dict[str, float]]:
    metric_names = ("auroc", "auprc", "brier", "ece")

    def _nan_ci() -> dict[str, float]:
        return {"low": float("nan"), "high": float("nan"), "valid_bootstrap_samples": 0.0}

    if n_bootstrap <= 0 or not (0.0 < alpha < 1.0):
        return {name: _nan_ci() for name in metric_names}
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {name: _nan_ci() for name in metric_names}

    rng = np.random.default_rng(seed)
    n = int(len(y_true))
    values: dict[str, list[float]] = {name: [] for name in metric_names}
    for _ in range(int(n_bootstrap)):
        sample_idx = rng.integers(0, n, size=n)
        y_boot = y_true[sample_idx]
        if len(np.unique(y_boot)) < 2:
            continue
        prob_boot = y_prob[sample_idx]
        boot_metrics = _binary_metrics(y_boot, prob_boot)
        for name in metric_names:
            value = float(boot_metrics[name])
            if np.isfinite(value):
                values[name].append(value)

    ci: dict[str, dict[str, float]] = {}
    lower_q = alpha / 2.0
    upper_q = 1.0 - lower_q
    for name in metric_names:
        samples = values[name]
        if not samples:
            ci[name] = _nan_ci()
            continue
        arr = np.asarray(samples, dtype=np.float64)
        ci[name] = {
            "low": float(np.quantile(arr, lower_q)),
            "high": float(np.quantile(arr, upper_q)),
            "valid_bootstrap_samples": float(len(arr)),
        }
    return ci


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
                "normalized_text": normalize_step_text(step_text),
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


def _metric_summary_for_tokens(step_tokens: np.ndarray, k_requested: int) -> dict[str, float | int]:
    token_count = int(step_tokens.shape[0]) if step_tokens.ndim == 2 else 0
    summary: dict[str, float | int] = {
        "k_effective": 0,
        "step_token_count": token_count,
        "lid": float("nan"),
        "abid": float("nan"),
        "twonn": float("nan"),
        "pr": float("nan"),
    }
    if step_tokens.ndim != 2 or token_count == 0:
        return summary
    if token_count >= 2:
        try:
            summary["pr"] = float(participation_ratio(step_tokens))
        except Exception:
            summary["pr"] = float("nan")
    if token_count >= 3:
        k_effective = max(2, min(int(k_requested), token_count - 1))
        summary["k_effective"] = int(k_effective)
        try:
            summary["lid"] = float(np.mean(lid_mle_batch(step_tokens, k=k_effective)))
        except Exception:
            summary["lid"] = float("nan")
        try:
            summary["abid"] = float(np.mean(abid_local_batch(step_tokens, k=k_effective)))
        except Exception:
            summary["abid"] = float("nan")
        try:
            summary["twonn"] = float(twonn_global_id(step_tokens))
        except Exception:
            summary["twonn"] = float("nan")
    return summary


def _fallback_trace_reason(
    trace: dict[str, Any],
    *,
    step_texts: list[str],
    step_signal_rows: list[dict[str, Any]],
) -> str:
    generated_text = str(trace.get("generated_text", "")).strip()
    layer_payloads = _trace_layer_payloads(trace)
    has_completion_tokens = any(
        payload["token_embeddings"].ndim == 2 and payload["token_embeddings"].shape[0] > 0 for payload in layer_payloads
    )
    if not generated_text or not has_completion_tokens:
        return "empty_completion"
    if not step_texts:
        return "trace_format_fail"
    if not step_signal_rows:
        return "trace_signal_fail"
    return "trace_signal_fail"


def _build_trace_failure_rows(
    *,
    model_name: str,
    split: str,
    dataset_index: int,
    question_id: str,
    question: str,
    sample_id: str,
    trace: dict[str, Any],
    step_texts: list[str],
    judged_steps: list[dict[str, Any]],
    final_correct: bool,
    k_values: list[int],
) -> list[dict[str, Any]]:
    reason = _fallback_trace_reason(trace, step_texts=step_texts, step_signal_rows=trace.get("step_signal_rows", []))
    raw_text = "\n".join(step_texts).strip() if step_texts else str(trace.get("generated_text", "")).strip()
    normalized_text = normalize_step_text(raw_text)
    layer_payloads = _trace_layer_payloads(trace)
    token_entropies = [float(value) for value in trace.get("token_entropy", []) if np.isfinite(float(value))]
    if token_entropies:
        entropy_value = float(np.mean(token_entropies))
    else:
        entropy_value = float(trace.get("entropy_summary", {}).get("mean_entropy", float("nan")))

    matched_values: list[str] = []
    for row in judged_steps:
        for value in row.get("matched_values", []):
            value_str = str(value)
            if value_str and value_str not in matched_values:
                matched_values.append(value_str)
    matched_values_text = "|".join(matched_values)

    rows: list[dict[str, Any]] = []
    for layer_payload in layer_payloads:
        metric_tokens = layer_payload["token_embeddings"]
        if metric_tokens.ndim != 2:
            metric_tokens = np.empty((0, 0), dtype=np.float64)
        for k_requested in k_values:
            metric_summary = _metric_summary_for_tokens(metric_tokens, int(k_requested))
            rows.append(
                {
                    "model_name": model_name,
                    "dataset_name": DATASET_NAME,
                    "dataset_config": DATASET_CONFIG,
                    "dataset_split": split,
                    "dataset_index": int(dataset_index),
                    "question_id": question_id,
                    "question_text": question,
                    "sample_id": sample_id,
                    "layer_name": str(layer_payload["layer_name"]),
                    "layer_index": int(layer_payload["layer_index"]),
                    "step_index": 0,
                    "step_text": raw_text,
                    "normalized_step_text": normalized_text,
                    "matched_values": matched_values_text,
                    "k_requested": int(k_requested),
                    "k_effective": int(metric_summary["k_effective"]),
                    "step_token_count": int(metric_summary["step_token_count"]),
                    "lid": metric_summary["lid"],
                    "abid": metric_summary["abid"],
                    "twonn": metric_summary["twonn"],
                    "pr": metric_summary["pr"],
                    "entropy": entropy_value,
                    "is_correct": False,
                    "parse_fail": True,
                    "reason": reason,
                    "final_correct": bool(final_correct),
                }
            )
    return rows


def _collect_step_signal_table(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    models = _parse_csv_str(args.models)
    k_values = _parse_csv_int(args.k_values)
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be >= 1")

    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=args.split)
    max_index = args.start_index + args.num_samples
    if max_index > len(dataset):
        raise ValueError(
            f"Requested range [{args.start_index}, {max_index}) exceeds split size {len(dataset)}."
        )

    rows: list[dict[str, Any]] = []
    total_samples = max_index - args.start_index
    models_root = out_dir / "models"
    models_root.mkdir(parents=True, exist_ok=True)
    model_iter = tqdm(
        models,
        desc="Models",
        unit="model",
        disable=args.quiet_progress,
    )
    for model_name in model_iter:
        model_dir = _model_output_dir(out_dir, model_name)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_metadata_path = model_dir / "model_metadata.json"
        model_metadata: dict[str, Any] = {
            "model_name": model_name,
            "safe_dir_name": _safe_model_dir_name(model_name),
            "dataset_name": DATASET_NAME,
            "dataset_config": DATASET_CONFIG,
            "dataset_split": args.split,
        }
        model_metadata_path.write_text(
            json.dumps(model_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        model_jsonl_path = model_dir / "step_signal_table.jsonl"
        model_csv_path = model_dir / "step_signal_table.csv"
        model_progress_path = model_dir / "progress.json"
        legacy_root_jsonl = out_dir / "step_signal_table.jsonl"

        if args.no_resume:
            LOGGER.info("[%s] --no-resume set; clearing prior checkpoints.", model_name)
            _reset_model_checkpoint_files(model_dir)
        elif (not model_jsonl_path.exists()) and legacy_root_jsonl.exists():
            legacy_df = _normalize_step_table(pd.DataFrame(_read_jsonl_rows(legacy_root_jsonl)))
            if not legacy_df.empty:
                legacy_df = legacy_df.loc[
                    (legacy_df["model_name"] == model_name)
                    & (legacy_df["dataset_split"].astype(str) == str(args.split))
                    & (legacy_df["dataset_index"] >= int(args.start_index))
                    & (legacy_df["dataset_index"] < int(max_index))
                ].copy()
                if not legacy_df.empty:
                    _write_step_table(legacy_df, jsonl_path=model_jsonl_path, csv_path=model_csv_path)
                    LOGGER.info(
                        "[%s] Migrated %d rows from legacy root checkpoint into model directory.",
                        model_name,
                        len(legacy_df),
                    )

        expected_metadata = _expected_resume_metadata(
            model_name=model_name,
            split=args.split,
            start_index=args.start_index,
            max_index=max_index,
            k_values=k_values,
            do_sample=args.do_sample,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )

        existing_df = _normalize_step_table(pd.DataFrame(_read_jsonl_rows(model_jsonl_path)))
        if not existing_df.empty:
            existing_df = existing_df.loc[
                (existing_df["model_name"] == model_name)
                & (existing_df["dataset_split"].astype(str) == str(args.split))
                & (existing_df["dataset_index"] >= int(args.start_index))
                & (existing_df["dataset_index"] < int(max_index))
            ].copy()
            existing_df = _write_step_table(existing_df, jsonl_path=model_jsonl_path, csv_path=model_csv_path)
            rows.extend(existing_df.to_dict(orient="records"))

        processed_indices = set()
        if not args.no_resume:
            processed_indices = _load_processed_indices(model_progress_path, expected_metadata)
        inferred_indices = set(existing_df["dataset_index"].astype(int).tolist()) if not existing_df.empty else set()
        processed_indices |= inferred_indices
        processed_indices = {idx for idx in processed_indices if args.start_index <= idx < max_index}
        _write_progress(
            model_progress_path,
            metadata=expected_metadata,
            processed_indices=processed_indices,
            total_target=total_samples,
        )

        pending_indices = [idx for idx in range(args.start_index, max_index) if idx not in processed_indices]
        LOGGER.info(
            "[%s] Resume status: %d/%d completed, %d pending.",
            model_name,
            len(processed_indices),
            total_samples,
            len(pending_indices),
        )
        if not pending_indices:
            LOGGER.info("[%s] Skipping generation, all target samples are already checkpointed.", model_name)
            continue

        model_start = time.perf_counter()
        LOGGER.info("Loading model: %s", model_name)
        model, tokenizer = load_model_and_tokenizer(model_name)
        LOGGER.info("Model loaded in %.1fs: %s", time.perf_counter() - model_start, model_name)
        cfg = GenerationConfig(
            model_name=model_name,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            collect_token_embeddings=True,
            collect_step_signals=True,
            seed=args.seed,
        )
        model_metadata.update(collect_model_metadata(model_name, model, tokenizer))
        model_metadata["capture_layer_names"] = [str(name) for name in cfg.capture_layer_names]
        model_metadata_path.write_text(
            json.dumps(model_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        sample_iter = tqdm(
            pending_indices,
            desc=f"{model_name.split('/')[-1]} samples",
            unit="sample",
            leave=False,
            total=total_samples,
            initial=len(processed_indices),
            disable=args.quiet_progress,
        )
        for local_idx, dataset_index in enumerate(sample_iter, start=1):
            sample_start = time.perf_counter()
            sample_rows: list[dict[str, Any]] = []
            sample = dataset[int(dataset_index)]
            question = str(sample["question"])
            gold_answer = str(sample["answer"])
            question_id = _build_question_id(args.split, int(dataset_index))
            sample_id = f"{model_name}::{question_id}"
            cfg.seed = int(args.seed + dataset_index)
            trace = generate_reasoning_trace(question, cfg, model=model, tokenizer=tokenizer)
            step_texts = [str(text) for text in trace.get("step_texts", [])]
            judged_steps = _collect_judged_steps(step_texts, question, gold_answer)
            judged_by_idx = {int(row["step_index"]): row for row in judged_steps}
            final_correct = _derive_final_correct(trace.get("generated_text", ""), judged_steps, question, gold_answer)

            layer_payloads = _trace_layer_payloads(trace)
            step_signal_rows = trace.get("step_signal_rows", [])
            for layer_payload in layer_payloads:
                token_embeddings = layer_payload["token_embeddings"]
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
                    if step_tokens.shape[0] == 0:
                        continue

                    judgement = judged_by_idx.get(
                        step_index,
                        {
                            "is_correct": False,
                            "parse_fail": True,
                            "reason": "missing_step_judgement",
                            "text": "",
                            "normalized_text": "",
                            "matched_values": [],
                        },
                    )
                    for k_requested in k_values:
                        metric_summary = _metric_summary_for_tokens(step_tokens, int(k_requested))
                        sample_rows.append(
                            {
                                "model_name": model_name,
                                "dataset_name": DATASET_NAME,
                                "dataset_config": DATASET_CONFIG,
                                "dataset_split": args.split,
                                "dataset_index": int(dataset_index),
                                "question_id": question_id,
                                "question_text": question,
                                "sample_id": sample_id,
                                "layer_name": str(layer_payload["layer_name"]),
                                "layer_index": int(layer_payload["layer_index"]),
                                "step_index": step_index,
                                "step_text": str(judgement.get("text", "")),
                                "normalized_step_text": str(judgement.get("normalized_text", "")),
                                "matched_values": "|".join(str(v) for v in judgement.get("matched_values", [])),
                                "k_requested": int(k_requested),
                                "k_effective": int(metric_summary["k_effective"]),
                                "step_token_count": int(metric_summary["step_token_count"]),
                                "lid": metric_summary["lid"],
                                "abid": metric_summary["abid"],
                                "twonn": metric_summary["twonn"],
                                "pr": metric_summary["pr"],
                                "entropy": float(signal["entropy_mean"]),
                                "is_correct": bool(judgement["is_correct"]),
                                "parse_fail": bool(judgement["parse_fail"]),
                                "reason": str(judgement["reason"]),
                                "final_correct": bool(final_correct),
                            }
                        )

            if not sample_rows:
                sample_rows = _build_trace_failure_rows(
                    model_name=model_name,
                    split=args.split,
                    dataset_index=int(dataset_index),
                    question_id=question_id,
                    question=question,
                    sample_id=sample_id,
                    trace=trace,
                    step_texts=step_texts,
                    judged_steps=judged_steps,
                    final_correct=bool(final_correct),
                    k_values=k_values,
                )

            _append_jsonl_rows(model_jsonl_path, sample_rows)
            rows.extend(sample_rows)
            processed_indices.add(int(dataset_index))
            _write_progress(
                model_progress_path,
                metadata=expected_metadata,
                processed_indices=processed_indices,
                total_target=total_samples,
            )
            if local_idx % args.checkpoint_every == 0 or local_idx == len(pending_indices):
                checkpoint_df = _normalize_step_table(pd.DataFrame(_read_jsonl_rows(model_jsonl_path)))
                if not checkpoint_df.empty:
                    checkpoint_df = checkpoint_df.loc[
                        (checkpoint_df["model_name"] == model_name)
                        & (checkpoint_df["dataset_split"].astype(str) == str(args.split))
                        & (checkpoint_df["dataset_index"] >= int(args.start_index))
                        & (checkpoint_df["dataset_index"] < int(max_index))
                    ].copy()
                _write_step_table(checkpoint_df, jsonl_path=model_jsonl_path, csv_path=model_csv_path)

            added_rows = len(sample_rows)
            sample_iter.set_postfix(
                idx=int(dataset_index),
                steps=len(step_signal_rows),
                rows=added_rows,
                final_ok=int(final_correct),
            )
            if args.verbose and args.log_every > 0 and local_idx % args.log_every == 0:
                completed_count = len(processed_indices)
                LOGGER.info(
                    "[%s] sample %d/%d (idx=%d): steps=%d rows=%d final_correct=%s elapsed=%.1fs",
                    model_name,
                    completed_count,
                    total_samples,
                    int(dataset_index),
                    len(step_signal_rows),
                    added_rows,
                    bool(final_correct),
                    time.perf_counter() - sample_start,
                )

        del model
        del tokenizer
        final_model_df = _normalize_step_table(pd.DataFrame(_read_jsonl_rows(model_jsonl_path)))
        if not final_model_df.empty:
            final_model_df = final_model_df.loc[
                (final_model_df["model_name"] == model_name)
                & (final_model_df["dataset_split"].astype(str) == str(args.split))
                & (final_model_df["dataset_index"] >= int(args.start_index))
                & (final_model_df["dataset_index"] < int(max_index))
            ].copy()
        _write_step_table(final_model_df, jsonl_path=model_jsonl_path, csv_path=model_csv_path)
        LOGGER.info("Completed model: %s", model_name)

    if not rows:
        raise RuntimeError("No step-signal rows collected. Increase num-samples or max-new-tokens.")
    return _normalize_step_table(pd.DataFrame(rows))


def _load_input_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        df = pd.DataFrame(rows)
    else:
        raise ValueError("Unsupported input format. Use .csv, .jsonl, or .parquet.")

    required = {"model_name", "step_index", "k_requested", "lid", "abid", "twonn", "pr", "entropy"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input table missing required columns: {missing}")

    if "dataset_index" not in df.columns:
        if "sample_id" in df.columns:
            parsed_idx = df["sample_id"].astype(str).str.extract(r"(\d+)$", expand=False)
            df["dataset_index"] = pd.to_numeric(parsed_idx, errors="coerce").fillna(-1).astype(int)
        else:
            raise ValueError("Input table must include either dataset_index or sample_id.")

    normalized = _normalize_step_table(df)
    if normalized.empty:
        raise ValueError("Input table produced no valid rows after normalization.")
    return normalized


def _persist_step_tables(step_df: pd.DataFrame, out_dir: Path) -> None:
    normalized = _normalize_step_table(step_df)
    combined_dir = out_dir / "combined"
    _write_step_table(
        normalized,
        jsonl_path=combined_dir / "step_signal_table.jsonl",
        csv_path=combined_dir / "step_signal_table.csv",
    )
    # Keep root-level files for backward compatibility.
    _write_step_table(
        normalized,
        jsonl_path=out_dir / "step_signal_table.jsonl",
        csv_path=out_dir / "step_signal_table.csv",
    )

    for model_name, model_df in normalized.groupby("model_name", sort=False):
        model_dir = _model_output_dir(out_dir, str(model_name))
        model_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = model_dir / "model_metadata.json"
        existing_metadata: dict[str, Any] = {}
        if metadata_path.exists():
            try:
                existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                existing_metadata = {}
        existing_metadata.update(
            {
                "model_name": str(model_name),
                "safe_dir_name": _safe_model_dir_name(str(model_name)),
            }
        )
        metadata_path.write_text(
            json.dumps(existing_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _write_step_table(
            model_df,
            jsonl_path=model_dir / "step_signal_table.jsonl",
            csv_path=model_dir / "step_signal_table.csv",
        )


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

        def _prefix_mean(column: str) -> float:
            if column not in prefix.columns:
                return float("nan")
            return float(prefix[column].mean())

        def _prefix_max(column: str) -> float:
            if column not in prefix.columns:
                return float("nan")
            numeric = pd.to_numeric(prefix[column], errors="coerce")
            return float(numeric.max()) if numeric.notna().any() else float("nan")

        def _prefix_last(column: str) -> float:
            if column not in prefix.columns:
                return float("nan")
            numeric = pd.to_numeric(prefix[column], errors="coerce")
            return float(numeric.iloc[-1]) if len(numeric) > 0 else float("nan")

        def _prefix_top2_mean(column: str) -> float:
            if column not in prefix.columns:
                return float("nan")
            numeric = pd.to_numeric(prefix[column], errors="coerce").dropna()
            if numeric.empty:
                return float("nan")
            return float(numeric.nlargest(min(2, len(numeric))).mean())

        warning_delta_series = (
            pd.to_numeric(prefix["warning_score_delta"], errors="coerce")
            if "warning_score_delta" in prefix.columns
            else pd.Series(np.nan, index=prefix.index, dtype=np.float64)
        )
        entropy_delta_series = (
            pd.to_numeric(prefix["entropy_delta"], errors="coerce")
            if "entropy_delta" in prefix.columns
            else pd.Series(np.nan, index=prefix.index, dtype=np.float64)
        )
        reason_weight = (
            pd.to_numeric(prefix["reason_weight"], errors="coerce")
            if "reason_weight" in prefix.columns
            else pd.Series(np.nan, index=prefix.index, dtype=np.float64)
        )
        positive_warning_delta = warning_delta_series.clip(lower=0.0)
        positive_entropy_delta = entropy_delta_series.clip(lower=0.0)
        trace_failure_prefix_any = bool(prefix["reason"].astype(str).isin(TRACE_FAILURE_REASONS).any())

        rows.append(
            {
                "sample_id": sample_id,
                "model_name": str(g["model_name"].iloc[0]),
                "dataset_index": int(g["dataset_index"].iloc[0]),
                "question_id": str(g["question_id"].iloc[0]),
                "layer_name": str(g["layer_name"].iloc[0]) if "layer_name" in g.columns else DEFAULT_ANALYSIS_LAYER,
                "layer_index": int(g["layer_index"].iloc[0]) if "layer_index" in g.columns else -1,
                "k_requested": int(g["k_requested"].iloc[0]),
                "final_failure": int(not bool(g["final_correct"].iloc[0])),
                "lid_prefix_mean": float(prefix["lid"].mean()),
                "abid_prefix_mean": float(prefix["abid"].mean()),
                "twonn_prefix_mean": float(prefix["twonn"].mean()),
                "pr_prefix_mean": float(prefix["pr"].mean()),
                "entropy_prefix_mean": float(prefix["entropy"].mean()),
                "parse_fail_prefix_rate": float(prefix["parse_fail"].astype(int).mean()),
                "parse_fail_prefix_any": int(prefix["parse_fail"].astype(bool).any()),
                "reason_weight_prefix_mean": float(reason_weight.mean()) if reason_weight.notna().any() else float("nan"),
                "reason_weight_prefix_max": float(reason_weight.max()) if reason_weight.notna().any() else float("nan"),
                "warning_prefix_mean": _prefix_mean("warning_score"),
                "warning_prefix_max": _prefix_max("warning_score"),
                "warning_prefix_last": _prefix_last("warning_score"),
                "warning_prefix_top2_mean": _prefix_top2_mean("warning_score"),
                "warning_delta_prefix_mean": _prefix_mean("warning_score_delta"),
                "warning_delta_prefix_max": float(positive_warning_delta.max())
                if positive_warning_delta.notna().any()
                else float("nan"),
                "hybrid_warning_prefix_mean": _prefix_mean("hybrid_warning_score"),
                "hybrid_warning_prefix_max": _prefix_max("hybrid_warning_score"),
                "entropy_delta_prefix_max": float(positive_entropy_delta.max())
                if positive_entropy_delta.notna().any()
                else float("nan"),
                "trace_failure_prefix_any": int(trace_failure_prefix_any),
                "observed_prefix_steps": int(len(prefix)),
                "observed_step_rows": int(len(g)),
                "step_count": int(g["step_index"].max()),
            }
        )
    return pd.DataFrame(rows)


def _prepare_feature_matrix(sample_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    feature_frame = sample_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)
    fill_values = feature_frame.median(numeric_only=True).fillna(0.0)
    return feature_frame.fillna(fill_values).fillna(0.0)


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
    bootstrap_iters: int,
    bootstrap_alpha: float,
    analysis_layer: str = DEFAULT_ANALYSIS_LAYER,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not (0.0 < float(bootstrap_alpha) < 1.0):
        raise ValueError("--bootstrap-alpha must be in (0, 1).")
    layer_df = _filter_analysis_layer(df, analysis_layer)
    k_use = _nearest_k(layer_df, requested_k)
    step_df = layer_df.loc[layer_df["k_requested"] == k_use].copy()
    alarm_payload = evaluate_alarm_policies(step_df, seed=seed)
    step_df = alarm_payload["selected_step_df"].copy()
    sample_df = _build_sample_features(step_df, early_n=early_n)
    if sample_df.empty:
        raise RuntimeError("No sample-level features could be built for experiment A.")

    feature_cols = [
        "lid_prefix_mean",
        "abid_prefix_mean",
        "twonn_prefix_mean",
        "pr_prefix_mean",
        "entropy_prefix_mean",
        "parse_fail_prefix_rate",
        "parse_fail_prefix_any",
        "reason_weight_prefix_mean",
        "reason_weight_prefix_max",
        "warning_prefix_mean",
        "warning_prefix_max",
        "warning_prefix_last",
        "warning_prefix_top2_mean",
        "warning_delta_prefix_mean",
        "warning_delta_prefix_max",
        "hybrid_warning_prefix_mean",
        "hybrid_warning_prefix_max",
        "entropy_delta_prefix_max",
        "trace_failure_prefix_any",
        "observed_prefix_steps",
        "observed_step_rows",
        "step_count",
    ]
    feature_matrix = _prepare_feature_matrix(sample_df, feature_cols)
    x = feature_matrix.to_numpy(dtype=np.float64)
    y = sample_df["final_failure"].to_numpy(dtype=np.int32)
    probs = _cross_validated_probs(x, y, seed=seed)

    warning_peak = _zscore(feature_matrix["warning_prefix_max"])
    warning_jump = _zscore(feature_matrix["warning_delta_prefix_max"])
    trace_failure = feature_matrix["trace_failure_prefix_any"].astype(np.float64)
    baseline_signal = warning_peak + 0.35 * warning_jump + 0.25 * trace_failure
    baseline_probs = _sigmoid(baseline_signal.to_numpy(dtype=np.float64))
    logistic_metrics = _binary_metrics(y, probs)
    baseline_metrics = _binary_metrics(y, baseline_probs)
    selected_alarm_policy = alarm_payload["selected_policy"]
    lead_time_metrics = alarm_payload["selected_metrics"]

    metrics = {
        "analysis_layer": str(analysis_layer),
        "analysis_layer_index": int(step_df["layer_index"].iloc[0]) if "layer_index" in step_df.columns and not step_df.empty else -1,
        "k_used": int(k_use),
        "n_samples": int(len(sample_df)),
        "failure_rate": float(np.mean(y)),
        "coverage": {
            "trace_failure_samples": int(sample_df["trace_failure_prefix_any"].astype(int).sum()),
            "zero_step_samples": int((sample_df["step_count"].astype(int) == 0).sum()),
        },
        "bootstrap": {
            "iterations": int(max(0, bootstrap_iters)),
            "alpha": float(bootstrap_alpha),
        },
        "logistic": logistic_metrics,
        "logistic_ci": _bootstrap_binary_metrics(
            y,
            probs,
            n_bootstrap=bootstrap_iters,
            alpha=bootstrap_alpha,
            seed=seed + 101,
        ),
        "baseline_threshold": baseline_metrics,
        "baseline_threshold_ci": _bootstrap_binary_metrics(
            y,
            baseline_probs,
            n_bootstrap=bootstrap_iters,
            alpha=bootstrap_alpha,
            seed=seed + 202,
        ),
        "lead_time": lead_time_metrics,
        "selected_alarm_policy": selected_alarm_policy,
        "warning_score_spec": {
            "base": (
                "zscore(lid) + zscore(entropy) + reason_weight(reason)"
                " + 0.20 * parse_fail"
            ),
            "reason_weight_table": dict(REASON_WARNING_WEIGHTS),
            "hybrid": "warning_score + 0.75 * zscore(max(delta_warning_score, 0)) + 0.35 * zscore(max(delta_entropy, 0))",
            "trajectory": (
                "warning_score + 0.45 * zscore(running_mean(warning_score))"
                " + 0.35 * zscore(cummax(warning_score))"
                " + 0.40 * zscore(max(delta_warning_score, 0))"
                " + 0.20 * zscore(max(delta_entropy, 0))"
            ),
        },
        "alarm_threshold": float(selected_alarm_policy["threshold"]),
        "alarm_policy_selection_objective": "1.8*alarm_before_error_rate - 0.9*false_alarm_before_any_error_rate - 0.7*late_alarm_rate - 0.5*missed_alarm_rate + 0.35*clipped_lead_time",
    }
    sample_df["failure_prob_logistic"] = probs
    sample_df["failure_prob_baseline"] = baseline_probs
    artifacts = {
        "policy_comparison_df": alarm_payload["policy_comparison"],
        "sample_alarm_df": alarm_payload["selected_sample_df"],
        "threshold_sweep_df": alarm_payload["threshold_sweep"],
        "warning_trajectory_df": alarm_payload["warning_trajectory"],
    }
    return metrics, sample_df, step_df, artifacts


def _build_experiment_a_layer_comparison(
    df: pd.DataFrame,
    *,
    requested_k: int,
    early_n: int,
    seed: int,
    bootstrap_iters: int,
    bootstrap_alpha: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for layer_name in _available_layers(df):
        metrics, _, _, _ = _run_experiment_a_core(
            df=df,
            requested_k=requested_k,
            early_n=early_n,
            seed=seed,
            bootstrap_iters=bootstrap_iters,
            bootstrap_alpha=bootstrap_alpha,
            analysis_layer=layer_name,
        )
        logistic_ci = metrics.get("logistic_ci", {})
        auroc_ci_low, auroc_ci_high = _metric_ci_bounds(logistic_ci, "auroc")
        auprc_ci_low, auprc_ci_high = _metric_ci_bounds(logistic_ci, "auprc")
        rows.append(
            {
                "analysis_layer": str(metrics["analysis_layer"]),
                "analysis_layer_index": int(metrics["analysis_layer_index"]),
                "k_used": int(metrics["k_used"]),
                "n_samples": int(metrics["n_samples"]),
                "failure_rate": float(metrics["failure_rate"]),
                "auroc": float(metrics["logistic"]["auroc"]),
                "auroc_ci_low": auroc_ci_low,
                "auroc_ci_high": auroc_ci_high,
                "auprc": float(metrics["logistic"]["auprc"]),
                "auprc_ci_low": auprc_ci_low,
                "auprc_ci_high": auprc_ci_high,
                "brier": float(metrics["logistic"]["brier"]),
                "ece": float(metrics["logistic"]["ece"]),
                "lead_time_mean": float(metrics["lead_time"]["lead_time_mean"]),
                "alarm_before_error_rate": float(metrics["lead_time"]["alarm_before_error_rate"]),
                "false_alarm_before_any_error_rate": float(metrics["lead_time"]["false_alarm_before_any_error_rate"]),
                "late_alarm_rate": float(metrics["lead_time"]["late_alarm_rate"]),
                "missed_alarm_rate": float(metrics["lead_time"]["missed_alarm_rate"]),
                "early_objective": float(metrics["lead_time"].get("early_objective", float("nan"))),
                "selected_policy_name": str(metrics["selected_alarm_policy"].get("policy_name", "")),
                "selected_policy_score_col": str(metrics["selected_alarm_policy"].get("score_col", "")),
                "selected_policy_threshold": float(metrics["selected_alarm_policy"].get("threshold", float("nan"))),
            }
        )
    layer_df = pd.DataFrame(rows)
    if not layer_df.empty:
        layer_df = layer_df.sort_values("auroc", ascending=False).reset_index(drop=True)
    return layer_df


def _layer_recommendation_from_df(
    layer_df: pd.DataFrame,
    *,
    objective: str = "classification",
) -> dict[str, Any]:
    if layer_df.empty:
        return {}
    objective_key = str(objective or "classification").strip().lower()
    if objective_key == "early_warning":
        sort_columns = [
            column
            for column in ["early_objective", "alarm_before_error_rate", "lead_time_mean", "auroc"]
            if column in layer_df.columns
        ]
        selection_metric = "early_objective_then_alarm_before_error_rate_then_lead_time_mean_then_auroc"
    else:
        sort_columns = [
            column
            for column in ["auroc", "alarm_before_error_rate", "lead_time_mean"]
            if column in layer_df.columns
        ]
        selection_metric = "classification_auroc_then_alarm_before_error_rate_then_lead_time_mean"
    ranked = layer_df.sort_values(
        by=sort_columns,
        ascending=[False] * len(sort_columns),
        kind="mergesort",
    ).reset_index(drop=True)
    best = ranked.iloc[0]
    recommendation = {
        "selection_metric": selection_metric,
        "analysis_layer": str(best["analysis_layer"]),
        "analysis_layer_index": int(best["analysis_layer_index"]),
        "auroc": float(best["auroc"]),
        "auprc": float(best["auprc"]),
        "brier": float(best["brier"]),
        "ece": float(best["ece"]),
        "lead_time_mean": float(best["lead_time_mean"]),
        "alarm_before_error_rate": float(best["alarm_before_error_rate"]),
    }
    if "false_alarm_before_any_error_rate" in best.index:
        recommendation["false_alarm_before_any_error_rate"] = float(best["false_alarm_before_any_error_rate"])
    if "late_alarm_rate" in best.index:
        recommendation["late_alarm_rate"] = float(best["late_alarm_rate"])
    if "missed_alarm_rate" in best.index:
        recommendation["missed_alarm_rate"] = float(best["missed_alarm_rate"])
    if "early_objective" in best.index:
        recommendation["early_objective"] = float(best["early_objective"])
    if "selected_policy_name" in best.index:
        recommendation["selected_policy_name"] = str(best["selected_policy_name"])
    if "selected_policy_score_col" in best.index:
        recommendation["selected_policy_score_col"] = str(best["selected_policy_score_col"])
    if "selected_policy_threshold" in best.index:
        recommendation["selected_policy_threshold"] = float(best["selected_policy_threshold"])
    return recommendation


def _best_layer_by_model(
    df: pd.DataFrame,
    *,
    requested_k: int,
    early_n: int,
    seed: int,
    bootstrap_iters: int,
    bootstrap_alpha: float,
    objective: str = "classification",
) -> dict[str, dict[str, Any]]:
    best_layers: dict[str, dict[str, Any]] = {}
    for model_name, model_df in df.groupby("model_name", sort=False):
        layer_df = _build_experiment_a_layer_comparison(
            model_df,
            requested_k=requested_k,
            early_n=early_n,
            seed=seed,
            bootstrap_iters=bootstrap_iters,
            bootstrap_alpha=bootstrap_alpha,
        )
        recommendation = _layer_recommendation_from_df(layer_df, objective=objective)
        if recommendation:
            best_layers[str(model_name)] = recommendation
    return best_layers


def _write_policy_outputs(out_dir: Path, artifacts: dict[str, Any]) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    policy_df = artifacts["policy_comparison_df"].copy()
    policy_csv = out_dir / "experiment_a_alarm_policy_comparison.csv"
    policy_json = out_dir / "experiment_a_alarm_policy_comparison.json"
    policy_parquet = _write_table_with_parquet(policy_df, policy_csv)
    policy_json.write_text(json.dumps(policy_df.to_dict(orient="records"), ensure_ascii=False, indent=2), encoding="utf-8")
    if not policy_df.empty:
        fig = go.Figure(
            data=[
                go.Bar(
                    x=policy_df["policy_name"].tolist(),
                    y=policy_df["early_objective"].tolist(),
                    name="Early Objective",
                )
            ]
        )
        fig.update_layout(title="Experiment A - Alarm Policy Comparison", template="plotly_white")
        fig.write_html(str(out_dir / "experiment_a_alarm_policy_comparison.html"))
    paths["policy_csv"] = str(policy_csv)
    paths["policy_parquet"] = str(policy_parquet)
    paths["policy_json"] = str(policy_json)

    sample_alarm_df = artifacts["sample_alarm_df"].copy()
    sample_alarm_csv = out_dir / "experiment_a_alarm_timing.csv"
    sample_alarm_parquet = _write_table_with_parquet(sample_alarm_df, sample_alarm_csv)
    paths["sample_alarm_csv"] = str(sample_alarm_csv)
    paths["sample_alarm_parquet"] = str(sample_alarm_parquet)

    sweep_df = artifacts["threshold_sweep_df"].copy()
    sweep_csv = out_dir / "experiment_a_threshold_sweep.csv"
    sweep_json = out_dir / "experiment_a_threshold_sweep.json"
    sweep_parquet = _write_table_with_parquet(sweep_df, sweep_csv)
    sweep_json.write_text(json.dumps(sweep_df.to_dict(orient="records"), ensure_ascii=False, indent=2), encoding="utf-8")
    if not sweep_df.empty:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=sweep_df["threshold"].tolist(),
                y=sweep_df["alarm_before_error_rate"].tolist(),
                mode="lines+markers",
                name="Alarm Before Error Rate",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sweep_df["threshold"].tolist(),
                y=sweep_df["false_alarm_before_any_error_rate"].tolist(),
                mode="lines+markers",
                name="False Alarm Rate",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sweep_df["threshold"].tolist(),
                y=sweep_df["lead_time_mean"].tolist(),
                mode="lines+markers",
                name="Lead Time Mean",
                yaxis="y2",
            )
        )
        fig.update_layout(
            title="Experiment A - Threshold Sweep",
            xaxis_title="Threshold",
            yaxis_title="Rate",
            yaxis2=dict(title="Lead Time", overlaying="y", side="right"),
            template="plotly_white",
        )
        fig.write_html(str(out_dir / "experiment_a_threshold_sweep.html"))
    paths["threshold_sweep_csv"] = str(sweep_csv)
    paths["threshold_sweep_parquet"] = str(sweep_parquet)
    paths["threshold_sweep_json"] = str(sweep_json)

    trajectory_df = artifacts["warning_trajectory_df"].copy()
    trajectory_csv = out_dir / "experiment_a_warning_trajectory.csv"
    trajectory_json = out_dir / "experiment_a_warning_trajectory.json"
    trajectory_parquet = _write_table_with_parquet(trajectory_df, trajectory_csv)
    trajectory_json.write_text(
        json.dumps(trajectory_df.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not trajectory_df.empty:
        fig = go.Figure()
        for outcome in trajectory_df["trace_outcome"].dropna().unique().tolist():
            subset = trajectory_df.loc[trajectory_df["trace_outcome"] == outcome].copy()
            fig.add_trace(
                go.Scatter(
                    x=subset["step_index"].tolist(),
                    y=subset["warning_score_mean"].tolist(),
                    mode="lines+markers",
                    name=f"{outcome} traces",
                )
            )
            if "hybrid_warning_score_mean" in subset.columns:
                fig.add_trace(
                    go.Scatter(
                        x=subset["step_index"].tolist(),
                        y=subset["hybrid_warning_score_mean"].tolist(),
                        mode="lines+markers",
                        name=f"{outcome} hybrid",
                        line=dict(dash="dot"),
                    )
                )
            if "trajectory_warning_score_mean" in subset.columns:
                fig.add_trace(
                    go.Scatter(
                        x=subset["step_index"].tolist(),
                        y=subset["trajectory_warning_score_mean"].tolist(),
                        mode="lines+markers",
                        name=f"{outcome} trajectory",
                        line=dict(dash="dash"),
                    )
                )
        fig.update_layout(
            title="Experiment A - Warning Score by Step Index",
            xaxis_title="Step Index",
            yaxis_title="Mean Warning Score",
            template="plotly_white",
        )
        fig.write_html(str(out_dir / "experiment_a_warning_trajectory.html"))
    paths["warning_trajectory_csv"] = str(trajectory_csv)
    paths["warning_trajectory_parquet"] = str(trajectory_parquet)
    paths["warning_trajectory_json"] = str(trajectory_json)
    return paths


def _write_calibration_outputs(
    out_dir: Path,
    calibration_payload: dict[str, Any],
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    comparison_df = calibration_payload["comparison"].copy()
    comparison_csv = out_dir / "experiment_a_raw_vs_calibrated.csv"
    comparison_json = out_dir / "experiment_a_raw_vs_calibrated.json"
    comparison_parquet = _write_table_with_parquet(comparison_df, comparison_csv)
    comparison_json.write_text(
        json.dumps(comparison_df.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["comparison_csv"] = str(comparison_csv)
    paths["comparison_parquet"] = str(comparison_parquet)
    paths["comparison_json"] = str(comparison_json)

    reliability_df = calibration_payload["reliability"].copy()
    reliability_csv = out_dir / "experiment_a_reliability_curve.csv"
    reliability_json = out_dir / "experiment_a_reliability_curve.json"
    reliability_parquet = _write_table_with_parquet(reliability_df, reliability_csv)
    reliability_json.write_text(
        json.dumps(reliability_df.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["reliability_csv"] = str(reliability_csv)
    paths["reliability_parquet"] = str(reliability_parquet)
    paths["reliability_json"] = str(reliability_json)

    if not reliability_df.empty:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[0.0, 1.0],
                y=[0.0, 1.0],
                mode="lines",
                name="Perfect calibration",
                line=dict(dash="dash"),
            )
        )
        for label in reliability_df["label"].dropna().unique().tolist():
            subset = reliability_df.loc[reliability_df["label"] == label].copy()
            fig.add_trace(
                go.Scatter(
                    x=subset["confidence_mean"].tolist(),
                    y=subset["accuracy_mean"].tolist(),
                    mode="lines+markers",
                    name=str(label),
                )
            )
        fig.update_layout(
            title="Experiment A - Reliability Curve",
            xaxis_title="Mean confidence",
            yaxis_title="Empirical accuracy",
            template="plotly_white",
        )
        fig.write_html(str(out_dir / "experiment_a_reliability_curve.html"))

    for method_name, artifact in calibration_payload["artifacts"].items():
        artifact_path = out_dir / f"calibration_artifact_{method_name}.json"
        artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
        paths[f"artifact_{method_name}"] = str(artifact_path)
    return paths


def _run_experiment_a(df: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    metrics, sample_df, step_df, artifacts = _run_experiment_a_core(
        df=df,
        requested_k=args.primary_k,
        early_n=args.early_n,
        seed=args.seed,
        bootstrap_iters=args.bootstrap_iters,
        bootstrap_alpha=args.bootstrap_alpha,
        analysis_layer=args.analysis_layer,
    )
    y = sample_df["final_failure"].to_numpy(dtype=np.int32)
    calibration_payload = evaluate_calibrators(
        sample_df["failure_prob_logistic"].to_numpy(dtype=np.float64),
        y,
        seed=args.seed + 303,
    )
    selected_method = str(calibration_payload["selected_method"])
    selected_artifact = calibration_payload["artifacts"][selected_method]
    layer_comparison_df = _build_experiment_a_layer_comparison(
        df,
        requested_k=args.primary_k,
        early_n=args.early_n,
        seed=args.seed,
        bootstrap_iters=args.bootstrap_iters,
        bootstrap_alpha=args.bootstrap_alpha,
    )
    classification_layer_recommendation = _layer_recommendation_from_df(
        layer_comparison_df,
        objective="classification",
    )
    early_warning_layer_recommendation = _layer_recommendation_from_df(
        layer_comparison_df,
        objective="early_warning",
    )
    metrics["layer_recommendation"] = classification_layer_recommendation
    metrics["layer_recommendation_classification"] = classification_layer_recommendation
    metrics["layer_recommendation_early_warning"] = early_warning_layer_recommendation
    metrics["primary_fixed_layer"] = {
        "analysis_layer": str(args.analysis_layer),
        "rationale": (
            "Keep the fixed analysis layer as the headline Experiment A view; "
            "treat per-layer recommendations as secondary diagnostics."
        ),
    }
    sample_df["failure_prob_calibrated"] = apply_calibration_artifact(
        sample_df["failure_prob_logistic"].to_numpy(dtype=np.float64),
        selected_artifact,
    )
    sample_df["calibration_method_selected"] = selected_method
    metrics["calibration"] = {
        "selected_method": selected_method,
        "rows": calibration_payload["comparison"].to_dict(orient="records"),
    }
    sample_path = out_dir / "experiment_a_sample_predictions.csv"
    step_path = out_dir / "experiment_a_step_scores.csv"
    summary_path = out_dir / "experiment_a_summary.json"
    layer_comparison_csv = out_dir / "experiment_a_layer_comparison.csv"
    layer_comparison_json = out_dir / "experiment_a_layer_comparison.json"
    sample_parquet = _write_table_with_parquet(sample_df, sample_path)
    step_parquet = _write_table_with_parquet(step_df, step_path)
    layer_comparison_parquet = _write_table_with_parquet(layer_comparison_df, layer_comparison_csv)
    layer_comparison_json.write_text(
        json.dumps(layer_comparison_df.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    policy_paths = _write_policy_outputs(out_dir, artifacts)
    calibration_paths = _write_calibration_outputs(out_dir, calibration_payload)

    per_model_outputs: dict[str, dict[str, Any]] = {}
    for model_name, model_df in df.groupby("model_name", sort=False):
        model_metrics, model_sample_df, model_step_df, model_artifacts = _run_experiment_a_core(
            df=model_df,
            requested_k=args.primary_k,
            early_n=args.early_n,
            seed=args.seed,
            bootstrap_iters=args.bootstrap_iters,
            bootstrap_alpha=args.bootstrap_alpha,
            analysis_layer=args.analysis_layer,
        )
        model_y = model_sample_df["final_failure"].to_numpy(dtype=np.int32)
        model_calibration_payload = evaluate_calibrators(
            model_sample_df["failure_prob_logistic"].to_numpy(dtype=np.float64),
            model_y,
            seed=args.seed + 303,
        )
        model_selected_method = str(model_calibration_payload["selected_method"])
        model_selected_artifact = model_calibration_payload["artifacts"][model_selected_method]
        model_layer_comparison_df = _build_experiment_a_layer_comparison(
            model_df,
            requested_k=args.primary_k,
            early_n=args.early_n,
            seed=args.seed,
            bootstrap_iters=args.bootstrap_iters,
            bootstrap_alpha=args.bootstrap_alpha,
        )
        model_classification_layer_recommendation = _layer_recommendation_from_df(
            model_layer_comparison_df,
            objective="classification",
        )
        model_early_warning_layer_recommendation = _layer_recommendation_from_df(
            model_layer_comparison_df,
            objective="early_warning",
        )
        model_metrics["layer_recommendation"] = model_classification_layer_recommendation
        model_metrics["layer_recommendation_classification"] = model_classification_layer_recommendation
        model_metrics["layer_recommendation_early_warning"] = model_early_warning_layer_recommendation
        model_metrics["primary_fixed_layer"] = {
            "analysis_layer": str(args.analysis_layer),
            "rationale": (
                "Keep the fixed analysis layer as the per-model headline view; "
                "use layer recommendations only as sensitivity analysis."
            ),
        }
        model_sample_df["failure_prob_calibrated"] = apply_calibration_artifact(
            model_sample_df["failure_prob_logistic"].to_numpy(dtype=np.float64),
            model_selected_artifact,
        )
        model_sample_df["calibration_method_selected"] = model_selected_method
        model_metrics["calibration"] = {
            "selected_method": model_selected_method,
            "rows": model_calibration_payload["comparison"].to_dict(orient="records"),
        }
        model_exp_dir = _model_output_dir(out_dir, str(model_name)) / "experiments"
        model_exp_dir.mkdir(parents=True, exist_ok=True)
        model_summary_path = model_exp_dir / "experiment_a_summary.json"
        model_sample_path = model_exp_dir / "experiment_a_sample_predictions.csv"
        model_step_path = model_exp_dir / "experiment_a_step_scores.csv"
        model_layer_comparison_csv = model_exp_dir / "experiment_a_layer_comparison.csv"
        model_layer_comparison_json = model_exp_dir / "experiment_a_layer_comparison.json"
        model_sample_parquet = _write_table_with_parquet(model_sample_df, model_sample_path)
        model_step_parquet = _write_table_with_parquet(model_step_df, model_step_path)
        model_layer_comparison_parquet = _write_table_with_parquet(model_layer_comparison_df, model_layer_comparison_csv)
        model_layer_comparison_json.write_text(
            json.dumps(model_layer_comparison_df.to_dict(orient="records"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        model_summary_path.write_text(
            json.dumps(model_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        model_policy_paths = _write_policy_outputs(model_exp_dir, model_artifacts)
        model_calibration_paths = _write_calibration_outputs(model_exp_dir, model_calibration_payload)
        per_model_outputs[str(model_name)] = {
            "summary": model_metrics,
            "sample_path": str(model_sample_path),
            "step_path": str(model_step_path),
            "layer_comparison_csv": str(model_layer_comparison_csv),
            "sample_parquet": str(model_sample_parquet),
            "step_parquet": str(model_step_parquet),
            "layer_comparison_parquet": str(model_layer_comparison_parquet),
            "policy_outputs": model_policy_paths,
            "calibration_outputs": model_calibration_paths,
        }

    return {
        "summary": metrics,
        "sample_path": str(sample_path),
        "step_path": str(step_path),
        "layer_comparison_csv": str(layer_comparison_csv),
        "sample_parquet": str(sample_parquet),
        "step_parquet": str(step_parquet),
        "layer_comparison_parquet": str(layer_comparison_parquet),
        "policy_outputs": policy_paths,
        "calibration_outputs": calibration_paths,
        "per_model": per_model_outputs,
    }


def _metric_ci_bounds(ci_payload: dict[str, Any], metric_name: str) -> tuple[float, float]:
    metric_payload = ci_payload.get(metric_name, {})
    low = metric_payload.get("low", float("nan"))
    high = metric_payload.get("high", float("nan"))
    return float(low), float(high)


def _build_experiment_b_rows(
    df: pd.DataFrame,
    args: argparse.Namespace,
    *,
    scope_label: str,
    analysis_layer_by_model: dict[str, str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, model_df in df.groupby("model_name", sort=False):
        selected_layer = (
            str(analysis_layer_by_model.get(str(model_name), args.analysis_layer))
            if analysis_layer_by_model is not None
            else str(args.analysis_layer)
        )
        metrics, _, step_df, _ = _run_experiment_a_core(
            df=model_df,
            requested_k=args.primary_k,
            early_n=args.early_n,
            seed=args.seed,
            bootstrap_iters=args.bootstrap_iters,
            bootstrap_alpha=args.bootstrap_alpha,
            analysis_layer=selected_layer,
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

        logistic_ci = metrics.get("logistic_ci", {})
        auroc_ci_low, auroc_ci_high = _metric_ci_bounds(logistic_ci, "auroc")
        auprc_ci_low, auprc_ci_high = _metric_ci_bounds(logistic_ci, "auprc")
        brier_ci_low, brier_ci_high = _metric_ci_bounds(logistic_ci, "brier")
        ece_ci_low, ece_ci_high = _metric_ci_bounds(logistic_ci, "ece")

        rows.append(
            {
                "comparison_scope": scope_label,
                "model_name": model_name,
                "analysis_layer": str(metrics["analysis_layer"]),
                "analysis_layer_index": int(metrics["analysis_layer_index"]),
                "k_used": int(metrics["k_used"]),
                "n_samples": int(metrics["n_samples"]),
                "failure_rate": float(metrics["failure_rate"]),
                "auroc": float(metrics["logistic"]["auroc"]),
                "auroc_ci_low": auroc_ci_low,
                "auroc_ci_high": auroc_ci_high,
                "auprc": float(metrics["logistic"]["auprc"]),
                "auprc_ci_low": auprc_ci_low,
                "auprc_ci_high": auprc_ci_high,
                "brier": float(metrics["logistic"]["brier"]),
                "brier_ci_low": brier_ci_low,
                "brier_ci_high": brier_ci_high,
                "ece": float(metrics["logistic"]["ece"]),
                "ece_ci_low": ece_ci_low,
                "ece_ci_high": ece_ci_high,
                "signal_gap_mean": float(np.mean(wrong_scores) - np.mean(correct_scores))
                if len(correct_scores) > 0 and len(wrong_scores) > 0
                else float("nan"),
                "cohen_d_wrong_vs_correct": _cohen_d(wrong_scores, correct_scores),
                "mann_whitney_pvalue": mwu_p,
            }
        )
    comparison_df = pd.DataFrame(rows)
    if not comparison_df.empty:
        comparison_df = comparison_df.sort_values("auroc", ascending=False).reset_index(drop=True)
    return comparison_df


def _common_dataset_indices_for_primary_k(
    df: pd.DataFrame,
    requested_k: int,
    analysis_layer: str = DEFAULT_ANALYSIS_LAYER,
) -> dict[str, Any]:
    shared_layer_map = {str(model_name): str(analysis_layer) for model_name in df["model_name"].astype(str).unique()}
    return _common_dataset_indices_for_layer_map(df, requested_k=requested_k, analysis_layer_by_model=shared_layer_map)


def _common_dataset_indices_for_layer_map(
    df: pd.DataFrame,
    *,
    requested_k: int,
    analysis_layer_by_model: dict[str, str],
) -> dict[str, Any]:
    per_model_indices: dict[str, set[int]] = {}
    k_used_by_model: dict[str, int] = {}
    layer_by_model: dict[str, str] = {}
    for model_name, model_df in df.groupby("model_name", sort=False):
        selected_layer = str(analysis_layer_by_model.get(str(model_name), DEFAULT_ANALYSIS_LAYER))
        model_layer_df = _filter_analysis_layer(model_df, selected_layer)
        k_used = _nearest_k(model_layer_df, requested_k)
        k_used_by_model[str(model_name)] = int(k_used)
        layer_by_model[str(model_name)] = selected_layer
        idx_set = set(model_layer_df.loc[model_layer_df["k_requested"] == k_used, "dataset_index"].astype(int).tolist())
        per_model_indices[str(model_name)] = idx_set

    if not per_model_indices:
        common_indices: list[int] = []
    elif len(per_model_indices) == 1:
        common_indices = sorted(next(iter(per_model_indices.values())))
    else:
        common = set.intersection(*(set(values) for values in per_model_indices.values()))
        common_indices = sorted(int(v) for v in common)

    return {
        "requested_primary_k": int(requested_k),
        "analysis_layer": str(next(iter(layer_by_model.values()), DEFAULT_ANALYSIS_LAYER))
        if len(set(layer_by_model.values())) <= 1
        else "",
        "analysis_layer_by_model": layer_by_model,
        "k_used_by_model": k_used_by_model,
        "per_model_index_counts": {name: int(len(values)) for name, values in per_model_indices.items()},
        "common_index_count": int(len(common_indices)),
        "common_dataset_indices": common_indices,
    }


def _write_experiment_b_table(
    comparison_df: pd.DataFrame,
    *,
    csv_path: Path,
    json_path: Path,
    html_path: Path | None = None,
    title: str = "Experiment B - Model Comparison (AUROC)",
) -> None:
    _write_table_with_parquet(comparison_df, csv_path)
    json_path.write_text(
        json.dumps(comparison_df.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if html_path is not None and not comparison_df.empty:
        fig = go.Figure(
            data=[
                go.Bar(
                    x=comparison_df["model_name"].tolist(),
                    y=comparison_df["auroc"].tolist(),
                    name="AUROC",
                )
            ]
        )
        fig.update_layout(title=title, template="plotly_white")
        fig.write_html(str(html_path))


def _run_experiment_b(df: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    all_df = _build_experiment_b_rows(df, args, scope_label="all_samples")
    all_csv_path = out_dir / "experiment_b_model_comparison_all_samples.csv"
    all_json_path = out_dir / "experiment_b_model_comparison_all_samples.json"
    all_html_path = out_dir / "experiment_b_model_comparison_all_samples.html"
    _write_experiment_b_table(
        all_df,
        csv_path=all_csv_path,
        json_path=all_json_path,
        html_path=all_html_path,
        title="Experiment B - Model Comparison (All Samples)",
    )

    common_meta = _common_dataset_indices_for_primary_k(
        df,
        requested_k=args.primary_k,
        analysis_layer=args.analysis_layer,
    )
    common_idx_path = out_dir / "experiment_b_common_dataset_indices.json"
    common_idx_path.write_text(json.dumps(common_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    common_indices = set(int(v) for v in common_meta["common_dataset_indices"])
    if common_indices:
        common_step_df = df.loc[df["dataset_index"].astype(int).isin(common_indices)].copy()
        common_df = _build_experiment_b_rows(common_step_df, args, scope_label="common_index")
    else:
        common_df = pd.DataFrame(columns=all_df.columns)

    common_csv_path = out_dir / "experiment_b_model_comparison_common_index.csv"
    common_json_path = out_dir / "experiment_b_model_comparison_common_index.json"
    common_html_path = out_dir / "experiment_b_model_comparison_common_index.html"
    _write_experiment_b_table(
        common_df,
        csv_path=common_csv_path,
        json_path=common_json_path,
        html_path=common_html_path,
        title="Experiment B - Model Comparison (Common Dataset Indices)",
    )

    best_layer_meta = _best_layer_by_model(
        df,
        requested_k=args.primary_k,
        early_n=args.early_n,
        seed=args.seed,
        bootstrap_iters=args.bootstrap_iters,
        bootstrap_alpha=args.bootstrap_alpha,
        objective="classification",
    )
    best_layer_map = {
        str(model_name): str(meta["analysis_layer"])
        for model_name, meta in best_layer_meta.items()
        if meta.get("analysis_layer")
    }
    best_common_meta = _common_dataset_indices_for_layer_map(
        df,
        requested_k=args.primary_k,
        analysis_layer_by_model=best_layer_map,
    )
    best_common_idx_path = out_dir / "experiment_b_best_layer_dataset_indices.json"
    best_common_idx_path.write_text(json.dumps(best_common_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    best_common_indices = set(int(v) for v in best_common_meta["common_dataset_indices"])
    if best_common_indices:
        best_step_df = df.loc[df["dataset_index"].astype(int).isin(best_common_indices)].copy()
        best_df = _build_experiment_b_rows(
            best_step_df,
            args,
            scope_label="per_model_best_layer",
            analysis_layer_by_model=best_layer_map,
        )
    else:
        best_df = pd.DataFrame(columns=all_df.columns)

    best_csv_path = out_dir / "experiment_b_model_comparison_best_layer.csv"
    best_json_path = out_dir / "experiment_b_model_comparison_best_layer.json"
    best_html_path = out_dir / "experiment_b_model_comparison_best_layer.html"
    _write_experiment_b_table(
        best_df,
        csv_path=best_csv_path,
        json_path=best_json_path,
        html_path=best_html_path,
        title="Experiment B - Model Comparison (Per-Model Best Layer)",
    )

    n_models = int(df["model_name"].nunique())
    primary_scope = "common_index" if n_models >= 2 and not common_df.empty else "all_samples"
    primary_df = common_df if primary_scope == "common_index" else all_df

    # Backward-compatible primary output paths now point to the fair comparison by default.
    primary_csv_path = out_dir / "experiment_b_model_comparison.csv"
    primary_json_path = out_dir / "experiment_b_model_comparison.json"
    primary_html_path = out_dir / "experiment_b_model_comparison.html"
    _write_experiment_b_table(
        primary_df,
        csv_path=primary_csv_path,
        json_path=primary_json_path,
        html_path=primary_html_path,
        title=f"Experiment B - Model Comparison ({primary_scope})",
    )

    rows_by_scope = {
        "all_samples": all_df.to_dict(orient="records"),
        "common_index": common_df.to_dict(orient="records"),
        "best_layer": best_df.to_dict(orient="records"),
    }
    for scope_label, rows in rows_by_scope.items():
        for row in rows:
            model_name = str(row["model_name"])
            model_exp_dir = _model_output_dir(out_dir, model_name) / "experiments"
            model_exp_dir.mkdir(parents=True, exist_ok=True)
            (model_exp_dir / f"experiment_b_row_{scope_label}.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    for row in primary_df.to_dict(orient="records"):
        model_name = str(row["model_name"])
        model_exp_dir = _model_output_dir(out_dir, model_name) / "experiments"
        model_exp_dir.mkdir(parents=True, exist_ok=True)
        (model_exp_dir / "experiment_b_row.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    primary_rows = primary_df.to_dict(orient="records")
    return {
        "primary_scope": primary_scope,
        "primary_scope_rationale": (
            "Use common_index when comparing multiple models to ensure fair, same-sample evaluation. "
            "Fallback to all_samples when common intersection is empty."
        ),
        "primary_analysis_layer": str(args.analysis_layer),
        "primary_analysis_layer_rationale": (
            "Keep the fixed analysis layer as the headline Experiment B view; "
            "use per-model best-layer results as a secondary sensitivity analysis."
        ),
        "common_dataset_index_count": int(common_meta["common_index_count"]),
        "common_dataset_indices_path": str(common_idx_path),
        "all_samples": {
            "rows": all_df.to_dict(orient="records"),
            "csv_path": str(all_csv_path),
            "json_path": str(all_json_path),
            "parquet_path": str(all_csv_path.with_suffix(".parquet")),
        },
        "common_index": {
            "rows": common_df.to_dict(orient="records"),
            "csv_path": str(common_csv_path),
            "json_path": str(common_json_path),
            "parquet_path": str(common_csv_path.with_suffix(".parquet")),
        },
        "best_layer": {
            "rows": best_df.to_dict(orient="records"),
            "csv_path": str(best_csv_path),
            "json_path": str(best_json_path),
            "parquet_path": str(best_csv_path.with_suffix(".parquet")),
            "dataset_indices_path": str(best_common_idx_path),
            "analysis_layer_by_model": best_layer_map,
            "recommendations_by_model": best_layer_meta,
            "reporting_role": "secondary_sensitivity_view",
        },
        "rows": primary_rows,
        "csv_path": str(primary_csv_path),
        "parquet_path": str(primary_csv_path.with_suffix(".parquet")),
    }


def _run_experiment_c(df: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    model_outputs: list[dict[str, Any]] = []
    for model_name, model_df in df.groupby("model_name", sort=False):
        available_k = sorted(set(int(v) for v in model_df["k_requested"].tolist()))
        k_rows: list[dict[str, Any]] = []
        preds_by_k: dict[int, pd.DataFrame] = {}
        for k in available_k:
            metrics, sample_df, _, _ = _run_experiment_a_core(
                df=model_df,
                requested_k=k,
                early_n=args.early_n,
                seed=args.seed,
                bootstrap_iters=args.bootstrap_iters,
                bootstrap_alpha=args.bootstrap_alpha,
                analysis_layer=args.analysis_layer,
            )
            logistic_ci = metrics.get("logistic_ci", {})
            auroc_ci_low, auroc_ci_high = _metric_ci_bounds(logistic_ci, "auroc")
            auprc_ci_low, auprc_ci_high = _metric_ci_bounds(logistic_ci, "auprc")
            brier_ci_low, brier_ci_high = _metric_ci_bounds(logistic_ci, "brier")
            ece_ci_low, ece_ci_high = _metric_ci_bounds(logistic_ci, "ece")
            k_rows.append(
                {
                    "model_name": model_name,
                    "analysis_layer": str(metrics["analysis_layer"]),
                    "analysis_layer_index": int(metrics["analysis_layer_index"]),
                    "k_requested": int(k),
                    "k_used": int(metrics["k_used"]),
                    "n_samples": int(metrics["n_samples"]),
                    "auroc": float(metrics["logistic"]["auroc"]),
                    "auroc_ci_low": auroc_ci_low,
                    "auroc_ci_high": auroc_ci_high,
                    "auprc": float(metrics["logistic"]["auprc"]),
                    "auprc_ci_low": auprc_ci_low,
                    "auprc_ci_high": auprc_ci_high,
                    "brier": float(metrics["logistic"]["brier"]),
                    "brier_ci_low": brier_ci_low,
                    "brier_ci_high": brier_ci_high,
                    "ece": float(metrics["logistic"]["ece"]),
                    "ece_ci_low": ece_ci_low,
                    "ece_ci_high": ece_ci_high,
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

        model_exp_dir = _model_output_dir(out_dir, str(model_name)) / "experiments"
        model_exp_dir.mkdir(parents=True, exist_ok=True)
        _write_table_with_parquet(metrics_df, model_exp_dir / "experiment_c_k_sensitivity.csv")
        (model_exp_dir / "experiment_c_k_sensitivity.json").write_text(
            json.dumps(
                {
                    "model_name": str(model_name),
                    "metrics_by_k": metrics_df.to_dict(orient="records"),
                    "auroc_cv_over_k": float(cv_auroc),
                    "kendall_tau_pairs": tau_rows,
                    "kendall_tau_mean": float(np.mean([row["kendall_tau"] for row in tau_rows]))
                    if tau_rows
                    else float("nan"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
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
            model_html_path = model_exp_dir / "experiment_c_k_sensitivity.html"
            fig.write_html(str(model_html_path))
            fig.write_html(str(out_dir / f"experiment_c_k_sensitivity_{safe_name}.html"))

    flat_rows: list[dict[str, Any]] = []
    for model_output in model_outputs:
        for row in model_output["metrics_by_k"]:
            flat_rows.append(row)
    metrics_csv = out_dir / "experiment_c_k_sensitivity.csv"
    metrics_parquet = _write_table_with_parquet(pd.DataFrame(flat_rows), metrics_csv)

    summary_json = out_dir / "experiment_c_k_sensitivity.json"
    summary_json.write_text(json.dumps(model_outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"model_outputs": model_outputs, "csv_path": str(metrics_csv), "parquet_path": str(metrics_parquet)}


def main() -> None:
    args = parse_args()
    if args.bootstrap_iters < 0:
        raise ValueError("--bootstrap-iters must be >= 0")
    if not (0.0 < float(args.bootstrap_alpha) < 1.0):
        raise ValueError("--bootstrap-alpha must be in (0, 1)")
    run_config = _run_config_payload(args)
    run_config_hash = _stable_payload_hash(run_config)
    _configure_logging(args.verbose)
    _apply_runtime_env(args)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Starting ablation run. Output dir: %s", out_dir)
    LOGGER.info(
        "Config | split=%s start=%d n=%d models=%s k=%s analysis_layer=%s",
        args.split,
        args.start_index,
        args.num_samples,
        args.models,
        args.k_values,
        args.analysis_layer,
    )

    if args.input:
        LOGGER.info("Loading precomputed step table: %s", args.input)
        step_df = _load_input_table(Path(args.input))
    else:
        LOGGER.info("Collecting step-signal table from model generations.")
        step_df = _collect_step_signal_table(args, out_dir=out_dir)

    _persist_step_tables(step_df, out_dir)
    model_names = sorted(set(step_df["model_name"].astype(str).tolist()))
    model_metadata_map = _load_model_metadata_map(out_dir, model_names)
    run_metadata_path = out_dir / "run_metadata.json"
    run_metadata = {
        "source_mode": "input_table" if args.input else "generation",
        "input_path": str(args.input or ""),
        "config_hash": run_config_hash,
        "config": run_config,
        "available_layers": _available_layers(step_df),
        "model_metadata_by_model": model_metadata_map,
    }
    run_metadata_path.write_text(json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    output: dict[str, Any] = {
        "rows": int(len(step_df)),
        "models": model_names,
        "available_layers": _available_layers(step_df),
        "analysis_layer": str(args.analysis_layer),
        "k_values": sorted(set(int(v) for v in step_df["k_requested"].tolist())),
        "primary_k_policy": {
            "requested_primary_k": int(args.primary_k),
            "default_primary_k": int(PRIMARY_K_DEFAULT),
            "rationale": PRIMARY_K_RATIONALE,
        },
        "bootstrap": {
            "iterations": int(args.bootstrap_iters),
            "alpha": float(args.bootstrap_alpha),
        },
        "seed": int(args.seed),
        "reproducibility": {
            "source_mode": "input_table" if args.input else "generation",
            "input_path": str(args.input or ""),
            "config_hash": run_config_hash,
            "run_metadata_path": str(run_metadata_path),
            "model_metadata_by_model": model_metadata_map,
        },
        "output_layout": {
            "models_root": str(out_dir / "models"),
            "combined_table_csv": str(out_dir / "combined" / "step_signal_table.csv"),
            "combined_table_jsonl": str(out_dir / "combined" / "step_signal_table.jsonl"),
            "combined_table_parquet": str(out_dir / "combined" / "step_signal_table.parquet"),
            "root_table_csv": str(out_dir / "step_signal_table.csv"),
            "root_table_jsonl": str(out_dir / "step_signal_table.jsonl"),
            "root_table_parquet": str(out_dir / "step_signal_table.parquet"),
        },
    }
    if args.experiment in {"A", "ALL"}:
        LOGGER.info("Running experiment A (early warning).")
        output["experiment_a"] = _run_experiment_a(step_df, args, out_dir)
    if args.experiment in {"B", "ALL"}:
        LOGGER.info("Running experiment B (model comparison).")
        output["experiment_b"] = _run_experiment_b(step_df, args, out_dir)
    if args.experiment in {"C", "ALL"}:
        LOGGER.info("Running experiment C (k sensitivity).")
        output["experiment_c"] = _run_experiment_c(step_df, args, out_dir)

    summary_path = out_dir / "ablation_summary.json"
    summary_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Finished ablation run. Rows=%d", output["rows"])
    print(f"Wrote ablation outputs to {out_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
