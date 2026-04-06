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

from src.evaluation.step_parser import normalize_step_text


DEFAULT_REASON_BUCKETS = [
    "no_math_signal",
    "sympy_parse_error",
    "unsupported_symbolic_form",
    "boundary_detected_but_empty_math",
    "trace_signal_fail",
    "trace_format_fail",
    "empty_completion",
    "equation_mismatch",
    "non_equational_numeric_reasoning",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a parser failure bank from a step-signal table.")
    parser.add_argument("--input", required=True, help="Input step table (.csv, .jsonl, or .parquet)")
    parser.add_argument("--out", default="data/debug/parser_failure_bank.csv", help="Output CSV path")
    parser.add_argument(
        "--summary-out",
        default="data/debug/parser_failure_bank_summary.json",
        help="Output JSON summary path",
    )
    parser.add_argument("--primary-k", type=int, default=20)
    parser.add_argument("--max-rows", type=int, default=150)
    parser.add_argument("--per-bucket", type=int, default=18)
    parser.add_argument(
        "--models",
        default="",
        help="Optional comma-separated model filter",
    )
    parser.add_argument(
        "--reasons",
        default=",".join(DEFAULT_REASON_BUCKETS),
        help="Comma-separated reason codes to include",
    )
    return parser.parse_args()


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return pd.DataFrame(rows)
    raise ValueError("Unsupported input format. Use .csv, .jsonl, or .parquet.")


def _csv_list(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def _expected_parseable_from_reason(reason: str) -> str:
    if reason in {"sympy_parse_error", "unsupported_symbolic_form"}:
        return "true"
    if reason in {"no_math_signal", "boundary_detected_but_empty_math", "trace_signal_fail", "trace_format_fail", "empty_completion"}:
        return "false"
    return ""


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    df = _read_table(input_path)
    if "k_requested" in df.columns:
        df = df.loc[pd.to_numeric(df["k_requested"], errors="coerce").fillna(-1).astype(int) == int(args.primary_k)].copy()

    reason_filter = set(_csv_list(args.reasons))
    if reason_filter:
        df = df.loc[df["reason"].astype(str).isin(reason_filter)].copy()

    models = _csv_list(args.models)
    if models:
        df = df.loc[df["model_name"].astype(str).isin(models)].copy()

    if df.empty:
        raise RuntimeError("No rows matched the requested filters.")

    if "step_text" not in df.columns:
        df["step_text"] = ""
    if "normalized_step_text" not in df.columns:
        df["normalized_step_text"] = df["step_text"].astype(str).map(normalize_step_text)
    df["step_text"] = df["step_text"].fillna("").astype(str)
    df["normalized_step_text"] = df["normalized_step_text"].fillna("").astype(str)
    if "question_text" not in df.columns:
        df["question_text"] = ""
    df["question_text"] = df["question_text"].fillna("").astype(str)
    df["model_name"] = df["model_name"].fillna("").astype(str)
    df["reason"] = df["reason"].fillna("").astype(str)
    df["dataset_index"] = pd.to_numeric(df["dataset_index"], errors="coerce").fillna(-1).astype(int)
    df["step_index"] = pd.to_numeric(df["step_index"], errors="coerce").fillna(-1).astype(int)
    df["source_has_step_text"] = df["step_text"].str.len() > 0

    selected_rows: list[pd.DataFrame] = []
    for (model_name, reason), group in df.groupby(["model_name", "reason"], sort=True):
        candidate = (
            group.sort_values(["dataset_index", "step_index"], kind="mergesort")
            .drop_duplicates(subset=["dataset_index", "step_index"], keep="first")
            .head(int(args.per_bucket))
            .copy()
        )
        candidate["expected_reason"] = ""
        candidate["expected_parseable"] = candidate["reason"].map(_expected_parseable_from_reason)
        candidate["review_status"] = "todo"
        candidate["notes"] = ""
        if not bool(candidate["source_has_step_text"].all()):
            candidate.loc[:, "notes"] = (
                "Legacy source table does not contain raw step text. "
                "Backfill by re-running ablation with the current schema."
            )
        selected_rows.append(candidate)

    bank = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    if bank.empty:
        raise RuntimeError("Could not assemble any parser failure rows.")

    bank = bank.head(int(args.max_rows)).copy()
    bank_out = bank[
        [
            "model_name",
            "dataset_index",
            "step_index",
            "question_text",
            "step_text",
            "normalized_step_text",
            "reason",
            "expected_reason",
            "expected_parseable",
            "review_status",
            "source_has_step_text",
            "notes",
        ]
    ].rename(columns={"reason": "current_reason", "step_text": "raw_step_text"})

    out_path = Path(args.out)
    summary_out = Path(args.summary_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    bank_out.to_csv(out_path, index=False)

    summary = {
        "input_path": str(input_path),
        "output_path": str(out_path),
        "rows": int(len(bank_out)),
        "primary_k": int(args.primary_k),
        "reason_counts": bank_out["current_reason"].value_counts(dropna=False).to_dict(),
        "model_counts": bank_out["model_name"].value_counts(dropna=False).to_dict(),
        "rows_with_raw_step_text": int(bank_out["source_has_step_text"].astype(bool).sum()),
    }
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote parser failure bank: {out_path}")
    print(f"Wrote summary:            {summary_out}")


if __name__ == "__main__":
    main()
