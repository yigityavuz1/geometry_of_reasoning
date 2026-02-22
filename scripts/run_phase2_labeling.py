from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.sympy_judge import (
    build_task_reference,
    judge_step_task_correctness,
    summarize_judgement_records,
)
from src.generation.extraction import split_steps
from src.generation.runner import GenerationConfig, generate_reasoning_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build phase-2 step-level labels on a GSM8K subset: "
            "generation -> task-correctness judge -> processed table + parse-fail report."
        )
    )
    parser.add_argument("--model", required=True, help="HF model id used for generation")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument(
        "--out-table-jsonl",
        default="data/processed/gsm8k_step_labels.jsonl",
        help="Output JSONL table path",
    )
    parser.add_argument(
        "--out-table-csv",
        default="data/processed/gsm8k_step_labels.csv",
        help="Output CSV table path",
    )
    parser.add_argument(
        "--out-summary",
        default="data/processed/gsm8k_step_labels_summary.json",
        help="Output summary JSON path",
    )
    return parser.parse_args()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = [
        "model_name",
        "dataset_split",
        "dataset_index",
        "step_index",
        "is_correct",
        "parse_fail",
        "reason",
        "matched_values",
        "step_text",
        "question",
        "gold_answer",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["matched_values"] = "|".join(row.get("matched_values", []))
            writer.writerow(csv_row)


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be >= 1")

    dataset = load_dataset("openai/gsm8k", "main", split=args.split)
    max_index = args.start_index + args.num_samples
    if max_index > len(dataset):
        raise ValueError(
            f"Requested range [{args.start_index}, {max_index}) exceeds split size {len(dataset)}."
        )

    generation_cfg = GenerationConfig(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        collect_token_embeddings=False,
    )

    rows: list[dict[str, object]] = []
    per_sample_summaries: list[dict[str, object]] = []

    for dataset_index in range(args.start_index, max_index):
        sample = dataset[int(dataset_index)]
        question = str(sample["question"])
        gold_answer = str(sample["answer"])
        trace = generate_reasoning_trace(question, generation_cfg)
        reference = build_task_reference(question, gold_answer)
        steps = split_steps(trace.get("generated_text", ""))

        sample_rows: list[dict[str, object]] = []
        for step_index, step_text in enumerate(steps, start=1):
            judgement = judge_step_task_correctness(step_text, reference)
            row = {
                "model_name": args.model,
                "dataset_split": args.split,
                "dataset_index": int(dataset_index),
                "step_index": step_index,
                "is_correct": judgement.is_correct,
                "parse_fail": judgement.parse_fail,
                "reason": judgement.reason,
                "matched_values": judgement.matched_values or [],
                "step_text": step_text,
                "question": question,
                "gold_answer": gold_answer,
            }
            rows.append(row)
            sample_rows.append(row)

        sample_summary = summarize_judgement_records(sample_rows)
        sample_summary["dataset_index"] = int(dataset_index)
        per_sample_summaries.append(sample_summary)

    summary = summarize_judgement_records(rows)
    summary.update(
        {
            "model_name": args.model,
            "dataset_name": "openai/gsm8k",
            "dataset_config": "main",
            "dataset_split": args.split,
            "start_index": int(args.start_index),
            "num_samples": int(args.num_samples),
            "sample_summaries": per_sample_summaries,
        }
    )

    out_table_jsonl = Path(args.out_table_jsonl)
    out_table_csv = Path(args.out_table_csv)
    out_summary = Path(args.out_summary)
    _write_jsonl(out_table_jsonl, rows)
    _write_csv(out_table_csv, rows)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote table JSONL: {out_table_jsonl}")
    print(f"Wrote table CSV:   {out_table_csv}")
    print(f"Wrote summary:     {out_summary}")


if __name__ == "__main__":
    main()
