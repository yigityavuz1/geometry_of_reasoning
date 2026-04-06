from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.sympy_judge import (
    build_task_reference,
    judge_step_equational_consistency,
    judge_step_task_correctness,
    summarize_judgement_records,
)
from src.generation.extraction import split_steps
from src.generation.runner import GenerationConfig, generate_reasoning_trace
from src.metrics.global_dim import participation_ratio
from src.metrics.lid_estimators import abid_local_batch, lid_mle_batch, twonn_global_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end smoke pipeline on one GSM8K sample: generation -> judge -> metrics."
    )
    parser.add_argument("--model", required=True, help="HF model id for generation")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--index", type=int, default=0, help="Sample index in selected split")
    parser.add_argument("--out-dir", default="results/smoke", help="Output directory")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--k", type=int, default=10, help="Requested k for local estimators")
    return parser.parse_args()


def build_step_judgements(
    generated_text: str,
    question: str,
    gold_answer: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    steps = split_steps(generated_text)
    reference = build_task_reference(question, gold_answer) if question and gold_answer else None
    judged_steps: list[dict[str, object]] = []
    for idx, step_text in enumerate(steps, start=1):
        if reference is not None:
            j = judge_step_task_correctness(step_text, reference)
        else:
            j = judge_step_equational_consistency(step_text)
        judged_steps.append(
            {
                "step_index": idx,
                "text": step_text,
                "is_correct": j.is_correct,
                "parse_fail": j.parse_fail,
                "reason": j.reason,
                "matched_values": j.matched_values or [],
            }
        )
    return judged_steps, summarize_judgement_records(judged_steps)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("openai/gsm8k", "main", split=args.split)
    sample = ds[int(args.index)]
    question = str(sample["question"])
    gold_answer = str(sample["answer"])

    cfg = GenerationConfig(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        collect_token_embeddings=True,
    )
    trace = generate_reasoning_trace(question, cfg)
    token_embeddings = np.asarray(trace.pop("token_embeddings", []), dtype=np.float32)
    token_embeddings_by_layer = trace.pop("token_embeddings_by_layer", {})

    trace["dataset"] = {"name": "openai/gsm8k", "config": "main", "split": args.split, "index": args.index}
    trace["gold_answer"] = gold_answer

    generation_path = out_dir / "generation_trace.json"
    generation_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    model_metadata = trace.get("model_metadata")
    if isinstance(model_metadata, dict):
        (out_dir / "model_metadata.json").write_text(
            json.dumps(model_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    judged_steps, judge_summary = build_step_judgements(
        trace.get("generated_text", ""),
        question,
        gold_answer,
    )
    judged_trace = dict(trace)
    judged_trace["judged_steps"] = judged_steps
    judged_trace["judge_summary"] = judge_summary
    judged_path = out_dir / "judged_trace.json"
    judged_path.write_text(json.dumps(judged_trace, ensure_ascii=False, indent=2), encoding="utf-8")

    embeddings_path = out_dir / "token_embeddings.npy"
    np.save(embeddings_path, token_embeddings)
    layer_metrics: dict[str, dict[str, object]] = {}
    if isinstance(token_embeddings_by_layer, dict):
        for layer_name, payload in token_embeddings_by_layer.items():
            layer_embeddings = np.asarray(payload.get("embeddings", []), dtype=np.float32)
            if layer_embeddings.ndim != 2:
                continue
            layer_path = out_dir / f"token_embeddings_{layer_name}.npy"
            np.save(layer_path, layer_embeddings)
            layer_summary: dict[str, object] = {
                "layer_index": int(payload.get("layer_index", -1)),
                "n_samples": int(layer_embeddings.shape[0]),
                "n_features": int(layer_embeddings.shape[1]) if layer_embeddings.ndim == 2 else 0,
            }
            if layer_embeddings.shape[0] >= 3:
                k_eff = max(2, min(args.k, int(layer_embeddings.shape[0]) - 1))
                layer_summary.update(
                    {
                        "k_effective": int(k_eff),
                        "lid_mle_mean": float(np.mean(lid_mle_batch(layer_embeddings, k=k_eff))),
                        "twonn_global_id": float(twonn_global_id(layer_embeddings)),
                        "abid_mean": float(np.mean(abid_local_batch(layer_embeddings, k=k_eff))),
                        "participation_ratio": float(participation_ratio(layer_embeddings)),
                    }
                )
            else:
                layer_summary["status"] = "insufficient_samples_for_lid"
            layer_metrics[str(layer_name)] = layer_summary

    metrics: dict[str, object] = {
        "n_samples": int(token_embeddings.shape[0]) if token_embeddings.ndim == 2 else 0,
        "n_features": int(token_embeddings.shape[1]) if token_embeddings.ndim == 2 else 0,
        "k_requested": int(args.k),
        "judge_parse_fail_rate": judge_summary["parse_fail_rate"],
        "judge_correct_rate": judge_summary["correct_rate"],
        "metrics_by_layer": layer_metrics,
    }
    if token_embeddings.ndim == 2 and token_embeddings.shape[0] >= 3:
        k_eff = max(2, min(args.k, int(token_embeddings.shape[0]) - 1))
        metrics.update(
            {
                "k_effective": k_eff,
                "lid_mle_mean": float(np.mean(lid_mle_batch(token_embeddings, k=k_eff))),
                "twonn_global_id": float(twonn_global_id(token_embeddings)),
                "abid_mean": float(np.mean(abid_local_batch(token_embeddings, k=k_eff))),
                "participation_ratio": float(participation_ratio(token_embeddings)),
            }
        )
    else:
        metrics["status"] = "insufficient_samples_for_lid"

    metrics_path = out_dir / "metrics_summary.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote: {generation_path}")
    print(f"Wrote: {judged_path}")
    print(f"Wrote: {embeddings_path}")
    print(f"Wrote: {metrics_path}")


if __name__ == "__main__":
    main()
