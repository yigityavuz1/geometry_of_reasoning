from __future__ import annotations

import re
from typing import Any, Iterable

import numpy as np
import torch


def split_steps(generated_text: str) -> list[str]:
    step_pattern = re.compile(r"Step\s+\d+\s*:", flags=re.IGNORECASE)
    matches = list(step_pattern.finditer(generated_text))
    if not matches:
        text = generated_text.strip()
        return [text] if text else []

    steps: list[str] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(generated_text)
        step_text = generated_text[start:end].strip()
        if step_text:
            steps.append(step_text)
    return steps


def find_step_boundaries(text: str) -> list[dict[str, int]]:
    """Return char ranges for `Step N:` blocks in generated text."""
    step_pattern = re.compile(r"(Step\s+\d+\s*:)", re.IGNORECASE)
    matches = list(step_pattern.finditer(text))
    if not matches:
        return []

    spans: list[dict[str, int]] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        spans.append({"step_index": i + 1, "start_char": start, "end_char": end})
    return spans


def entropy_from_logits(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    probs = torch.softmax(logits, dim=dim)
    log_probs = torch.log(probs.clamp_min(1e-12))
    return -(probs * log_probs).sum(dim=dim)


def summarize_token_entropies(scores: Iterable[torch.Tensor]) -> dict[str, float]:
    entropies = [entropy_from_logits(step_logits).mean().item() for step_logits in scores]
    if not entropies:
        return {"mean_entropy": 0.0, "max_entropy": 0.0, "min_entropy": 0.0}
    arr = np.asarray(entropies, dtype=np.float64)
    return {
        "mean_entropy": float(arr.mean()),
        "max_entropy": float(arr.max()),
        "min_entropy": float(arr.min()),
    }


def estimate_step_token_spans(
    generated_text: str,
    step_boundaries: list[dict[str, int]],
    tokenizer: Any,
    n_completion_tokens: int,
) -> list[dict[str, int]]:
    """Approximate per-step token spans from char boundaries.

    This uses tokenizer lengths on each step substring and keeps monotonic
    token pointers. It is robust enough for step-level aggregation on generated
    traces without relying on exact char-token alignment.
    """
    if not step_boundaries or n_completion_tokens <= 0:
        return []

    first_step_start = max(0, int(step_boundaries[0]["start_char"]))
    prefix_text = generated_text[:first_step_start]
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False).get("input_ids", [])
    cursor = min(len(prefix_ids), n_completion_tokens)

    spans: list[dict[str, int]] = []
    for boundary in step_boundaries:
        step_index = int(boundary["step_index"])
        start_char = max(0, int(boundary["start_char"]))
        end_char = max(start_char, int(boundary["end_char"]))
        step_text = generated_text[start_char:end_char]
        step_ids = tokenizer(step_text, add_special_tokens=False).get("input_ids", [])
        span_len = max(0, len(step_ids))

        start_token = min(cursor, n_completion_tokens)
        end_token = min(start_token + span_len, n_completion_tokens)
        if end_token > start_token:
            spans.append(
                {
                    "step_index": step_index,
                    "start_token": int(start_token),
                    "end_token": int(end_token),
                }
            )
        cursor = end_token

    return spans
