from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import torch


def split_steps(generated_text: str) -> list[str]:
    chunks = re.split(r"(?=Step\s+\d+\s*:)", generated_text, flags=re.IGNORECASE)
    return [c.strip() for c in chunks if c.strip()]


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
