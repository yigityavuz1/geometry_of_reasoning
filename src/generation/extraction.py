from __future__ import annotations

import re
from typing import Any, Iterable

import numpy as np
import torch

_STEP_START_PATTERN = re.compile(
    r"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?(?:step\s*\d+\s*:|\(\d+\)\s+|\d+[\)\.:\-]\s+)"
)
_FINAL_ANSWER_PATTERN = re.compile(r"(?im)^\s*final\s+answer\s*:")
_INLINE_MARKER_SPLIT_PATTERN = re.compile(
    r"(?i)(?<!^)(?<!\n)\s+(?=(?:[-*]\s*)?(?:\*\*)?(?:step\s*\d+\s*:|final\s+answer\s*:))"
)


def _strip_step_markup(text: str) -> str:
    cleaned = text.strip()
    while True:
        updated = _STEP_START_PATTERN.sub("", cleaned, count=1).strip()
        if updated == cleaned:
            break
        cleaned = updated
    cleaned = _FINAL_ANSWER_PATTERN.sub("", cleaned, count=1).strip()
    return cleaned


def _separate_inline_markers(text: str) -> str:
    return _INLINE_MARKER_SPLIT_PATTERN.sub("\n", text)


def _trim_empty_leading_headers(text: str) -> tuple[str, int]:
    candidate = text
    offset = 0
    while True:
        match = _STEP_START_PATTERN.match(candidate)
        if match is None:
            return candidate, offset
        remainder = candidate[match.end() :]
        stripped_remainder = remainder.lstrip()
        whitespace = len(remainder) - len(stripped_remainder)
        if not stripped_remainder:
            return "", len(text)
        if _STEP_START_PATTERN.match(stripped_remainder) or _FINAL_ANSWER_PATTERN.match(stripped_remainder):
            offset += match.end() + whitespace
            candidate = stripped_remainder
            continue
        return candidate, offset


def _truncate_before_final_answer(text: str) -> tuple[str, int]:
    matches = list(_FINAL_ANSWER_PATTERN.finditer(text))
    if not matches:
        return text, len(text)

    for match in matches:
        if _STEP_START_PATTERN.search(text, match.end()) is None:
            return text[: match.start()], match.start()
    return text, len(text)


def _fallback_line_spans(text: str) -> list[dict[str, int]]:
    spans: list[dict[str, int]] = []
    cursor = 0
    step_index = 1
    for raw_line in text.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        start = cursor
        end = cursor + len(line)
        cursor += len(raw_line)
        stripped = line.strip()
        if not stripped:
            continue
        if not _strip_step_markup(stripped):
            continue
        if "=" not in stripped and not re.search(r"\d", stripped):
            continue
        left_ws = len(line) - len(line.lstrip())
        spans.append(
            {
                "step_index": step_index,
                "start_char": start + left_ws,
                "end_char": end,
            }
        )
        step_index += 1
    return spans


def split_steps(generated_text: str) -> list[str]:
    normalized_text = _separate_inline_markers(generated_text)
    body, _ = _truncate_before_final_answer(normalized_text)
    matches = list(_STEP_START_PATTERN.finditer(body))
    if not matches:
        fallback = _fallback_line_spans(body)
        if fallback:
            return [body[int(span["start_char"]) : int(span["end_char"])].strip() for span in fallback]
        text = body.strip()
        return [text] if text else []

    steps: list[str] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        step_text, _ = _trim_empty_leading_headers(body[start:end].strip())
        if step_text and _strip_step_markup(step_text):
            steps.append(step_text)
    return steps


def find_step_boundaries(text: str) -> list[dict[str, int]]:
    """Return char ranges for `Step N:` blocks in generated text."""
    body, body_end = _truncate_before_final_answer(text)
    matches = list(_STEP_START_PATTERN.finditer(body))
    if not matches:
        fallback = _fallback_line_spans(body)
        if fallback:
            return fallback
        stripped = body.strip()
        if not stripped:
            return []
        start = len(body) - len(body.lstrip())
        return [{"step_index": 1, "start_char": start, "end_char": body_end}]

    spans: list[dict[str, int]] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else body_end
        step_text, offset = _trim_empty_leading_headers(body[start:end])
        if not _strip_step_markup(step_text):
            continue
        spans.append({"step_index": len(spans) + 1, "start_char": start + offset, "end_char": end})
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
