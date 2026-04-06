from __future__ import annotations

from src.generation.runner import (
    _build_step_prompt,
    _placeholder_retry_reason,
    _resolve_capture_layers,
)


def test_build_step_prompt_avoids_literal_placeholders() -> None:
    prompt = _build_step_prompt("What is 2 + 2?")
    assert "<single number>" not in prompt
    assert "<equation>" not in prompt
    assert "Final Answer: 8" in prompt


def test_placeholder_retry_reason_detects_template_copy() -> None:
    invalid = "Step 1: <equation>\nFinal Answer: <single number>"
    valid = "Step 1: 2 + 2 = 4\nFinal Answer: 4"
    assert _placeholder_retry_reason(invalid) == "placeholder_output"
    assert _placeholder_retry_reason(valid) is None


def test_resolve_capture_layers_uses_early_middle_late_snapshots() -> None:
    assert _resolve_capture_layers(28, ("early", "middle", "late")) == [
        ("early", 7),
        ("middle", 14),
        ("late", 28),
    ]


def test_resolve_capture_layers_accepts_numeric_indices() -> None:
    assert _resolve_capture_layers(12, ("early", "9", "late")) == [
        ("early", 3),
        ("layer_9", 9),
        ("late", 12),
    ]
