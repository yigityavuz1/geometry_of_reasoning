from __future__ import annotations

from src.evaluation.step_parser import extract_equation_pairs, extract_numeric_tokens
from src.generation.extraction import split_steps


def test_split_steps_drops_preamble_and_keeps_numbered_steps() -> None:
    text = (
        "We first inspect the question.\n"
        "Step 1: 2 + 3 = 5\n"
        "Step 2: 5 * 2 = 10"
    )
    steps = split_steps(text)
    assert len(steps) == 2
    assert steps[0].startswith("Step 1:")
    assert steps[1].startswith("Step 2:")


def test_extract_equation_pairs_supports_multiple_equalities() -> None:
    step = "Step 3: x = y = 5"
    pairs = extract_equation_pairs(step)
    assert pairs == [("x", "y"), ("y", "5")]


def test_extract_numeric_tokens_ignores_step_index() -> None:
    step = "Step 9: Janet has 16 eggs and uses 4."
    tokens = extract_numeric_tokens(step)
    assert tokens == ["16", "4"]
