from __future__ import annotations

from src.evaluation.sympy_judge import judge_step_equational_consistency


def test_judge_accepts_correct_equation() -> None:
    step = "Step 1: 2 + 3 = 5"
    result = judge_step_equational_consistency(step)
    assert result.is_correct is True
    assert result.parse_fail is False


def test_judge_rejects_incorrect_equation() -> None:
    step = "Step 2: 7 * 8 = 54"
    result = judge_step_equational_consistency(step)
    assert result.is_correct is False
    assert result.parse_fail is False


def test_judge_flags_parse_failure_when_no_equation() -> None:
    step = "Step 3: We now conclude the argument."
    result = judge_step_equational_consistency(step)
    assert result.is_correct is False
    assert result.parse_fail is True
