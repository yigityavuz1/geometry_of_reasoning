from __future__ import annotations

from src.evaluation.sympy_judge import (
    build_task_reference,
    judge_step_equational_consistency,
    judge_step_task_correctness,
)


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


def test_task_judge_accepts_textual_step_with_supported_number() -> None:
    question = "Janet's ducks lay 16 eggs per day and she eats 3."
    gold_answer = "She sells 16 - 3 = <<16-3=13>>13 eggs.\n#### 13"
    reference = build_task_reference(question, gold_answer)
    step = "Step 1: Janet's ducks lay 16 eggs per day."
    result = judge_step_task_correctness(step, reference)
    assert result.is_correct is True
    assert result.parse_fail is False


def test_task_judge_rejects_wrong_equation() -> None:
    question = "Janet's ducks lay 16 eggs per day and she eats 3."
    gold_answer = "She sells 16 - 3 = <<16-3=13>>13 eggs.\n#### 13"
    reference = build_task_reference(question, gold_answer)
    step = "Step 2: 16 - 3 = 99"
    result = judge_step_task_correctness(step, reference)
    assert result.is_correct is False
    assert result.parse_fail is False
    assert result.reason == "equation_mismatch"
