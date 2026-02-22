from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import sympy as sp

from src.evaluation.step_parser import (
    extract_equation_pairs,
    extract_gsm8k_final_answer_text,
    extract_gsm8k_inline_equations,
    extract_numeric_tokens,
    normalize_math_text,
)

@dataclass
class StepJudgement:
    is_correct: bool
    parse_fail: bool
    reason: str
    matched_values: list[str] | None = None


@dataclass(frozen=True)
class TaskReference:
    question_values: tuple[sp.Expr, ...]
    target_values: tuple[sp.Expr, ...]
    final_answer: sp.Expr | None = None


def _safe_sympify(expr: str) -> sp.Expr:
    cleaned = normalize_math_text(expr)
    if not cleaned:
        raise ValueError("Empty expression after normalization.")
    return sp.sympify(cleaned, evaluate=True)


def _expr_equal(lhs: sp.Expr, rhs: sp.Expr) -> bool:
    try:
        return bool(sp.simplify(lhs - rhs) == 0)
    except Exception:
        try:
            equals_value = lhs.equals(rhs)
            return bool(equals_value)
        except Exception:
            return False


def _parse_numeric_tokens(tokens: Sequence[str]) -> list[sp.Expr]:
    parsed: list[sp.Expr] = []
    for token in tokens:
        try:
            parsed.append(_safe_sympify(token))
        except Exception:
            continue
    return parsed


def _expr_key(value: sp.Expr) -> str:
    try:
        return sp.sstr(sp.nsimplify(value))
    except Exception:
        return sp.sstr(value)


def _unique_exprs(values: Iterable[sp.Expr]) -> list[sp.Expr]:
    dedup: dict[str, sp.Expr] = {}
    for value in values:
        dedup[_expr_key(value)] = value
    return list(dedup.values())


def _find_matches(candidates: Sequence[sp.Expr], references: Sequence[sp.Expr]) -> list[sp.Expr]:
    matches: list[sp.Expr] = []
    for candidate in candidates:
        if any(_expr_equal(candidate, ref) for ref in references):
            matches.append(candidate)
    return _unique_exprs(matches)


def build_task_reference(question_text: str, gold_answer: str) -> TaskReference:
    question_values = _parse_numeric_tokens(extract_numeric_tokens(question_text))
    target_values = _parse_numeric_tokens(extract_numeric_tokens(gold_answer))

    for inline_eq in extract_gsm8k_inline_equations(gold_answer):
        for lhs_text, rhs_text in extract_equation_pairs(inline_eq):
            try:
                lhs_expr = _safe_sympify(lhs_text)
                rhs_expr = _safe_sympify(rhs_text)
                target_values.extend([lhs_expr, rhs_expr])
            except Exception:
                continue

    final_answer_expr: sp.Expr | None = None
    final_text = extract_gsm8k_final_answer_text(gold_answer)
    if final_text:
        final_candidates = _parse_numeric_tokens(extract_numeric_tokens(final_text))
        if final_candidates:
            final_answer_expr = final_candidates[-1]
            target_values.append(final_answer_expr)
        else:
            try:
                final_answer_expr = _safe_sympify(final_text)
                target_values.append(final_answer_expr)
            except Exception:
                final_answer_expr = None

    return TaskReference(
        question_values=tuple(_unique_exprs(question_values)),
        target_values=tuple(_unique_exprs(target_values)),
        final_answer=final_answer_expr,
    )


def judge_step_equational_consistency(step_text: str) -> StepJudgement:
    """Judge whether equations inside one step are symbolically consistent.

    Note: this checks internal consistency, not full task correctness.
    """
    equation_pairs = extract_equation_pairs(step_text)
    if not equation_pairs:
        return StepJudgement(is_correct=False, parse_fail=True, reason="no_equation_found")

    parsed_equations = 0
    try:
        for lhs, rhs in equation_pairs:
            lhs_expr = _safe_sympify(lhs)
            rhs_expr = _safe_sympify(rhs)
            parsed_equations += 1
            if not _expr_equal(lhs_expr, rhs_expr):
                return StepJudgement(
                    is_correct=False,
                    parse_fail=False,
                    reason="equation_mismatch",
                )
    except Exception:
        return StepJudgement(is_correct=False, parse_fail=True, reason="sympy_parse_error")

    if parsed_equations == 0:
        return StepJudgement(is_correct=False, parse_fail=True, reason="sympy_parse_error")

    return StepJudgement(is_correct=True, parse_fail=False, reason="ok")


def judge_step_task_correctness(step_text: str, reference: TaskReference) -> StepJudgement:
    """Judge one step against GSM8K task references with parser-aware heuristics."""
    equation_pairs = extract_equation_pairs(step_text)
    text_values = _parse_numeric_tokens(extract_numeric_tokens(step_text))
    equation_values: list[sp.Expr] = []

    parsed_equations = 0
    for lhs_text, rhs_text in equation_pairs:
        try:
            lhs_expr = _safe_sympify(lhs_text)
            rhs_expr = _safe_sympify(rhs_text)
        except Exception:
            continue

        parsed_equations += 1
        if not _expr_equal(lhs_expr, rhs_expr):
            return StepJudgement(is_correct=False, parse_fail=False, reason="equation_mismatch")
        equation_values.extend([lhs_expr, rhs_expr])

    if not equation_pairs and not text_values:
        return StepJudgement(is_correct=False, parse_fail=True, reason="no_math_signal")

    equation_target_matches = _find_matches(equation_values, reference.target_values)
    text_target_matches = _find_matches(text_values, reference.target_values)
    text_question_matches = _find_matches(text_values, reference.question_values)
    matched_values = [
        _expr_key(value)
        for value in _unique_exprs(equation_target_matches + text_target_matches + text_question_matches)
    ]

    if equation_pairs and parsed_equations == 0:
        if text_target_matches:
            return StepJudgement(
                is_correct=True,
                parse_fail=False,
                reason="text_matches_reference_value",
                matched_values=matched_values,
            )
        if text_question_matches:
            return StepJudgement(
                is_correct=True,
                parse_fail=False,
                reason="text_matches_question_value",
                matched_values=matched_values,
            )
        return StepJudgement(is_correct=False, parse_fail=True, reason="sympy_parse_error")

    if equation_pairs:
        if equation_target_matches:
            return StepJudgement(
                is_correct=True,
                parse_fail=False,
                reason="equation_matches_reference",
                matched_values=matched_values,
            )
        if parsed_equations > 0 and (text_target_matches or text_question_matches):
            return StepJudgement(
                is_correct=True,
                parse_fail=False,
                reason="equation_consistent_supported",
                matched_values=matched_values,
            )
        return StepJudgement(is_correct=False, parse_fail=False, reason="equation_consistent_unanchored")

    if text_target_matches:
        return StepJudgement(
            is_correct=True,
            parse_fail=False,
            reason="text_matches_reference_value",
            matched_values=matched_values,
        )
    if text_question_matches:
        return StepJudgement(
            is_correct=True,
            parse_fail=False,
            reason="text_matches_question_value",
            matched_values=matched_values,
        )
    return StepJudgement(is_correct=False, parse_fail=False, reason="numeric_off_reference")


def summarize_judgement_records(records: list[dict[str, object]]) -> dict[str, object]:
    total_steps = len(records)
    if total_steps == 0:
        return {
            "total_steps": 0,
            "parse_fail_count": 0,
            "parse_fail_rate": 0.0,
            "correct_count": 0,
            "correct_rate": 0.0,
            "correct_rate_non_parse_fail": 0.0,
            "reason_counts": {},
        }

    parse_fail_count = sum(bool(row.get("parse_fail", False)) for row in records)
    correct_count = sum(bool(row.get("is_correct", False)) for row in records)
    non_parse_fail = max(1, total_steps - parse_fail_count)

    reason_counts: dict[str, int] = {}
    for row in records:
        reason = str(row.get("reason", "unknown"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    return {
        "total_steps": total_steps,
        "parse_fail_count": parse_fail_count,
        "parse_fail_rate": float(parse_fail_count / total_steps),
        "correct_count": correct_count,
        "correct_rate": float(correct_count / total_steps),
        "correct_rate_non_parse_fail": float(correct_count / non_parse_fail),
        "reason_counts": reason_counts,
    }
