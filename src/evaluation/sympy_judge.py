from __future__ import annotations

from dataclasses import dataclass
import re
import string
from typing import Iterable, Sequence

import sympy as sp
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from src.evaluation.step_parser import (
    extract_equation_pairs,
    extract_gsm8k_final_answer_text,
    extract_gsm8k_inline_equations,
    extract_numeric_tokens,
    normalize_math_text,
    normalize_step_text,
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


_UNSUPPORTED_SYMBOL_PATTERN = re.compile(r"[<>]|[A-Za-z]{2,}")
_SYMPY_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)
_SYMPY_LOCAL_DICT = {letter: sp.Symbol(letter) for letter in string.ascii_letters}


def _safe_sympify(expr: str) -> sp.Expr:
    cleaned = normalize_math_text(expr, convert_units=True)
    if not cleaned:
        raise ValueError("Empty expression after normalization.")
    return parse_expr(
        cleaned,
        transformations=_SYMPY_TRANSFORMATIONS,
        local_dict=_SYMPY_LOCAL_DICT,
        evaluate=True,
    )


def _expr_equal(lhs: sp.Expr, rhs: sp.Expr) -> bool:
    try:
        return bool(sp.simplify(lhs - rhs) == 0)
    except Exception:
        try:
            equals_value = lhs.equals(rhs)
            return bool(equals_value)
        except Exception:
            return False


def _expr_close(lhs: sp.Expr, rhs: sp.Expr) -> bool:
    if _is_symbolic_relation(lhs, rhs):
        return False
    try:
        lhs_value = complex(sp.N(lhs))
        rhs_value = complex(sp.N(rhs))
    except Exception:
        return False
    if abs(lhs_value.imag) > 1e-9 or abs(rhs_value.imag) > 1e-9:
        return False
    scale = max(1.0, abs(lhs_value.real), abs(rhs_value.real))
    tolerance = max(1e-3, 5e-3 * scale)
    return abs(lhs_value.real - rhs_value.real) <= tolerance


def _is_symbolic_relation(lhs: sp.Expr, rhs: sp.Expr) -> bool:
    return bool(getattr(lhs, "free_symbols", set()) or getattr(rhs, "free_symbols", set()))


def _expr_percent_display_match(lhs: sp.Expr, rhs: sp.Expr, raw_text: str | None = None) -> bool:
    if not raw_text or "%" not in raw_text or _is_symbolic_relation(lhs, rhs):
        return False
    try:
        lhs_value = complex(sp.N(lhs))
        rhs_value = complex(sp.N(rhs))
    except Exception:
        return False
    if abs(lhs_value.imag) > 1e-9 or abs(rhs_value.imag) > 1e-9:
        return False
    scale = max(
        1.0,
        abs(lhs_value.real),
        abs(rhs_value.real),
        abs(lhs_value.real * 100.0),
        abs(rhs_value.real * 100.0),
    )
    tolerance = max(1e-3, 5e-3 * scale)
    return (
        abs(lhs_value.real - rhs_value.real * 100.0) <= tolerance
        or abs(lhs_value.real * 100.0 - rhs_value.real) <= tolerance
    )


def _expr_matches(lhs: sp.Expr, rhs: sp.Expr, raw_text: str | None = None) -> bool:
    return _expr_equal(lhs, rhs) or _expr_close(lhs, rhs) or _expr_percent_display_match(lhs, rhs, raw_text)


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


def _expr_numeric_atoms(expr: sp.Expr) -> list[sp.Expr]:
    numeric_atoms = list(getattr(expr, "atoms", lambda *_: set())(sp.Number))
    return _unique_exprs(numeric_atoms)


def _unique_exprs(values: Iterable[sp.Expr]) -> list[sp.Expr]:
    dedup: dict[str, sp.Expr] = {}
    for value in values:
        dedup[_expr_key(value)] = value
    return list(dedup.values())


def _find_matches(candidates: Sequence[sp.Expr], references: Sequence[sp.Expr]) -> list[sp.Expr]:
    matches: list[sp.Expr] = []
    for candidate in candidates:
        if any(_expr_matches(candidate, ref) for ref in references):
            matches.append(candidate)
    return _unique_exprs(matches)


def _parse_failure_reason(body: str) -> str:
    if not body.strip():
        return "boundary_detected_but_empty_math"
    if "=" in body and _UNSUPPORTED_SYMBOL_PATTERN.search(body):
        return "unsupported_symbolic_form"
    if "=" in body:
        return "sympy_parse_error"
    return "no_math_signal"


def _is_simple_symbol(expr: sp.Expr) -> bool:
    return bool(getattr(expr, "is_Symbol", False))


def _assignment_value_expr(lhs: sp.Expr, rhs: sp.Expr) -> sp.Expr | None:
    """Return the bound value/expression for simple symbolic assignments.

    Examples:
    - ``x = 5`` -> ``5``
    - ``30 = x`` -> ``30``
    - ``C = 4S`` -> ``4*S``

    We only treat these as assignments when the bound symbol does not appear
    inside the value expression, so ``x = x + 1`` is still judged as a mismatch.
    """
    if _is_simple_symbol(lhs) and lhs not in getattr(rhs, "free_symbols", set()):
        return rhs
    if _is_simple_symbol(rhs) and rhs not in getattr(lhs, "free_symbols", set()):
        return lhs
    return None


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
    body = normalize_step_text(step_text)
    equation_pairs = extract_equation_pairs(step_text)
    if not body:
        return StepJudgement(is_correct=False, parse_fail=True, reason="boundary_detected_but_empty_math")
    if not equation_pairs:
        return StepJudgement(is_correct=False, parse_fail=True, reason=_parse_failure_reason(body))

    parsed_equations = 0
    try:
        for lhs, rhs in equation_pairs:
            lhs_expr = _safe_sympify(lhs)
            rhs_expr = _safe_sympify(rhs)
            parsed_equations += 1
            if _assignment_value_expr(lhs_expr, rhs_expr) is not None:
                continue
            if _expr_matches(lhs_expr, rhs_expr, raw_text=step_text):
                continue
            if _is_symbolic_relation(lhs_expr, rhs_expr):
                continue
            if not _expr_matches(lhs_expr, rhs_expr, raw_text=step_text):
                return StepJudgement(
                    is_correct=False,
                    parse_fail=False,
                    reason="equation_mismatch",
                )
    except Exception:
        return StepJudgement(is_correct=False, parse_fail=True, reason=_parse_failure_reason(body))

    if parsed_equations == 0:
        return StepJudgement(is_correct=False, parse_fail=True, reason=_parse_failure_reason(body))

    return StepJudgement(is_correct=True, parse_fail=False, reason="ok")


def judge_step_task_correctness(step_text: str, reference: TaskReference) -> StepJudgement:
    """Judge one step against GSM8K task references with parser-aware heuristics."""
    body = normalize_step_text(step_text)
    if not body:
        return StepJudgement(is_correct=False, parse_fail=True, reason="boundary_detected_but_empty_math")

    has_explicit_equation = "=" in body
    equation_pairs = extract_equation_pairs(step_text)
    text_values = _parse_numeric_tokens(extract_numeric_tokens(step_text))
    equation_values: list[sp.Expr] = []
    support_values: list[sp.Expr] = list(text_values)

    parsed_equations = 0
    parse_errors = 0
    for lhs_text, rhs_text in equation_pairs:
        try:
            lhs_expr = _safe_sympify(lhs_text)
            rhs_expr = _safe_sympify(rhs_text)
        except Exception:
            parse_errors += 1
            continue

        parsed_equations += 1
        assignment_value = _assignment_value_expr(lhs_expr, rhs_expr)
        if assignment_value is not None:
            equation_values.append(assignment_value)
            support_values.extend(_expr_numeric_atoms(assignment_value))
            continue
        support_values.extend(_expr_numeric_atoms(lhs_expr))
        support_values.extend(_expr_numeric_atoms(rhs_expr))
        if _expr_matches(lhs_expr, rhs_expr, raw_text=step_text):
            equation_values.extend([lhs_expr, rhs_expr])
            continue
        if _is_symbolic_relation(lhs_expr, rhs_expr):
            equation_values.extend([lhs_expr, rhs_expr])
            continue
        if not _expr_matches(lhs_expr, rhs_expr, raw_text=step_text):
            return StepJudgement(is_correct=False, parse_fail=False, reason="equation_mismatch")

    if not equation_pairs and not text_values:
        return StepJudgement(is_correct=False, parse_fail=True, reason=_parse_failure_reason(body))

    equation_target_matches = _find_matches(equation_values, reference.target_values)
    support_target_matches = _find_matches(support_values, reference.target_values)
    support_question_matches = _find_matches(support_values, reference.question_values)
    matched_values = [
        _expr_key(value)
        for value in _unique_exprs(equation_target_matches + support_target_matches + support_question_matches)
    ]

    if has_explicit_equation and parsed_equations == 0:
        rhs_text = body.rsplit("=", 1)[-1].strip()
        rhs_looks_math = bool(rhs_text) and not _UNSUPPORTED_SYMBOL_PATTERN.search(rhs_text)
        if support_target_matches and rhs_looks_math:
            return StepJudgement(
                is_correct=True,
                parse_fail=False,
                reason="equation_consistent_supported",
                matched_values=matched_values,
            )
        return StepJudgement(
            is_correct=False,
            parse_fail=True,
            reason=_parse_failure_reason(body if parse_errors > 0 else body),
            matched_values=matched_values,
        )

    if equation_pairs:
        if equation_target_matches:
            return StepJudgement(
                is_correct=True,
                parse_fail=False,
                reason="equation_matches_reference",
                matched_values=matched_values,
            )
        if parsed_equations > 0 and (support_target_matches or support_question_matches):
            return StepJudgement(
                is_correct=True,
                parse_fail=False,
                reason="equation_consistent_supported",
                matched_values=matched_values,
            )
        return StepJudgement(
            is_correct=False,
            parse_fail=False,
            reason="equation_consistent_unanchored",
            matched_values=matched_values,
        )

    if support_target_matches or support_question_matches:
        return StepJudgement(
            is_correct=True,
            parse_fail=False,
            reason="non_equational_numeric_reasoning",
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
