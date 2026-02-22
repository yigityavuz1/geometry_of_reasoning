from __future__ import annotations

import re
from dataclasses import dataclass

import sympy as sp


@dataclass
class StepJudgement:
    is_correct: bool
    parse_fail: bool
    reason: str


_EQ_PATTERN = re.compile(r"([^\n=]+)=([^\n=]+)")


def _safe_sympify(expr: str) -> sp.Expr:
    expr = expr.strip().replace("^", "**")
    return sp.sympify(expr)


def judge_step_equational_consistency(step_text: str) -> StepJudgement:
    """Judge whether equations inside one step are symbolically consistent.

    Note: this checks internal consistency, not full task correctness.
    """
    matches = _EQ_PATTERN.findall(step_text)
    if not matches:
        return StepJudgement(is_correct=False, parse_fail=True, reason="no_equation_found")

    try:
        for lhs, rhs in matches:
            lhs_expr = _safe_sympify(lhs)
            rhs_expr = _safe_sympify(rhs)
            if sp.simplify(lhs_expr - rhs_expr) != 0:
                return StepJudgement(
                    is_correct=False,
                    parse_fail=False,
                    reason="equation_mismatch",
                )
    except Exception:
        return StepJudgement(is_correct=False, parse_fail=True, reason="sympy_parse_error")

    return StepJudgement(is_correct=True, parse_fail=False, reason="ok")
