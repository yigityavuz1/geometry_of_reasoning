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


def test_judge_accepts_symbolic_assignment_chain() -> None:
    step = "Step 5: C = 4 * 20 = 80"
    result = judge_step_equational_consistency(step)
    assert result.is_correct is True
    assert result.parse_fail is False


def test_judge_accepts_symbolic_equation_relation() -> None:
    step = "Step 2: 0.75x = 19.50"
    result = judge_step_equational_consistency(step)
    assert result.is_correct is True
    assert result.parse_fail is False


def test_judge_flags_parse_failure_when_no_equation() -> None:
    step = "Step 3: We now conclude the argument."
    result = judge_step_equational_consistency(step)
    assert result.is_correct is False
    assert result.parse_fail is True
    assert result.reason == "no_math_signal"


def test_judge_accepts_rounded_decimal_equation() -> None:
    step = "Step 1: 10 * (2/3) = 6.666..."
    result = judge_step_equational_consistency(step)
    assert result.is_correct is True
    assert result.parse_fail is False


def test_judge_accepts_percentage_display_equation() -> None:
    step = "Step 4: 0.3333 * 100 = 33.33%"
    result = judge_step_equational_consistency(step)
    assert result.is_correct is True
    assert result.parse_fail is False


def test_judge_accepts_percent_of_chain() -> None:
    step = "Step 2: 25% of (20 - 4) = 25% of 16 = 4"
    result = judge_step_equational_consistency(step)
    assert result.is_correct is True
    assert result.parse_fail is False


def test_judge_accepts_imperial_length_conversion_chain() -> None:
    step = "Step 2: 4 feet - 6 inches = 3 feet 6 inches = 42 inches"
    result = judge_step_equational_consistency(step)
    assert result.is_correct is True
    assert result.parse_fail is False


def test_judge_accepts_descriptive_lhs_label_before_numeric_chain() -> None:
    step = "Step 2: Total cost of lemons = $3 * 20 = $60"
    result = judge_step_equational_consistency(step)
    assert result.is_correct is True
    assert result.parse_fail is False


def test_judge_keeps_equation_before_self_correction_marker() -> None:
    step = "Step 4: 20 - 28 = -9. Wait, that can't be right."
    result = judge_step_equational_consistency(step)
    assert result.is_correct is False
    assert result.parse_fail is False
    assert result.reason == "equation_mismatch"


def test_task_judge_accepts_textual_step_with_supported_number() -> None:
    question = "Janet's ducks lay 16 eggs per day and she eats 3."
    gold_answer = "She sells 16 - 3 = <<16-3=13>>13 eggs.\n#### 13"
    reference = build_task_reference(question, gold_answer)
    step = "Step 1: Janet's ducks lay 16 eggs per day."
    result = judge_step_task_correctness(step, reference)
    assert result.is_correct is True
    assert result.parse_fail is False
    assert result.reason == "non_equational_numeric_reasoning"


def test_task_judge_rejects_wrong_equation() -> None:
    question = "Janet's ducks lay 16 eggs per day and she eats 3."
    gold_answer = "She sells 16 - 3 = <<16-3=13>>13 eggs.\n#### 13"
    reference = build_task_reference(question, gold_answer)
    step = "Step 2: 16 - 3 = 99"
    result = judge_step_task_correctness(step, reference)
    assert result.is_correct is False
    assert result.parse_fail is False
    assert result.reason == "equation_mismatch"


def test_task_judge_accepts_question_value_assignment() -> None:
    question = (
        "Toulouse has twice as many sheep as Charleston. "
        "Charleston has 4 times as many sheep as Seattle. "
        "How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?"
    )
    gold_answer = (
        "Charleston has 4 * 20 = <<4*20=80>>80 sheep.\n"
        "Toulouse has 2 * 80 = <<2*80=160>>160 sheep.\n"
        "Together they have 20 + 80 + 160 = <<20+80+160=260>>260 sheep.\n"
        "#### 260"
    )
    reference = build_task_reference(question, gold_answer)
    step = "Step 4: Given that S = 20, we can substitute to find C and T."
    result = judge_step_task_correctness(step, reference)
    assert result.is_correct is True
    assert result.parse_fail is False
    assert result.reason == "equation_matches_reference"


def test_task_judge_accepts_assignment_chain_with_reference_value() -> None:
    question = (
        "Toulouse has twice as many sheep as Charleston. "
        "Charleston has 4 times as many sheep as Seattle. "
        "How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?"
    )
    gold_answer = (
        "Charleston has 4 * 20 = <<4*20=80>>80 sheep.\n"
        "Toulouse has 2 * 80 = <<2*80=160>>160 sheep.\n"
        "Together they have 20 + 80 + 160 = <<20+80+160=260>>260 sheep.\n"
        "#### 260"
    )
    reference = build_task_reference(question, gold_answer)
    step = "Step 5: C = 4 * 20 = 80"
    result = judge_step_task_correctness(step, reference)
    assert result.is_correct is True
    assert result.parse_fail is False
    assert result.reason == "equation_matches_reference"


def test_task_judge_accepts_symbolic_multiplier_assignment() -> None:
    question = (
        "Toulouse has twice as many sheep as Charleston. "
        "Charleston has 4 times as many sheep as Seattle. "
        "How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?"
    )
    gold_answer = (
        "Charleston has 4 * 20 = <<4*20=80>>80 sheep.\n"
        "Toulouse has 2 * 80 = <<2*80=160>>160 sheep.\n"
        "Together they have 20 + 80 + 160 = <<20+80+160=260>>260 sheep.\n"
        "#### 260"
    )
    reference = build_task_reference(question, gold_answer)
    step = "Step 6: T = 2C is the number of sheep in Toulouse."
    result = judge_step_task_correctness(step, reference)
    assert result.is_correct is True
    assert result.parse_fail is False
    assert result.reason == "equation_consistent_supported"


def test_task_judge_accepts_reversed_symbol_assignment_to_target() -> None:
    question = (
        "John plans to sell all his toys and use the money to buy video games. "
        "He has 13 lego sets and he sells them for $15 each. "
        "He ends up buying 8 video games for $20 each and has $5 left. "
        "How many lego sets does he still have?"
    )
    gold_answer = (
        "He makes 13 * 15 = <<13*15=195>>195 dollars.\n"
        "He spends 8 * 20 = <<8*20=160>>160 dollars.\n"
        "That leaves 195 - 160 - 5 = <<195-160-5=30>>30 dollars.\n"
        "#### 30"
    )
    reference = build_task_reference(question, gold_answer)
    step = "Step 5: Simplifying the equation gives us 30 = x."
    result = judge_step_task_correctness(step, reference)
    assert result.is_correct is True
    assert result.parse_fail is False
    assert result.reason == "equation_matches_reference"


def test_task_judge_accepts_symbolic_relation_with_reference_support() -> None:
    question = "A shirt that is 25% off costs $19.50. What was the original price?"
    gold_answer = "0.75 * 26 = <<0.75*26=19.5>>19.5\n#### 26"
    reference = build_task_reference(question, gold_answer)
    step = "Step 2: The discounted price is 75% of the original price, so we have 0.75x = $19.50."
    result = judge_step_task_correctness(step, reference)
    assert result.is_correct is True
    assert result.parse_fail is False
    assert result.reason == "equation_matches_reference"


def test_task_judge_accepts_symbolic_relation_after_operation_phrase() -> None:
    question = "There are 110 coins total and 30 more gold coins than silver coins. How many silver coins are there?"
    gold_answer = "2 * 40 + 30 = <<2*40+30=110>>110\n#### 40"
    reference = build_task_reference(question, gold_answer)
    step = "Step 6: Subtract 30 from both sides: 2S = 80."
    result = judge_step_task_correctness(step, reference)
    assert result.is_correct is True
    assert result.parse_fail is False
    assert result.reason == "equation_consistent_supported"


def test_task_judge_accepts_clause_before_assignment_symbol() -> None:
    question = "Darrell and Allen's ages are in the ratio of 7:11. If their total age now is 162, calculate Allen's age 10 years from now."
    gold_answer = (
        "Total parts = 7 + 11 = <<7+11=18>>18\n"
        "Allen's current age = 11 * 162 / 18 = <<11*162/18=99>>99\n"
        "Allen in 10 years = 99 + 10 = <<99+10=109>>109\n"
        "#### 109"
    )
    reference = build_task_reference(question, gold_answer)
    step = "Step 5: Multiply both sides by 11/18 to isolate A, giving us A = 11 * 162 / 18."
    result = judge_step_task_correctness(step, reference)
    assert result.is_correct is True
    assert result.parse_fail is False
    assert result.reason == "equation_matches_reference"


def test_task_judge_accepts_labelled_numeric_expression_with_target_match() -> None:
    question = (
        "Eliza's rate per hour for the first 40 hours she works each week is $10. "
        "She also receives an overtime pay of 1.2 times her regular hourly rate. "
        "If Eliza worked for 45 hours this week, how much are her earnings for this week?"
    )
    gold_answer = (
        "Regular earnings = 40 * 10 = <<40*10=400>>400\n"
        "Overtime earnings = 5 * (1.2 * 10) = <<5*(1.2*10)=60>>60\n"
        "Total earnings = 400 + 60 = <<400+60=460>>460\n"
        "#### 460"
    )
    reference = build_task_reference(question, gold_answer)
    step = "Step 9: Step 8: Total earnings = $400 + $60"
    result = judge_step_task_correctness(step, reference)
    assert result.is_correct is True
    assert result.parse_fail is False
    assert result.reason == "equation_consistent_supported"


def test_task_judge_accepts_equation_with_deepseek_trailer() -> None:
    question = "He makes 13 * 15 dollars and spends 8 * 20 dollars, leaving 30 dollars."
    gold_answer = (
        "He makes 13 * 15 = <<13*15=195>>195 dollars.\n"
        "He spends 8 * 20 = <<8*20=160>>160 dollars.\n"
        "That leaves 195 - 160 - 5 = <<195-160-5=30>>30 dollars.\n"
        "#### 30"
    )
    reference = build_task_reference(question, gold_answer)
    step = "Step 7: 195 - 160 - 5 = 30 Final Answer: 30 </think>"
    result = judge_step_task_correctness(step, reference)
    assert result.is_correct is True
    assert result.parse_fail is False
    assert result.reason == "equation_matches_reference"


def test_task_judge_flags_empty_boundary() -> None:
    question = "A shelf has 12 books and then 9 more are added."
    gold_answer = "12 + 9 = <<12+9=21>>21\n#### 21"
    reference = build_task_reference(question, gold_answer)
    result = judge_step_task_correctness("Step 2: ", reference)
    assert result.is_correct is False
    assert result.parse_fail is True
    assert result.reason == "boundary_detected_but_empty_math"


def test_task_judge_flags_unsupported_symbolic_form() -> None:
    question = "A price increases by 150% from 80 to what value?"
    gold_answer = "80 * 2.5 = <<80*2.5=200>>200\n#### 200"
    reference = build_task_reference(question, gold_answer)
    step = "Step 1: increase = old price -> 80 with markup"
    result = judge_step_task_correctness(step, reference)
    assert result.is_correct is False
    assert result.parse_fail is True
    assert result.reason in {"unsupported_symbolic_form", "sympy_parse_error"}
