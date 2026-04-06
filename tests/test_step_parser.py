from __future__ import annotations

from src.evaluation.step_parser import extract_equation_pairs, extract_numeric_tokens, normalize_math_text, normalize_step_text
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


def test_split_steps_supports_numeric_list_format() -> None:
    text = (
        "1) 2 + 3 = 5\n"
        "2) 5 * 2 = 10\n"
    )
    steps = split_steps(text)
    assert len(steps) == 2
    assert steps[0].startswith("1)")
    assert steps[1].startswith("2)")


def test_split_steps_excludes_final_answer_block() -> None:
    text = (
        "Step 1: 12 - 5 = 7\n"
        "Step 2: 7 + 9 = 16\n"
        "Final Answer: 16\n"
    )
    steps = split_steps(text)
    assert len(steps) == 2
    assert all("Final Answer" not in step for step in steps)


def test_split_steps_skips_header_only_blocks() -> None:
    text = "Step 1: Step 2: 2 + 3 = 5\nFinal Answer: 5"
    steps = split_steps(text)
    assert steps == ["Step 2: 2 + 3 = 5"]


def test_split_steps_keeps_inline_steps_after_early_final_answer() -> None:
    text = "Final Answer: 82 Step 1: 38 + 6 = 44\nStep 2: 38 + 44 = 82"
    steps = split_steps(text)
    assert steps == ["Step 1: 38 + 6 = 44", "Step 2: 38 + 44 = 82"]


def test_extract_equation_pairs_supports_multiple_equalities() -> None:
    step = "Step 3: x = y = 5"
    pairs = extract_equation_pairs(step)
    assert pairs == [("x", "y"), ("y", "5")]


def test_extract_equation_pairs_handles_explanatory_clause() -> None:
    step = "Step 1: 32, because 40 - 8 = 32."
    pairs = extract_equation_pairs(step)
    assert pairs == [("40 - 8", "32")]


def test_extract_equation_pairs_strips_trailing_symbol_mentions() -> None:
    step = "Step 4: Given that S = 20, we can substitute to find C and T."
    pairs = extract_equation_pairs(step)
    assert pairs == [("S", "20")]


def test_extract_equation_pairs_strips_duplicate_symbol_prefix() -> None:
    step = "Step 3: Solving for x gives us x = $19.50 / 0.75 = $26."
    pairs = extract_equation_pairs(step)
    assert pairs == [("x", "19.50 / 0.75"), ("19.50 / 0.75", "26")]


def test_extract_equation_pairs_strips_inline_step_references_from_lhs() -> None:
    step = "Step 7: Substituting the value of R from Step 4 into Step 5, we get A = (40/2) + 5 = 20 + 5 = 25."
    pairs = extract_equation_pairs(step)
    assert pairs == [("A", "(40/2) + 5"), ("(40/2) + 5", "20 + 5"), ("20 + 5", "25")]


def test_extract_equation_pairs_prefers_rhs_after_colon_label() -> None:
    step = "Step 6: Subtract 30 from both sides: 2S = 80."
    pairs = extract_equation_pairs(step)
    assert pairs == [("2S", "80")]


def test_extract_equation_pairs_drops_descriptive_lhs_label_before_chain() -> None:
    step = "Step 2: Total cost of lemons = $3 * 20 = $60."
    pairs = extract_equation_pairs(step)
    assert pairs == [("3 * 20", "60")]


def test_extract_equation_pairs_keeps_symbol_after_operation_phrase() -> None:
    step = "Step 7: Divide by 2: S = 40."
    pairs = extract_equation_pairs(step)
    assert pairs == [("S", "40")]


def test_extract_equation_pairs_keeps_trailing_symbol_after_clause() -> None:
    step = "Step 5: Multiply both sides by 11/18 to isolate A, giving us A = 11 * 162 / 18."
    pairs = extract_equation_pairs(step)
    assert pairs == [("A", "11 * 162 / 18")]


def test_extract_equation_pairs_strips_units_and_percentages() -> None:
    step = "Step 2: 80,000 + 50,000 = 130,000 dollars and 150% = 1.5"
    pairs = extract_equation_pairs(step)
    assert ("80000 + 50000", "130000") in pairs
    assert ("(150/100)", "1.5") in pairs


def test_extract_equation_pairs_treats_of_as_multiplication() -> None:
    step = "Step 2: 25% of (20 - 4) = 25% of 16 = 4"
    pairs = extract_equation_pairs(step)
    assert pairs == [("(25/100) * (20 - 4)", "(25/100) * 16"), ("(25/100) * 16", "4")]


def test_extract_numeric_tokens_ignores_step_index() -> None:
    step = "Step 9: Janet has 16 eggs and uses 4."
    tokens = extract_numeric_tokens(step)
    assert tokens == ["16", "4"]


def test_extract_numeric_tokens_ignores_numeric_list_index() -> None:
    step = "1) Janet has 16 eggs and uses 4."
    tokens = extract_numeric_tokens(step)
    assert tokens == ["16", "4"]


def test_normalize_step_text_removes_final_answer_prefix() -> None:
    text = "Final Answer: 84 dollars"
    assert normalize_step_text(text) == "84"


def test_normalize_step_text_handles_repeated_headers_and_latex() -> None:
    text = r"Step 2: Step 1: W = \frac{3}{2}"
    assert normalize_step_text(text) == "W = (3)/(2)"


def test_normalize_step_text_strips_trailing_final_answer_and_think_tag() -> None:
    text = "Step 9: 400 + 60 = 460 Final Answer: 460 </think>"
    assert normalize_step_text(text) == "400 + 60 = 460"


def test_extract_equation_pairs_ignores_problem_reentry_after_think_tag() -> None:
    step = "Step 4: 24 - 10 = 14 Final Answer: 14 </think> Problem: Two girls shared the water."
    pairs = extract_equation_pairs(step)
    assert pairs == [("24 - 10", "14")]


def test_normalize_math_text_keeps_length_units_by_default() -> None:
    assert normalize_math_text("3 miles") == "3 miles"
    assert normalize_math_text("3 miles", convert_units=True) == "((3) * 5280)"


def test_extract_numeric_tokens_keeps_raw_length_numbers_without_global_conversion() -> None:
    step = "Step 3: The board is 4 feet 6 inches long."
    tokens = extract_numeric_tokens(step)
    assert tokens == ["4", "6"]
