from __future__ import annotations

import re

_STEP_HEADER_PATTERN = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\*\*)?(?:Step\s*\d+\s*:|\(\d+\)\s+|\d+[\)\.:\-]\s+)",
    flags=re.IGNORECASE,
)
_FINAL_ANSWER_PATTERN = re.compile(r"(?im)^\s*final\s+answer\s*:\s*")
_GSM8K_INLINE_PATTERN = re.compile(r"<<([^<>]+)>>")
_GSM8K_FINAL_PATTERN = re.compile(r"####\s*([^\n]+)")
_NUMBER_PATTERN = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?(?![A-Za-z0-9_])")
_PERCENT_PATTERN = re.compile(r"(?<![A-Za-z0-9_])([-+]?\d+(?:\.\d+)?)\s*%")
_LATEX_FRACTION_PATTERN = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
_FEET_INCHES_COMPOUND_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])([-+]?\d+(?:\.\d+)?)\s*(?:feet|foot)\s+([-+]?\d+(?:\.\d+)?)\s*(?:inches|inch)\b",
    flags=re.IGNORECASE,
)
_MILES_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])([-+]?\d+(?:\.\d+)?)\s*(?:miles|mile)\b",
    flags=re.IGNORECASE,
)
_FEET_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])([-+]?\d+(?:\.\d+)?)\s*(?:feet|foot)\b",
    flags=re.IGNORECASE,
)
_INCHES_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])([-+]?\d+(?:\.\d+)?)\s*(?:inches|inch)\b",
    flags=re.IGNORECASE,
)
_OF_MULTIPLICATION_PATTERN = re.compile(r"(?i)(?<=\d|\))\s+of\s+(?=[A-Za-z0-9(])")
_CONNECTOR_SPLIT_PATTERN = re.compile(
    r"(?i)(?:,\s*)?\b(?:and|because|therefore|thus|hence|so|then|which means|meaning)\b"
)
_SELF_CORRECTION_SPLIT_PATTERN = re.compile(
    r"(?i)(?:[.;]\s*|\s+)(?:but\s+wait|wait|hold\s+on|oops)\b"
)
_LEADING_FILLER_PATTERN = re.compile(
    r"(?i)^(?:final\s+answer|answer|result|remaining|total|profit|cost|distance|there\s+(?:are|is)|"
    r"we\s+(?:have|get|find|write|obtain)|we\s+can\s+(?:write|get|set)|giving\s+us|this\s+gives|"
    r"gives|equals?|is|are|so|thus|therefore|hence|because|then)\b"
    r"[\s:,\-]*"
)
_EQUATION_CUE_PATTERN = re.compile(r"(?i)\b(?:have|gives?|get|equals?|is|are|yields?|results?|therefore|thus|hence|so|then)\b")
_LONG_WORD_PATTERN = re.compile(r"\b[A-Za-z]{2,}\b")
_NON_MATH_PARENS_PATTERN = re.compile(r"\(([^()]*)\)")
_EXTRA_PUNCT_PATTERN = re.compile(r"[\"']")
_THINK_TAG_PATTERN = re.compile(r"(?is)</?think\s*>")
_TRAILING_META_MARKER_PATTERN = re.compile(r"(?i)\b(?:final\s+answer|problem|question|solution)\s*:")
_TRAILING_SYMBOL_MENTION_PATTERN = re.compile(r"(?<=[\d\)])\s*[,;]\s*(?:[A-Za-z](?:\s+[A-Za-z])*)[.]*$")
_INLINE_STEP_REFERENCE_PATTERN = re.compile(
    r"(?i)\b(?:from|into|in|at|on|to)?\s*step\s*\d+\s*:?"
)
_RIGHTMOST_OPERATOR_FRAGMENT_PATTERN = re.compile(
    r"([A-Za-z0-9().]+(?:\s*[+\-*/]\s*[A-Za-z0-9().]+)+)$"
)
_UNIT_SUFFIXES = (
    "dollar",
    "dollars",
    "cent",
    "cents",
    "egg",
    "eggs",
    "bolt",
    "bolts",
    "book",
    "books",
    "meter",
    "meters",
    "mile",
    "miles",
    "hour",
    "hours",
    "minute",
    "minutes",
    "day",
    "days",
    "week",
    "weeks",
    "month",
    "months",
    "year",
    "years",
    "lap",
    "laps",
    "sandwich",
    "sandwiches",
    "drink",
    "drinks",
    "student",
    "students",
    "person",
    "people",
    "item",
    "items",
    "car",
    "cars",
    "house",
    "houses",
    "problem",
    "problems",
    "question",
    "questions",
    "ticket",
    "tickets",
    "apple",
    "apples",
    "orange",
    "oranges",
    "bag",
    "bags",
    "box",
    "boxes",
    "shirt",
    "shirts",
    "page",
    "pages",
)
_UNIT_SUFFIX_PATTERN = re.compile(
    r"(?<=\d)\s+(?:" + "|".join(re.escape(unit) for unit in _UNIT_SUFFIXES) + r")\b",
    flags=re.IGNORECASE,
)


def _replace_latex_fractions(text: str) -> str:
    updated = text
    while True:
        replaced = _LATEX_FRACTION_PATTERN.sub(r"(\1)/(\2)", updated)
        if replaced == updated:
            return replaced
        updated = replaced


def _normalize_length_units(text: str) -> str:
    normalized = _FEET_INCHES_COMPOUND_PATTERN.sub(
        lambda match: f"(({match.group(1)}) + ({match.group(2)})/12)",
        text,
    )
    normalized = _MILES_PATTERN.sub(lambda match: f"(({match.group(1)}) * 5280)", normalized)
    normalized = _FEET_PATTERN.sub(lambda match: match.group(1), normalized)
    normalized = _INCHES_PATTERN.sub(lambda match: f"(({match.group(1)})/12)", normalized)
    return normalized


def normalize_math_text(text: str, *, convert_units: bool = False) -> str:
    """Normalize noisy math text into a parser-friendly form."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\*\*(.*?)\*\*", r"\1", normalized)
    normalized = _replace_latex_fractions(normalized)
    replacements = {
        "\u2212": "-",  # unicode minus
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u00d7": "*",  # multiplication sign
        "\u00f7": "/",  # division sign
        "\u2264": "<=",
        "\u2265": ">=",
        "\u2248": "=",
        "\u2243": "=",
        "\u2245": "=",
        "\u2026": "...",
        "^": "**",
        "$": "",
        "`": "",
        "<<": "",
        ">>": "",
        "\\(": "",
        "\\)": "",
        "\\[": "",
        "\\]": "",
        "\\times": "*",
        "\\cdot": "*",
        "\\div": "/",
        "\\left": "",
        "\\right": "",
        "{": "",
        "}": "",
        "\\": "",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    if convert_units:
        normalized = _normalize_length_units(normalized)
    normalized = _PERCENT_PATTERN.sub(lambda match: f"({match.group(1)}/100)", normalized)
    normalized = _OF_MULTIPLICATION_PATTERN.sub(" * ", normalized)
    normalized = re.sub(r"(?<=\d),(?=\d)", "", normalized)
    normalized = _EXTRA_PUNCT_PATTERN.sub("", normalized)
    return normalized.strip()


def strip_step_header(text: str) -> str:
    stripped = text.strip()
    while True:
        updated = _STEP_HEADER_PATTERN.sub("", stripped, count=1).strip()
        if updated == stripped:
            return stripped
        stripped = updated


def _remove_non_math_parentheticals(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        inner = match.group(1)
        if re.search(r"\d|[=+\-*/]", inner):
            return match.group(0)
        return " "

    return _NON_MATH_PARENS_PATTERN.sub(_replace, text)


def _strip_trailing_meta_sections(text: str) -> str:
    cleaned = _THINK_TAG_PATTERN.sub(" ", text)
    earliest_cutoff = len(cleaned)
    for match in _TRAILING_META_MARKER_PATTERN.finditer(cleaned):
        prefix = cleaned[: match.start()]
        if not re.search(r"\d|=", prefix):
            continue
        earliest_cutoff = min(earliest_cutoff, match.start())
    return cleaned[:earliest_cutoff]


def _dedupe_terminal_numeric_echo(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if "=" not in compact:
        return compact
    lhs, rhs = compact.rsplit("=", 1)
    tokens = [token for token in rhs.strip().split() if token]
    while len(tokens) >= 2:
        last = tokens[-1].strip(" .,:;").replace(",", "")
        prev = tokens[-2].strip(" .,:;").replace(",", "")
        if last == prev and _NUMBER_PATTERN.fullmatch(last):
            tokens.pop()
            continue
        break
    deduped_rhs = " ".join(tokens)
    if not deduped_rhs:
        return lhs.strip()
    return f"{lhs.strip()} = {deduped_rhs}".strip()


def normalize_step_text(text: str) -> str:
    body = strip_step_header(normalize_math_text(text))
    body = _FINAL_ANSWER_PATTERN.sub("", body, count=1)
    body = _remove_non_math_parentheticals(body)
    body = _strip_trailing_meta_sections(body)
    body = _UNIT_SUFFIX_PATTERN.sub("", body)
    body = _dedupe_terminal_numeric_echo(body)
    body = re.sub(r"\s+", " ", body.replace("\n", " \n ")).replace(" \n ", "\n")
    return body.strip()


def _strip_inline_step_references(text: str) -> str:
    without_refs = _INLINE_STEP_REFERENCE_PATTERN.sub(" ", text)
    return re.sub(r"(?i)\bstep\s*\d+\s*:", " ", without_refs)


def _looks_descriptive_numeric_label(original: str, cleaned: str) -> bool:
    if re.search(r"[A-Za-z]", cleaned):
        return False
    if re.search(r"[+\-*/()]", cleaned):
        return False
    if re.match(r"\s*[-+]?(?:\d|\.\d|\()", original):
        return False
    if _EQUATION_CUE_PATTERN.search(original):
        return False
    return bool(_LONG_WORD_PATTERN.search(original))


def _looks_descriptive_lhs_label(original: str) -> bool:
    candidate = original.strip(" .,:;")
    if re.search(r"\d|[+\-*/()]", candidate):
        return False
    tokens = [token.strip(" .,:;") for token in candidate.split() if token.strip(" .,:;")]
    if len(tokens) < 2:
        return False
    if any(re.fullmatch(r"[A-Za-z]", token) for token in tokens):
        return False
    alpha_tokens = [token for token in tokens if re.fullmatch(r"[A-Za-z]+", token)]
    return len(alpha_tokens) == len(tokens)


def _prefer_equation_side_tail(text: str, *, side: str) -> str:
    if side != "lhs":
        return text

    candidate = text
    for delimiter in (",", ":"):
        if delimiter not in candidate:
            continue
        tail = candidate.rsplit(delimiter, 1)[-1].strip()
        if tail and re.search(r"[A-Za-z0-9()]", tail):
            candidate = tail
    return candidate


def _finalize_cleaned_side(cleaned: str, original: str) -> str:
    candidate = cleaned.strip(" .,:;")
    if ":" in candidate:
        tail = candidate.rsplit(":", 1)[-1].strip()
        if tail:
            candidate = tail

    if re.search(r"[A-Za-z]", original):
        operator_match = _RIGHTMOST_OPERATOR_FRAGMENT_PATTERN.search(candidate)
        if operator_match:
            candidate = operator_match.group(1).strip()

    tokens = candidate.split()
    if len(tokens) == 1:
        token = tokens[0]
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", token) and _looks_descriptive_numeric_label(original, candidate):
            return ""
        return token.strip(" .,:;")

    if tokens and not re.search(r"[+\-*/()]", candidate):
        symbol_tokens = [token for token in tokens if re.search(r"[A-Za-z]", token)]
        if symbol_tokens:
            return symbol_tokens[-1].strip(" .,:;")
        if _looks_descriptive_numeric_label(original, candidate):
            return ""
        return tokens[-1].strip(" .,:;")

    return candidate.strip(" .,:;")


def _clean_equation_side(text: str, *, side: str = "either") -> str:
    cleaned = normalize_math_text(text, convert_units=True)
    cleaned = _remove_non_math_parentheticals(cleaned)
    cleaned = _strip_inline_step_references(cleaned)
    cleaned = _prefer_equation_side_tail(cleaned, side=side)
    cleaned = _LEADING_FILLER_PATTERN.sub("", cleaned)
    cleaned = _UNIT_SUFFIX_PATTERN.sub("", cleaned)
    original = cleaned
    if side == "lhs" and _looks_descriptive_lhs_label(original):
        return ""
    cleaned = _LONG_WORD_PATTERN.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = _TRAILING_SYMBOL_MENTION_PATTERN.sub("", cleaned)
    return _finalize_cleaned_side(cleaned, original)


def _equation_chunks(text: str) -> list[str]:
    body = normalize_step_text(text)
    if not body:
        return []
    expanded = body.replace(";", "\n")
    expanded = _CONNECTOR_SPLIT_PATTERN.sub("\n", expanded)
    expanded = _SELF_CORRECTION_SPLIT_PATTERN.sub("\n", expanded)
    return [chunk.strip() for chunk in expanded.splitlines() if chunk.strip()]


def extract_equation_pairs(step_text: str) -> list[tuple[str, str]]:
    """Extract lhs=rhs equation pairs from a step string."""
    equation_pairs: list[tuple[str, str]] = []
    for chunk in _equation_chunks(step_text):
        if "=" not in chunk:
            continue
        raw_parts = chunk.split("=")
        parts = [
            _clean_equation_side(part, side="lhs" if idx < len(raw_parts) - 1 else "rhs")
            for idx, part in enumerate(raw_parts)
        ]
        parts = [part for part in parts if part]
        for idx in range(len(parts) - 1):
            lhs, rhs = parts[idx], parts[idx + 1]
            if lhs and rhs:
                equation_pairs.append((lhs, rhs))
    return equation_pairs


def extract_numeric_tokens(text: str) -> list[str]:
    body = normalize_step_text(text)
    return [match.group(0).replace(",", "") for match in _NUMBER_PATTERN.finditer(body)]


def extract_gsm8k_inline_equations(gold_answer: str) -> list[str]:
    return [chunk.strip() for chunk in _GSM8K_INLINE_PATTERN.findall(gold_answer) if chunk.strip()]


def extract_gsm8k_final_answer_text(gold_answer: str) -> str:
    match = _GSM8K_FINAL_PATTERN.search(gold_answer)
    if not match:
        return ""
    return normalize_math_text(match.group(1)).strip()
