from __future__ import annotations

import re

_STEP_HEADER_PATTERN = re.compile(r"^\s*Step\s+\d+\s*:\s*", flags=re.IGNORECASE)
_GSM8K_INLINE_PATTERN = re.compile(r"<<([^<>]+)>>")
_GSM8K_FINAL_PATTERN = re.compile(r"####\s*([^\n]+)")
_NUMBER_PATTERN = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?(?![A-Za-z0-9_])")


def normalize_math_text(text: str) -> str:
    """Normalize noisy math text into a parser-friendly form."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    replacements = {
        "\u2212": "-",  # unicode minus
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u00d7": "*",  # multiplication sign
        "\u00f7": "/",  # division sign
        "^": "**",
        "$": "",
        "<<": "",
        ">>": "",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"(?<=\d),(?=\d)", "", normalized)
    return normalized.strip()


def strip_step_header(text: str) -> str:
    return _STEP_HEADER_PATTERN.sub("", text, count=1).strip()


def extract_equation_pairs(step_text: str) -> list[tuple[str, str]]:
    """Extract lhs=rhs equation pairs from a step string."""
    body = strip_step_header(normalize_math_text(step_text))
    equation_pairs: list[tuple[str, str]] = []
    for raw_line in body.splitlines():
        line = raw_line.strip().strip(".,;")
        if not line or "=" not in line:
            continue
        parts = [part.strip(" .,:;") for part in line.split("=")]
        parts = [part for part in parts if part]
        for idx in range(len(parts) - 1):
            lhs, rhs = parts[idx], parts[idx + 1]
            if lhs and rhs:
                equation_pairs.append((lhs, rhs))
    return equation_pairs


def extract_numeric_tokens(text: str) -> list[str]:
    body = strip_step_header(normalize_math_text(text))
    return [match.group(0).replace(",", "") for match in _NUMBER_PATTERN.finditer(body)]


def extract_gsm8k_inline_equations(gold_answer: str) -> list[str]:
    return [chunk.strip() for chunk in _GSM8K_INLINE_PATTERN.findall(gold_answer) if chunk.strip()]


def extract_gsm8k_final_answer_text(gold_answer: str) -> str:
    match = _GSM8K_FINAL_PATTERN.search(gold_answer)
    if not match:
        return ""
    return normalize_math_text(match.group(1)).strip()
