from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.sympy_judge import judge_step_equational_consistency
from src.generation.extraction import split_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simple step-level SymPy judge.")
    parser.add_argument("--in", dest="input_path", required=True, help="Generation trace JSON path")
    parser.add_argument("--out", default="results/judged_trace.json", help="Output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace = json.loads(Path(args.input_path).read_text(encoding="utf-8"))
    steps = split_steps(trace.get("generated_text", ""))

    judged_steps = []
    for idx, step_text in enumerate(steps, start=1):
        judgement = judge_step_equational_consistency(step_text)
        judged_steps.append(
            {
                "step_index": idx,
                "text": step_text,
                "is_correct": judgement.is_correct,
                "parse_fail": judgement.parse_fail,
                "reason": judgement.reason,
            }
        )

    trace["judged_steps"] = judged_steps
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote judged trace to {output_path}")


if __name__ == "__main__":
    main()
