from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A/B/C ablation scaffolding.")
    parser.add_argument(
        "--experiment",
        choices=["A", "B", "C"],
        required=True,
        help="A: early warning, B: model comparison, C: estimator sensitivity",
    )
    parser.add_argument("--input", required=True, help="Path to prepared experiment table")
    parser.add_argument("--out", default="results/ablation", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Ablation scaffold")
    print(f"Experiment: {args.experiment}")
    print(f"Input: {args.input}")
    print(f"Output: {args.out}")
    print("TODO: implement experiment-specific statistical pipeline.")


if __name__ == "__main__":
    main()
