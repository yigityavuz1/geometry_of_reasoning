from __future__ import annotations

import argparse


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
