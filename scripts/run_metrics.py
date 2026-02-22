from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.global_dim import participation_ratio
from src.metrics.lid_estimators import (
    abid_local_batch,
    coefficient_of_variation,
    k_sweep_local_id,
    lid_mle_batch,
    twonn_global_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run metric calculations on a saved embedding file.")
    parser.add_argument("--embeddings", required=True, help="Path to .npy embeddings")
    parser.add_argument("--out", default="results/metrics_summary.json", help="Output JSON path")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--k-values",
        default="",
        help="Optional comma-separated k values for sensitivity sweep, e.g. 5,10,20,40",
    )
    return parser.parse_args()


def _parse_k_values(raw: str, fallback_k: int) -> list[int]:
    if not raw.strip():
        return [int(fallback_k)]
    out: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        return [int(fallback_k)]
    return out


def main() -> None:
    args = parse_args()
    x = np.load(args.embeddings)
    if x.ndim != 2:
        raise ValueError("Embeddings must be a 2D array (n_samples, n_features).")
    n_samples = int(x.shape[0])
    if n_samples < 3:
        raise ValueError(
            "Need at least 3 embedding samples to compute LID/TwoNN robustly. "
            "Increase generated tokens or use a richer trace."
        )

    k_eff = max(2, min(args.k, n_samples - 1))
    k_values = _parse_k_values(args.k_values, args.k)
    k_sweep = k_sweep_local_id(x, k_values=k_values)
    lid_values = [float(row["lid_mle_mean"]) for row in k_sweep]
    abid_values = [float(row["abid_mean"]) for row in k_sweep]

    summary = {
        "n_samples": n_samples,
        "n_features": int(x.shape[1]),
        "k_requested": args.k,
        "k_effective": k_eff,
        "lid_mle_mean": float(np.mean(lid_mle_batch(x, k=k_eff))),
        "twonn_global_id": float(twonn_global_id(x)),
        "abid_mean": float(np.mean(abid_local_batch(x, k=k_eff))),
        "participation_ratio": float(participation_ratio(x)),
        "k_sweep": k_sweep,
        "lid_mle_cv_over_k": coefficient_of_variation(lid_values),
        "abid_cv_over_k": coefficient_of_variation(abid_values),
    }

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote metric summary to {output_path}")


if __name__ == "__main__":
    main()
