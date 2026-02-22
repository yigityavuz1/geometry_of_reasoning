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
from src.metrics.lid_estimators import abid_local_batch, lid_mle_batch, twonn_global_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run metric calculations on a saved embedding file.")
    parser.add_argument("--embeddings", required=True, help="Path to .npy embeddings")
    parser.add_argument("--out", default="results/metrics_summary.json", help="Output JSON path")
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


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
    summary = {
        "n_samples": n_samples,
        "n_features": int(x.shape[1]),
        "k_requested": args.k,
        "k_effective": k_eff,
        "lid_mle_mean": float(np.mean(lid_mle_batch(x, k=k_eff))),
        "twonn_global_id": float(twonn_global_id(x)),
        "abid_mean": float(np.mean(abid_local_batch(x, k=k_eff))),
        "participation_ratio": float(participation_ratio(x)),
    }

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote metric summary to {output_path}")


if __name__ == "__main__":
    main()
