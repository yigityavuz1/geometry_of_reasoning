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
    coefficient_of_variation,
    k_sweep_local_id,
    twonn_global_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase-3 synthetic validation: checks expected ID trends and k-sensitivity "
            "for LID/ABID/TwoNN/PR estimators."
        )
    )
    parser.add_argument("--out", default="results/phase3/synthetic_validation.json")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--ambient-dim", type=int, default=64)
    parser.add_argument("--noise-std", type=float, default=1e-4)
    parser.add_argument(
        "--intrinsic-dims",
        default="4,8,12",
        help="Comma-separated intrinsic dimensions to evaluate.",
    )
    parser.add_argument(
        "--k-values",
        default="5,10,20,40",
        help="Comma-separated k values for sensitivity.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _parse_int_csv(raw: str) -> list[int]:
    values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _make_subspace_data(
    intrinsic_dim: int,
    ambient_dim: int,
    n_samples: int,
    noise_std: float,
    seed: int,
) -> np.ndarray:
    if intrinsic_dim > ambient_dim:
        raise ValueError("intrinsic_dim must be <= ambient_dim.")
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.normal(size=(ambient_dim, intrinsic_dim)))
    latent = rng.normal(size=(n_samples, intrinsic_dim))
    signal = latent @ basis.T
    noise = noise_std * rng.normal(size=(n_samples, ambient_dim))
    return signal + noise


def _is_increasing(values: list[float]) -> bool:
    return all(curr > prev for prev, curr in zip(values, values[1:]))


def main() -> None:
    args = parse_args()
    intrinsic_dims = _parse_int_csv(args.intrinsic_dims)
    k_values = _parse_int_csv(args.k_values)
    if len(intrinsic_dims) < 2:
        raise ValueError("--intrinsic-dims must contain at least two values.")

    rows: list[dict[str, object]] = []
    twonn_values: list[float] = []
    pr_values: list[float] = []

    for intrinsic_dim in intrinsic_dims:
        x = _make_subspace_data(
            intrinsic_dim=intrinsic_dim,
            ambient_dim=args.ambient_dim,
            n_samples=args.n_samples,
            noise_std=args.noise_std,
            seed=args.seed + intrinsic_dim,
        )
        sweep = k_sweep_local_id(x, k_values=k_values)
        lid_values = [float(item["lid_mle_mean"]) for item in sweep]
        abid_values = [float(item["abid_mean"]) for item in sweep]
        twonn = float(twonn_global_id(x))
        pr = float(participation_ratio(x))
        twonn_values.append(twonn)
        pr_values.append(pr)

        rows.append(
            {
                "intrinsic_dim": intrinsic_dim,
                "k_sweep": sweep,
                "twonn_global_id": twonn,
                "participation_ratio": pr,
                "lid_mle_cv_over_k": coefficient_of_variation(lid_values),
                "abid_cv_over_k": coefficient_of_variation(abid_values),
            }
        )

    first_lid = [float(row["k_sweep"][0]["lid_mle_mean"]) for row in rows]
    first_abid = [float(row["k_sweep"][0]["abid_mean"]) for row in rows]
    trend_checks = {
        "lid_mle_increasing_at_first_k": _is_increasing(first_lid),
        "abid_increasing_at_first_k": _is_increasing(first_abid),
        "twonn_increasing": _is_increasing(twonn_values),
        "pr_increasing": _is_increasing(pr_values),
    }

    output = {
        "config": {
            "n_samples": args.n_samples,
            "ambient_dim": args.ambient_dim,
            "noise_std": args.noise_std,
            "intrinsic_dims": intrinsic_dims,
            "k_values": k_values,
            "seed": args.seed,
        },
        "results": rows,
        "trend_checks": trend_checks,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote phase-3 synthetic validation to {out_path}")


if __name__ == "__main__":
    main()
