from __future__ import annotations

import numpy as np

from src.metrics.global_dim import participation_ratio
from src.metrics.lid_estimators import (
    abid_local_batch,
    coefficient_of_variation,
    k_sweep_local_id,
    lid_mle_batch,
    twonn_global_id,
)


def test_lid_outputs_are_positive() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(64, 8))
    lid = lid_mle_batch(x, k=10)
    assert lid.shape == (64,)
    assert np.all(lid > 0.0)


def test_twonn_is_positive() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(100, 6))
    d = twonn_global_id(x)
    assert d > 0.0


def test_participation_ratio_reasonable_bounds() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 16))
    pr = participation_ratio(x)
    assert 0.0 <= pr <= 16.0


def test_abid_outputs_are_positive() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(80, 10))
    abid = abid_local_batch(x, k=12)
    assert abid.shape == (80,)
    assert np.all(abid > 0.0)


def _make_subspace_data(intrinsic_dim: int, ambient_dim: int, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.normal(size=(ambient_dim, intrinsic_dim)))
    latent = rng.normal(size=(n_samples, intrinsic_dim))
    return latent @ basis.T + 1e-4 * rng.normal(size=(n_samples, ambient_dim))


def test_estimators_follow_expected_low_vs_high_id_trend() -> None:
    low = _make_subspace_data(intrinsic_dim=4, ambient_dim=64, n_samples=500, seed=101)
    high = _make_subspace_data(intrinsic_dim=12, ambient_dim=64, n_samples=500, seed=202)

    low_lid = float(np.mean(lid_mle_batch(low, k=20)))
    high_lid = float(np.mean(lid_mle_batch(high, k=20)))
    low_abid = float(np.mean(abid_local_batch(low, k=20)))
    high_abid = float(np.mean(abid_local_batch(high, k=20)))
    low_twonn = float(twonn_global_id(low))
    high_twonn = float(twonn_global_id(high))
    low_pr = float(participation_ratio(low))
    high_pr = float(participation_ratio(high))

    assert high_lid > low_lid
    assert high_abid > low_abid
    assert high_twonn > low_twonn
    assert high_pr > low_pr


def test_k_sweep_produces_rows_and_cv() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(size=(120, 12))
    sweep = k_sweep_local_id(x, k_values=[5, 10, 20, 40])
    lid_values = [float(row["lid_mle_mean"]) for row in sweep]
    abid_values = [float(row["abid_mean"]) for row in sweep]

    assert len(sweep) >= 3
    assert all(int(row["k_effective"]) >= 2 for row in sweep)
    assert all(value > 0.0 for value in lid_values)
    assert all(value > 0.0 for value in abid_values)
    assert coefficient_of_variation(lid_values) >= 0.0
    assert coefficient_of_variation(abid_values) >= 0.0
