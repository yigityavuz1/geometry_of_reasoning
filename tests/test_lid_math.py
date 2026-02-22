from __future__ import annotations

import numpy as np

from src.metrics.global_dim import participation_ratio
from src.metrics.lid_estimators import lid_mle_batch, twonn_global_id


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
