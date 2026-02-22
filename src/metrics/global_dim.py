from __future__ import annotations

import numpy as np


def participation_ratio(x: np.ndarray, center: bool = True) -> float:
    """Compute participation ratio from covariance eigenvalue spectrum."""
    if x.ndim != 2:
        raise ValueError("Input must be 2D array with shape (n_samples, n_features).")
    if x.shape[0] < 2:
        raise ValueError("Need at least two samples for covariance.")

    data = x - x.mean(axis=0, keepdims=True) if center else x
    cov = np.cov(data, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov).clip(min=0.0)
    denom = float(np.square(eigvals).sum())
    if denom == 0.0:
        return 0.0
    return float(np.square(eigvals.sum()) / denom)
