from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def _sorted_neighbor_distances(x: np.ndarray) -> np.ndarray:
    distances = cdist(x, x, metric="euclidean")
    # Remove self-distance by setting diagonal to +inf.
    np.fill_diagonal(distances, np.inf)
    return np.sort(distances, axis=1)


def lid_mle_batch(x: np.ndarray, k: int = 20) -> np.ndarray:
    """Levina-Bickel local intrinsic dimension estimate for each sample."""
    if x.ndim != 2:
        raise ValueError("Input must be 2D array with shape (n_samples, n_features).")
    if k < 2 or k >= x.shape[0]:
        raise ValueError("k must satisfy 2 <= k < n_samples.")

    dists = _sorted_neighbor_distances(x)
    local = np.clip(dists[:, :k], 1e-12, None)
    tk = local[:, -1]
    ratios = np.clip(tk[:, None] / local[:, : k - 1], 1.0 + 1e-12, None)
    denom = np.mean(np.log(ratios), axis=1)
    return 1.0 / np.clip(denom, 1e-12, None)


def twonn_global_id(x: np.ndarray) -> float:
    """TwoNN global intrinsic dimension estimate."""
    if x.ndim != 2:
        raise ValueError("Input must be 2D array with shape (n_samples, n_features).")
    if x.shape[0] < 3:
        raise ValueError("TwoNN requires at least 3 samples.")

    dists = _sorted_neighbor_distances(x)
    r1 = np.clip(dists[:, 0], 1e-12, None)
    r2 = np.clip(dists[:, 1], 1e-12, None)
    mu = (r2 / np.clip(r1, 1e-12, None)).clip(min=1.0 + 1e-12)
    return float(1.0 / np.mean(np.log(mu)))


def abid_local_batch(x: np.ndarray, k: int = 20) -> np.ndarray:
    """Angle-based intrinsic dimension proxy per sample.

    This is a practical ABID-style proxy based on local angular concentration.
    """
    if x.ndim != 2:
        raise ValueError("Input must be 2D array with shape (n_samples, n_features).")
    if k < 2 or k >= x.shape[0]:
        raise ValueError("k must satisfy 2 <= k < n_samples.")

    dmat = cdist(x, x, metric="euclidean")
    np.fill_diagonal(dmat, np.inf)
    nn_idx = np.argsort(dmat, axis=1)[:, :k]

    outputs = np.zeros(x.shape[0], dtype=np.float64)
    for i in range(x.shape[0]):
        neighbors = x[nn_idx[i]] - x[i]
        norms = np.linalg.norm(neighbors, axis=1, keepdims=True).clip(min=1e-12)
        unit = neighbors / norms
        gram = unit @ unit.T
        off_diag = gram[~np.eye(k, dtype=bool)]
        angle_var = float(np.var(off_diag))
        # Low angular variance often indicates lower effective local dimension.
        outputs[i] = 1.0 / max(angle_var, 1e-8)
    return outputs
