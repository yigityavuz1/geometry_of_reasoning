from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def _validate_input_matrix(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("Input must be 2D array with shape (n_samples, n_features).")
    if arr.shape[0] < 3:
        raise ValueError("Need at least 3 samples for intrinsic-dimension estimators.")
    return arr


def _effective_k(k: int, n_samples: int) -> int:
    if n_samples < 3:
        raise ValueError("Need at least 3 samples.")
    return max(2, min(int(k), n_samples - 1))


def _parse_k_values(k_values: list[int], n_samples: int) -> list[int]:
    if not k_values:
        raise ValueError("k_values must not be empty.")
    effective = sorted({_effective_k(k, n_samples) for k in k_values})
    if not effective:
        raise ValueError("No valid k value after clipping to sample size.")
    return effective


def _sorted_neighbor_distances(x: np.ndarray) -> np.ndarray:
    distances = cdist(x, x, metric="euclidean")
    # Remove self-distance by setting diagonal to +inf.
    np.fill_diagonal(distances, np.inf)
    return np.sort(distances, axis=1)


def lid_mle_batch(x: np.ndarray, k: int = 20) -> np.ndarray:
    """Levina-Bickel local intrinsic dimension estimate for each sample."""
    arr = _validate_input_matrix(x)
    k_eff = _effective_k(k, arr.shape[0])

    dists = _sorted_neighbor_distances(arr)
    local = np.clip(dists[:, :k_eff], 1e-12, None)
    tk = local[:, -1]
    ratios = np.clip(tk[:, None] / local[:, : k_eff - 1], 1.0 + 1e-12, None)
    denom = np.mean(np.log(ratios), axis=1)
    return 1.0 / np.clip(denom, 1e-12, None)


def twonn_global_id(x: np.ndarray) -> float:
    """TwoNN global intrinsic dimension estimate."""
    arr = _validate_input_matrix(x)

    dists = _sorted_neighbor_distances(arr)
    r1 = np.clip(dists[:, 0], 1e-12, None)
    r2 = np.clip(dists[:, 1], 1e-12, None)
    mu = (r2 / np.clip(r1, 1e-12, None)).clip(min=1.0 + 1e-12)
    return float(1.0 / np.mean(np.log(mu)))


def abid_local_batch(x: np.ndarray, k: int = 20) -> np.ndarray:
    """Angle-based local intrinsic dimension estimate.

    Uses the identity E[cos(theta)^2] ~= 1 / d on isotropic data:
    local_id ~= 1 / mean(cos(theta_ij)^2), i != j over k-neighborhood.
    """
    arr = _validate_input_matrix(x)
    k_eff = _effective_k(k, arr.shape[0])

    dmat = cdist(arr, arr, metric="euclidean")
    np.fill_diagonal(dmat, np.inf)
    nn_idx = np.argsort(dmat, axis=1)[:, :k_eff]

    outputs = np.zeros(arr.shape[0], dtype=np.float64)
    for i in range(arr.shape[0]):
        neighbors = arr[nn_idx[i]] - arr[i]
        norms = np.linalg.norm(neighbors, axis=1, keepdims=True).clip(min=1e-12)
        unit = neighbors / norms
        gram = unit @ unit.T
        off_diag = gram[~np.eye(k_eff, dtype=bool)]
        cos_sq_mean = float(np.mean(np.square(off_diag)))
        outputs[i] = 1.0 / max(cos_sq_mean, 1e-12)
    return outputs


def coefficient_of_variation(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    mean = float(np.mean(arr))
    if abs(mean) < 1e-12:
        return 0.0
    return float(np.std(arr) / abs(mean))


def k_sweep_local_id(x: np.ndarray, k_values: list[int]) -> list[dict[str, float | int]]:
    arr = _validate_input_matrix(x)
    effective_k_values = _parse_k_values(k_values, arr.shape[0])
    rows: list[dict[str, float]] = []
    for k_eff in effective_k_values:
        lid = lid_mle_batch(arr, k=k_eff)
        abid = abid_local_batch(arr, k=k_eff)
        rows.append(
            {
                "k_effective": int(k_eff),
                "lid_mle_mean": float(np.mean(lid)),
                "abid_mean": float(np.mean(abid)),
            }
        )
    return rows
