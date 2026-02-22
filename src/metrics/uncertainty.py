from __future__ import annotations

import numpy as np
import torch


def conditional_entropy_from_logits(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    probs = torch.softmax(logits, dim=dim)
    log_probs = torch.log(probs.clamp_min(1e-12))
    return -(probs * log_probs).sum(dim=dim)


def mean_entropy_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(logits)
    probs = exp / np.sum(exp, axis=axis, keepdims=True)
    return -np.sum(probs * np.log(np.clip(probs, 1e-12, None)), axis=axis)
