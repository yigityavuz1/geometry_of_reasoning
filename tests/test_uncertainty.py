from __future__ import annotations

import math

import numpy as np
import torch

from src.metrics.uncertainty import conditional_entropy_from_logits, mean_entropy_np


def test_uniform_logits_entropy_matches_log_vocab_size() -> None:
    vocab_size = 7
    logits = torch.zeros((3, vocab_size), dtype=torch.float32)
    entropy = conditional_entropy_from_logits(logits, dim=-1)
    expected = math.log(vocab_size)
    assert torch.allclose(entropy, torch.full_like(entropy, expected), atol=1e-5)


def test_peaked_distribution_has_lower_entropy() -> None:
    flat = np.zeros((1, 5), dtype=np.float64)
    peaked = np.array([[8.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    flat_entropy = float(mean_entropy_np(flat)[0])
    peaked_entropy = float(mean_entropy_np(peaked)[0])
    assert peaked_entropy < flat_entropy


def test_torch_and_numpy_entropy_are_consistent() -> None:
    logits_np = np.array([[1.2, -0.5, 0.3], [0.0, 0.0, 0.0]], dtype=np.float64)
    logits_torch = torch.tensor(logits_np, dtype=torch.float32)
    entropy_torch = conditional_entropy_from_logits(logits_torch, dim=-1).detach().cpu().numpy()
    entropy_np = mean_entropy_np(logits_np, axis=-1)
    assert np.allclose(entropy_torch, entropy_np, atol=1e-5)
