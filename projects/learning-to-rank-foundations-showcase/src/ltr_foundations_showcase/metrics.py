from __future__ import annotations

from collections import defaultdict

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import ndcg_score


def grouped_ndcg(
    y_true: NDArray[np.float64],
    y_score: NDArray[np.float64],
    group_ids: list[str],
    *,
    k: int,
) -> float:
    """Mean query NDCG with 2**grade-1 gains; zero-IDCG queries score 1 (LightGBM)."""
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    if y_true.ndim != 1 or y_score.ndim != 1 or not len(y_true):
        raise ValueError("Expected nonempty one-dimensional relevance and score arrays.")
    if y_true.shape[0] != y_score.shape[0] or y_true.shape[0] != len(group_ids):
        raise ValueError("Mismatched lengths for y_true, y_score, or group_ids.")
    if not np.isfinite(y_true).all() or not np.isfinite(y_score).all() or (y_true < 0).any():
        raise ValueError("Relevance must be finite and nonnegative; scores must be finite.")

    groups: dict[str, list[int]] = defaultdict(list)
    for idx, group in enumerate(group_ids):
        groups[group].append(idx)

    scores: list[float] = []
    for indices in groups.values():
        if len(indices) == 1 or not y_true[indices].any():
            scores.append(1.0)
            continue
        true_block = np.expm1(y_true[indices] * np.log(2)).reshape(1, -1)
        pred_block = y_score[indices].reshape(1, -1)
        effective_k = min(k, true_block.shape[1])
        scores.append(float(ndcg_score(true_block, pred_block, k=effective_k)))

    return float(np.mean(scores))
