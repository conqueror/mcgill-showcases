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
    if y_true.shape[0] != y_score.shape[0] or y_true.shape[0] != len(group_ids):
        raise ValueError("Mismatched lengths for y_true, y_score, or group_ids.")

    groups: dict[str, list[int]] = defaultdict(list)
    for idx, group in enumerate(group_ids):
        groups[group].append(idx)

    scores: list[float] = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        true_block = y_true[indices].reshape(1, -1)
        pred_block = y_score[indices].reshape(1, -1)
        effective_k = min(k, true_block.shape[1])
        scores.append(float(ndcg_score(true_block, pred_block, k=effective_k)))

    if not scores:
        return 0.0
    return float(np.mean(scores))
