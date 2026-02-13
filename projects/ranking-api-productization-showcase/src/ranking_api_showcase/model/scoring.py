from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from lightgbm import Booster
from numpy.typing import NDArray


def build_feature_matrix(
    feature_dicts: Sequence[Mapping[str, float]],
    feature_names: Sequence[str],
) -> NDArray[np.float64]:
    expected = set(feature_names)

    for idx, features in enumerate(feature_dicts):
        extra = set(features).difference(expected)
        if extra:
            extra_preview = sorted(extra)[:20]
            raise ValueError(
                f"Record {idx} has unexpected feature keys: {extra_preview}. "
                "Call GET /model/schema to see the expected feature names."
            )

    matrix = np.zeros((len(feature_dicts), len(feature_names)), dtype=np.float64)
    for row_index, features in enumerate(feature_dicts):
        for col_index, name in enumerate(feature_names):
            matrix[row_index, col_index] = float(features.get(name, 0.0))
    return matrix


def score(booster: Booster, feature_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(booster.predict(feature_matrix), dtype=np.float64)


def argsort_desc(values: Sequence[float]) -> list[int]:
    indexed = list(enumerate(values))
    indexed.sort(key=lambda item: (-item[1], item[0]))
    return [idx for idx, _ in indexed]
