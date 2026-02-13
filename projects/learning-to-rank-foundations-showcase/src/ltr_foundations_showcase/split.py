from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ltr_foundations_showcase.data import RankingDataset


@dataclass(frozen=True)
class RankingSplit:
    x_train: NDArray[np.float64]
    y_train: NDArray[np.float64]
    q_train: list[int]
    x_val: NDArray[np.float64]
    y_val: NDArray[np.float64]
    q_val: list[int]
    x_test: NDArray[np.float64]
    y_test: NDArray[np.float64]
    q_test: list[int]
    val_group_ids: list[str]
    test_group_ids: list[str]
    train_groups: list[str]
    val_groups: list[str]
    test_groups: list[str]
    feature_names: list[str]
    test_indices: NDArray[np.int64]


def compute_group_sizes(group_values: list[str]) -> list[int]:
    sizes: list[int] = []
    current: str | None = None
    count = 0

    for value in group_values:
        if current is None:
            current = value
            count = 1
            continue
        if value == current:
            count += 1
            continue
        sizes.append(count)
        current = value
        count = 1

    if count > 0:
        sizes.append(count)

    return sizes


def build_group_split(dataset: RankingDataset, *, group_col: str = "season") -> RankingSplit:
    groups = sorted(dataset.frame[group_col].astype(str).unique().tolist())
    if len(groups) < 4:
        raise ValueError("Need at least 4 groups for grouped train/val/test split.")

    train_groups = groups[:-2]
    val_groups = [groups[-2]]
    test_groups = [groups[-1]]

    group_values = dataset.frame[group_col].astype(str)
    train_mask = group_values.isin(train_groups)
    val_mask = group_values.isin(val_groups)
    test_mask = group_values.isin(test_groups)

    matrix = dataset.feature_frame.to_numpy(dtype=np.float64)
    relevance = dataset.relevance.to_numpy(dtype=np.float64)

    train_group_values = group_values[train_mask].tolist()
    val_group_values = group_values[val_mask].tolist()
    test_group_values = group_values[test_mask].tolist()

    return RankingSplit(
        x_train=matrix[train_mask.to_numpy()],
        y_train=relevance[train_mask.to_numpy()],
        q_train=compute_group_sizes(train_group_values),
        x_val=matrix[val_mask.to_numpy()],
        y_val=relevance[val_mask.to_numpy()],
        q_val=compute_group_sizes(val_group_values),
        x_test=matrix[test_mask.to_numpy()],
        y_test=relevance[test_mask.to_numpy()],
        q_test=compute_group_sizes(test_group_values),
        val_group_ids=val_group_values,
        test_group_ids=test_group_values,
        train_groups=train_groups,
        val_groups=val_groups,
        test_groups=test_groups,
        feature_names=dataset.feature_names,
        test_indices=np.flatnonzero(test_mask.to_numpy()),
    )
