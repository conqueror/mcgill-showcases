from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
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

    order = np.argsort(group_values.to_numpy(), kind="stable")
    train_indices = order[train_mask.to_numpy()[order]]
    val_indices = order[val_mask.to_numpy()[order]]
    test_indices = order[test_mask.to_numpy()[order]]

    medians = dataset.feature_frame.iloc[train_indices].select_dtypes(include="number").median()
    features = dataset.feature_frame.fillna(medians.fillna(0.0)).fillna("UNKNOWN")
    categorical = list(features.select_dtypes(exclude="number").columns)
    training_columns = pd.get_dummies(features.iloc[train_indices], columns=categorical).columns
    encoded = pd.get_dummies(features, columns=categorical).reindex(
        columns=training_columns, fill_value=0
    )
    matrix = encoded.to_numpy(dtype=np.float64)
    relevance = dataset.relevance.to_numpy(dtype=np.float64)

    train_group_values = group_values.iloc[train_indices].tolist()
    val_group_values = group_values.iloc[val_indices].tolist()
    test_group_values = group_values.iloc[test_indices].tolist()

    return RankingSplit(
        x_train=matrix[train_indices],
        y_train=relevance[train_indices],
        q_train=compute_group_sizes(train_group_values),
        x_val=matrix[val_indices],
        y_val=relevance[val_indices],
        q_val=compute_group_sizes(val_group_values),
        x_test=matrix[test_indices],
        y_test=relevance[test_indices],
        q_test=compute_group_sizes(test_group_values),
        val_group_ids=val_group_values,
        test_group_ids=test_group_values,
        train_groups=train_groups,
        val_groups=val_groups,
        test_groups=test_groups,
        feature_names=list(encoded.columns),
        test_indices=test_indices,
    )
