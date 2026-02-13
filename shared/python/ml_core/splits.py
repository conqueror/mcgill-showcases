"""Split utilities for supervised datasets and CV manifest generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit,
    KFold,
    StratifiedKFold,
    train_test_split,
)


@dataclass(frozen=True)
class SplitBundle3:
    """Container for train/validation/test feature and target partitions."""

    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


@dataclass(frozen=True)
class SplitSpec:
    """Specification metadata for a supervised split strategy."""

    strategy: str
    random_state: int
    task_type: str
    group_column: str | None = None
    time_column: str | None = None


def _assert_no_index_overlap(split: SplitBundle3) -> bool:
    train_idx = set(split.x_train.index.tolist())
    val_idx = set(split.x_val.index.tolist())
    test_idx = set(split.x_test.index.tolist())
    return not (train_idx & val_idx or train_idx & test_idx or val_idx & test_idx)


def _split_stratified_or_random(
    frame: pd.DataFrame,
    target: pd.Series,
    *,
    random_state: int,
    val_size: float,
    test_size: float,
    stratify: bool,
) -> SplitBundle3:
    strat = target if stratify else None
    x_train, x_temp, y_train, y_temp = train_test_split(
        frame,
        target,
        test_size=val_size + test_size,
        random_state=random_state,
        stratify=strat,
    )
    rel_test = test_size / (val_size + test_size)
    strat_temp = y_temp if stratify else None
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=rel_test,
        random_state=random_state,
        stratify=strat_temp,
    )
    return SplitBundle3(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )


def _split_group(
    frame: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    *,
    random_state: int,
    val_size: float,
    test_size: float,
) -> SplitBundle3:
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=val_size + test_size,
        random_state=random_state,
    )
    train_idx, temp_idx = next(splitter.split(frame, target, groups=groups))

    temp_frame = frame.iloc[temp_idx]
    temp_target = target.iloc[temp_idx]
    temp_groups = groups.iloc[temp_idx]

    rel_test = test_size / (val_size + test_size)
    splitter_2 = GroupShuffleSplit(
        n_splits=1,
        test_size=rel_test,
        random_state=random_state,
    )
    val_rel, test_rel = next(splitter_2.split(temp_frame, temp_target, groups=temp_groups))

    x_train = frame.iloc[train_idx]
    y_train = target.iloc[train_idx]
    x_val = temp_frame.iloc[val_rel]
    y_val = temp_target.iloc[val_rel]
    x_test = temp_frame.iloc[test_rel]
    y_test = temp_target.iloc[test_rel]

    return SplitBundle3(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )


def _split_timeseries(
    frame: pd.DataFrame,
    target: pd.Series,
    time_values: pd.Series,
    *,
    val_size: float,
    test_size: float,
) -> SplitBundle3:
    order = np.argsort(time_values.to_numpy())
    ordered_x = frame.iloc[order]
    ordered_y = target.iloc[order]

    n_total = len(ordered_x)
    n_test = max(1, int(round(n_total * test_size)))
    n_val = max(1, int(round(n_total * val_size)))
    n_train = n_total - n_val - n_test
    if n_train <= 0:
        raise ValueError("Not enough rows for train/val/test split with requested sizes.")

    x_train = ordered_x.iloc[:n_train]
    y_train = ordered_y.iloc[:n_train]
    x_val = ordered_x.iloc[n_train : n_train + n_val]
    y_val = ordered_y.iloc[n_train : n_train + n_val]
    x_test = ordered_x.iloc[n_train + n_val :]
    y_test = ordered_y.iloc[n_train + n_val :]

    return SplitBundle3(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )


def build_supervised_split(
    frame: pd.DataFrame,
    target: pd.Series,
    *,
    strategy: str = "stratified",
    random_state: int = 42,
    val_size: float = 0.2,
    test_size: float = 0.2,
    groups: pd.Series | None = None,
    time_values: pd.Series | None = None,
) -> SplitBundle3:
    """Build a 3-way train/validation/test split using a named strategy.

    Args:
        frame: Feature matrix.
        target: Target series aligned with ``frame``.
        strategy: One of ``stratified``, ``random``, ``group``, or ``timeseries``.
        random_state: Seed used by randomized splitting strategies.
        val_size: Validation ratio in (0, 1).
        test_size: Test ratio in (0, 1).
        groups: Optional group IDs for ``group`` strategy.
        time_values: Optional sortable timestamps for ``timeseries`` strategy.

    Returns:
        SplitBundle3 containing disjoint train/val/test partitions.

    Raises:
        ValueError: If ratios are invalid, required strategy inputs are missing,
            or an unsupported strategy is requested.
    """

    if val_size <= 0 or test_size <= 0 or (val_size + test_size) >= 1.0:
        raise ValueError("val_size and test_size must be >0 and sum to <1.")

    if strategy == "stratified":
        return _split_stratified_or_random(
            frame,
            target,
            random_state=random_state,
            val_size=val_size,
            test_size=test_size,
            stratify=True,
        )
    if strategy == "random":
        return _split_stratified_or_random(
            frame,
            target,
            random_state=random_state,
            val_size=val_size,
            test_size=test_size,
            stratify=False,
        )
    if strategy == "group":
        if groups is None:
            raise ValueError("groups must be provided for group strategy")
        return _split_group(
            frame,
            target,
            groups,
            random_state=random_state,
            val_size=val_size,
            test_size=test_size,
        )
    if strategy == "timeseries":
        if time_values is None:
            raise ValueError("time_values must be provided for timeseries strategy")
        return _split_timeseries(
            frame,
            target,
            time_values,
            val_size=val_size,
            test_size=test_size,
        )
    raise ValueError(f"Unsupported split strategy: {strategy}")


def split_manifest_dict(
    split: SplitBundle3,
    *,
    task_type: str,
    strategy: str,
    random_state: int,
    group_column: str | None = None,
    time_column: str | None = None,
) -> dict[str, int | bool | str | None]:
    """Build a JSON-serializable split manifest payload.

    The returned dictionary is written to ``artifacts/splits/split_manifest.json``
    by pipeline scripts and consumed by contract validation tooling.
    """

    return {
        "task_type": task_type,
        "strategy": strategy,
        "train_rows": int(len(split.x_train)),
        "val_rows": int(len(split.x_val)),
        "test_rows": int(len(split.x_test)),
        "group_column": group_column,
        "time_column": time_column,
        "random_state": int(random_state),
        "no_overlap_checks_passed": _assert_no_index_overlap(split),
    }


def cv_manifest_dict(
    target: pd.Series,
    *,
    strategy: str = "stratified_kfold",
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, int | bool | str | list[dict[str, int | float]]]:
    """Build fold-level metadata for CV diagnostics and contract artifacts.

    Args:
        target: Target series used to derive fold statistics.
        strategy: CV strategy, currently ``stratified_kfold`` or ``kfold``.
        n_splits: Number of folds.
        random_state: Random seed used when fold shuffling is enabled.

    Returns:
        A dictionary containing fold sizes, validation positive rates, and overlap checks.
    """

    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    dummy_features = np.zeros((len(target), 1), dtype=float)
    if strategy == "stratified_kfold":
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        iterator = splitter.split(dummy_features, target.to_numpy())
    elif strategy == "kfold":
        splitter = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        iterator = splitter.split(dummy_features)
    else:
        raise ValueError(f"Unsupported CV strategy: {strategy}")

    folds: list[dict[str, int | float]] = []
    no_overlap_checks_passed = True
    seen_validation_indices: set[int] = set()
    for fold_id, (train_idx, val_idx) in enumerate(iterator):
        train_set = set(train_idx.tolist())
        val_set = set(val_idx.tolist())
        if train_set & val_set:
            no_overlap_checks_passed = False
        if seen_validation_indices & val_set:
            no_overlap_checks_passed = False
        seen_validation_indices |= val_set

        val_target = target.iloc[val_idx]
        folds.append(
            {
                "fold_id": int(fold_id),
                "train_rows": int(len(train_idx)),
                "val_rows": int(len(val_idx)),
                "val_positive_rate": float(val_target.mean()),
            }
        )

    if len(seen_validation_indices) != len(target):
        no_overlap_checks_passed = False

    return {
        "strategy": strategy,
        "n_splits": int(n_splits),
        "random_state": int(random_state),
        "total_rows": int(len(target)),
        "no_overlap_checks_passed": no_overlap_checks_passed,
        "folds": folds,
    }
