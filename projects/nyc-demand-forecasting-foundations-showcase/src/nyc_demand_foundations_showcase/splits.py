from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]
    target_column: str


def build_time_split(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> TimeSplit:
    if not (0.0 < train_frac < 1.0):
        raise ValueError("train_frac must be in (0, 1)")
    if not (0.0 < val_frac < 1.0):
        raise ValueError("val_frac must be in (0, 1)")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1")

    if "pickup_hour" not in frame.columns:
        raise KeyError("Expected pickup_hour column.")

    unique_hours = frame["pickup_hour"].drop_duplicates().sort_values().to_list()
    if len(unique_hours) < 12:
        raise ValueError("Need at least 12 unique pickup hours for train/val/test splitting.")

    train_cut = max(int(len(unique_hours) * train_frac), 1)
    val_cut = max(int(len(unique_hours) * (train_frac + val_frac)), train_cut + 1)
    val_cut = min(val_cut, len(unique_hours) - 1)

    train_end = unique_hours[train_cut - 1]
    val_end = unique_hours[val_cut - 1]

    train_mask = frame["pickup_hour"] <= train_end
    val_mask = (frame["pickup_hour"] > train_end) & (frame["pickup_hour"] <= val_end)
    test_mask = frame["pickup_hour"] > val_end

    train = frame.loc[train_mask].copy()
    val = frame.loc[val_mask].copy()
    test = frame.loc[test_mask].copy()

    if train.empty or val.empty or test.empty:
        raise ValueError("Time split produced an empty partition.")

    return TimeSplit(
        train=train,
        val=val,
        test=test,
        feature_columns=feature_columns,
        target_column=target_column,
    )
