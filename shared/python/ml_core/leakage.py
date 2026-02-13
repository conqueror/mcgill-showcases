"""Leakage diagnostics shared by supervised showcase pipelines."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .splits import SplitBundle3


def _row_hashes(frame: pd.DataFrame) -> set[str]:
    hashes: set[str] = set()
    for row in frame.fillna("<NA>").astype(str).itertuples(index=False, name=None):
        payload = "|".join(row).encode("utf-8")
        hashes.add(hashlib.sha1(payload).hexdigest())
    return hashes


def _exact_target_leakage(frame: pd.DataFrame, target: pd.Series) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    y = target.reset_index(drop=True)

    for col in frame.columns:
        x = frame[col].reset_index(drop=True)

        same_mask = x.astype(str) == y.astype(str)
        exact_match_ratio = float(same_mask.mean())
        if exact_match_ratio >= 0.98:
            rows.append(
                {
                    "check": "exact_target_match",
                    "feature": str(col),
                    "severity": "high",
                    "value": exact_match_ratio,
                }
            )
            continue

        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            aligned = pd.concat([x, y], axis=1).dropna()
            if aligned.shape[0] > 2:
                corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
                if np.isfinite(corr) and abs(corr) >= 0.995:
                    rows.append(
                        {
                            "check": "near_perfect_target_corr",
                            "feature": str(col),
                            "severity": "medium",
                            "value": corr,
                        }
                    )

    return rows


def split_leakage_rows(split: SplitBundle3) -> list[dict[str, str | float]]:
    """Detect duplicate row overlap across train/val/test partitions."""

    train_hashes = _row_hashes(split.x_train)
    val_hashes = _row_hashes(split.x_val)
    test_hashes = _row_hashes(split.x_test)

    train_val_overlap = len(train_hashes & val_hashes)
    train_test_overlap = len(train_hashes & test_hashes)
    val_test_overlap = len(val_hashes & test_hashes)

    rows: list[dict[str, str | float]] = []
    for pair_name, overlap_count in (
        ("train_val", train_val_overlap),
        ("train_test", train_test_overlap),
        ("val_test", val_test_overlap),
    ):
        if overlap_count > 0:
            rows.append(
                {
                    "check": "duplicate_row_overlap",
                    "feature": pair_name,
                    "severity": "high",
                    "value": float(overlap_count),
                }
            )
    return rows


def run_leakage_checks(frame: pd.DataFrame, target: pd.Series, split: SplitBundle3) -> pd.DataFrame:
    """Run leakage checks and return a normalized diagnostics table.

    The output is written to ``artifacts/leakage/leakage_report.csv`` by
    contract-writing helpers.
    """

    rows = _exact_target_leakage(frame, target)
    rows.extend(split_leakage_rows(split))
    if not rows:
        rows.append(
            {
                "check": "no_high_risk_leakage_detected",
                "feature": "",
                "severity": "info",
                "value": 0.0,
            }
        )
    return pd.DataFrame(rows)
