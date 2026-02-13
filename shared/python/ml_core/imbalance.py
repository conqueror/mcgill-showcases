"""Imbalance handling utilities for binary classification showcases."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.utils import resample


def resample_binary(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    method: str,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Resample a binary classification training set using a named strategy.

    Supported methods:
    - ``none``
    - ``upsample_minority``
    - ``downsample_majority``
    - ``smote``
    - ``smotetomek``
    - ``smoteenn``
    """

    if method == "none":
        return x_train, y_train

    if method in {"upsample_minority", "downsample_majority"}:
        counts = y_train.value_counts(dropna=False)
        if len(counts) != 2:
            raise ValueError("Binary resampling expects exactly two target classes.")
        minority_label = counts.idxmin()
        majority_label = counts.idxmax()

        minority_mask = y_train == minority_label
        x_minority = x_train.loc[minority_mask]
        y_minority = y_train.loc[minority_mask]
        x_majority = x_train.loc[~minority_mask]
        y_majority = y_train.loc[~minority_mask]

        if method == "upsample_minority":
            x_minority_up, y_minority_up = resample(
                x_minority,
                y_minority,
                replace=True,
                n_samples=len(y_majority),
                random_state=random_state,
            )
            out_x = pd.concat([x_majority, x_minority_up], axis=0)
            out_y = pd.concat([y_majority, y_minority_up], axis=0)
            return out_x.reset_index(drop=True), out_y.reset_index(drop=True)

        x_majority_down, y_majority_down = resample(
            x_majority,
            y_majority,
            replace=False,
            n_samples=len(y_minority),
            random_state=random_state,
        )
        out_x = pd.concat([x_minority, x_majority_down], axis=0)
        out_y = pd.concat([y_minority, y_majority_down], axis=0)
        return out_x.reset_index(drop=True), out_y.reset_index(drop=True)

    try:
        from imblearn.combine import SMOTEENN, SMOTETomek
        from imblearn.over_sampling import SMOTE
    except Exception as exc:
        raise RuntimeError(
            "Imbalanced-learn is required for SMOTE methods. Install optional extras."
        ) from exc

    x_np = x_train.to_numpy()
    y_np = y_train.to_numpy()

    if method == "smote":
        sampler = SMOTE(random_state=random_state)
    elif method == "smotetomek":
        sampler = SMOTETomek(random_state=random_state)
    elif method == "smoteenn":
        sampler = SMOTEENN(random_state=random_state)
    else:
        raise ValueError(f"Unsupported method: {method}")

    x_res, y_res = sampler.fit_resample(x_np, y_np)
    x_df = pd.DataFrame(x_res, columns=x_train.columns)
    y_series = pd.Series(y_res, name=y_train.name)
    return x_df, y_series


def class_balance(y: pd.Series) -> pd.DataFrame:
    """Return class counts and ratios for quick imbalance diagnostics."""

    counts = y.value_counts(dropna=False).sort_index()
    total = max(1, int(counts.sum()))
    return pd.DataFrame(
        {
            "class": counts.index.to_list(),
            "count": counts.to_list(),
            "ratio": [float(v / total) for v in counts.to_list()],
        }
    )
