from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitData:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    g_train: pd.Series
    g_test: pd.Series


def make_audit_dataset(
    *,
    n_samples: int = 1800,
    n_features: int = 10,
    random_state: int = 42,
) -> SplitData:
    """Create a dataset with a synthetic sensitive attribute."""
    x_values, y_values = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=6,
        n_redundant=1,
        flip_y=0.03,
        random_state=random_state,
    )

    feature_names = [f"f_{idx}" for idx in range(n_features)]
    x_df = pd.DataFrame(x_values, columns=feature_names)
    y_series = pd.Series(y_values, name="target")

    # Sensitive attribute g in {0,1} correlated with one feature but noisy.
    rng = np.random.default_rng(random_state)
    group_signal = x_df["f_0"].to_numpy() + rng.normal(0.0, 0.7, size=n_samples)
    g_series = pd.Series((group_signal > np.median(group_signal)).astype(int), name="group")

    x_train, x_test, y_train, y_test, g_train, g_test = train_test_split(
        x_df,
        y_series,
        g_series,
        test_size=0.3,
        random_state=random_state,
        stratify=y_series,
    )

    return SplitData(
        x_train=x_train.reset_index(drop=True),
        x_test=x_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        g_train=g_train.reset_index(drop=True),
        g_test=g_test.reset_index(drop=True),
    )
