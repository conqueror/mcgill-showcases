from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def make_synthetic_dataset(*, random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    x_values, y_values = make_classification(
        n_samples=1400,
        n_features=16,
        n_informative=10,
        n_redundant=2,
        random_state=random_state,
    )
    columns = [f"f_{idx}" for idx in range(x_values.shape[1])]
    return pd.DataFrame(x_values, columns=columns), pd.Series(y_values, name="target")


def _split_three_way(
    x_values: pd.DataFrame,
    y_values: pd.Series,
    *,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    x_train, x_temp, y_train, y_temp = train_test_split(
        x_values,
        y_values,
        test_size=0.4,
        random_state=random_state,
        stratify=y_values,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        random_state=random_state,
        stratify=y_temp,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def score_config(
    *,
    n_estimators: int,
    max_depth: int,
    min_samples_split: int,
    random_state: int = 42,
) -> float:
    x_values, y_values = make_synthetic_dataset(random_state=random_state)
    x_train, x_val, _, y_train, y_val, _ = _split_three_way(
        x_values,
        y_values,
        random_state=random_state,
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(x_train, y_train)
    probs = model.predict_proba(x_val)[:, 1]
    return float(roc_auc_score(y_val, probs))


def evaluate_on_test(
    *,
    n_estimators: int,
    max_depth: int,
    min_samples_split: int,
    random_state: int = 42,
) -> float:
    x_values, y_values = make_synthetic_dataset(random_state=random_state)
    x_train, _, x_test, y_train, _, y_test = _split_three_way(
        x_values,
        y_values,
        random_state=random_state,
    )
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(x_train, y_train)
    probs = model.predict_proba(x_test)[:, 1]
    return float(roc_auc_score(y_test, probs))


def random_config(seed: int) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    return {
        "n_estimators": int(rng.integers(30, 151)),
        "max_depth": int(rng.integers(2, 13)),
        "min_samples_split": int(rng.integers(2, 13)),
    }
