from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

EXPECTED_COLUMNS = {
    "test_group",
    "converted",
    "total_ads",
    "most_ads_day",
    "most_ads_hour",
}


@dataclass(frozen=True)
class PreparedData:
    """Container for model-ready data."""

    X: pd.DataFrame
    treatment: np.ndarray
    outcome: np.ndarray
    feature_names: list[str]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [c.strip().lower().replace(" ", "_") for c in normalized.columns]
    return normalized


def load_marketing_ab_data(csv_path: Path) -> PreparedData:
    """
    Load Kaggle marketing A/B data and prepare features for CATE/uplift modeling.

    The expected Kaggle schema includes:
    - test_group: ad vs psa (treatment vs control)
    - converted: binary outcome
    - total_ads, most_ads_day, most_ads_hour: covariates
    """
    df = pd.read_csv(csv_path)
    df = _normalize_columns(df)

    missing = sorted(EXPECTED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{missing}. Found columns: {sorted(df.columns)}"
        )

    encoded = pd.get_dummies(
        df[["total_ads", "most_ads_hour", "most_ads_day"]],
        columns=["most_ads_day"],
        drop_first=False,
        dtype=float,
    )

    treatment = (df["test_group"].str.lower() == "ad").astype(int).to_numpy(dtype=int)
    outcome = df["converted"].astype(int).to_numpy(dtype=int)

    return PreparedData(
        X=encoded,
        treatment=treatment,
        outcome=outcome,
        feature_names=list(encoded.columns),
    )


def train_test_split_prepared(
    prepared: PreparedData,
    *,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[PreparedData, PreparedData]:
    """Create reproducible train/test splits while preserving treatment mix."""
    x_train, x_test, w_train, w_test, y_train, y_test = train_test_split(
        prepared.X,
        prepared.treatment,
        prepared.outcome,
        test_size=test_size,
        random_state=random_state,
        stratify=prepared.treatment,
    )

    train_data = PreparedData(
        X=x_train.reset_index(drop=True),
        treatment=w_train,
        outcome=y_train,
        feature_names=prepared.feature_names,
    )
    test_data = PreparedData(
        X=x_test.reset_index(drop=True),
        treatment=w_test,
        outcome=y_test,
        feature_names=prepared.feature_names,
    )
    return train_data, test_data
