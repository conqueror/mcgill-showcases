"""Regression workflows for a self-guided supervised learning tutorial."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

from .config import RANDOM_STATE
from .data import RegressionSplit


def _score_regression_model(
    model: object,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    return {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_test, y_pred)),
    }


def evaluate_regression_models(split: RegressionSplit) -> pd.DataFrame:
    """Evaluate baseline, linear regression, and gradient boosting regressors."""
    models: list[tuple[str, object]] = [
        ("baseline_dummy_mean", DummyRegressor(strategy="mean")),
        ("linear_regression", LinearRegression()),
        (
            "gradient_boosting_regression",
            GradientBoostingRegressor(
                n_estimators=220,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_STATE,
            ),
        ),
    ]

    rows: list[dict[str, float | str]] = []
    for model_name, model in models:
        scores = _score_regression_model(
            model=model,
            x_train=split.x_train,
            y_train=split.y_train,
            x_test=split.x_test,
            y_test=split.y_test,
        )
        rows.append({"model": model_name, **scores})
    return pd.DataFrame(rows).sort_values("rmse")


def manual_gradient_boosting_example(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_new: np.ndarray,
) -> np.ndarray:
    """
    Reproduce the slide's "fit residuals sequentially" GBRT intuition.

    This is intentionally simple and used as a didactic code example.
    """
    tree_reg_1 = DecisionTreeRegressor(max_depth=2, random_state=RANDOM_STATE)
    tree_reg_1.fit(x_train, y_train)

    residual_1 = y_train - tree_reg_1.predict(x_train)
    tree_reg_2 = DecisionTreeRegressor(max_depth=2, random_state=RANDOM_STATE)
    tree_reg_2.fit(x_train, residual_1)

    residual_2 = residual_1 - tree_reg_2.predict(x_train)
    tree_reg_3 = DecisionTreeRegressor(max_depth=2, random_state=RANDOM_STATE)
    tree_reg_3.fit(x_train, residual_2)

    return (
        tree_reg_1.predict(x_new)
        + tree_reg_2.predict(x_new)
        + tree_reg_3.predict(x_new)
    )
