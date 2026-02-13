from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from lightgbm import LGBMRegressor
from numpy.typing import NDArray
from sklearn.metrics import mean_absolute_error, mean_squared_error

from nyc_demand_foundations_showcase.splits import TimeSplit

FEATURE_COLUMNS = [
    "pickup_zone_id",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_peak_hour",
]


@dataclass(frozen=True)
class TrainingOutput:
    model: LGBMRegressor
    val_predictions: NDArray[np.float64]
    test_predictions: NDArray[np.float64]
    metric_rows: list[dict[str, float | str]]


def _smape(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    denominator = np.abs(y_true) + np.abs(y_pred) + 1e-8
    values = 200.0 * np.abs(y_pred - y_true) / denominator
    return float(np.mean(values))


def _metric_row(
    split_name: str,
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> dict[str, float | str]:
    return {
        "split": split_name,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "smape": _smape(y_true, y_pred),
    }


def train_forecaster(
    split: TimeSplit,
    *,
    random_state: int,
    quick: bool,
) -> TrainingOutput:
    x_train = split.train[split.feature_columns]
    y_train = split.train[split.target_column]
    x_val = split.val[split.feature_columns]
    y_val = split.val[split.target_column]
    x_test = split.test[split.feature_columns]
    y_test = split.test[split.target_column]

    model = LGBMRegressor(
        objective="poisson",
        n_estimators=140 if quick else 320,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="l1",
    )

    val_pred = np.asarray(model.predict(x_val), dtype=np.float64)
    test_pred = np.asarray(model.predict(x_test), dtype=np.float64)

    rows = [
        _metric_row("val", y_val.to_numpy(dtype=np.float64), val_pred),
        _metric_row("test", y_test.to_numpy(dtype=np.float64), test_pred),
    ]

    return TrainingOutput(
        model=model,
        val_predictions=val_pred,
        test_predictions=test_pred,
        metric_rows=rows,
    )
