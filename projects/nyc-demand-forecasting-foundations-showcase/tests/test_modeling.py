import numpy as np
import pandas as pd
import pytest

from nyc_demand_foundations_showcase.data import add_time_features, generate_synthetic_grouped_data
from nyc_demand_foundations_showcase.modeling import FEATURE_COLUMNS, train_forecaster
from nyc_demand_foundations_showcase.splits import TimeSplit, build_time_split


def test_training_produces_forecast_metrics() -> None:
    grouped = generate_synthetic_grouped_data(n_hours=24 * 9, n_zones=12, random_state=21)
    featured = add_time_features(grouped)
    split = build_time_split(
        featured,
        feature_columns=FEATURE_COLUMNS,
        target_column="pickups",
    )

    output = train_forecaster(split, random_state=21, quick=True)
    assert {(row["model"], row["split"]) for row in output.metric_rows} == {
        ("lightgbm", "val"), ("lightgbm", "test"),
        ("last_train_naive", "val"), ("last_train_naive", "test"),
    }
    for row in output.metric_rows:
        assert float(row["mae"]) >= 0.0
        assert float(row["rmse"]) >= 0.0
        assert float(row["smape"]) >= 0.0


def test_naive_forecast_uses_last_training_count_and_exact_metrics() -> None:
    train = pd.DataFrame({"pickup_zone_id": [1, 1], "pickups": [1.0, 2.0]})
    train["pickup_hour"] = pd.date_range("2024-01-01", periods=2, freq="h")
    val = pd.DataFrame({"pickup_zone_id": [1, 1], "pickups": [0.0, 4.0]})
    split = TimeSplit(train, val, val.copy(), ["pickup_zone_id"], "pickups")
    result = train_forecaster(split, random_state=1, quick=True)
    row = next(r for r in result.metric_rows if r["model"] == "last_train_naive")
    # Forecast [2, 2]: absolute errors [2, 2], squared errors [4, 4].
    # sMAPE = (200 * 2/2 + 200 * 2/6) / 2 = 400/3 percent.
    assert row["mae"] == 2.0
    assert row["rmse"] == 2.0
    assert float(row["smape"]) == pytest.approx(400.0 / 3.0)
    changed = TimeSplit(train, val, val.assign(pickups=[9.0, 8.0]),
                        ["pickup_zone_id"], "pickups")
    changed_result = train_forecaster(changed, random_state=1, quick=True)
    np.testing.assert_array_equal(result.val_predictions, changed_result.val_predictions)
    assert result.model.booster_.model_to_string() == (
        changed_result.model.booster_.model_to_string()
    )
