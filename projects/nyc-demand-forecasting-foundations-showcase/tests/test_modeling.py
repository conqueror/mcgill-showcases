from nyc_demand_foundations_showcase.data import add_time_features, generate_synthetic_grouped_data
from nyc_demand_foundations_showcase.modeling import FEATURE_COLUMNS, train_forecaster
from nyc_demand_foundations_showcase.splits import build_time_split


def test_training_produces_forecast_metrics() -> None:
    grouped = generate_synthetic_grouped_data(n_hours=24 * 9, n_zones=12, random_state=21)
    featured = add_time_features(grouped)
    split = build_time_split(
        featured,
        feature_columns=FEATURE_COLUMNS,
        target_column="pickups",
    )

    output = train_forecaster(split, random_state=21, quick=True)
    assert len(output.metric_rows) == 2
    for row in output.metric_rows:
        assert float(row["mae"]) >= 0.0
        assert float(row["rmse"]) >= 0.0
        assert float(row["smape"]) >= 0.0
