import pandas as pd
import pytest

from nyc_demand_foundations_showcase.data import add_time_features, generate_synthetic_grouped_data


def test_synthetic_generation_and_features() -> None:
    grouped = generate_synthetic_grouped_data(n_hours=24 * 5, n_zones=12, random_state=9)
    assert {"pickup_zone_id", "pickup_hour", "pickups"}.issubset(set(grouped.columns))
    assert (grouped["pickups"] >= 0.0).all()

    featured = add_time_features(grouped)
    for col in ["hour", "day_of_week", "month", "is_weekend", "is_peak_hour"]:
        assert col in featured.columns


def test_tlc_aggregation_includes_empty_zone_hours() -> None:
    from nyc_demand_foundations_showcase.data import _group_tlc_trips

    trips = pd.DataFrame(
        {
            "tpep_pickup_datetime": ["2024-01-01 00:10", "2024-01-01 02:20"],
            "PULocationID": [1, 2],
        }
    )
    grouped = _group_tlc_trips(trips).set_index(["pickup_zone_id", "pickup_hour"])
    assert len(grouped) == 6  # Two observed zones times three consecutive hours.
    assert grouped.loc[(1, pd.Timestamp("2024-01-01 01:00")), "pickups"] == 0.0
    assert grouped.loc[(2, pd.Timestamp("2024-01-01 00:00")), "pickups"] == 0.0
    assert grouped["pickups"].sum() == 2.0


def test_tlc_aggregation_rejects_no_valid_trips() -> None:
    from nyc_demand_foundations_showcase.data import _group_tlc_trips

    trips = pd.DataFrame({"tpep_pickup_datetime": ["invalid"], "PULocationID": [1]})
    with pytest.raises(ValueError, match="No valid TLC trips"):
        _group_tlc_trips(trips)
