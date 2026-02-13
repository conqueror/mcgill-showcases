from nyc_demand_foundations_showcase.data import add_time_features, generate_synthetic_grouped_data


def test_synthetic_generation_and_features() -> None:
    grouped = generate_synthetic_grouped_data(n_hours=24 * 5, n_zones=12, random_state=9)
    assert {"pickup_zone_id", "pickup_hour", "pickups"}.issubset(set(grouped.columns))
    assert (grouped["pickups"] >= 0.0).all()

    featured = add_time_features(grouped)
    for col in ["hour", "day_of_week", "month", "is_weekend", "is_peak_hour"]:
        assert col in featured.columns
