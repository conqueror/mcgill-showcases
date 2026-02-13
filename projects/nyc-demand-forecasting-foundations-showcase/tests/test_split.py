from nyc_demand_foundations_showcase.data import add_time_features, generate_synthetic_grouped_data
from nyc_demand_foundations_showcase.modeling import FEATURE_COLUMNS
from nyc_demand_foundations_showcase.splits import build_time_split


def test_time_split_has_three_non_overlapping_partitions() -> None:
    grouped = generate_synthetic_grouped_data(n_hours=24 * 8, n_zones=10, random_state=13)
    featured = add_time_features(grouped)

    split = build_time_split(
        featured,
        feature_columns=FEATURE_COLUMNS,
        target_column="pickups",
    )

    assert not split.train.empty
    assert not split.val.empty
    assert not split.test.empty
    assert split.train["pickup_hour"].max() < split.val["pickup_hour"].min()
    assert split.val["pickup_hour"].max() < split.test["pickup_hour"].min()
