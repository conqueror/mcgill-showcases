import numpy as np

from ltr_foundations_showcase.data import make_synthetic_player_dataset, prepare_ranking_dataset
from ltr_foundations_showcase.split import build_group_split


def test_group_split_shapes_and_group_sizes() -> None:
    frame = make_synthetic_player_dataset(n_seasons=5, players_per_season=24, random_state=11)
    dataset = prepare_ranking_dataset(frame)
    split = build_group_split(dataset)

    assert split.x_train.shape[0] > 0
    assert split.x_val.shape[0] > 0
    assert split.x_test.shape[0] > 0

    assert sum(split.q_train) == split.x_train.shape[0]
    assert sum(split.q_val) == split.x_val.shape[0]
    assert sum(split.q_test) == split.x_test.shape[0]

    assert np.isfinite(split.x_train).all()
    assert np.isfinite(split.x_val).all()
    assert np.isfinite(split.x_test).all()


def test_future_seasons_cannot_change_training_features() -> None:
    frame = make_synthetic_player_dataset(n_seasons=5, players_per_season=24, random_state=11)
    original = build_group_split(prepare_ranking_dataset(frame))
    held_out = frame["season"].isin(["season_2021", "season_2022"])
    frame.loc[held_out, "shots"] = 1e9
    frame.loc[held_out, "position"] = "future_only"
    modified = build_group_split(prepare_ranking_dataset(frame))
    assert original.feature_names == modified.feature_names
    np.testing.assert_array_equal(original.x_train, modified.x_train)


def test_shuffled_rows_remain_contiguous_queries() -> None:
    frame = make_synthetic_player_dataset(n_seasons=5, players_per_season=24, random_state=11)
    shuffled = frame.sample(frac=1.0, random_state=9).reset_index(drop=True)
    split = build_group_split(prepare_ranking_dataset(shuffled))
    assert split.q_train == [24, 24, 24]
    assert split.q_val == split.q_test == [24]
    assert shuffled.iloc[split.test_indices]["season"].tolist() == ["season_2022"] * 24
