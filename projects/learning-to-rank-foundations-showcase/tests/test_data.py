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

    assert not dataset.feature_frame.isna().any().any()
