from ltr_foundations_showcase.data import make_synthetic_player_dataset, prepare_ranking_dataset
from ltr_foundations_showcase.split import build_group_split
from ltr_foundations_showcase.training import train_and_evaluate


def test_training_returns_ndcg_metrics() -> None:
    frame = make_synthetic_player_dataset(n_seasons=5, players_per_season=30, random_state=13)
    dataset = prepare_ranking_dataset(frame)
    split = build_group_split(dataset)

    booster, result = train_and_evaluate(split, random_state=13, quick=True)
    assert booster.params["label_gain"] == [0, 1, 3, 7]
    assert booster.params["eval_at"] == [5, 10]

    assert result.metrics["val_ndcg_at_5"] >= 0.0
    assert result.metrics["val_ndcg_at_5"] <= 1.0
    assert result.metrics["test_ndcg_at_10"] >= 0.0
    assert result.metrics["test_ndcg_at_10"] <= 1.0
