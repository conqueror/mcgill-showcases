from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
from numpy.typing import NDArray

from ltr_foundations_showcase.metrics import grouped_ndcg
from ltr_foundations_showcase.split import RankingSplit


@dataclass(frozen=True)
class TrainingResult:
    val_scores: NDArray[np.float64]
    test_scores: NDArray[np.float64]
    metrics: dict[str, float]


def train_and_evaluate(
    split: RankingSplit,
    *,
    random_state: int = 42,
    quick: bool = False,
) -> tuple[lgb.Booster, TrainingResult]:
    train_set = lgb.Dataset(
        split.x_train,
        label=split.y_train,
        group=split.q_train,
        feature_name=split.feature_names,
    )
    valid_set = lgb.Dataset(
        split.x_val,
        label=split.y_val,
        group=split.q_val,
        feature_name=split.feature_names,
    )

    params: dict[str, object] = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": random_state,
    }

    num_boost_round = 120 if quick else 300
    booster = lgb.train(
        params=params,
        train_set=train_set,
        valid_sets=[valid_set],
        valid_names=["valid"],
        num_boost_round=num_boost_round,
        callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)],
    )

    val_scores = np.asarray(booster.predict(split.x_val), dtype=np.float64)
    test_scores = np.asarray(booster.predict(split.x_test), dtype=np.float64)

    metrics = {
        "val_ndcg_at_5": grouped_ndcg(split.y_val, val_scores, split.val_group_ids, k=5),
        "val_ndcg_at_10": grouped_ndcg(split.y_val, val_scores, split.val_group_ids, k=10),
        "test_ndcg_at_5": grouped_ndcg(split.y_test, test_scores, split.test_group_ids, k=5),
        "test_ndcg_at_10": grouped_ndcg(split.y_test, test_scores, split.test_group_ids, k=10),
        "best_iteration": float(booster.best_iteration or booster.current_iteration()),
    }

    return booster, TrainingResult(val_scores=val_scores, test_scores=test_scores, metrics=metrics)
