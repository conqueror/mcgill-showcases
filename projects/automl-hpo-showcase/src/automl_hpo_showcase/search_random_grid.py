from __future__ import annotations

from itertools import product

import pandas as pd

from automl_hpo_showcase.objective import random_config, score_config
from automl_hpo_showcase.search_space import GRID_SPACE


def run_grid_search(*, budget: int, random_state: int = 42) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for trial_id, (n_estimators, max_depth, min_samples_split) in enumerate(
        product(
            GRID_SPACE["n_estimators"],
            GRID_SPACE["max_depth"],
            GRID_SPACE["min_samples_split"],
        )
    ):
        if trial_id >= budget:
            break
        score = score_config(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
        )
        rows.append(
            {
                "strategy": "grid",
                "trial_id": trial_id,
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "score": score,
            }
        )
    return pd.DataFrame(rows)


def run_random_search(*, budget: int, seed: int = 99, random_state: int = 42) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for trial_id in range(budget):
        cfg = random_config(seed + trial_id)
        score = score_config(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_split=cfg["min_samples_split"],
            random_state=random_state,
        )
        rows.append(
            {
                "strategy": "random",
                "trial_id": trial_id,
                "n_estimators": cfg["n_estimators"],
                "max_depth": cfg["max_depth"],
                "min_samples_split": cfg["min_samples_split"],
                "score": score,
            }
        )
    return pd.DataFrame(rows)
