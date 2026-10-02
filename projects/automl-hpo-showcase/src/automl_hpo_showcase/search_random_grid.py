from __future__ import annotations

from itertools import product

import pandas as pd

from automl_hpo_showcase.objective import random_config, score_config
from automl_hpo_showcase.search_space import GRID_SPACE


def run_grid_search(*, budget: int, random_state: int = 42) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    # Cycle all levels before revisiting them, so short budgets cover each parameter.
    for trial_id, (depth_offset, split_offset, tree_index) in enumerate(
        product(
            range(len(GRID_SPACE["max_depth"])),
            range(len(GRID_SPACE["min_samples_split"])),
            range(len(GRID_SPACE["n_estimators"])),
        )
    ):
        if trial_id >= budget:
            break
        n_estimators = GRID_SPACE["n_estimators"][tree_index]
        max_depth = GRID_SPACE["max_depth"][
            (tree_index + depth_offset) % len(GRID_SPACE["max_depth"])
        ]
        min_samples_split = GRID_SPACE["min_samples_split"][
            (tree_index + split_offset) % len(GRID_SPACE["min_samples_split"])
        ]
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
