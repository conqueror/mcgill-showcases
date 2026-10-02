from __future__ import annotations

import numpy as np
import pandas as pd

from automl_hpo_showcase.objective import score_config
from automl_hpo_showcase.search_space import GRID_SPACE


def run_hyperopt_search(*, budget: int, seed: int = 42) -> pd.DataFrame:
    try:
        from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
    except ImportError as exc:
        raise RuntimeError("Hyperopt was requested but could not be imported.") from exc

    space = {name: hp.choice(name, values) for name, values in GRID_SPACE.items()}

    trials = Trials()

    def objective(params: dict[str, float]) -> dict[str, float | str]:
        n_estimators = int(params["n_estimators"])
        max_depth = int(params["max_depth"])
        min_samples_split = int(params["min_samples_split"])
        score = score_config(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=seed,
        )
        return {"loss": -score, "status": STATUS_OK}

    fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=budget,
        trials=trials,
        rstate=np.random.default_rng(seed),
        show_progressbar=False,
    )

    rows: list[dict[str, float | int | str]] = []
    for trial_id, trial in enumerate(trials.trials):
        vals = trial["misc"]["vals"]
        score = float(-trial["result"]["loss"])
        rows.append(
            {
                "strategy": "hyperopt_tpe",
                "trial_id": trial_id,
                "n_estimators": GRID_SPACE["n_estimators"][int(vals["n_estimators"][0])],
                "max_depth": GRID_SPACE["max_depth"][int(vals["max_depth"][0])],
                "min_samples_split": GRID_SPACE["min_samples_split"][
                    int(vals["min_samples_split"][0])
                ],
                "score": score,
            }
        )

    return pd.DataFrame(rows)
