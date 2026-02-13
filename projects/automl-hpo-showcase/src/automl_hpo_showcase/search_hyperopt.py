from __future__ import annotations

import pandas as pd

from automl_hpo_showcase.objective import score_config


def run_hyperopt_search(*, budget: int, seed: int = 42) -> pd.DataFrame:
    try:
        from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
    except Exception:
        return pd.DataFrame(
            columns=[
                "strategy",
                "trial_id",
                "n_estimators",
                "max_depth",
                "min_samples_split",
                "score",
            ]
        )

    space = {
        "n_estimators": hp.quniform("n_estimators", 30, 150, 1),
        "max_depth": hp.quniform("max_depth", 2, 12, 1),
        "min_samples_split": hp.quniform("min_samples_split", 2, 12, 1),
    }

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
        rstate=None,
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
                "n_estimators": int(vals["n_estimators"][0]),
                "max_depth": int(vals["max_depth"][0]),
                "min_samples_split": int(vals["min_samples_split"][0]),
                "score": score,
            }
        )

    return pd.DataFrame(rows)
