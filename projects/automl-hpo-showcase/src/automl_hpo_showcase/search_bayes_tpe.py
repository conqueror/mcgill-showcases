from __future__ import annotations

import optuna
import pandas as pd

from automl_hpo_showcase.objective import score_config


def run_tpe_search(*, budget: int, seed: int = 42) -> pd.DataFrame:
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 30, 150),
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
        }
        return score_config(**params, random_state=seed)

    study.optimize(objective, n_trials=budget)

    rows: list[dict[str, float | int | str]] = []
    for trial in study.trials:
        if trial.value is None:
            continue
        rows.append(
            {
                "strategy": "tpe",
                "trial_id": int(trial.number),
                "n_estimators": int(trial.params["n_estimators"]),
                "max_depth": int(trial.params["max_depth"]),
                "min_samples_split": int(trial.params["min_samples_split"]),
                "score": float(trial.value),
            }
        )
    return pd.DataFrame(rows)
