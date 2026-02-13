from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def summarize_trials(trials: pd.DataFrame) -> pd.DataFrame:
    grouped = trials.groupby("strategy", as_index=False).agg(
        best_score=("score", "max"),
        mean_score=("score", "mean"),
        n_trials=("trial_id", "count"),
    )
    return grouped.sort_values(by="best_score", ascending=False).reset_index(drop=True)


def write_best_configs(trials: pd.DataFrame, output_path: Path) -> None:
    payload: dict[str, dict[str, float | int]] = {}
    for strategy in sorted(trials["strategy"].unique()):
        best = (
            trials[trials["strategy"] == strategy].sort_values(by="score", ascending=False).iloc[0]
        )
        payload[strategy] = {
            "score": float(best["score"]),
            "n_estimators": int(best["n_estimators"]),
            "max_depth": int(best["max_depth"]),
            "min_samples_split": int(best["min_samples_split"]),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
