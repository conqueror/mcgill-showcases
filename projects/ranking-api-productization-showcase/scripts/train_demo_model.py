#!/usr/bin/env python
"""Train a deterministic synthetic ranking model for API demonstrations."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np


def _write_json(path: Path, payload: object) -> None:
    """Write JSON payload with stable formatting for version control."""

    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def train_demo_model(out_dir: Path) -> None:
    """Train and persist a LightGBM LambdaRank demo model under ``out_dir``."""

    out_dir.mkdir(parents=True, exist_ok=True)

    feature_names = ["goals", "assists", "shots", "toi_minutes", "plus_minus"]

    rng = np.random.default_rng(7)
    seasons = ["2019-2020", "2020-2021", "2021-2022"]
    players_per_season = 60

    rows: list[list[float]] = []
    labels: list[float] = []
    group_sizes: list[int] = []

    for season in seasons:
        _ = season
        group_sizes.append(players_per_season)

        goals = rng.poisson(lam=18, size=players_per_season)
        assists = rng.poisson(lam=22, size=players_per_season)
        shots = rng.poisson(lam=150, size=players_per_season)
        toi = rng.normal(loc=15.0, scale=3.0, size=players_per_season).clip(min=5.0)
        plus_minus = rng.normal(loc=0.0, scale=10.0, size=players_per_season)

        for idx in range(players_per_season):
            rows.append(
                [
                    float(goals[idx]),
                    float(assists[idx]),
                    float(shots[idx]),
                    float(toi[idx]),
                    float(plus_minus[idx]),
                ]
            )
            points = goals[idx] + assists[idx]
            labels.append(float(min(points // 10, 3)))

    matrix = np.asarray(rows, dtype=np.float64)
    target = np.asarray(labels, dtype=np.float64)

    train_set = lgb.Dataset(matrix, label=target, group=group_sizes, feature_name=feature_names)
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
        "seed": 7,
    }

    booster = lgb.train(params=params, train_set=train_set, num_boost_round=100)
    booster.save_model(str(out_dir / "model.txt"))

    _write_json(out_dir / "feature_names.json", feature_names)
    _write_json(
        out_dir / "model_meta.json",
        {
            "kind": "demo",
            "trained_at": datetime.now(UTC).isoformat(),
            "objective": params["objective"],
            "feature_count": len(feature_names),
            "notes": "Synthetic ranking model for API integration demos.",
        },
    )


def main() -> None:
    """CLI entrypoint for demo model training."""

    parser = argparse.ArgumentParser(description="Train a synthetic LightGBM ranking model.")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    train_demo_model(out_dir=args.out_dir)


if __name__ == "__main__":
    main()
