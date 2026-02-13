#!/usr/bin/env python
"""Run the end-to-end learning-to-rank foundations pipeline.

Outputs:
    Writes dataset samples, model artifacts, evaluation tables, split manifest,
    and ``artifacts/manifest.json`` required files list.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ltr_foundations_showcase.data import make_synthetic_player_dataset, prepare_ranking_dataset
from ltr_foundations_showcase.split import build_group_split
from ltr_foundations_showcase.training import train_and_evaluate

REQUIRED_FILES = [
    "artifacts/data/ranking_dataset_sample.csv",
    "artifacts/data/feature_schema.json",
    "artifacts/model/model.txt",
    "artifacts/model/model_meta.json",
    "artifacts/eval/ranking_metrics.json",
    "artifacts/eval/test_rankings_top10.csv",
    "artifacts/splits/group_split_manifest.json",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for reproducible ranking pipeline runs."""

    parser = argparse.ArgumentParser(description="Run learning-to-rank foundations pipeline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Execute the ranking data prep, training, evaluation, and artifact writes."""

    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    n_seasons = 5 if args.quick else 7
    players_per_season = 80 if args.quick else 140

    frame = make_synthetic_player_dataset(
        n_seasons=n_seasons,
        players_per_season=players_per_season,
        random_state=args.seed,
    )
    dataset = prepare_ranking_dataset(frame)
    split = build_group_split(dataset)
    booster, result = train_and_evaluate(split, random_state=args.seed, quick=args.quick)

    data_dir = root / "artifacts/data"
    model_dir = root / "artifacts/model"
    eval_dir = root / "artifacts/eval"
    split_dir = root / "artifacts/splits"
    for directory in [data_dir, model_dir, eval_dir, split_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    dataset.frame.head(200).to_csv(data_dir / "ranking_dataset_sample.csv", index=False)
    (data_dir / "feature_schema.json").write_text(
        json.dumps({"feature_names": split.feature_names}, indent=2),
        encoding="utf-8",
    )

    booster.save_model(str(model_dir / "model.txt"))
    (model_dir / "model_meta.json").write_text(
        json.dumps(
            {
                "task": "learning_to_rank",
                "objective": "lambdarank",
                "train_groups": split.train_groups,
                "val_groups": split.val_groups,
                "test_groups": split.test_groups,
                "seed": args.seed,
                "best_iteration": result.metrics["best_iteration"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (eval_dir / "ranking_metrics.json").write_text(
        json.dumps(result.metrics, indent=2),
        encoding="utf-8",
    )

    test_rankings = dataset.frame.iloc[split.test_indices][["season", "player_id", "points"]].copy()
    test_rankings["pred_score"] = result.test_scores
    test_rankings["pred_rank"] = test_rankings.groupby("season")["pred_score"].rank(
        ascending=False,
        method="first",
    )
    top10 = (
        test_rankings.sort_values(["season", "pred_rank"])  # deterministic output for demos
        .groupby("season", group_keys=False)
        .head(10)
        .reset_index(drop=True)
    )
    top10.to_csv(eval_dir / "test_rankings_top10.csv", index=False)

    split_manifest = {
        "strategy": "grouped_time_order",
        "group_column": "season",
        "train_groups": split.train_groups,
        "val_groups": split.val_groups,
        "test_groups": split.test_groups,
        "train_rows": int(split.x_train.shape[0]),
        "val_rows": int(split.x_val.shape[0]),
        "test_rows": int(split.x_test.shape[0]),
        "q_train": split.q_train,
        "q_val": split.q_val,
        "q_test": split.q_test,
    }
    (split_dir / "group_split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2),
        encoding="utf-8",
    )

    (root / "artifacts/manifest.json").write_text(
        json.dumps({"version": 1, "required_files": REQUIRED_FILES}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
