#!/usr/bin/env python
"""Run the NYC demand forecasting foundations pipeline.

Outputs:
    Writes training samples, fitted model bundle, forecast metrics, prediction
    examples, split manifest, and required artifact manifest.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd

from nyc_demand_foundations_showcase.data import add_time_features, load_grouped_data
from nyc_demand_foundations_showcase.modeling import FEATURE_COLUMNS, train_forecaster
from nyc_demand_foundations_showcase.splits import TimeSplit, build_time_split

REQUIRED_FILES = [
    "artifacts/data/training_dataset_sample.csv",
    "artifacts/model/model.joblib",
    "artifacts/model/model_meta.json",
    "artifacts/eval/metrics_summary.csv",
    "artifacts/eval/prediction_examples.csv",
    "artifacts/splits/time_split_manifest.json",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI options for quick/full and synthetic/real-data runs."""

    parser = argparse.ArgumentParser(description="Run NYC demand forecasting showcase pipeline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--data-path", type=Path, default=None)
    return parser.parse_args()


def _split_manifest(split: TimeSplit) -> dict[str, object]:
    """Build JSON payload describing chronological train/val/test boundaries."""

    return {
        "strategy": "time_ordered_train_val_test",
        "time_column": "pickup_hour",
        "train_rows": int(split.train.shape[0]),
        "val_rows": int(split.val.shape[0]),
        "test_rows": int(split.test.shape[0]),
        "train_start": str(split.train["pickup_hour"].min()),
        "train_end": str(split.train["pickup_hour"].max()),
        "val_start": str(split.val["pickup_hour"].min()),
        "val_end": str(split.val["pickup_hour"].max()),
        "test_start": str(split.test["pickup_hour"].min()),
        "test_end": str(split.test["pickup_hour"].max()),
        "feature_columns": split.feature_columns,
        "target_column": split.target_column,
    }


def main() -> None:
    """Execute data preparation, model training, and artifact persistence."""

    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    grouped_bundle = load_grouped_data(
        data_path=args.data_path,
        quick=args.quick,
        random_state=args.seed,
    )
    featured = add_time_features(grouped_bundle.frame)

    split = build_time_split(
        featured,
        feature_columns=FEATURE_COLUMNS,
        target_column="pickups",
    )
    output = train_forecaster(split, random_state=args.seed, quick=args.quick)

    data_dir = root / "artifacts/data"
    model_dir = root / "artifacts/model"
    eval_dir = root / "artifacts/eval"
    split_dir = root / "artifacts/splits"
    for directory in [data_dir, model_dir, eval_dir, split_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    featured.head(500).to_csv(data_dir / "training_dataset_sample.csv", index=False)

    joblib.dump(output.model, model_dir / "model.joblib")
    model_meta = {
        "source": grouped_bundle.source,
        "trained_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "features": FEATURE_COLUMNS,
        "task": "demand_forecasting",
    }
    (model_dir / "model_meta.json").write_text(json.dumps(model_meta, indent=2), encoding="utf-8")

    metrics_df = pd.DataFrame(output.metric_rows)
    metrics_df.to_csv(eval_dir / "metrics_summary.csv", index=False)

    pred_examples = split.test[["pickup_hour", "pickup_zone_id", "pickups"]].copy()
    pred_examples["predicted_pickups"] = output.test_predictions
    pred_examples["residual"] = pred_examples["predicted_pickups"] - pred_examples["pickups"]
    pred_examples = pred_examples.sort_values("pickup_hour").head(300)
    pred_examples.to_csv(eval_dir / "prediction_examples.csv", index=False)

    split_payload = _split_manifest(split)
    split_payload["source"] = grouped_bundle.source
    (split_dir / "time_split_manifest.json").write_text(
        json.dumps(split_payload, indent=2),
        encoding="utf-8",
    )

    (root / "artifacts/manifest.json").write_text(
        json.dumps({"version": 1, "required_files": REQUIRED_FILES}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
