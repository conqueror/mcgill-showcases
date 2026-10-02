#!/usr/bin/env python
"""Validate required artifacts declared in ``artifacts/manifest.json``."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    """Fail with missing-path details when required outputs are absent."""

    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "artifacts/manifest.json").read_text(encoding="utf-8"))
    required = manifest.get("required_files")
    if not isinstance(required, list) or not required or not all(
        isinstance(rel, str) and rel for rel in required
    ):
        raise SystemExit("required_files must be a nonempty list of paths.")
    missing = [rel for rel in required if not (root / rel).is_file()
               or (root / rel).stat().st_size == 0]
    if missing:
        raise SystemExit(f"Missing required artifacts: {missing}")
    metrics = pd.read_csv(root / "artifacts/eval/metrics_summary.csv")
    columns = ["mae", "rmse", "smape"]
    if metrics.empty or not {"model", "split", *columns}.issubset(metrics.columns):
        raise SystemExit("Forecast metrics must contain model, split, mae, rmse, smape.")
    if set(zip(metrics["model"], metrics["split"], strict=True)) != {
        ("lightgbm", "val"), ("lightgbm", "test"),
        ("last_train_naive", "val"), ("last_train_naive", "test"),
    }:
        raise SystemExit("Missing validation/test metrics for the model or naive baseline.")
    values = metrics[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise SystemExit("Forecast metrics must be finite and nonnegative.")
    split = json.loads((root / "artifacts/splits/time_split_manifest.json").read_text())
    bounds = []
    for name in ["train", "val", "test"]:
        count = split.get(f"{name}_rows")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise SystemExit("Split row counts must be positive integers.")
        start = datetime.fromisoformat(split[f"{name}_start"])
        end = datetime.fromisoformat(split[f"{name}_end"])
        if start > end:
            raise SystemExit("Split start must not follow its end.")
        bounds.append((start, end))
    if not bounds[0][1] < bounds[1][0] or not bounds[1][1] < bounds[2][0]:
        raise SystemExit("Time split boundaries must be strictly separated.")


if __name__ == "__main__":
    main()
