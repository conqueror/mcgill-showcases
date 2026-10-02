#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from mlops_drift_showcase.train import load_model

CSV_COLUMNS = {
    "train_eval_summary.csv": ["metric", "value"],
    "metrics_summary.csv": ["metric", "value"],
    "runs.csv": ["run_name", "timestamp_utc", "notes", "roc_auc", "accuracy"],
    "univariate_summary.csv": [
        "feature",
        "missing_ratio",
        "n_unique",
        "mean",
        "std",
        "min",
        "p50",
        "max",
        "mode",
    ],
    "bivariate_vs_target.csv": ["feature", "analysis_type", "stat", "value"],
    "missingness_summary.csv": ["feature", "missing_count", "missing_ratio"],
    "leakage_report.csv": ["check", "feature", "severity", "value"],
    "threshold_analysis.csv": ["threshold", "precision", "recall", "f1"],
    "experiment_log.csv": [
        "run_name",
        "timestamp_utc",
        "split_strategy",
        "primary_metric",
        "primary_metric_value",
        "notes",
    ],
    "drift_report.csv": ["feature", "ks_stat", "ks_pvalue", "psi", "drift_flag"],
}
TEXT_COLUMNS = {
    "metric",
    "run_name",
    "timestamp_utc",
    "notes",
    "feature",
    "mode",
    "analysis_type",
    "stat",
    "check",
    "severity",
    "split_strategy",
    "primary_metric",
    "Unnamed: 0",
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "artifacts/manifest.json"

    if not manifest_path.exists():
        raise SystemExit("Missing manifest: artifacts/manifest.json")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_files = manifest.get("required_files", [])
    if (
        not isinstance(required_files, list)
        or not required_files
        or not all(isinstance(name, str) for name in required_files)
    ):
        raise SystemExit("Manifest field 'required_files' must be a nonempty list of paths")

    missing: list[str] = []
    for file_name in required_files:
        if not (root / file_name).exists():
            missing.append(file_name)

    if missing:
        raise SystemExit(f"Missing required artifacts: {missing}")

    try:
        model = load_model(root / "artifacts/model/model.joblib")
        if not isinstance(model, Pipeline):
            raise ValueError("Model artifact must contain the training pipeline")
        features = list(model.feature_names_in_)
        for name in required_files:
            path = root / name
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Empty or invalid artifact: {name}")
            if path.suffix == ".csv":
                frame = pd.read_csv(path)
                if frame.empty:
                    raise ValueError(f"CSV artifact has no rows: {name}")
                if path.name == "train_features.csv":
                    columns = features
                elif path.name == "holdout_predictions.csv":
                    columns = features + ["y_true", "y_pred_proba", "y_pred"]
                elif path.name == "correlation_matrix.csv":
                    columns = ["Unnamed: 0", *features]
                else:
                    columns = CSV_COLUMNS[path.name]
                if frame.columns.tolist() != columns:
                    raise ValueError(f"Invalid CSV columns: {name}")
                for column in columns:
                    if column not in TEXT_COLUMNS:
                        if not np.isfinite(pd.to_numeric(frame[column])).all():
                            raise ValueError(f"Nonfinite numeric values: {name}: {column}")
            elif path.suffix == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict) or not payload:
                    raise ValueError(f"Invalid JSON object: {name}")
                if path.name == "split_manifest.json":
                    if not all(
                        type(payload.get(key)) is int and payload[key] > 0
                        for key in ("train_rows", "val_rows", "test_rows")
                    ):
                        raise ValueError("Split counts must be positive integers")
                    if payload.get("no_overlap_checks_passed") is not True:
                        raise ValueError("Split overlap check did not pass")
    except Exception as exc:
        raise SystemExit(f"Invalid artifact contents: {exc}") from exc

    print("All required artifacts have valid contents.")


if __name__ == "__main__":
    main()
