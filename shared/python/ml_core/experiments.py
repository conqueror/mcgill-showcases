"""Experiment logging helpers for CSV and optional MLflow backends."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


def log_experiment_csv(
    output_path: Path,
    *,
    run_name: str,
    split_strategy: str,
    primary_metric: str,
    primary_metric_value: float,
    notes: str = "",
) -> pd.DataFrame:
    """Append one experiment row to a CSV log and return the updated table.

    Side effects:
        Creates parent directories as needed and writes the CSV to ``output_path``.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "run_name": run_name,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "split_strategy": split_strategy,
        "primary_metric": primary_metric,
        "primary_metric_value": float(primary_metric_value),
        "notes": notes,
    }

    if output_path.exists():
        frame = pd.read_csv(output_path)
    else:
        frame = pd.DataFrame()

    updated = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    updated.to_csv(output_path, index=False)
    return updated


def log_experiment_mlflow(
    *,
    enabled: bool,
    run_name: str,
    params: dict[str, int | float | str],
    metrics: dict[str, float],
    tracking_uri: str | None = None,
    experiment_name: str = "mcgill_showcases",
) -> str:
    """Log params/metrics to MLflow when enabled and dependency is present.

    Returns:
        ``disabled`` when logging is not requested,
        ``skipped_missing_dependency`` when MLflow is unavailable,
        ``logged`` when metrics are written successfully.
    """

    if not enabled:
        return "disabled"

    try:
        import mlflow
    except Exception:
        return "skipped_missing_dependency"

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

    return "logged"
