"""Artifact contract writers and manifest helpers for supervised showcases."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from .eda import bivariate_vs_target, correlation_matrix, missingness_summary, univariate_summary
from .experiments import log_experiment_csv
from .leakage import run_leakage_checks
from .splits import SplitBundle3, split_manifest_dict


def _threshold_table(y_true: pd.Series, y_score: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    thresholds = np.linspace(0.1, 0.9, 9)
    y_true_np = y_true.to_numpy()
    for threshold in thresholds:
        preds = (y_score >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(y_true_np, preds, zero_division=0)),
                "recall": float(recall_score(y_true_np, preds, zero_division=0)),
                "f1": float(f1_score(y_true_np, preds, zero_division=0)),
            }
        )
    return pd.DataFrame(rows)


def write_supervised_contract_artifacts(
    *,
    project_root: Path,
    frame: pd.DataFrame,
    target: pd.Series,
    split: SplitBundle3,
    task_type: str,
    strategy: str,
    random_state: int,
    metrics: dict[str, float],
    run_name: str,
    threshold_scores: np.ndarray | None = None,
    group_column: str | None = None,
    time_column: str | None = None,
) -> list[str]:
    """Write required supervised artifact files and return required manifest entries.

    Side effects:
        Writes split, EDA, leakage, evaluation, and experiment artifacts under
        ``project_root / artifacts``.
    """

    required: list[str] = []

    split_manifest = split_manifest_dict(
        split,
        task_type=task_type,
        strategy=strategy,
        random_state=random_state,
        group_column=group_column,
        time_column=time_column,
    )
    split_manifest_path = project_root / "artifacts/splits/split_manifest.json"
    split_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    split_manifest_path.write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    required.append("artifacts/splits/split_manifest.json")

    univariate = univariate_summary(frame)
    bivariate = bivariate_vs_target(frame, target)
    missingness = missingness_summary(frame)
    corr = correlation_matrix(frame)

    univariate_path = project_root / "artifacts/eda/univariate_summary.csv"
    bivariate_path = project_root / "artifacts/eda/bivariate_vs_target.csv"
    missingness_path = project_root / "artifacts/eda/missingness_summary.csv"
    corr_path = project_root / "artifacts/eda/correlation_matrix.csv"

    univariate_path.parent.mkdir(parents=True, exist_ok=True)
    univariate.to_csv(univariate_path, index=False)
    bivariate.to_csv(bivariate_path, index=False)
    missingness.to_csv(missingness_path, index=False)
    corr.to_csv(corr_path, index=True)

    required.extend(
        [
            "artifacts/eda/univariate_summary.csv",
            "artifacts/eda/bivariate_vs_target.csv",
            "artifacts/eda/missingness_summary.csv",
            "artifacts/eda/correlation_matrix.csv",
        ]
    )

    leakage_path = project_root / "artifacts/leakage/leakage_report.csv"
    leakage_path.parent.mkdir(parents=True, exist_ok=True)
    run_leakage_checks(frame, target, split).to_csv(leakage_path, index=False)
    required.append("artifacts/leakage/leakage_report.csv")

    eval_path = project_root / "artifacts/eval/metrics_summary.csv"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"metric": k, "value": float(v)} for k, v in metrics.items()]).to_csv(
        eval_path,
        index=False,
    )
    required.append("artifacts/eval/metrics_summary.csv")

    is_binary_target = int(split.y_test.nunique()) == 2
    if task_type == "classification" and threshold_scores is not None and is_binary_target:
        threshold_path = project_root / "artifacts/eval/threshold_analysis.csv"
        threshold_table = _threshold_table(split.y_test, threshold_scores)
        threshold_table.to_csv(threshold_path, index=False)
        required.append("artifacts/eval/threshold_analysis.csv")

    exp_path = project_root / "artifacts/experiments/experiment_log.csv"
    primary_metric, primary_metric_value = next(iter(metrics.items()))
    log_experiment_csv(
        exp_path,
        run_name=run_name,
        split_strategy=strategy,
        primary_metric=primary_metric,
        primary_metric_value=float(primary_metric_value),
        notes="supervised_contract",
    )
    required.append("artifacts/experiments/experiment_log.csv")

    return required


def merge_required_files(manifest_path: Path, required_files: list[str]) -> None:
    """Merge required artifact paths into ``artifacts/manifest.json``."""

    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        payload = {"version": 1, "required_files": []}

    merged = set(payload.get("required_files", []))
    merged.update(required_files)
    payload["required_files"] = sorted(merged)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
