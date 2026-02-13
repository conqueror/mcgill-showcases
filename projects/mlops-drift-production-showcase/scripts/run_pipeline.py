#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlops_drift_showcase.data import generate_reference_data
from mlops_drift_showcase.tracking import append_run_tracking
from mlops_drift_showcase.train import save_model, train_and_evaluate

REQUIRED_ARTIFACTS = [
    "artifacts/metrics/train_eval_summary.csv",
    "artifacts/reference/train_features.csv",
    "artifacts/reference/holdout_predictions.csv",
    "artifacts/tracking/runs.csv",
    "artifacts/model/model.joblib",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT / "shared/python"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MLOps training and tracking pipeline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Use a smaller dataset for smoke runs")
    parser.add_argument("--with-mlflow", action="store_true")
    return parser.parse_args()


def main() -> None:
    from ml_core.contracts import (
        merge_required_files,
        write_supervised_contract_artifacts,
    )
    from ml_core.experiments import log_experiment_mlflow
    from ml_core.splits import build_supervised_split

    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    sample_size = 600 if args.quick else 1200
    bundle = generate_reference_data(n_samples=sample_size, random_state=args.seed)

    model, metrics_df, holdout_df = train_and_evaluate(
        bundle.features,
        bundle.target,
        random_state=args.seed,
    )

    metrics_path = root / "artifacts/metrics/train_eval_summary.csv"
    ref_path = root / "artifacts/reference/train_features.csv"
    holdout_path = root / "artifacts/reference/holdout_predictions.csv"
    tracking_path = root / "artifacts/tracking/runs.csv"
    model_path = root / "artifacts/model/model.joblib"
    manifest_path = root / "artifacts/manifest.json"

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    ref_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(metrics_path, index=False)
    bundle.features.to_csv(ref_path, index=False)
    holdout_df.to_csv(holdout_path, index=False)
    save_model(model, model_path)

    metric_map = {
        row["metric"]: float(row["value"])
        for row in metrics_df.to_dict(orient="records")
        if row["metric"] in {"roc_auc", "accuracy"}
    }
    append_run_tracking(
        tracking_path,
        run_name="baseline_logreg",
        metrics=metric_map,
        notes="pipeline_run",
    )

    contract_split = build_supervised_split(
        bundle.features,
        bundle.target,
        strategy="stratified",
        random_state=args.seed,
    )
    contract_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=400, random_state=args.seed)),
        ]
    )
    contract_model.fit(contract_split.x_train, contract_split.y_train)
    contract_probs = contract_model.predict_proba(contract_split.x_test)[:, 1]
    contract_preds = (contract_probs >= 0.5).astype(int)
    contract_metrics = {
        "test_roc_auc": float(roc_auc_score(contract_split.y_test, contract_probs)),
        "test_accuracy": float(accuracy_score(contract_split.y_test, contract_preds)),
    }
    contract_required = write_supervised_contract_artifacts(
        project_root=root,
        frame=bundle.features,
        target=bundle.target,
        split=contract_split,
        task_type="classification",
        strategy="stratified",
        random_state=args.seed,
        metrics=contract_metrics,
        run_name="mlops_contract_baseline",
        threshold_scores=contract_probs,
    )
    mlflow_status = log_experiment_mlflow(
        enabled=args.with_mlflow,
        run_name="mlops_drift_pipeline",
        params={
            "sample_size": sample_size,
            "seed": args.seed,
        },
        metrics=contract_metrics,
        tracking_uri=f"file://{(root / 'artifacts/mlruns').resolve()}",
        experiment_name="mcgill_mlops_drift_showcase",
    )
    (root / "artifacts/tracking/mlflow_status.txt").write_text(
        f"{mlflow_status}\n",
        encoding="utf-8",
    )

    merge_required_files(
        manifest_path,
        REQUIRED_ARTIFACTS + contract_required + ["artifacts/tracking/mlflow_status.txt"],
    )


if __name__ == "__main__":
    main()
