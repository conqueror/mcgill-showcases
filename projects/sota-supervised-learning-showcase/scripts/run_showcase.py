"""Run end-to-end supervised learning demos and export learning artifacts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sota_supervised_showcase.classification import (
    build_classification_benchmark,
    build_model_selection_summary,
    evaluate_binary_classification,
    evaluate_multiclass_strategies,
    evaluate_multilabel_classification,
    evaluate_multioutput_denoising,
)
from sota_supervised_showcase.config import ARTIFACTS_DIR, TENANT_ID, TRACE_ID
from sota_supervised_showcase.data import load_digits_split, load_regression_split
from sota_supervised_showcase.regression import evaluate_regression_models

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT / "shared/python"))


def log_event(route: str, status: str, started_at: float) -> None:
    latency_ms = int((time.perf_counter() - started_at) * 1_000)
    payload = {
        "trace_id": TRACE_ID,
        "tenant_id": TENANT_ID,
        "route": route,
        "status": status,
        "latency_ms": latency_ms,
    }
    print(json.dumps(payload))


def render_markdown_summary(
    output_path: Path,
    binary_metrics_path: Path,
    multiclass_metrics_path: Path,
    multilabel_metrics_path: Path,
    multioutput_metrics_path: Path,
    classification_benchmark_path: Path,
    regression_benchmark_path: Path,
    model_selection_summary_path: Path,
) -> None:
    content = f"""# Supervised Learning Showcase Summary

This summary points to the generated artifacts.

## Classification artifacts
- Binary metrics: `{binary_metrics_path.name}`
- Multi-class metrics: `{multiclass_metrics_path.name}`
- Multi-label metrics: `{multilabel_metrics_path.name}`
- Multi-output metrics: `{multioutput_metrics_path.name}`
- Classifier benchmark: `{classification_benchmark_path.name}`

## Regression artifacts
- Regression benchmark: `{regression_benchmark_path.name}`

## Model selection artifacts
- Validation curve / learning curve summary JSON: `{model_selection_summary_path.name}`

Use these files to walk from baseline models to advanced ensembles.
"""
    output_path.write_text(content, encoding="utf-8")


def run(output_dir: Path) -> None:
    from ml_core.contracts import (
        merge_required_files,
        write_supervised_contract_artifacts,
    )
    from ml_core.splits import build_supervised_split

    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()
    digits_split = load_digits_split()
    binary_result = evaluate_binary_classification(digits_split)
    multiclass_df = evaluate_multiclass_strategies(digits_split)
    multilabel_df = evaluate_multilabel_classification(digits_split)
    multioutput_df = evaluate_multioutput_denoising(digits_split)
    benchmark_df = build_classification_benchmark(digits_split)
    selection_result = build_model_selection_summary(digits_split)
    log_event(route="classification_pipeline", status="ok", started_at=started_at)

    started_at = time.perf_counter()
    regression_split = load_regression_split()
    regression_df = evaluate_regression_models(regression_split)
    log_event(route="regression_pipeline", status="ok", started_at=started_at)

    binary_metrics_path = output_dir / "binary_metrics.csv"
    multiclass_metrics_path = output_dir / "multiclass_metrics.csv"
    multilabel_metrics_path = output_dir / "multilabel_metrics.csv"
    multioutput_metrics_path = output_dir / "multioutput_metrics.csv"
    classification_benchmark_path = output_dir / "classification_benchmark.csv"
    regression_benchmark_path = output_dir / "regression_benchmark.csv"
    pr_curves_path = output_dir / "pr_curves.csv"
    roc_curves_path = output_dir / "roc_curves.csv"
    validation_curve_path = output_dir / "validation_curve.csv"
    learning_curve_path = output_dir / "learning_curve.csv"
    model_selection_summary_path = output_dir / "model_selection_summary.json"
    markdown_summary_path = output_dir / "summary.md"

    binary_result.metrics.to_csv(binary_metrics_path, index=False)
    multiclass_df.to_csv(multiclass_metrics_path, index=False)
    multilabel_df.to_csv(multilabel_metrics_path, index=False)
    multioutput_df.to_csv(multioutput_metrics_path, index=False)
    benchmark_df.to_csv(classification_benchmark_path, index=False)
    regression_df.to_csv(regression_benchmark_path, index=False)
    binary_result.pr_curves.to_csv(pr_curves_path, index=False)
    binary_result.roc_curves.to_csv(roc_curves_path, index=False)
    selection_result.validation_curve.to_csv(validation_curve_path, index=False)
    selection_result.learning_curve.to_csv(learning_curve_path, index=False)
    model_selection_summary_path.write_text(
        json.dumps(selection_result.summary, indent=2),
        encoding="utf-8",
    )
    render_markdown_summary(
        output_path=markdown_summary_path,
        binary_metrics_path=binary_metrics_path,
        multiclass_metrics_path=multiclass_metrics_path,
        multilabel_metrics_path=multilabel_metrics_path,
        multioutput_metrics_path=multioutput_metrics_path,
        classification_benchmark_path=classification_benchmark_path,
        regression_benchmark_path=regression_benchmark_path,
        model_selection_summary_path=model_selection_summary_path,
    )

    started_at = time.perf_counter()
    log_event(route="artifact_export", status="ok", started_at=started_at)

    digits = load_digits()
    x_df = pd.DataFrame(
        digits.data,
        columns=[f"pixel_{index}" for index in range(digits.data.shape[1])],
    )
    y = pd.Series((digits.target == 0).astype(int), name="target")
    contract_split = build_supervised_split(
        x_df,
        y,
        strategy="stratified",
        random_state=42,
    )
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )
    model.fit(contract_split.x_train, contract_split.y_train)
    probs = model.predict_proba(contract_split.x_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    contract_metrics = {
        "test_roc_auc": float(roc_auc_score(contract_split.y_test, probs)),
        "test_f1": float(f1_score(contract_split.y_test, preds)),
    }
    contract_required = write_supervised_contract_artifacts(
        project_root=output_dir.parent,
        frame=x_df,
        target=y,
        split=contract_split,
        task_type="classification",
        strategy="stratified",
        random_state=42,
        metrics=contract_metrics,
        run_name="sota_supervised_binary_baseline",
        threshold_scores=probs,
    )

    merge_required_files(
        output_dir.parent / "artifacts/manifest.json",
        contract_required,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACTS_DIR,
        help="Directory where generated CSV/JSON/Markdown artifacts are stored.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.output_dir)
