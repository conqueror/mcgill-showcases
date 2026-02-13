#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from feature_dimred_showcase.evaluation import evaluate_classifier
from feature_dimred_showcase.feature_selection import compute_selection_scores
from feature_dimred_showcase.preprocessing import build_preprocessor, make_split, transform_split

REQUIRED_ARTIFACTS = [
    "artifacts/features/feature_matrix_summary.csv",
    "artifacts/selection/selection_scores.csv",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT / "shared/python"))


def _build_dataset() -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    raw = load_iris(as_frame=True)
    x_df = raw.data.copy()
    target = raw.target.copy()

    # Add categorical and missing data to practice preprocessing choices.
    x_df["petal_bucket"] = pd.cut(
        x_df["petal length (cm)"],
        bins=[0.0, 2.5, 4.5, 8.0],
        labels=["short", "medium", "long"],
        include_lowest=True,
    ).astype(str)
    x_df.loc[x_df.index[::11], "sepal width (cm)"] = float("nan")

    numeric = [col for col in x_df.columns if col != "petal_bucket"]
    categorical = ["petal_bucket"]
    return x_df, target, numeric, categorical


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run feature engineering comparison")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    from ml_core.contracts import (
        merge_required_files,
        write_supervised_contract_artifacts,
    )
    from ml_core.eda import (
        maybe_write_missingness_plot,
        maybe_write_profile_report,
    )
    from ml_core.splits import build_supervised_split

    _ = parse_args()
    root = Path(__file__).resolve().parents[1]

    x_df, target, numeric, categorical = _build_dataset()
    split = make_split(x_df, target, random_state=42)

    summary_rows: list[dict[str, float | str | int]] = []
    selection_frame: pd.DataFrame | None = None

    for encoding in ["onehot", "ordinal"]:
        x_train, x_test, names, _ = transform_split(
            split,
            numeric_features=numeric,
            categorical_features=categorical,
            encoding=encoding,
        )
        metrics = evaluate_classifier(x_train, x_test, split.y_train, split.y_test)
        summary_rows.append(
            {
                "encoding": encoding,
                "n_features": int(x_train.shape[1]),
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
            }
        )

        if encoding == "onehot":
            selection_frame = compute_selection_scores(x_train, split.y_train, names)

    summary = pd.DataFrame(summary_rows)
    if selection_frame is None:
        raise RuntimeError("selection_frame was not produced")

    summary_path = root / "artifacts/features/feature_matrix_summary.csv"
    selection_path = root / "artifacts/selection/selection_scores.csv"

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(summary_path, index=False)
    selection_frame.to_csv(selection_path, index=False)

    contract_split = build_supervised_split(
        x_df,
        target,
        strategy="stratified",
        random_state=42,
    )
    preprocessor = build_preprocessor(
        numeric_features=numeric,
        categorical_features=categorical,
        encoding="onehot",
    )
    x_train_arr = preprocessor.fit_transform(contract_split.x_train)
    x_test_arr = preprocessor.transform(contract_split.x_test)

    clf = LogisticRegression(max_iter=400, random_state=42)
    clf.fit(x_train_arr, contract_split.y_train)
    test_proba_matrix = clf.predict_proba(x_test_arr)
    test_probs = test_proba_matrix.max(axis=1)
    test_preds = clf.predict(x_test_arr)
    metrics = {
        "test_roc_auc_ovr": float(
            roc_auc_score(contract_split.y_test, test_proba_matrix, multi_class="ovr")
        ),
        "test_f1_macro": float(f1_score(contract_split.y_test, test_preds, average="macro")),
    }
    contract_required = write_supervised_contract_artifacts(
        project_root=root,
        frame=x_df,
        target=target,
        split=contract_split,
        task_type="classification",
        strategy="stratified",
        random_state=42,
        metrics=metrics,
        run_name="feature_engineering_baseline",
        threshold_scores=test_probs,
    )
    profile_status = maybe_write_profile_report(
        x_df,
        root / "artifacts/eda/profile_report.html",
    )
    missing_plot_status = maybe_write_missingness_plot(
        x_df,
        root / "artifacts/eda/missingness_matrix.png",
    )
    (root / "artifacts/eda/profile_status.txt").write_text(f"{profile_status}\n", encoding="utf-8")
    (root / "artifacts/eda/missing_plot_status.txt").write_text(
        f"{missing_plot_status}\n",
        encoding="utf-8",
    )

    manifest_path = root / "artifacts/manifest.json"
    merge_required_files(
        manifest_path,
        REQUIRED_ARTIFACTS
        + contract_required
        + [
            "artifacts/eda/profile_status.txt",
            "artifacts/eda/missing_plot_status.txt",
        ],
    )


if __name__ == "__main__":
    main()
