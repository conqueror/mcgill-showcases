#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from eda_leakage_showcase.data import make_dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT / "shared/python"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EDA and leakage showcase")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    from ml_core.contracts import merge_required_files, write_supervised_contract_artifacts
    from ml_core.eda import maybe_write_missingness_plot, maybe_write_profile_report
    from ml_core.splits import build_supervised_split, cv_manifest_dict, split_manifest_dict

    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    n_samples = 500 if args.quick else 1200
    bundle = make_dataset(n_samples=n_samples, random_state=args.seed)

    frame_for_model = bundle.frame.drop(columns=["event_time", "group_id", "leak_target_copy"])
    encoded = pd.get_dummies(frame_for_model, columns=["segment", "region"], dummy_na=True)
    encoded = encoded.fillna(encoded.median(numeric_only=True)).fillna(0.0)

    split = build_supervised_split(
        encoded,
        bundle.target,
        strategy="stratified",
        random_state=args.seed,
    )

    model = LogisticRegression(max_iter=800)
    model.fit(split.x_train, split.y_train)
    probs = model.predict_proba(split.x_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "test_roc_auc": float(roc_auc_score(split.y_test, probs)),
        "test_f1": float(f1_score(split.y_test, preds)),
    }
    required = write_supervised_contract_artifacts(
        project_root=root,
        frame=bundle.frame,
        target=bundle.target,
        split=split,
        task_type="classification",
        strategy="stratified",
        random_state=args.seed,
        metrics=metrics,
        run_name="eda_leakage_baseline",
        threshold_scores=probs,
        group_column="group_id",
        time_column="event_time",
    )

    group_split = build_supervised_split(
        encoded,
        bundle.target,
        strategy="group",
        random_state=args.seed,
        groups=bundle.frame["group_id"],
    )
    group_manifest = split_manifest_dict(
        group_split,
        task_type="classification",
        strategy="group",
        random_state=args.seed,
        group_column="group_id",
    )
    group_manifest_path = root / "artifacts/splits/group_split_manifest.json"
    group_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    group_manifest_path.write_text(json.dumps(group_manifest, indent=2), encoding="utf-8")

    time_split = build_supervised_split(
        encoded,
        bundle.target,
        strategy="timeseries",
        random_state=args.seed,
        time_values=bundle.frame["event_time"],
    )
    time_manifest = split_manifest_dict(
        time_split,
        task_type="classification",
        strategy="timeseries",
        random_state=args.seed,
        time_column="event_time",
    )
    time_manifest_path = root / "artifacts/splits/timeseries_split_manifest.json"
    time_manifest_path.write_text(json.dumps(time_manifest, indent=2), encoding="utf-8")

    cv_manifest = cv_manifest_dict(
        bundle.target,
        strategy="stratified_kfold",
        n_splits=5,
        random_state=args.seed,
    )
    cv_manifest_path = root / "artifacts/splits/cv_split_manifest.json"
    cv_manifest_path.write_text(json.dumps(cv_manifest, indent=2), encoding="utf-8")

    profile_status = maybe_write_profile_report(
        bundle.frame,
        root / "artifacts/eda/profile_report.html",
    )
    missing_plot_status = maybe_write_missingness_plot(
        bundle.frame,
        root / "artifacts/eda/missingness_matrix.png",
    )

    (root / "artifacts/eda/profile_status.txt").write_text(f"{profile_status}\n", encoding="utf-8")
    (root / "artifacts/eda/missing_plot_status.txt").write_text(
        f"{missing_plot_status}\n",
        encoding="utf-8",
    )

    required.extend(
        [
            "artifacts/splits/group_split_manifest.json",
            "artifacts/splits/timeseries_split_manifest.json",
            "artifacts/splits/cv_split_manifest.json",
            "artifacts/eda/profile_status.txt",
            "artifacts/eda/missing_plot_status.txt",
        ]
    )
    merge_required_files(root / "artifacts/manifest.json", required)


if __name__ == "__main__":
    main()
