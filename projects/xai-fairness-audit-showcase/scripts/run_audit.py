#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from xai_fairness_showcase.data import make_audit_dataset
from xai_fairness_showcase.explainability import (
    global_feature_importance,
    local_linear_contributions,
)
from xai_fairness_showcase.fairness import disparity_table, subgroup_metrics
from xai_fairness_showcase.mitigation import train_baseline_model
from xai_fairness_showcase.modeling import evaluate_binary, predict_probabilities
from xai_fairness_showcase.reporting import build_disparity_markdown

REQUIRED_ARTIFACTS = [
    "artifacts/explainability/global_importance.csv",
    "artifacts/explainability/local_explanations_sample.csv",
    "artifacts/fairness/group_metrics.csv",
    "artifacts/fairness/disparity_summary.md",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT / "shared/python"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fairness and explainability baseline audit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--with-lime", action="store_true")
    parser.add_argument("--with-shap", action="store_true")
    return parser.parse_args()


def main() -> None:
    from ml_core.contracts import (
        merge_required_files,
        write_supervised_contract_artifacts,
    )
    from ml_core.explainability import (
        run_lime_local_explanations,
        run_shap_importance,
    )
    from ml_core.splits import build_supervised_split

    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    n_samples = 900 if args.quick else 1800
    data = make_audit_dataset(n_samples=n_samples, random_state=args.seed)
    model = train_baseline_model(data.x_train, data.y_train)

    probs = predict_probabilities(model, data.x_test)
    g_importance = global_feature_importance(
        model,
        data.x_test,
        data.y_test,
        random_state=args.seed,
    )
    local = local_linear_contributions(model, data.x_test, n_rows=20)
    group = subgroup_metrics(data.y_test, probs, data.g_test)
    disparities = disparity_table(group)

    global_path = root / "artifacts/explainability/global_importance.csv"
    local_path = root / "artifacts/explainability/local_explanations_sample.csv"
    group_path = root / "artifacts/fairness/group_metrics.csv"
    summary_path = root / "artifacts/fairness/disparity_summary.md"

    global_path.parent.mkdir(parents=True, exist_ok=True)
    group_path.parent.mkdir(parents=True, exist_ok=True)

    g_importance.to_csv(global_path, index=False)
    local.to_csv(local_path, index=False)
    group.to_csv(group_path, index=False)
    summary_path.write_text(build_disparity_markdown(disparities), encoding="utf-8")

    all_x = pd.concat([data.x_train, data.x_test], axis=0, ignore_index=True).reset_index(drop=True)
    all_y = pd.concat([data.y_train, data.y_test], axis=0, ignore_index=True).reset_index(drop=True)
    contract_split = build_supervised_split(
        all_x,
        all_y,
        strategy="stratified",
        random_state=args.seed,
    )
    contract_model = train_baseline_model(contract_split.x_train, contract_split.y_train)
    contract_probs = predict_probabilities(contract_model, contract_split.x_test)
    contract_metrics = evaluate_binary(contract_split.y_test, contract_probs)
    contract_required = write_supervised_contract_artifacts(
        project_root=root,
        frame=all_x,
        target=all_y,
        split=contract_split,
        task_type="classification",
        strategy="stratified",
        random_state=args.seed,
        metrics=contract_metrics,
        run_name="xai_fairness_baseline",
        threshold_scores=contract_probs,
    )

    shap_status = "disabled"
    if args.with_shap:
        shap_status = run_shap_importance(
            contract_model,
            contract_split.x_test,
            output_path=root / "artifacts/explainability/shap_importance.csv",
        )
    lime_status = "disabled"
    if args.with_lime:
        lime_status = run_lime_local_explanations(
            contract_model.predict_proba,
            contract_split.x_train,
            contract_split.x_test,
            output_path=root / "artifacts/explainability/lime_local_explanations.csv",
        )

    (root / "artifacts/explainability/shap_status.txt").write_text(
        f"{shap_status}\n", encoding="utf-8"
    )
    (root / "artifacts/explainability/lime_status.txt").write_text(
        f"{lime_status}\n", encoding="utf-8"
    )

    manifest_path = root / "artifacts/manifest.json"
    merge_required_files(
        manifest_path,
        REQUIRED_ARTIFACTS
        + contract_required
        + [
            "artifacts/explainability/shap_status.txt",
            "artifacts/explainability/lime_status.txt",
        ],
    )


if __name__ == "__main__":
    main()
