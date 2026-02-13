#!/usr/bin/env python
"""Run the credit-risk capstone pipeline and write contract artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from credit_risk_capstone.data import (
    build_target_from_status,
    clean_and_encode_features,
    make_credit_risk_dataset,
)
from credit_risk_capstone.modeling import (
    best_model_summary,
    evaluate_imbalance_strategies,
    model_benchmark,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI options for quick/full capstone execution."""

    parser = argparse.ArgumentParser(description="Run the credit risk capstone pipeline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Execute capstone data prep, modeling, evaluation, and artifact writing."""

    from ml_core.contracts import (
        merge_required_files,
        write_supervised_contract_artifacts,
    )
    from ml_core.eda import (
        infer_feature_types,
        maybe_write_missingness_plot,
        maybe_write_profile_report,
    )
    from ml_core.imbalance import class_balance
    from ml_core.splits import build_supervised_split

    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    n_samples = 1400 if args.quick else 3600
    bundle = make_credit_risk_dataset(n_samples=n_samples, random_state=args.seed)

    target = build_target_from_status(bundle.frame)
    diagnostics_frame, model_frame = clean_and_encode_features(bundle.frame)

    split = build_supervised_split(
        model_frame,
        target,
        strategy="stratified",
        random_state=args.seed,
    )

    strategy_df, _, best_strategy = evaluate_imbalance_strategies(
        split,
        random_state=args.seed,
    )
    benchmark_df = model_benchmark(split, random_state=args.seed)
    summary_payload = best_model_summary(benchmark_df)
    summary_payload["best_imbalance_strategy"] = best_strategy

    diag_dir = root / "artifacts/diagnostics"
    model_dir = root / "artifacts/models"
    eda_dir = root / "artifacts/eda"
    diag_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    eda_dir.mkdir(parents=True, exist_ok=True)

    infer_feature_types(diagnostics_frame).to_csv(
        diag_dir / "feature_type_summary.csv",
        index=False,
    )
    class_balance(split.y_train).to_csv(diag_dir / "class_balance_train.csv", index=False)
    strategy_df.to_csv(model_dir / "strategy_comparison.csv", index=False)
    benchmark_df.to_csv(model_dir / "model_benchmark.csv", index=False)
    (model_dir / "best_model_summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )

    profile_status = maybe_write_profile_report(
        diagnostics_frame,
        eda_dir / "profile_report.html",
    )
    missing_plot_status = maybe_write_missingness_plot(
        diagnostics_frame,
        eda_dir / "missingness_matrix.png",
    )
    (eda_dir / "profile_status.txt").write_text(f"{profile_status}\n", encoding="utf-8")
    (eda_dir / "missing_plot_status.txt").write_text(
        f"{missing_plot_status}\n",
        encoding="utf-8",
    )

    final_model = RandomForestClassifier(
        n_estimators=320,
        max_depth=8,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        random_state=args.seed,
        n_jobs=1,
    )
    final_model.fit(split.x_train, split.y_train)
    test_scores = final_model.predict_proba(split.x_test)[:, 1]
    test_preds = (test_scores >= 0.5).astype(int)

    metrics = {
        "test_f1": float(f1_score(split.y_test, test_preds, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(split.y_test, test_scores)),
        "test_pr_auc": float(average_precision_score(split.y_test, test_scores)),
        "best_val_strategy_f1": float(strategy_df.iloc[0]["val_f1"]),
    }

    contract_required = write_supervised_contract_artifacts(
        project_root=root,
        frame=diagnostics_frame,
        target=target,
        split=split,
        task_type="classification",
        strategy="stratified",
        random_state=args.seed,
        metrics=metrics,
        run_name="credit_risk_capstone",
        threshold_scores=test_scores,
    )

    merge_required_files(
        root / "artifacts/manifest.json",
        contract_required
        + [
            "artifacts/diagnostics/feature_type_summary.csv",
            "artifacts/diagnostics/class_balance_train.csv",
            "artifacts/models/strategy_comparison.csv",
            "artifacts/models/model_benchmark.csv",
            "artifacts/models/best_model_summary.json",
            "artifacts/eda/profile_status.txt",
            "artifacts/eda/missing_plot_status.txt",
        ],
    )


if __name__ == "__main__":
    main()
