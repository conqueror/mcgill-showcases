#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from automl_hpo_showcase.benchmark import summarize_trials, write_best_configs
from automl_hpo_showcase.objective import evaluate_on_test, make_synthetic_dataset
from automl_hpo_showcase.search_bayes_tpe import run_tpe_search
from automl_hpo_showcase.search_hyperopt import run_hyperopt_search
from automl_hpo_showcase.search_random_grid import run_grid_search, run_random_search

REQUIRED_ARTIFACTS = [
    "artifacts/hpo/trials.csv",
    "artifacts/hpo/strategy_comparison.csv",
    "artifacts/hpo/best_configs.json",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT / "shared/python"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HPO strategy benchmark")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--with-hyperopt", action="store_true")
    parser.add_argument("--with-mlflow", action="store_true")
    return parser.parse_args()


def _best_trial(trials: pd.DataFrame) -> dict[str, float | int]:
    best = trials.sort_values(by="score", ascending=False).iloc[0]
    return {
        "n_estimators": int(best["n_estimators"]),
        "max_depth": int(best["max_depth"]),
        "min_samples_split": int(best["min_samples_split"]),
        "score": float(best["score"]),
    }


def main() -> None:
    from ml_core.contracts import (
        merge_required_files,
        write_supervised_contract_artifacts,
    )
    from ml_core.experiments import log_experiment_mlflow
    from ml_core.splits import build_supervised_split

    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    budget = 6 if args.quick else 18
    grid = run_grid_search(budget=budget, random_state=args.seed)
    random = run_random_search(budget=budget, seed=args.seed, random_state=args.seed)
    tpe = run_tpe_search(budget=budget, seed=args.seed)
    frames = [grid, random, tpe]
    if args.with_hyperopt:
        frames.append(run_hyperopt_search(budget=budget, seed=args.seed))

    trials = pd.concat(frames, ignore_index=True)
    comparison = summarize_trials(trials)

    trials_path = root / "artifacts/hpo/trials.csv"
    comparison_path = root / "artifacts/hpo/strategy_comparison.csv"
    best_path = root / "artifacts/hpo/best_configs.json"

    trials_path.parent.mkdir(parents=True, exist_ok=True)
    trials.to_csv(trials_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    write_best_configs(trials, best_path)

    best_cfg = _best_trial(trials)
    test_roc_auc = evaluate_on_test(
        n_estimators=int(best_cfg["n_estimators"]),
        max_depth=int(best_cfg["max_depth"]),
        min_samples_split=int(best_cfg["min_samples_split"]),
        random_state=args.seed,
    )

    x_df, y_series = make_synthetic_dataset(random_state=args.seed)
    split = build_supervised_split(
        x_df,
        y_series,
        strategy="stratified",
        random_state=args.seed,
    )
    model = RandomForestClassifier(
        n_estimators=int(best_cfg["n_estimators"]),
        max_depth=int(best_cfg["max_depth"]),
        min_samples_split=int(best_cfg["min_samples_split"]),
        random_state=args.seed,
        n_jobs=1,
    )
    model.fit(split.x_train, split.y_train)
    test_scores = model.predict_proba(split.x_test)[:, 1]
    test_auc = float(roc_auc_score(split.y_test, test_scores))
    metrics = {
        "best_validation_roc_auc": float(best_cfg["score"]),
        "best_test_roc_auc": test_auc,
        "eval_on_test_from_objective": test_roc_auc,
    }
    contract_required = write_supervised_contract_artifacts(
        project_root=root,
        frame=x_df,
        target=y_series,
        split=split,
        task_type="classification",
        strategy="stratified",
        random_state=args.seed,
        metrics=metrics,
        run_name="hpo_best_model",
        threshold_scores=test_scores,
    )

    mlflow_status = log_experiment_mlflow(
        enabled=args.with_mlflow,
        run_name="automl_hpo_showcase",
        params={
            "budget": budget,
            "with_hyperopt": args.with_hyperopt,
            "best_n_estimators": int(best_cfg["n_estimators"]),
            "best_max_depth": int(best_cfg["max_depth"]),
            "best_min_samples_split": int(best_cfg["min_samples_split"]),
        },
        metrics=metrics,
        tracking_uri=f"file://{(root / 'artifacts/mlruns').resolve()}",
        experiment_name="mcgill_automl_hpo_showcase",
    )
    (root / "artifacts/hpo/mlflow_status.txt").write_text(f"{mlflow_status}\n", encoding="utf-8")

    manifest_path = root / "artifacts/manifest.json"
    merge_required_files(
        manifest_path,
        REQUIRED_ARTIFACTS + contract_required + ["artifacts/hpo/mlflow_status.txt"],
    )


if __name__ == "__main__":
    main()
