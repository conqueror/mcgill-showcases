#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT / "shared/python"))


def main() -> None:
    from ml_core.contracts import (
        merge_required_files,
        write_supervised_contract_artifacts,
    )
    from ml_core.splits import build_supervised_split

    root = Path(__file__).resolve().parents[1]

    x_values, y_values = make_classification(
        n_samples=1200,
        n_features=10,
        n_informative=7,
        n_redundant=1,
        random_state=42,
    )
    frame = pd.DataFrame(x_values, columns=[f"feature_{idx}" for idx in range(x_values.shape[1])])
    target = pd.Series(y_values, name="target")

    split = build_supervised_split(
        frame,
        target,
        strategy="stratified",
        random_state=42,
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(split.x_train, split.y_train)
    probs = model.predict_proba(split.x_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "test_roc_auc": float(roc_auc_score(split.y_test, probs)),
        "test_f1": float(f1_score(split.y_test, preds)),
    }
    required = write_supervised_contract_artifacts(
        project_root=root,
        frame=frame,
        target=target,
        split=split,
        task_type="classification",
        strategy="stratified",
        random_state=42,
        metrics=metrics,
        run_name="causal_contract_baseline_synthetic",
        threshold_scores=probs,
    )

    merge_required_files(root / "artifacts/manifest.json", required)


if __name__ == "__main__":
    main()
