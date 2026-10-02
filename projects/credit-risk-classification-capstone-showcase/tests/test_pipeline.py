import argparse
import json
import runpy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from ml_core.splits import build_supervised_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from credit_risk_capstone import modeling
from credit_risk_capstone.data import (
    build_target_from_status,
    clean_and_encode_features,
    make_credit_risk_dataset,
)

ROOT = Path(__file__).resolve().parents[1]


def test_diagnostics_preserve_original_missingness() -> None:
    frame = make_credit_risk_dataset(n_samples=500, random_state=11).frame
    split = build_supervised_split(frame, build_target_from_status(frame), random_state=11)
    diagnostics, _ = clean_and_encode_features(frame, train_index=split.x_train.index)
    for column in ["annual_income", "dti", "employment_length", "home_ownership"]:
        assert diagnostics[column].isna().sum() == frame[column].isna().sum()


def test_held_out_rows_cannot_change_preprocessing() -> None:
    frame = make_credit_risk_dataset(n_samples=500, random_state=11).frame
    split = build_supervised_split(frame, build_target_from_status(frame), random_state=11)
    _, original = clean_and_encode_features(frame, train_index=split.x_train.index)
    changed = frame.copy()
    held_out = split.x_val.index.union(split.x_test.index)
    changed.loc[held_out, "annual_income"] = 1e9
    changed.loc[held_out, "home_ownership"] = "held_out_only"
    _, modified = clean_and_encode_features(changed, train_index=split.x_train.index)
    pd.testing.assert_frame_equal(
        original.loc[split.x_train.index], modified.loc[split.x_train.index]
    )


def test_benchmark_does_not_read_test_rows() -> None:
    frame = make_credit_risk_dataset(n_samples=500, random_state=11).frame
    target = build_target_from_status(frame)
    features = frame[["loan_amount", "fico_score"]].copy()
    split = build_supervised_split(features, target, random_state=11)
    original = modeling.model_benchmark(split, random_state=11)
    split.x_test.iloc[:, :] = np.nan
    split.y_test.iloc[:] = 1 - split.y_test
    modified = modeling.model_benchmark(split, random_state=11)
    pd.testing.assert_frame_equal(original, modified)


def test_model_summary_selects_validation_winner() -> None:
    benchmark = pd.DataFrame(
        {
            "model": ["test_winner", "validation_winner"],
            "val_f1": [0.2, 0.8],
            "val_roc_auc": [0.6, 0.9],
            "val_average_precision": [0.4, 0.7],
            "test_f1": [0.9, 0.1],
            "test_roc_auc": [0.9, 0.6],
            "test_pr_auc": [0.8, 0.3],
        }
    )
    assert modeling.best_model_summary(benchmark)["best_model"] == "validation_winner"


def test_average_precision_is_named_and_computed_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    from ml_core.splits import SplitBundle3

    monkeypatch.setattr(modeling, "list_imbalance_methods", lambda: ["none"])

    def predict(self: Any, features: Any) -> Any:
        return np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.9, 0.1]])

    monkeypatch.setattr(Pipeline, "predict_proba", predict)
    x = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    y = pd.Series([1, 0, 1, 0])
    result, _, _ = modeling.evaluate_imbalance_strategies(SplitBundle3(x, x, x, y, y, y))
    # Positive ranks are 1 and 3; AP=(precision@1 + precision@3)/2=(1+2/3)/2=5/6.
    assert result.loc[0, "val_average_precision"] == pytest.approx(5 / 6)


def test_final_model_uses_selected_strategy_and_validation_thresholds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = runpy.run_path(str(ROOT / "scripts/run_pipeline.py"))
    globals_ = namespace["main"].__globals__
    monkeypatch.setitem(globals_, "__file__", str(tmp_path / "scripts/run_pipeline.py"))
    monkeypatch.setitem(globals_, "parse_args", lambda: argparse.Namespace(quick=True, seed=11))
    monkeypatch.setattr(modeling, "list_imbalance_methods", lambda: ["upsample_minority"])
    fitted: list[tuple[Any, pd.Series]] = []
    for cls in [Pipeline, RandomForestClassifier]:
        fit = cls.fit

        def observe(self: Any, x: Any, y: pd.Series, fit: Any = fit, **kwargs: Any) -> Any:
            fitted.append((self, y.copy()))
            return fit(self, x, y, **kwargs)

        monkeypatch.setattr(cls, "fit", observe)
    namespace["main"]()
    model, training_labels = fitted[-1]
    assert training_labels.value_counts().nunique() == 1
    summary = json.loads((tmp_path / "artifacts/models/best_model_summary.json").read_text())
    assert isinstance(model, Pipeline) == (summary["best_model"] == "logistic_regression")
    assert summary["threshold_selection_split"] == "validation"
    threshold_table = pd.read_csv(tmp_path / "artifacts/eval/threshold_analysis.csv")
    bundle = make_credit_risk_dataset(n_samples=1400, random_state=11)
    target = build_target_from_status(bundle.frame)
    raw = build_supervised_split(bundle.frame, target, random_state=11)
    _, features = clean_and_encode_features(bundle.frame, train_index=raw.x_train.index)
    scores = model.predict_proba(features.loc[raw.x_val.index])[:, 1]
    from sklearn.metrics import f1_score

    for row in threshold_table.itertuples():
        assert row.f1 == pytest.approx(
            f1_score(raw.y_val, scores >= row.threshold, zero_division=0)
        )
