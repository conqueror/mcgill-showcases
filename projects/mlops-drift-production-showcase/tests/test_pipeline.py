from __future__ import annotations

import importlib
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pytest import MonkeyPatch
from sklearn.metrics import accuracy_score, roc_auc_score

from mlops_drift_showcase.data import DatasetBundle, generate_reference_data
from mlops_drift_showcase.train import load_model


def _run_pipeline(root: Path, bundle: DatasetBundle, monkeypatch: MonkeyPatch) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/run_pipeline.py"
    main = runpy.run_path(str(script))["main"]
    monkeypatch.setitem(main.__globals__, "__file__", str(root / "scripts/run_pipeline.py"))
    monkeypatch.setitem(main.__globals__, "generate_reference_data", lambda **kwargs: bundle)
    monkeypatch.setattr(sys, "argv", [str(script), "--quick"])
    main()


def test_saved_model_and_contract_describe_the_same_split(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    bundle = generate_reference_data(n_samples=200)
    _run_pipeline(tmp_path, bundle, monkeypatch)
    split = importlib.import_module("ml_core.splits").build_supervised_split(
        bundle.features, bundle.target
    )
    model = load_model(tmp_path / "artifacts/model/model.joblib")
    np.testing.assert_allclose(model.named_steps["scaler"].mean_, split.x_train.mean())
    reference = pd.read_csv(tmp_path / "artifacts/reference/train_features.csv")
    pd.testing.assert_frame_equal(reference, split.x_train.reset_index(drop=True))
    holdout = pd.read_csv(tmp_path / "artifacts/reference/holdout_predictions.csv")
    pd.testing.assert_frame_equal(
        holdout[bundle.features.columns], split.x_test.reset_index(drop=True)
    )
    probs = model.predict_proba(split.x_test)[:, 1]
    np.testing.assert_allclose(holdout["y_pred_proba"], probs)
    metrics = pd.read_csv(tmp_path / "artifacts/eval/metrics_summary.csv").set_index("metric")
    assert metrics.loc["test_roc_auc", "value"] == roc_auc_score(split.y_test, probs)
    assert metrics.loc["test_accuracy", "value"] == accuracy_score(split.y_test, probs >= 0.5)
    manifest = json.loads((tmp_path / "artifacts/splits/split_manifest.json").read_text())
    assert [manifest[key] for key in ("train_rows", "val_rows", "test_rows")] == [120, 40, 40]


def test_changing_held_out_rows_cannot_change_saved_fit(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    bundle = generate_reference_data(n_samples=200)
    # Give both runs the same pandas array layout, so only held-out values change.
    bundle = DatasetBundle(bundle.features.copy(), bundle.target)
    _run_pipeline(tmp_path / "original", bundle, monkeypatch)
    split = importlib.import_module("ml_core.splits").build_supervised_split(
        bundle.features, bundle.target
    )
    changed = bundle.features.copy()
    changed.loc[split.x_val.index.union(split.x_test.index)] += 1000.0
    _run_pipeline(tmp_path / "changed", DatasetBundle(changed, bundle.target), monkeypatch)
    original = load_model(tmp_path / "original/artifacts/model/model.joblib")
    modified = load_model(tmp_path / "changed/artifacts/model/model.joblib")
    np.testing.assert_array_equal(
        original.named_steps["scaler"].mean_, modified.named_steps["scaler"].mean_
    )
    np.testing.assert_array_equal(
        original.named_steps["clf"].coef_, modified.named_steps["clf"].coef_
    )


@pytest.mark.parametrize(
    "corruption",
    [
        "empty_manifest",
        "wrong_csv",
        "broken_model",
        "empty_split",
        "wrong_reference",
        "wrong_holdout",
        "wrong_eda",
        "nonfinite_reference",
    ],
)
def test_verifier_rejects_empty_or_corrupt_artifacts(
    tmp_path: Path, monkeypatch: MonkeyPatch, corruption: str
) -> None:
    _run_pipeline(tmp_path, generate_reference_data(n_samples=200), monkeypatch)
    script = Path(__file__).resolve().parents[1] / "scripts/verify_artifacts.py"
    verify = runpy.run_path(str(script))["main"]
    monkeypatch.setitem(
        verify.__globals__, "__file__", str(tmp_path / "scripts/verify_artifacts.py")
    )
    verify()
    corruptions = {
        "empty_manifest": ("artifacts/manifest.json", '{"required_files": []}'),
        "wrong_csv": ("artifacts/metrics/train_eval_summary.csv", "junk\n1\n"),
        "broken_model": ("artifacts/model/model.joblib", "broken model"),
        "empty_split": ("artifacts/splits/split_manifest.json", "{}"),
        "wrong_reference": ("artifacts/reference/train_features.csv", "junk\n1\n"),
        "wrong_holdout": ("artifacts/reference/holdout_predictions.csv", "junk\n1\n"),
        "wrong_eda": ("artifacts/eda/univariate_summary.csv", "junk\n1\n"),
    }
    if corruption == "nonfinite_reference":
        path = tmp_path / "artifacts/reference/train_features.csv"
        frame = pd.read_csv(path)
        frame.iloc[0, 0] = float("inf")
        frame.to_csv(path, index=False)
    else:
        filename, content = corruptions[corruption]
        (tmp_path / filename).write_text(content)
    with pytest.raises(SystemExit):
        verify()
