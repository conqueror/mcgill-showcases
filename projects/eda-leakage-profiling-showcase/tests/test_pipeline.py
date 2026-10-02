import argparse
import runpy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from eda_leakage_showcase.data import DatasetBundle, make_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parents[1] / "shared/python"))


def run_pipeline(
    bundle: DatasetBundle, output: Path, monkeypatch: pytest.MonkeyPatch, *, seed: int = 7
) -> list[Any]:
    namespace = runpy.run_path(str(ROOT / "scripts/run_pipeline.py"))
    globals_ = namespace["main"].__globals__
    monkeypatch.setitem(globals_, "__file__", str(output / "scripts/run_pipeline.py"))
    monkeypatch.setitem(globals_, "parse_args", lambda: argparse.Namespace(quick=True, seed=seed))
    monkeypatch.setitem(globals_, "make_dataset", lambda **kwargs: bundle)
    fitted: list[Any] = []
    fit = LogisticRegression.fit

    def observe(self: Any, x: Any, y: Any, **kwargs: Any) -> Any:
        fitted.append(self)
        return fit(self, x, y, **kwargs)

    with monkeypatch.context() as fit_patch:
        fit_patch.setattr(LogisticRegression, "fit", observe)
        namespace["main"]()
    return fitted


def test_default_quick_run_distinguishes_records_from_feature_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = make_dataset(n_samples=500, random_state=42)
    features = bundle.frame.drop(columns=["event_time", "group_id", "leak_target_copy"])
    assert features.duplicated().any()
    assert not bundle.frame.duplicated().any()
    assert run_pipeline(bundle, tmp_path, monkeypatch, seed=42)


def test_copied_source_record_blocks_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ml_core.splits import build_supervised_split

    bundle = make_dataset(n_samples=200, random_state=7)
    split = build_supervised_split(bundle.frame, bundle.target, random_state=7)
    bundle.frame.loc[split.x_test.index[0]] = bundle.frame.loc[split.x_train.index[0]]
    with pytest.raises(ValueError, match="[Ll]eakage"):
        run_pipeline(bundle, tmp_path, monkeypatch)


def test_held_out_values_cannot_change_fitted_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ml_core.splits import build_supervised_split

    bundle = make_dataset(n_samples=200, random_state=7)
    raw_split = build_supervised_split(bundle.frame, bundle.target, random_state=7)
    changed = bundle.frame.copy()
    held_out = raw_split.x_val.index.union(raw_split.x_test.index)
    changed.loc[held_out, "amount"] = 1e9 + np.arange(len(held_out))
    changed.loc[held_out, "segment"] = "held_out_only"
    original = run_pipeline(bundle, tmp_path / "original", monkeypatch)[-1]
    modified = run_pipeline(
        DatasetBundle(changed, bundle.target), tmp_path / "modified", monkeypatch
    )[-1]
    np.testing.assert_array_equal(original.coef_, modified.coef_)
    np.testing.assert_array_equal(original.intercept_, modified.intercept_)


def test_target_copy_blocks_training_and_writes_diagnostics_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = make_dataset(n_samples=200, random_state=7)
    bundle.frame["future_outcome"] = bundle.target
    with pytest.raises(ValueError, match="[Ll]eakage"):
        run_pipeline(bundle, tmp_path, monkeypatch)
    report = pd.read_csv(tmp_path / "artifacts/leakage/leakage_report.csv")
    assert "future_outcome" in report["feature"].tolist()
    assert not (tmp_path / "artifacts/eval/metrics_summary.csv").exists()
