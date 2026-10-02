import argparse
import json
import runpy
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from feature_dimred_showcase.dimensionality_reduction import run_embeddings
from feature_dimred_showcase.feature_selection import compute_selection_scores

ROOT = Path(__file__).resolve().parents[1]


def test_selection_includes_coefficients_for_every_class(monkeypatch: pytest.MonkeyPatch) -> None:
    def fit(self: Any, *args: Any, **kwargs: Any) -> Any:
        self.coef_ = np.array([[0.0, 1.0], [3.0, -2.0], [-4.0, 0.0]])
        return self

    monkeypatch.setattr(LogisticRegression, "fit", fit)
    values = np.arange(60, dtype=float).reshape(30, 2)
    scores = compute_selection_scores(values, pd.Series([0, 1, 2] * 10), ["a", "b"])
    # By hand: max(|0|, |3|, |-4|) = 4; max(|1|, |-2|, |0|) = 2.
    assert scores.set_index("feature")["abs_l1_coef"].to_dict() == {"a": 4.0, "b": 2.0}


def test_featuretools_never_receives_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = ModuleType("featuretools")

    class EntitySet:
        def __init__(self, **kwargs: Any) -> None:
            self.frame = pd.DataFrame()

        def add_dataframe(self, *, dataframe: pd.DataFrame, **kwargs: Any) -> "EntitySet":
            self.frame = dataframe.copy()
            return self

    def dfs(*, entityset: EntitySet, **kwargs: Any) -> tuple[pd.DataFrame, list[Any]]:
        return entityset.frame, []

    module.EntitySet = EntitySet  # type: ignore[attr-defined]
    module.dfs = dfs  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "featuretools", module)
    namespace = runpy.run_path(str(ROOT / "scripts/run_advanced_features.py"))
    frame = pd.DataFrame({"sample_id": [0, 1], "value": [2.0, 3.0]})
    output = tmp_path / "features.csv"
    namespace["_maybe_featuretools"](frame, pd.Series([0, 1]), output)
    assert pd.read_csv(output).columns.tolist() == ["sample_id", "value"]
    first = output.read_text()
    namespace["_maybe_featuretools"](frame, pd.Series([1, 0]), output)
    assert output.read_text() == first


def test_installed_umap_failure_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("umap")

    class BrokenUMAP:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def fit_transform(self, values: Any) -> Any:
            raise RuntimeError("installed UMAP failed")

    module.UMAP = BrokenUMAP  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "umap", module)
    with pytest.raises(RuntimeError, match="installed UMAP failed"):
        run_embeddings(np.random.default_rng(7).normal(size=(40, 4)), quick=True)


def test_encoding_tables_use_contract_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = runpy.run_path(str(ROOT / "scripts/run_feature_pipeline.py"))
    globals_ = namespace["main"].__globals__
    monkeypatch.setitem(globals_, "__file__", str(tmp_path / "scripts/run_feature_pipeline.py"))
    monkeypatch.setitem(globals_, "parse_args", lambda: argparse.Namespace(quick=True))
    evaluate = globals_["evaluate_classifier"]
    counts: list[tuple[int, int]] = []

    def observe(x_train: Any, x_test: Any, *args: Any) -> Any:
        counts.append((len(x_train), len(x_test)))
        return evaluate(x_train, x_test, *args)

    monkeypatch.setitem(globals_, "evaluate_classifier", observe)
    namespace["main"]()
    manifest = json.loads((tmp_path / "artifacts/splits/split_manifest.json").read_text())
    assert counts == [(manifest["train_rows"], manifest["test_rows"])] * 2
