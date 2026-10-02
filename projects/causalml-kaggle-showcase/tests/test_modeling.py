from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from joblib import parallel_backend

from causal_showcase import modeling
from causal_showcase.data import PreparedData


def _constant_data() -> PreparedData:
    return PreparedData(pd.DataFrame({"x": np.zeros(8)}), np.array([0, 0, 0, 0, 1, 1, 1, 1]),
                        np.array([0, 0, 1, 1, 0, 1, 1, 1]), ["x"])


def test_uplift_tree_returns_treatment_contrast() -> None:
    data = _constant_data()
    result = modeling.fit_uplift_tree(data, data)
    # One leaf: treatment converts 3/4, control converts 2/4; uplift = 1/4.
    np.testing.assert_allclose(result.uplift_scores, np.full(8, 0.25))


def test_tree_exercise_retrains_with_displayed_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    models: list[Any] = []
    original = modeling.UpliftTreeClassifier

    def capture(**kwargs: Any) -> Any:
        model = original(**kwargs)
        models.append(model)
        return model

    monkeypatch.setattr(modeling, "UpliftTreeClassifier", capture)
    path = Path(__file__).resolve().parents[1] / "notebooks/03_uplift_tree_interpretability.ipynb"
    code = "".join(json.loads(path.read_text())["cells"][8]["source"])
    exec(code, {"train_data": _constant_data(), "test_data": _constant_data(),
                "fit_uplift_tree": modeling.fit_uplift_tree})
    assert len(models) == 1
    assert models[0].max_depth == 3
    assert models[0].min_samples_leaf == 400
    assert models[0].min_samples_treatment == 200
    assert models[0].random_state == 42


def test_shap_surrogate_scores_rows_it_did_not_fit() -> None:
    path = Path(__file__).resolve().parents[1] / "notebooks/07_shap_interpretability.ipynb"
    code = "".join(json.loads(path.read_text())["cells"][4]["source"])
    rng = np.random.default_rng(4)
    data = PreparedData(pd.DataFrame({"x": np.arange(80)}), np.tile([0, 1], 40),
                        np.tile([0, 1], 40), ["x"])
    scope: dict[str, Any] = {"test_data": data, "best_scores": rng.normal(size=80)}
    exec(code, scope)
    # Independent noise cannot be predicted out of sample; memorizing it gives a high training R².
    assert scope["surrogate_r2"] < 0.1
    changed_scores = scope["best_scores"].copy()
    changed_scores[scope["x_test"].index.to_numpy()] += 100.0
    changed: dict[str, Any] = {"test_data": data, "best_scores": changed_scores}
    exec(code, changed)
    np.testing.assert_array_equal(scope["surrogate"].predict(data.X),
                                  changed["surrogate"].predict(data.X))


def test_meta_learner_fits_repeat_and_ignore_held_out_outcomes() -> None:
    rng = np.random.default_rng(8)
    x = pd.DataFrame(rng.normal(size=(120, 3)), columns=["a", "b", "c"])
    treatment = np.tile([0, 1], 60)
    outcome = rng.binomial(1, 1 / (1 + np.exp(-(x["a"].to_numpy() + 0.5 * treatment))))
    train = PreparedData(x, treatment, outcome, list(x.columns))
    held = PreparedData(x.iloc[:12].copy(), treatment[:12], outcome[:12], list(x.columns))
    changed = PreparedData(held.X, held.treatment, 1 - held.outcome, held.feature_names)
    with parallel_backend("threading", n_jobs=1):
        first = modeling.fit_meta_learners(train, held)
        second = modeling.fit_meta_learners(train, changed)
    for name in first:
        np.testing.assert_array_equal(first[name].uplift_scores, second[name].uplift_scores,
                                      err_msg=name)
        assert first[name].ate == second[name].ate
