from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from xai_fairness_showcase import mitigation
from xai_fairness_showcase.data import make_audit_dataset
from xai_fairness_showcase.mitigation import run_mitigation_benchmark


def test_mitigation_table_contains_all_strategies() -> None:
    split = make_audit_dataset(n_samples=500, random_state=7)
    table = run_mitigation_benchmark(
        split.x_train,
        split.y_train,
        split.g_train,
        split.x_test,
        split.y_test,
        split.g_test,
    )

    strategies = set(table["strategy"].tolist())
    assert strategies == {
        "baseline",
        "pre_processing_reweight",
        "in_processing_balanced",
        "post_processing_threshold",
    }


def test_postprocessing_metrics_describe_adjusted_decisions() -> None:
    row = mitigation._score_postprocessing(
        "post", np.array([0.9, 0.8, 0.4, 0.3]), np.array([1, 0, 1, 0]),
        pd.Series([1, 0, 1, 0]), pd.Series([0, 0, 1, 1]),
    )
    # Each group selects one of two, detects its one positive, and has no false positives.
    assert row["accuracy"] == 1.0
    assert row["selection_rate_gap"] == 0.0
    assert row["tpr_gap"] == 0.0
    assert row["fpr_gap"] == 0.0


def test_test_cohort_cannot_refit_thresholds(monkeypatch: pytest.MonkeyPatch) -> None:
    train_x = pd.DataFrame({"score": [0.1, 0.3, 0.7, 0.9] * 10})
    train_y = pd.Series([0, 0, 1, 1] * 10)
    train_g = pd.Series([0, 1, 0, 1] * 10)
    test_x = pd.DataFrame({"score": [0.9, 0.8, 0.7, 0.4, 0.3, 0.2]})
    test_y = pd.Series([1, 0, 1, 0, 1, 0])
    test_g = pd.Series([0, 0, 0, 1, 1, 1])
    monkeypatch.setattr(mitigation, "predict_probabilities",
                        lambda model, x: x["score"].to_numpy(dtype=float))
    decisions: list[np.ndarray] = []
    fitted_thresholds: list[pd.Series] = []
    postprocess = mitigation._postprocess_group_thresholds

    def capture(probas: np.ndarray, group: pd.Series, **kwargs: Any) -> np.ndarray:
        if "thresholds" in kwargs:
            fitted_thresholds.append(kwargs["thresholds"].copy())
        result = postprocess(probas, group, **kwargs)
        decisions.append(result.copy())
        return result

    monkeypatch.setattr(mitigation, "_postprocess_group_thresholds", capture)
    run_mitigation_benchmark(train_x, train_y, train_g, test_x, test_y, test_g)
    run_mitigation_benchmark(
        train_x, train_y, train_g,
        pd.concat([test_x, pd.DataFrame({"score": [0.99]})], ignore_index=True),
        pd.concat([test_y, pd.Series([1])], ignore_index=True),
        pd.concat([test_g, pd.Series([0])], ignore_index=True),
    )
    np.testing.assert_array_equal(decisions[0], decisions[1][:6])
    assert len(fitted_thresholds) == 2
    pd.testing.assert_series_equal(fitted_thresholds[0], fitted_thresholds[1])


def test_group_reweighting_equalizes_group_mass() -> None:
    groups = pd.Series(["north", "north", "south", "east"])
    weights = mitigation._reweigh_samples(groups)
    # Four rows / three groups: each group gets weight 4/3; north divides it in two.
    np.testing.assert_allclose(weights, [2 / 3, 2 / 3, 4 / 3, 4 / 3])
    np.testing.assert_allclose(pd.Series(weights).groupby(groups).sum(), np.full(3, 4 / 3))
