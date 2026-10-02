from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xai_fairness_showcase.fairness import disparity_table, subgroup_metrics


def test_group_metrics_and_disparity_shape() -> None:
    y_true = pd.Series([1, 0, 1, 0, 1, 0])
    probs = np.array([0.9, 0.2, 0.8, 0.1, 0.3, 0.7])
    group = pd.Series([0, 0, 0, 1, 1, 1])

    metrics = subgroup_metrics(y_true, probs, group)
    disparities = disparity_table(metrics)

    assert set(metrics["group"].tolist()) == {0, 1}
    assert set(disparities["metric"].tolist()) == {
        "selection_rate",
        "tpr",
        "fpr",
        "fnr",
        "precision",
    }


def test_undefined_rates_keep_denominators_and_block_gap() -> None:
    metrics = subgroup_metrics(pd.Series([0, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.2]),
                               pd.Series([0, 0, 1, 1])).set_index("group")
    # Group 0 has TP=FN=0: TPR and FNR have no denominator, not a zero rate.
    assert np.isnan(metrics.loc[0, "tpr"])
    assert np.isnan(metrics.loc[0, "fnr"])
    assert metrics.loc[0, "positive_count"] == 0
    assert metrics.loc[0, "negative_count"] == 2
    assert metrics.loc[0, "predicted_positive_count"] == 1
    assert metrics.loc[0, "fpr"] == 0.5  # One false positive / two negatives.
    gap = disparity_table(metrics.reset_index()).set_index("metric")
    assert np.isnan(gap.loc["tpr", "gap"])
    assert gap.loc["tpr", "supported_groups"] == 1
    assert gap.loc["tpr", "total_groups"] == 2


def test_named_group_labels_are_preserved() -> None:
    result = subgroup_metrics(pd.Series([1, 0, 1, 0]), np.array([0.9, 0.8, 0.4, 0.3]),
                              pd.Series(["north", "north", "south", "south"]))
    assert result["group"].tolist() == ["north", "south"]
    assert result["selection_rate"].tolist() == [1.0, 0.0]


@pytest.mark.parametrize("labels,groups", [([2, 0], [0, 1]), ([1, 0], [0, None])])
def test_subgroup_metrics_reject_invalid_labels_or_missing_groups(
    labels: list[int], groups: list[int | None],
) -> None:
    with pytest.raises(ValueError):
        subgroup_metrics(pd.Series(labels), np.array([0.8, 0.2]), pd.Series(groups))


def test_subgroup_metrics_reject_misaligned_series() -> None:
    with pytest.raises(ValueError):
        subgroup_metrics(pd.Series([1, 0], index=[0, 1]), np.array([0.8, 0.2]),
                         pd.Series([0, 1], index=[1, 0]))
