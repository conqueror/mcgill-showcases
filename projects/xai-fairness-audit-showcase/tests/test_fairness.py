from __future__ import annotations

import numpy as np
import pandas as pd

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
