from __future__ import annotations

import pandas as pd

from mlops_drift_showcase.policy import recommend_action


def test_retrain_recommendation_when_many_features_drift() -> None:
    report = pd.DataFrame(
        {
            "feature": ["a", "b", "c"],
            "drift_flag": [1, 1, 1],
        }
    )
    result = recommend_action(report, max_drifted_features_before_retrain=1)
    assert result["action"] == "retrain"
    assert result["drifted_features"] == 3
