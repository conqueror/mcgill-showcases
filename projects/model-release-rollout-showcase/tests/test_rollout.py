from __future__ import annotations

import pandas as pd

from model_release_showcase.rollout import evaluate_canary


def test_promote_when_gain_is_large_enough() -> None:
    champion = pd.Series([0.70, 0.71, 0.72])
    challenger = pd.Series([0.73, 0.74, 0.75])
    result = evaluate_canary(champion, challenger, min_gain=0.005, max_regression=0.01)
    assert result.decision == "promote"


def test_rollback_when_regression_is_large() -> None:
    champion = pd.Series([0.74, 0.73, 0.75])
    challenger = pd.Series([0.68, 0.69, 0.70])
    result = evaluate_canary(champion, challenger, min_gain=0.005, max_regression=0.01)
    assert result.decision == "rollback"
