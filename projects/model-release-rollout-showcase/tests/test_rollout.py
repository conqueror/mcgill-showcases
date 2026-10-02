from __future__ import annotations

import pandas as pd
import pytest

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


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")], [0.7, float("nan")]])
def test_invalid_scores_do_not_become_hold(values: list[float]) -> None:
    with pytest.raises(ValueError, match="finite|empty"):
        evaluate_canary(
            pd.Series(values, dtype=float), pd.Series([0.7]), min_gain=0.005, max_regression=0.01
        )


@pytest.mark.parametrize(
    "gain,regression", [(-1.0, 0.01), (0.005, -1.0), (float("nan"), 0.01), (0.005, float("inf"))]
)
def test_invalid_thresholds_are_rejected(gain: float, regression: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        evaluate_canary(
            pd.Series([0.7]), pd.Series([0.7]), min_gain=gain, max_regression=regression
        )


@pytest.mark.parametrize(
    "champion,challenger,gain,want",
    [
        (0.5, 0.75, 0.25, "promote"),
        (0.75, 0.5, 0.25, "rollback"),
        (0.5, 0.625, 0.25, "hold"),
        (0.5, 0.5, 0.0, "promote"),
    ],
)
def test_hold_inclusive_boundaries_and_zero_gain(
    champion: float, challenger: float, gain: float, want: str
) -> None:
    # Hand differences: +1/4, -1/4, +1/8, and zero; quarters are exact binary floats.
    result = evaluate_canary(
        pd.Series([champion]), pd.Series([challenger]), min_gain=gain, max_regression=0.25
    )
    assert result.decision == want
