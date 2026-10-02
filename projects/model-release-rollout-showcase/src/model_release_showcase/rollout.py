from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DecisionResult:
    decision: str
    reason: str


def evaluate_canary(
    champion_scores: pd.Series,
    challenger_scores: pd.Series,
    *,
    min_gain: float,
    max_regression: float,
) -> DecisionResult:
    """Compare means of finite, nonempty scores for a higher-is-better metric."""
    if any(not isfinite(value) or value < 0 for value in (min_gain, max_regression)):
        raise ValueError("Decision thresholds must be finite and nonnegative")
    for scores in (champion_scores, challenger_scores):
        if scores.empty or not np.isfinite(scores).all():
            raise ValueError("Canary scores must be finite and nonempty")
    champ_mean = float(champion_scores.mean())
    chall_mean = float(challenger_scores.mean())
    delta = chall_mean - champ_mean

    if delta >= min_gain:
        return DecisionResult(decision="promote", reason="challenger_gain_exceeds_threshold")
    if delta <= -max_regression:
        return DecisionResult(decision="rollback", reason="challenger_regression_exceeds_threshold")
    return DecisionResult(decision="hold", reason="insufficient_signal")
