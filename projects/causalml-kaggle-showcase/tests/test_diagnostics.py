from __future__ import annotations

import numpy as np
import pandas as pd

from causal_showcase.diagnostics import covariate_balance_table, propensity_diagnostics


def test_covariate_balance_table_contains_smd_columns() -> None:
    X = pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5, 6],
            "x2": [0, 0, 1, 1, 0, 1],
        }
    )
    treatment = np.array([0, 0, 0, 1, 1, 1])

    balance = covariate_balance_table(X, treatment)

    assert set(balance.columns) == {
        "feature",
        "treated_mean",
        "control_mean",
        "standardized_mean_difference",
        "abs_smd",
    }
    assert balance.shape[0] == X.shape[1]


def test_propensity_diagnostics_returns_valid_metrics() -> None:
    rng = np.random.default_rng(7)
    n = 300
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    logits = 0.8 * x1 - 0.3 * x2
    p = 1.0 / (1.0 + np.exp(-logits))
    treatment = rng.binomial(1, p)

    X = pd.DataFrame({"x1": x1, "x2": x2})
    diagnostics = propensity_diagnostics(X, treatment)

    assert diagnostics.scores.shape[0] == n
    assert 0.0 <= diagnostics.auc <= 1.0
    assert 0.0 <= diagnostics.overlap_share <= 1.0
