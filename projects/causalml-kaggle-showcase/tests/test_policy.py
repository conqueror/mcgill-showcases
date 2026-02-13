from __future__ import annotations

import numpy as np

from causal_showcase.policy import select_best_model_per_budget, simulate_policy_table


def test_simulate_policy_table_shape_and_columns() -> None:
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    treatment = np.array([1, 0, 1, 0, 1, 0, 0, 1])

    score_by_model = {
        "Model-A": np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4]),
        "Model-B": np.array([0.4, 0.5, 0.3, 0.2, 0.8, 0.7, 0.1, 0.9]),
    }

    budgets = [0.25, 0.5]
    policy_df = simulate_policy_table(y, treatment, score_by_model, budgets)

    assert policy_df.shape[0] == len(score_by_model) * len(budgets)
    assert set(policy_df.columns) == {
        "model",
        "budget_fraction",
        "targeted_users",
        "uplift_rate",
        "expected_incremental_conversions",
    }


def test_select_best_model_per_budget_returns_one_per_budget() -> None:
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    treatment = np.array([1, 0, 1, 0, 1, 0, 0, 1])
    score_by_model = {
        "Strong": np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4]),
        "Weak": np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]),
    }

    policy_df = simulate_policy_table(y, treatment, score_by_model, [0.25, 0.5, 0.75])
    best_df = select_best_model_per_budget(policy_df)

    assert best_df["budget_fraction"].nunique() == 3
    assert best_df.shape[0] == 3
