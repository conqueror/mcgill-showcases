from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_empirical_ate(y: np.ndarray, treatment: np.ndarray) -> float:
    treated = y[treatment == 1]
    control = y[treatment == 0]
    if len(treated) == 0 or len(control) == 0:
        return float("nan")
    return float(treated.mean() - control.mean())


def simulate_policy_table(
    y: np.ndarray,
    treatment: np.ndarray,
    score_by_model: dict[str, np.ndarray],
    budgets: list[float],
) -> pd.DataFrame:
    """
    Simulate treatment targeting policies for multiple models and budget levels.

    Budgets are fractions in (0, 1], interpreted as top-k users selected
    by each model's uplift scores.
    """
    rows: list[dict[str, float | int | str]] = []
    n_total = len(y)

    for budget in budgets:
        if not 0 < budget <= 1:
            raise ValueError(f"Budget must be in (0, 1], got {budget}.")

    for model_name, scores in score_by_model.items():
        if len(scores) != n_total:
            raise ValueError(
                "Model "
                f"`{model_name}` score length {len(scores)} "
                f"does not match y length {n_total}."
            )

        for budget in budgets:
            n_target = max(1, int(n_total * budget))
            ranked_idx = np.argsort(-scores)[:n_target]

            y_target = y[ranked_idx]
            w_target = treatment[ranked_idx]
            uplift = _safe_empirical_ate(y_target, w_target)

            rows.append(
                {
                    "model": model_name,
                    "budget_fraction": budget,
                    "targeted_users": int(n_target),
                    "uplift_rate": float(uplift),
                    "expected_incremental_conversions": float(uplift * n_target),
                }
            )

    return pd.DataFrame(rows)


def select_best_model_per_budget(policy_df: pd.DataFrame) -> pd.DataFrame:
    """Return the top model for each budget based on expected incremental conversions."""
    required = {
        "model",
        "budget_fraction",
        "expected_incremental_conversions",
        "uplift_rate",
        "targeted_users",
    }
    missing = sorted(required - set(policy_df.columns))
    if missing:
        raise ValueError(f"Policy dataframe missing columns: {missing}")

    ranking = policy_df.sort_values(
        ["budget_fraction", "expected_incremental_conversions"],
        ascending=[True, False],
    )
    best = ranking.groupby("budget_fraction", as_index=False).head(1).reset_index(drop=True)
    return best
