from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


def global_feature_importance(
    model: Pipeline,
    x_eval: pd.DataFrame,
    y_eval: pd.Series,
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute global feature importance using permutation scores."""
    result = permutation_importance(
        model,
        x_eval,
        y_eval,
        scoring="roc_auc",
        n_repeats=8,
        random_state=random_state,
    )
    return (
        pd.DataFrame(
            {
                "feature": x_eval.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values(by="importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def local_linear_contributions(
    model: Pipeline,
    x_eval: pd.DataFrame,
    *,
    n_rows: int = 25,
) -> pd.DataFrame:
    """Approximate local feature contributions for a linear model pipeline."""
    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]
    transformed = scaler.transform(x_eval.iloc[:n_rows])
    coefs = clf.coef_[0]
    contributions = transformed * coefs

    rows: list[dict[str, float | int | str]] = []
    for idx, contrib_row in enumerate(contributions):
        for feat_name, value in zip(x_eval.columns, contrib_row, strict=True):
            rows.append(
                {
                    "sample_id": idx,
                    "feature": str(feat_name),
                    "contribution": float(value),
                    "abs_contribution": float(np.abs(value)),
                }
            )

    return pd.DataFrame(rows)
