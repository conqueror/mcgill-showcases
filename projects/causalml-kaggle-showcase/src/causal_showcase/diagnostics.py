from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class PropensityDiagnostics:
    scores: np.ndarray
    auc: float
    overlap_share: float


def covariate_balance_table(X: pd.DataFrame, treatment: np.ndarray) -> pd.DataFrame:
    """Compute standardized mean differences (SMD) for all numeric covariates."""
    if len(X) != len(treatment):
        raise ValueError("X and treatment must have the same number of rows.")

    treated_mask = treatment == 1
    control_mask = treatment == 0

    rows: list[dict[str, float | str]] = []
    for col in X.columns:
        x_col = X[col].to_numpy(dtype=float)
        treated = x_col[treated_mask]
        control = x_col[control_mask]

        treated_mean = float(np.mean(treated))
        control_mean = float(np.mean(control))
        treated_var = float(np.var(treated, ddof=1)) if len(treated) > 1 else 0.0
        control_var = float(np.var(control, ddof=1)) if len(control) > 1 else 0.0

        pooled_std = float(np.sqrt((treated_var + control_var) / 2.0))
        smd = 0.0 if pooled_std == 0.0 else float((treated_mean - control_mean) / pooled_std)

        rows.append(
            {
                "feature": str(col),
                "treated_mean": treated_mean,
                "control_mean": control_mean,
                "standardized_mean_difference": smd,
                "abs_smd": abs(smd),
            }
        )

    return pd.DataFrame(rows).sort_values("abs_smd", ascending=False).reset_index(drop=True)


def propensity_diagnostics(
    X: pd.DataFrame,
    treatment: np.ndarray,
    *,
    random_state: int = 42,
) -> PropensityDiagnostics:
    """
    Estimate propensity diagnostics.

    AUC near 0.5 suggests treatment assignment is hard to predict from observed features.
    Higher AUC suggests stronger covariate-treatment dependence (possible confounding risk
    in observational settings).
    """
    x_train, x_test, w_train, w_test = train_test_split(
        X,
        treatment,
        test_size=0.25,
        random_state=random_state,
        stratify=treatment,
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(x_train.to_numpy(), w_train)

    scores_all = model.predict_proba(X.to_numpy())[:, 1]
    scores_test = model.predict_proba(x_test.to_numpy())[:, 1]

    auc = float(roc_auc_score(w_test, scores_test))

    in_overlap = (scores_all >= 0.05) & (scores_all <= 0.95)
    overlap_share = float(np.mean(in_overlap))

    return PropensityDiagnostics(scores=scores_all, auc=auc, overlap_share=overlap_share)
