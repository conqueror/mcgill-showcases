from __future__ import annotations

import numpy as np
import pandas as pd

from xai_fairness_showcase.data import make_audit_dataset
from xai_fairness_showcase.explainability import (
    global_feature_importance,
    local_linear_contributions,
)
from xai_fairness_showcase.mitigation import train_baseline_model


def test_explainability_outputs_not_empty() -> None:
    split = make_audit_dataset(n_samples=500, random_state=12)
    model = train_baseline_model(split.x_train, split.y_train)

    global_scores = global_feature_importance(model, split.x_test, split.y_test, random_state=12)
    local_scores = local_linear_contributions(model, split.x_test, n_rows=5)

    assert not global_scores.empty
    assert not local_scores.empty
    assert {"feature", "importance_mean", "importance_std"}.issubset(global_scores.columns)


def test_local_terms_reconstruct_log_odds_with_intercept() -> None:
    split = make_audit_dataset(n_samples=500, random_state=12)
    model = train_baseline_model(split.x_train, split.y_train)
    local = local_linear_contributions(model, split.x_test, n_rows=5)
    assert (local["feature"] == "intercept").sum() == 5
    terms = local.groupby("sample_id")["contribution"].sum().to_numpy()
    np.testing.assert_allclose(terms, model.decision_function(split.x_test.iloc[:5]), atol=1e-12)


def test_local_terms_match_hand_computed_log_odds() -> None:
    x = pd.DataFrame({"a": [-1.0, 1.0, -1.0, 1.0], "b": [-2.0, -2.0, 2.0, 2.0]})
    model = train_baseline_model(x, pd.Series([0, 1, 0, 1]))
    model.named_steps["clf"].coef_ = np.array([[2.0, -1.0]])
    model.named_steps["clf"].intercept_ = np.array([0.25])
    local = local_linear_contributions(model, pd.DataFrame({"a": [1.0], "b": [2.0]}))
    # Training means are 0, scales are [1, 2]; standardized input [1, 1].
    # Log odds = intercept .25 + 2*1 - 1*1 = 1.25.
    assert local.set_index("feature")["contribution"].to_dict() == {
        "intercept": 0.25, "a": 2.0, "b": -1.0,
    }
    assert local["contribution"].sum() == 1.25
