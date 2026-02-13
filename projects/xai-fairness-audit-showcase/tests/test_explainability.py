from __future__ import annotations

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
