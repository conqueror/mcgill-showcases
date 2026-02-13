from __future__ import annotations

from xai_fairness_showcase.data import make_audit_dataset
from xai_fairness_showcase.mitigation import run_mitigation_benchmark


def test_mitigation_table_contains_all_strategies() -> None:
    split = make_audit_dataset(n_samples=500, random_state=7)
    table = run_mitigation_benchmark(
        split.x_train,
        split.y_train,
        split.g_train,
        split.x_test,
        split.y_test,
        split.g_test,
    )

    strategies = set(table["strategy"].tolist())
    assert strategies == {
        "baseline",
        "pre_processing_reweight",
        "in_processing_balanced",
        "post_processing_threshold",
    }
