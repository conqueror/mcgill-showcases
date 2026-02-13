from __future__ import annotations

from automl_hpo_showcase.objective import score_config


def test_objective_is_deterministic_with_fixed_seed() -> None:
    score_a = score_config(
        n_estimators=60,
        max_depth=5,
        min_samples_split=4,
        random_state=21,
    )
    score_b = score_config(
        n_estimators=60,
        max_depth=5,
        min_samples_split=4,
        random_state=21,
    )
    assert score_a == score_b
