from __future__ import annotations

from automl_hpo_showcase.search_random_grid import run_random_search


def test_trials_have_required_columns() -> None:
    frame = run_random_search(budget=3, seed=11, random_state=11)
    required = {"strategy", "trial_id", "n_estimators", "max_depth", "min_samples_split", "score"}
    assert required.issubset(frame.columns)
