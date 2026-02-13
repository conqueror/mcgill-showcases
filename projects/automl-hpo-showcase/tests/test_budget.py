from __future__ import annotations

from automl_hpo_showcase.search_bayes_tpe import run_tpe_search
from automl_hpo_showcase.search_random_grid import run_grid_search, run_random_search


def test_trial_counts_respect_budget() -> None:
    budget = 5
    assert len(run_grid_search(budget=budget, random_state=2)) <= budget
    assert len(run_random_search(budget=budget, seed=2, random_state=2)) <= budget
    assert len(run_tpe_search(budget=budget, seed=2)) <= budget
