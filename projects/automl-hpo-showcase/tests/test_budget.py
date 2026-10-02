from __future__ import annotations

import runpy
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from automl_hpo_showcase import search_random_grid
from automl_hpo_showcase.search_bayes_tpe import run_tpe_search
from automl_hpo_showcase.search_random_grid import run_grid_search, run_random_search
from automl_hpo_showcase.search_space import GRID_SPACE


def test_trial_counts_respect_budget() -> None:
    budget = 5
    assert len(run_grid_search(budget=budget, random_state=2)) == budget
    assert len(run_random_search(budget=budget, seed=2, random_state=2)) == budget
    assert len(run_tpe_search(budget=budget, seed=2)) == budget


@pytest.mark.parametrize("budget", [6, 18])
def test_grid_balances_every_parameter_level(budget: int) -> None:
    trials = run_grid_search(budget=budget, random_state=2)
    for column, values in [("n_estimators", [40, 80, 120]), ("max_depth", [3, 5, 8]),
                           ("min_samples_split", [2, 4, 8])]:
        assert trials[column].value_counts().to_dict() == dict.fromkeys(values, budget // 3)
    assert len(trials.drop_duplicates(["n_estimators", "max_depth", "min_samples_split"])) == budget


def test_full_grid_contains_every_candidate_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(search_random_grid, "score_config", lambda **kwargs: 0.5)
    trials = run_grid_search(budget=27, random_state=2)
    combinations = list(trials[list(GRID_SPACE)].itertuples(index=False, name=None))
    assert len(combinations) == len(set(combinations)) == 27
    assert set(combinations) == set(product(*GRID_SPACE.values()))


def test_searches_use_the_same_candidate_space() -> None:
    random = run_random_search(budget=6, seed=2, random_state=2)
    tpe = run_tpe_search(budget=6, seed=2)
    for trials in [random, tpe]:
        assert set(trials["n_estimators"]).issubset({40, 80, 120})
        assert set(trials["max_depth"]).issubset({3, 5, 8})
        assert set(trials["min_samples_split"]).issubset({2, 4, 8})


def test_budget_plot_and_readme_describe_trial_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    main = runpy.run_path(str(root / "scripts/run_budget_sensitivity.py"))["main"]
    scope = main.__globals__
    scope["__file__"] = str(tmp_path / "scripts/run_budget_sensitivity.py")
    scope["parse_args"] = lambda: SimpleNamespace(quick=True, seed=42, with_hyperopt=False)
    for name in ["run_grid_search", "run_random_search", "run_tpe_search"]:
        scope[name] = lambda **kwargs: pd.DataFrame({"score": [0.75]})
    titles: list[str] = []
    monkeypatch.setattr(scope["plt"], "title", titles.append)
    main()
    scope["plt"].close("all")
    assert titles == ["HPO Trial Budget vs Score"]
    assert "fixed trial budgets" in (root / "README.md").read_text()
