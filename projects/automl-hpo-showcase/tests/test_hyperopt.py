from __future__ import annotations

import builtins
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from automl_hpo_showcase.search_hyperopt import run_hyperopt_search
from automl_hpo_showcase.search_space import GRID_SPACE


def test_explicit_hyperopt_request_fails_when_import_is_broken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = builtins.__import__

    def fail_hyperopt(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "hyperopt":
            raise ImportError("missing optional dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_hyperopt)
    with pytest.raises(RuntimeError, match="Hyperopt"):
        run_hyperopt_search(budget=2, seed=7)


def test_hyperopt_supplies_a_seeded_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    draws: list[list[int]] = []

    def fmin(**kwargs: Any) -> None:
        rng = kwargs["rstate"]
        assert isinstance(rng, np.random.Generator)
        draws.append(rng.integers(0, 10000, size=5).tolist())

    hyperopt = ModuleType("hyperopt")
    hyperopt.STATUS_OK = "ok"  # type: ignore[attr-defined]
    hyperopt.Trials = lambda: SimpleNamespace(trials=[])  # type: ignore[attr-defined]
    hyperopt.fmin = fmin  # type: ignore[attr-defined]
    hyperopt.hp = SimpleNamespace(  # type: ignore[attr-defined]
        quniform=lambda *args: None, choice=lambda *args: None,
    )
    hyperopt.tpe = SimpleNamespace(suggest=None)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hyperopt", hyperopt)
    run_hyperopt_search(budget=2, seed=7)
    run_hyperopt_search(budget=2, seed=7)
    assert draws[0] == draws[1]


def test_real_hyperopt_repeats_trial_sequence() -> None:
    pytest.importorskip("hyperopt", reason="Optional Hyperopt is not installed")
    first = run_hyperopt_search(budget=3, seed=7)
    second = run_hyperopt_search(budget=3, seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 3


def test_hyperopt_records_candidate_values_instead_of_choice_indexes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fmin(**kwargs: Any) -> None:
        config = {name: values[1] for name, values in kwargs["space"].items()}
        result = kwargs["fn"](config)
        kwargs["trials"].trials.append({
            "result": result, "misc": {"vals": {name: [1] for name in config}},
        })

    hyperopt = ModuleType("hyperopt")
    hyperopt.STATUS_OK = "ok"  # type: ignore[attr-defined]
    hyperopt.Trials = lambda: SimpleNamespace(trials=[])  # type: ignore[attr-defined]
    hyperopt.fmin = fmin  # type: ignore[attr-defined]
    hyperopt.hp = SimpleNamespace(choice=lambda name, values: values)  # type: ignore[attr-defined]
    hyperopt.tpe = SimpleNamespace(suggest=None)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hyperopt", hyperopt)
    monkeypatch.setattr("automl_hpo_showcase.search_hyperopt.score_config", lambda **kwargs: 0.75)
    trials = run_hyperopt_search(budget=1, seed=7)
    assert trials[list(GRID_SPACE)].iloc[0].to_dict() == {
        "n_estimators": 80, "max_depth": 5, "min_samples_split": 4,
    }
    assert trials["score"].tolist() == [0.75]
