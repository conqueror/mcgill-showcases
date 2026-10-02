from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from causal_showcase.data import PreparedData
from causal_showcase.modeling import LearnerResult, UpliftTreeResult


@pytest.mark.parametrize("script", ["run_pipeline.py", "policy_simulator.py"])
def test_test_outcomes_cannot_choose_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, script: str,
) -> None:
    path = Path(__file__).resolve().parents[1] / "scripts" / script
    main = runpy.run_path(str(path))["main"]
    scope = main.__globals__
    scope["ARTIFACTS_DIR"] = tmp_path / "artifacts"
    scope["FIGURES_DIR"] = tmp_path / "artifacts/figures"
    scope["REPORT_PATH"] = tmp_path / "artifacts/metrics_summary.csv"
    scope["TREE_SUMMARY_PATH"] = tmp_path / "artifacts/uplift_tree.txt"
    ids = np.arange(1200)
    effect = (ids // 2) % 2
    treatment = ids % 2
    prepared = PreparedData(pd.DataFrame({"row_id": ids, "effect": effect}), treatment,
                            treatment * effect, ["row_id", "effect"])
    holder = [prepared]
    scope["load_marketing_ab_data"] = lambda _: holder[0]
    fitted_outcomes: list[np.ndarray] = []
    selected: list[str] = []

    def learners(train: PreparedData, evaluate: PreparedData) -> dict[str, LearnerResult]:
        fitted_outcomes.append(train.outcome.copy())
        scores = evaluate.X["effect"].to_numpy(dtype=float)
        return {name: LearnerResult(name, 0.5, 0.4, 0.6, sign * scores)
                for name, sign in [("S", 1), ("T", -1)]}

    scope["fit_meta_learners"] = learners
    scope["fit_uplift_tree"] = lambda train, evaluate: UpliftTreeResult(
        np.zeros(len(evaluate.X)), "root -> control and treated leaves\n",
    )
    scope["plot_qini_curves"] = lambda *args: None
    scope["plot_uplift_distribution"] = lambda scores, name, path: selected.append(name)
    import ml_core.contracts as contracts
    monkeypatch.setattr(contracts, "write_supervised_contract_artifacts", lambda **kwargs: [])
    monkeypatch.setattr(contracts, "merge_required_files", lambda *args: None)
    csv = tmp_path / "fixture.csv"
    csv.write_text("fixture supplied by loader\n")

    def run() -> set[str]:
        if script == "policy_simulator.py":
            main(data_path=csv, budgets="0.3,0.5")
            return set(pd.read_csv(tmp_path / "artifacts/policy_best_models.csv")["model"])
        main(data_path=csv)
        return {selected[-1]}

    first = run()
    if "train_val_test_split_prepared" in scope:
        _, _, test = scope["train_val_test_split_prepared"](prepared)
    else:
        _, test = scope["train_test_split_prepared"](prepared)
    test_ids = test.X["row_id"].to_numpy(dtype=int)
    changed = prepared.outcome.copy()
    changed[test_ids] = treatment[test_ids] * (1 - effect[test_ids])
    holder[0] = PreparedData(prepared.X, treatment, changed, prepared.feature_names)
    second = run()
    np.testing.assert_array_equal(fitted_outcomes[0], fitted_outcomes[1])
    assert first == second
    if script == "run_pipeline.py":
        report = pd.read_csv(scope["REPORT_PATH"])
        assert set(report["baseline_ate_population"]) == {"test"}
        assert set(report.loc[report["model"] != "Uplift Tree (KL)",
                              "estimated_ate_population"]) == {"train"}


@pytest.mark.parametrize("notebook,cells", [
    ("04_qini_and_targeting_policy", [2, 4, 8, 10]),
    ("05_capstone_policy_simulation", [2, 4, 6]),
    ("07_shap_interpretability", [2]),
])
def test_notebook_selection_ignores_test_outcomes(
    monkeypatch: pytest.MonkeyPatch, notebook: str, cells: list[int],
) -> None:
    from causal_showcase import data, modeling

    ids = np.arange(1200)
    effect, treatment = (ids // 2) % 2, ids % 2
    prepared = PreparedData(pd.DataFrame({"row_id": ids, "effect": effect}), treatment,
                            treatment * effect, ["row_id", "effect"])
    holder = [prepared]
    monkeypatch.setattr(data, "load_marketing_ab_data", lambda _: holder[0])

    def learners(train: PreparedData, evaluate: PreparedData) -> dict[str, LearnerResult]:
        scores = evaluate.X["effect"].to_numpy(dtype=float)
        return {name: LearnerResult(name, 0.5, 0.4, 0.6, sign * scores)
                for name, sign in [("S", 1), ("T", -1)]}

    monkeypatch.setattr(modeling, "fit_meta_learners", learners)
    monkeypatch.setattr(modeling, "fit_uplift_tree", lambda train, evaluate: UpliftTreeResult(
        np.zeros(len(evaluate.X)), "root -> control and treated leaves\n",
    ))
    path = Path(__file__).resolve().parents[1] / "notebooks" / f"{notebook}.ipynb"
    content = json.loads(path.read_text())

    def run() -> tuple[set[str], PreparedData]:
        scope: dict[str, Any] = {}
        for cell in cells:
            exec("".join(content["cells"][cell]["source"]), scope)
        if "best_df" in scope:
            winners = set(scope["best_df"]["model"])
        elif "best_model_by_budget" in scope:
            best = scope["best_model_by_budget"](pd.DataFrame(scope["budget_rows"]))
            winners = set(best["model"])
        else:
            winners = {scope["best_model"]}
        return winners, scope["test_data"]

    first, test = run()
    test_ids = test.X["row_id"].to_numpy(dtype=int)
    changed = prepared.outcome.copy()
    changed[test_ids] = treatment[test_ids] * (1 - effect[test_ids])
    holder[0] = PreparedData(prepared.X, treatment, changed, prepared.feature_names)
    second, _ = run()
    assert first == second
