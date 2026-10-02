from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pandas as pd
import pytest
from pytest import MonkeyPatch


@pytest.mark.parametrize("check", ["thresholds", "registry", "metric"])
def test_rollout_records_thresholds_active_version_and_metric(
    tmp_path: Path, monkeypatch: MonkeyPatch, check: str
) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/run_rollout.py"
    main = runpy.run_path(str(script))["main"]
    monkeypatch.setitem(main.__globals__, "__file__", str(tmp_path / "scripts/run_rollout.py"))
    monkeypatch.setattr(sys, "argv", [str(script), "--quick", "--seed", "42"])
    main()
    decision = json.loads((tmp_path / "artifacts/rollout/decision_log.json").read_text())
    assert decision["decision"] == "promote"
    if check == "thresholds":
        assert decision["min_gain"] == 0.005
        assert decision["max_regression"] == 0.01
    elif check == "registry":
        registry = json.loads((tmp_path / "artifacts/registry/model_versions.json").read_text())
        assert registry["active_version"] == "v1.5.0"
        assert registry["rollback_target"] == "v1.4.0"
    else:
        assert decision["metric"] == "synthetic_score"
        assert decision["metric_direction"] == "higher_is_better"
        assert decision["metric_definition"] == "mean of generated scores; not measured ROC AUC"
        evaluation = pd.read_csv(tmp_path / "artifacts/rollout/canary_eval.csv")
        assert evaluation.columns.tolist() == ["champion_score", "challenger_score", "delta"]


@pytest.mark.parametrize(
    "corruption",
    [
        "empty_manifest",
        "wrong_csv",
        "empty_decision",
        "wrong_mean",
        "wrong_delta",
        "wrong_registry_decision",
    ],
)
def test_verifier_rejects_empty_or_corrupt_artifacts(
    tmp_path: Path, monkeypatch: MonkeyPatch, corruption: str
) -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    generate = runpy.run_path(str(scripts / "run_rollout.py"))["main"]
    verify = runpy.run_path(str(scripts / "verify_artifacts.py"))["main"]
    for main in (generate, verify):
        monkeypatch.setitem(main.__globals__, "__file__", str(tmp_path / "scripts/run.py"))
    monkeypatch.setattr(sys, "argv", ["run.py", "--quick"])
    generate()
    verify()
    if corruption in {"wrong_mean", "wrong_registry_decision"}:
        relative = (
            "artifacts/rollout/decision_log.json"
            if corruption == "wrong_mean"
            else "artifacts/registry/model_versions.json"
        )
        path = tmp_path / relative
        payload = json.loads(path.read_text())
        if corruption == "wrong_mean":
            payload["champion_mean_score"] = float("nan")
        else:
            payload["decision"] = "hold"
        path.write_text(json.dumps(payload))
    elif corruption == "wrong_delta":
        path = tmp_path / "artifacts/rollout/canary_eval.csv"
        evaluation = pd.read_csv(path)
        evaluation["delta"] = 100.0
        evaluation.to_csv(path, index=False)
    else:
        corruptions = {
            "empty_manifest": ("artifacts/manifest.json", '{"required_files": []}'),
            "wrong_csv": ("artifacts/rollout/canary_eval.csv", "junk\n1\n"),
            "empty_decision": ("artifacts/rollout/decision_log.json", "{}"),
        }
        filename, content = corruptions[corruption]
        (tmp_path / filename).write_text(content)
    with pytest.raises(SystemExit):
        verify()
