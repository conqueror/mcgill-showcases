from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest


def test_optional_explainers_use_the_audited_model_and_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = Path(__file__).resolve().parents[1] / "scripts/run_audit.py"
    namespace = runpy.run_path(str(path))
    main = namespace["main"]
    scope = main.__globals__
    scope["__file__"] = str(tmp_path / "scripts/run_audit.py")
    monkeypatch.setattr(sys, "argv", [str(path), "--quick", "--with-shap", "--with-lime"])
    audited: list[tuple[Any, pd.DataFrame]] = []
    original = scope["global_feature_importance"]

    def capture(model: Any, x: pd.DataFrame, y: pd.Series, **kwargs: Any) -> pd.DataFrame:
        audited.append((model, x))
        return original(model, x, y, **kwargs)

    scope["global_feature_importance"] = capture
    import ml_core.contracts as contracts
    import ml_core.explainability as explainers

    def shap(model: Any, x: pd.DataFrame, **kwargs: Any) -> str:
        assert model is audited[0][0]
        pd.testing.assert_frame_equal(x, audited[0][1])
        return "written"

    def lime(predict: Any, train: pd.DataFrame, x: pd.DataFrame, **kwargs: Any) -> str:
        assert predict.__self__ is audited[0][0]
        pd.testing.assert_frame_equal(x, audited[0][1])
        return "written"

    monkeypatch.setattr(explainers, "run_shap_importance", shap)
    monkeypatch.setattr(explainers, "run_lime_local_explanations", lime)
    monkeypatch.setattr(contracts, "write_supervised_contract_artifacts", lambda **kwargs: [])
    monkeypatch.setattr(contracts, "merge_required_files", lambda *args: None)
    main()


def test_artifact_verifier_rejects_empty_contracts(tmp_path: Path) -> None:
    import json
    import shutil
    import subprocess
    import sys

    script = tmp_path / "scripts/verify_artifacts.py"
    script.parent.mkdir()
    shutil.copyfile(Path(__file__).resolve().parents[1] / "scripts/verify_artifacts.py", script)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "empty.csv").touch()
    for payload in [{}, {"required_files": []}, {"required_files": ["artifacts/empty.csv"]}]:
        (artifacts / "manifest.json").write_text(json.dumps(payload))
        result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
        assert result.returncode != 0


@pytest.mark.parametrize("column,value", [("tpr", float("nan")), ("fpr", float("inf")),
                                          ("precision", 1.5), ("positive_count", -1),
                                          ("positive_count", 0)])
def test_artifact_verifier_rejects_corrupt_group_metrics(
    tmp_path: Path, column: str, value: float,
) -> None:
    import json
    import shutil
    import subprocess

    script = tmp_path / "scripts/verify_artifacts.py"
    script.parent.mkdir()
    shutil.copyfile(Path(__file__).resolve().parents[1] / "scripts/verify_artifacts.py", script)
    path = tmp_path / "artifacts/fairness/group_metrics.csv"
    path.parent.mkdir(parents=True)
    frame = pd.DataFrame({"group": ["north"], "count": [2], "positive_count": [1],
                          "negative_count": [1], "predicted_positive_count": [1],
                          "selection_rate": [0.5], "tpr": [1.0], "fpr": [0.0],
                          "fnr": [0.0], "precision": [1.0]})
    frame.to_csv(path, index=False)
    (tmp_path / "artifacts/manifest.json").write_text(json.dumps({
        "required_files": ["artifacts/fairness/group_metrics.csv"],
    }))
    valid = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert valid.returncode == 0, valid.stderr
    frame.loc[0, column] = value
    frame.to_csv(path, index=False)
    invalid = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert invalid.returncode != 0
