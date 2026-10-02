from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from demand_api_observability_showcase.api.app import create_app
from demand_api_observability_showcase.model.demo_training import train_demo_model


@pytest.mark.parametrize(
    "filename,content",
    [
        ("artifacts/model.joblib", "broken model"),
        ("artifacts/metrics.json", "{}"),
        ("openapi.json", "{}"),
    ],
)
def test_verifier_rejects_corrupt_artifacts(
    tmp_path: Path, monkeypatch: MonkeyPatch, filename: str, content: str
) -> None:
    train_demo_model(tmp_path / "artifacts")
    (tmp_path / "openapi.json").write_text(json.dumps(create_app().openapi()))
    script = Path(__file__).resolve().parents[1] / "scripts/verify_artifacts.py"
    verify = runpy.run_path(str(script))["main"]
    monkeypatch.setitem(
        verify.__globals__, "__file__", str(tmp_path / "scripts/verify_artifacts.py")
    )
    verify()
    (tmp_path / filename).write_text(content)
    with pytest.raises(SystemExit):
        verify()
