from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


def _run_script(tmp_path: Path, name: str, *args: str) -> subprocess.CompletedProcess[str]:
    source = Path(__file__).resolve().parents[1] / "scripts" / name
    script = tmp_path / "scripts" / name
    script.parent.mkdir(exist_ok=True)
    shutil.copyfile(source, script)
    return subprocess.run([sys.executable, str(script), *args], capture_output=True,
                          text=True, env=os.environ.copy())


@pytest.mark.parametrize("manifest", [{}, {"required_files": []}])
def test_verifier_rejects_empty_manifest(tmp_path: Path, manifest: dict[str, object]) -> None:
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts/manifest.json").write_text(json.dumps(manifest))
    result = _run_script(tmp_path, "verify_artifacts.py")
    assert result.returncode != 0
    assert "required_files" in result.stderr


def test_run_metadata_is_reproducible(tmp_path: Path) -> None:
    first = _run_script(tmp_path, "run_pipeline.py", "--quick", "--seed", "9")
    assert first.returncode == 0, first.stderr
    meta_path = tmp_path / "artifacts/model/model_meta.json"
    first_meta = json.loads(meta_path.read_text())
    metrics_path = tmp_path / "artifacts/eval/metrics_summary.csv"
    predictions_path = tmp_path / "artifacts/eval/prediction_examples.csv"
    first_metrics = metrics_path.read_text()
    first_predictions = predictions_path.read_text()
    second = _run_script(tmp_path, "run_pipeline.py", "--quick", "--seed", "9")
    assert second.returncode == 0, second.stderr
    assert json.loads(meta_path.read_text()) == first_meta
    assert metrics_path.read_text() == first_metrics
    assert predictions_path.read_text() == first_predictions
    assert first_meta["seed"] == 9
    assert len(first_meta["data_sha256"]) == 64
    assert first_meta["environment"]["python"]
    assert "trained_at_utc" not in first_meta
    changed = _run_script(tmp_path, "run_pipeline.py", "--quick", "--seed", "8")
    assert changed.returncode == 0, changed.stderr
    changed_meta = json.loads(meta_path.read_text())
    assert changed_meta["seed"] == 8
    assert changed_meta["data_sha256"] != first_meta["data_sha256"]
    assert _run_script(tmp_path, "verify_artifacts.py").returncode == 0


def test_verifier_rejects_overlapping_split_boundaries(tmp_path: Path) -> None:
    result = _run_script(tmp_path, "run_pipeline.py", "--quick")
    assert result.returncode == 0, result.stderr
    path = tmp_path / "artifacts/splits/time_split_manifest.json"
    payload = json.loads(path.read_text())
    payload["val_start"] = payload["train_end"]
    path.write_text(json.dumps(payload))
    assert _run_script(tmp_path, "verify_artifacts.py").returncode != 0


def test_verifier_rejects_nonfinite_metrics(tmp_path: Path) -> None:
    result = _run_script(tmp_path, "run_pipeline.py", "--quick")
    assert result.returncode == 0, result.stderr
    path = tmp_path / "artifacts/eval/metrics_summary.csv"
    metrics = pd.read_csv(path)
    metrics.loc[0, "mae"] = float("nan")
    metrics.to_csv(path, index=False)
    assert _run_script(tmp_path, "verify_artifacts.py").returncode != 0
