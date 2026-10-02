"""Regression tests for generated artifacts and their integrity checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pytorch_training_regularization_showcase import config
from scripts.run_showcase import main as run_showcase_main
from scripts.verify_artifacts import main as verify_artifacts_main

REQUIRED_FILES = [
    "baseline_metrics.json",
    "training_history.csv",
    "optimizer_comparison.csv",
    "learning_rate_schedule_comparison.csv",
    "regularization_ablation.csv",
    "gradient_health_report.md",
    "error_analysis.csv",
    "summary.md",
]


def _run(output_dir: Path) -> None:
    tmp_path = output_dir
    assert (
        run_showcase_main(
            ["--dataset", "synthetic", "--quick", "--output-dir", str(tmp_path)]
        )
        == 0
    )


def test_verify_artifacts_fails_when_required_files_are_missing(tmp_path: Path) -> None:
    assert verify_artifacts_main(["--output-dir", str(tmp_path)]) == 1


def test_verify_artifacts_succeeds_for_complete_outputs(tmp_path: Path) -> None:
    _run(tmp_path)
    assert verify_artifacts_main(["--output-dir", str(tmp_path)]) == 0


@pytest.mark.parametrize("content", ["", "placeholder"])
def test_verifier_rejects_empty_or_placeholder_outputs(
    tmp_path: Path, content: str
) -> None:
    for name in REQUIRED_FILES:
        (tmp_path / name).write_text(content, encoding="utf-8")
    assert verify_artifacts_main(["--output-dir", str(tmp_path)]) == 1


@pytest.mark.parametrize(
    "name",
    ["training_history.csv", "summary.md", "manifest.json", "baseline_metrics.json"],
)
def test_verifier_rejects_corruption(tmp_path: Path, name: str) -> None:
    _run(tmp_path)
    assert verify_artifacts_main(["--output-dir", str(tmp_path)]) == 0
    (tmp_path / name).write_text("corrupt", encoding="utf-8")
    assert verify_artifacts_main(["--output-dir", str(tmp_path)]) == 1


def test_manifest_cannot_remove_the_required_contract(tmp_path: Path) -> None:
    _run(tmp_path)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"required_files": []}), encoding="utf-8"
    )
    assert verify_artifacts_main(["--output-dir", str(tmp_path)]) == 1


def test_verifier_uses_the_selected_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "selected"
    default_dir = tmp_path / "default"
    _run(output_dir)
    default_dir.mkdir()
    (default_dir / "manifest.json").write_text(
        json.dumps({"required_files": ["unrelated.csv"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "ARTIFACTS_DIR", default_dir)
    assert verify_artifacts_main(["--output-dir", str(output_dir)]) == 0
