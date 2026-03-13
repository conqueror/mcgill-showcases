"""Tests for artifact verification helpers."""

from __future__ import annotations

from pathlib import Path

from scripts.verify_artifacts import main as verify_artifacts_main


def test_verify_artifacts_fails_when_required_files_are_missing(tmp_path: Path) -> None:
    """Artifact verification should fail loudly on incomplete output directories."""

    exit_code = verify_artifacts_main(["--output-dir", str(tmp_path)])
    assert exit_code == 1


def test_verify_artifacts_succeeds_for_complete_outputs(tmp_path: Path) -> None:
    """Artifact verification should pass when the expected outputs exist."""

    for artifact_name in (
        "baseline_metrics.json",
        "training_history.csv",
        "optimizer_comparison.csv",
        "learning_rate_schedule_comparison.csv",
        "regularization_ablation.csv",
        "gradient_health_report.md",
        "error_analysis.csv",
        "summary.md",
    ):
        (tmp_path / artifact_name).write_text("placeholder", encoding="utf-8")

    exit_code = verify_artifacts_main(["--output-dir", str(tmp_path)])
    assert exit_code == 0
