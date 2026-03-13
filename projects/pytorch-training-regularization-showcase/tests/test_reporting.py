"""Tests for artifact-reporting helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pytorch_training_regularization_showcase import reporting


def test_expected_artifact_paths_are_stable(tmp_path: Path) -> None:
    """Artifact path planning should remain deterministic."""

    paths = reporting.expected_artifact_paths(tmp_path)

    assert paths["baseline_metrics"] == tmp_path / "baseline_metrics.json"
    assert paths["optimizer_comparison"] == tmp_path / "optimizer_comparison.csv"
    assert paths["summary"] == tmp_path / "summary.md"


def test_summary_markdown_contains_required_sections() -> None:
    """The top-level summary should stay easy to scan."""

    summary = reporting.build_summary_markdown(
        project_title="PyTorch Training Regularization Showcase",
        highlights=["Batch norm stabilizes hidden activations."],
        next_steps=["Inspect artifacts/optimizer_comparison.csv."],
    )

    assert "# PyTorch Training Regularization Showcase" in summary
    assert "## Highlights" in summary
    assert "## Next Steps" in summary


def test_write_csv_artifact_creates_file(tmp_path: Path) -> None:
    """CSV artifact helpers should write to disk predictably."""

    destination = tmp_path / "optimizer_comparison.csv"
    table = pd.DataFrame([{"optimizer": "adam", "test_accuracy": 0.9}])

    reporting.write_csv_artifact(table, destination)

    assert destination.exists()
