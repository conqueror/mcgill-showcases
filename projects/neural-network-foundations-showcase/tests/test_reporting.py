"""Tests for artifact-reporting helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from neural_network_foundations_showcase import reporting


def test_expected_artifact_paths_are_stable(tmp_path: Path) -> None:
    """Artifact path planning should remain deterministic."""

    paths = reporting.expected_artifact_paths(tmp_path)

    assert paths["activation_comparison"] == tmp_path / "activation_comparison.csv"
    assert paths["decision_boundaries"] == tmp_path / "decision_boundaries.png"
    assert paths["summary"] == tmp_path / "summary.md"


def test_summary_markdown_contains_required_sections() -> None:
    """The top-level summary should be easy to scan."""

    summary = reporting.build_summary_markdown(
        project_title="Neural Network Foundations Showcase",
        highlights=["Backprop turns prediction errors into weight updates."],
        next_steps=["Open artifacts/decision_boundary_summary.csv."],
    )

    assert "# Neural Network Foundations Showcase" in summary
    assert "## Highlights" in summary
    assert "## Next Steps" in summary


def test_write_csv_artifact_creates_file(tmp_path: Path) -> None:
    """CSV artifact helpers should write to disk predictably."""

    destination = tmp_path / "activation_comparison.csv"
    table = pd.DataFrame([{"input": 0.0, "sigmoid": 0.5}])

    reporting.write_csv_artifact(table, destination)

    assert destination.exists()
