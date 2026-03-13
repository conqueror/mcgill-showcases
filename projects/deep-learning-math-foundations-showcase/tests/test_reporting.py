"""Tests for reporting helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from deep_learning_math_foundations_showcase import reporting


def test_expected_artifact_paths_are_stable(tmp_path: Path) -> None:
    """Artifact path planning should be deterministic."""

    paths = reporting.expected_artifact_paths(tmp_path)
    assert paths["vector_operations"] == tmp_path / "vector_operations.csv"
    assert paths["summary"] == tmp_path / "summary.md"


def test_summary_markdown_contains_required_sections() -> None:
    """The summary should include predictable section headers for docs."""

    summary = reporting.build_summary_markdown(
        project_title="Deep Learning Math Foundations Showcase",
        highlights=["Gradient descent reduces loss.", "Entropy measures uncertainty."],
        next_steps=[
            "Read docs/learning-flow.md",
            "Inspect artifacts/gradient_descent_trace.csv",
        ],
    )

    assert "# Deep Learning Math Foundations Showcase" in summary
    assert "## Highlights" in summary
    assert "## Next Steps" in summary


def test_write_csv_artifact_creates_file(tmp_path: Path) -> None:
    """CSV artifact helpers should write stable outputs to disk."""

    table = pd.DataFrame([{"operation": "add", "result": "[4.0, 6.0]"}])
    destination = tmp_path / "vector_operations.csv"

    reporting.write_csv_artifact(table, destination)

    assert destination.exists()
