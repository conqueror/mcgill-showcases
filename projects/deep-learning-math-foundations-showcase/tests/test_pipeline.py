"""Integration tests for the showcase pipeline."""

from __future__ import annotations

from pathlib import Path

from scripts.run_showcase import main as run_showcase_main


def test_run_showcase_generates_required_artifacts(tmp_path: Path) -> None:
    """Running the pipeline should create the agreed artifact set."""

    exit_code = run_showcase_main(["--output-dir", str(tmp_path)])

    assert exit_code == 0
    for artifact_name in (
        "vector_operations.csv",
        "matrix_transformations.csv",
        "derivative_examples.csv",
        "gradient_descent_trace.csv",
        "probability_simulations.csv",
        "information_theory_summary.md",
        "summary.md",
    ):
        assert (tmp_path / artifact_name).exists(), artifact_name
