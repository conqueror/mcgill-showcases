"""Integration tests for the showcase pipeline."""

from __future__ import annotations

from pathlib import Path

from scripts.run_showcase import main as run_showcase_main


def test_run_showcase_generates_required_artifacts(tmp_path: Path) -> None:
    """Running the pipeline should create the agreed artifact set."""

    exit_code = run_showcase_main(["--output-dir", str(tmp_path)])

    assert exit_code == 0
    for artifact_name in (
        "activation_comparison.csv",
        "loss_function_comparison.csv",
        "backprop_gradient_trace.csv",
        "initialization_comparison.csv",
        "underfit_overfit_examples.csv",
        "training_curves.csv",
        "decision_boundary_summary.csv",
        "decision_boundaries.png",
        "summary.md",
    ):
        assert (tmp_path / artifact_name).exists(), artifact_name
