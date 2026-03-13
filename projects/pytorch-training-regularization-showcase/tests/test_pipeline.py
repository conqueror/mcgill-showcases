"""Integration tests for the PyTorch showcase pipeline."""

from __future__ import annotations

from pathlib import Path

from scripts.run_optimizer_comparison import main as run_optimizer_comparison_main
from scripts.run_regularization_ablation import main as run_regularization_main
from scripts.run_showcase import main as run_showcase_main


def test_run_showcase_generates_required_artifacts(tmp_path: Path) -> None:
    """Running the full pipeline should create the agreed artifact set."""

    exit_code = run_showcase_main(
        [
            "--dataset",
            "synthetic",
            "--quick",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert exit_code == 0
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
        assert (tmp_path / artifact_name).exists(), artifact_name


def test_auxiliary_scripts_generate_their_target_artifacts(tmp_path: Path) -> None:
    """Specialized experiment scripts should write their own outputs."""

    optimizer_exit_code = run_optimizer_comparison_main(
        ["--dataset", "synthetic", "--quick", "--output-dir", str(tmp_path)],
    )
    regularization_exit_code = run_regularization_main(
        ["--dataset", "synthetic", "--quick", "--output-dir", str(tmp_path)],
    )

    assert optimizer_exit_code == 0
    assert regularization_exit_code == 0
    assert (tmp_path / "optimizer_comparison.csv").exists()
    assert (tmp_path / "regularization_ablation.csv").exists()
