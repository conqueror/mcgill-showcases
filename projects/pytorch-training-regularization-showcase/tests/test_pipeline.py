"""Integration tests for the PyTorch showcase pipeline."""

from __future__ import annotations

from pathlib import Path

from scripts.run_optimizer_comparison import main as run_optimizer_comparison_main
from scripts.run_regularization_ablation import main as run_regularization_main
from scripts.run_showcase import main as run_showcase_main


def test_run_receipt_records_outputs_and_repeated_run_hashes(tmp_path: Path) -> None:
    """The advertised manifest must bind reproducible outputs to source and inputs."""

    import hashlib
    import json

    args = ["--dataset", "synthetic", "--quick"]
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    for output_dir in (first_dir, second_dir):
        assert run_showcase_main([*args, "--output-dir", str(output_dir)]) == 0
    first = json.loads((first_dir / "manifest.json").read_text(encoding="utf-8"))
    second = json.loads((second_dir / "manifest.json").read_text(encoding="utf-8"))
    assert first["sha256"] == second["sha256"]
    assert len(first["git_revision"]) == 40
    assert first["command"] == [
        "scripts/run_showcase.py",
        *args,
        "--output-dir",
        str(first_dir),
    ]
    assert first["random_states"] == [7]
    assert first["environment"]["python"]
    assert first["environment"]["packages"]["numpy"]
    assert first["source_sha256"]
    assert set(first["required_files"]) == set(first["sha256"])
    for name, digest in first["sha256"].items():
        assert hashlib.sha256((first_dir / name).read_bytes()).hexdigest() == digest


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
