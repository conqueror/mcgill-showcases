"""Integration tests for the showcase pipeline."""

from __future__ import annotations

from pathlib import Path

from scripts.run_showcase import main as run_showcase_main


def test_run_receipt_records_outputs_and_repeated_run_hashes(tmp_path: Path) -> None:
    """The advertised manifest must bind reproducible outputs to source and inputs."""

    import hashlib
    import json

    args: list[str] = []
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
