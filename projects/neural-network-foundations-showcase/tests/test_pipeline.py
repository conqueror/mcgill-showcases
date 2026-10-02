"""Integration tests for the showcase pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import run_showcase
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
    assert first["random_states"] == [7, 8, 9, 10, 11]
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


def test_best_xor_highlight_selects_the_maximum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The last XOR recipe need not have the highest validation accuracy."""

    original = run_showcase._build_decision_boundary_outputs

    def controlled_results():
        table, experiments = original()
        table["validation_accuracy"] = [1.0, 0.8, 0.6]
        return table, experiments

    monkeypatch.setattr(
        run_showcase, "_build_decision_boundary_outputs", controlled_results
    )
    assert run_showcase_main(["--output-dir", str(tmp_path)]) == 0
    assert (
        "Best XOR validation accuracy: 0.800" in (tmp_path / "summary.md").read_text()
    )
