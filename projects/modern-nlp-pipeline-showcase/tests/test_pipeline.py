import subprocess
import sys
from pathlib import Path

from modern_nlp_pipeline_showcase.reporting import required_artifact_paths


def test_required_artifact_paths_match_contract() -> None:
    required = required_artifact_paths()

    assert "artifacts/manifest.json" in required
    assert "artifacts/classification/metrics_summary.csv" in required
    assert "artifacts/retrieval/retrieval_metrics.csv" in required
    assert "artifacts/generation/qa_outputs.csv" in required
    assert "artifacts/summary.md" in required


def test_quick_pipeline_creates_summary_artifact() -> None:
    project_root = Path(__file__).resolve().parents[1]
    summary_path = project_root / "artifacts/summary.md"

    if summary_path.exists():
        summary_path.unlink()

    subprocess.run(
        [sys.executable, "scripts/run_pipeline.py", "--quick"],
        cwd=project_root,
        check=True,
    )

    assert summary_path.exists()
