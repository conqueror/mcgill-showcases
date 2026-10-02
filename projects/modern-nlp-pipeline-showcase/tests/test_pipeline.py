import subprocess
import sys
from pathlib import Path

from modern_nlp_pipeline_showcase.reporting import (
    required_artifact_paths,
    verify_required_artifacts,
)


def test_required_artifact_paths_match_contract() -> None:
    required = required_artifact_paths()

    assert "artifacts/manifest.json" in required
    assert "artifacts/classification/metrics_summary.csv" in required
    assert "artifacts/retrieval/retrieval_metrics.csv" in required
    assert "artifacts/generation/qa_outputs.csv" in required
    assert "artifacts/summary.md" in required


def test_quick_pipeline_creates_summary_artifact(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    existing_summary = project_root / "artifacts/summary.md"
    original_mtime = existing_summary.stat().st_mtime_ns if existing_summary.exists() else None
    summary_path = tmp_path / "artifacts/summary.md"

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; from pathlib import Path; "
            "from modern_nlp_pipeline_showcase import config; "
            "config.ARTIFACTS_DIR = Path(sys.argv[1]) / 'artifacts'; "
            "sys.argv = ['scripts/run_pipeline.py', '--quick']; "
            "runpy.run_path('scripts/run_pipeline.py', run_name='__main__')",
            str(tmp_path),
        ],
        cwd=project_root,
        check=True,
    )

    assert summary_path.exists()
    assert verify_required_artifacts(tmp_path, required_artifact_paths()) == []
    assert (
        existing_summary.stat().st_mtime_ns if existing_summary.exists() else None
    ) == original_mtime


def test_pipeline_rejects_nonpositive_top_k() -> None:
    project_root = Path(__file__).resolve().parents[1]
    for value in ("0", "-1"):
        result = subprocess.run(
            [sys.executable, "scripts/run_pipeline.py", "--quick", "--top-k", value],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 2
        assert "--top-k must be positive" in result.stderr
