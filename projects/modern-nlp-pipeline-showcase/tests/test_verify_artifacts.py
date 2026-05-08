from pathlib import Path

from modern_nlp_pipeline_showcase.reporting import (
    required_artifact_paths,
    verify_required_artifacts,
)


def test_verify_required_artifacts_detects_missing_files(tmp_path: Path) -> None:
    missing = verify_required_artifacts(tmp_path, required_artifact_paths())

    assert "artifacts/manifest.json" in missing
