"""Verify the showcase artifact contract."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> None:
    from modern_nlp_pipeline_showcase.reporting import (
        required_artifact_paths,
        verify_required_artifacts,
    )

    project_root = Path(__file__).resolve().parents[1]
    missing = verify_required_artifacts(project_root, required_artifact_paths())
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise SystemExit(f"Missing required artifacts:\n{formatted}")
    print("Artifact contract verified.")


if __name__ == "__main__":
    main()
