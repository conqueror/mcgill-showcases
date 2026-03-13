from __future__ import annotations

import json
from pathlib import Path

from autoresearch_showcase.reporting import build_showcase


def test_build_showcase_writes_manifest_and_required_artifacts(tmp_path: Path) -> None:
    written = build_showcase(tmp_path)
    assert written

    manifest_path = tmp_path / "artifacts/manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative_path in payload["required_files"]:
        assert (tmp_path / relative_path).exists()
