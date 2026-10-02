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

    summary = (tmp_path / "artifacts/summary.md").read_text()
    assert "recorded macOS and Unix upstream snapshots, not a live source check" in summary


def test_artifact_verifier_rejects_empty_contracts(tmp_path: Path) -> None:
    import json
    import shutil
    import subprocess
    import sys

    script = tmp_path / "scripts/verify_artifacts.py"
    script.parent.mkdir()
    shutil.copyfile(Path(__file__).resolve().parents[1] / "scripts/verify_artifacts.py", script)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "empty.csv").touch()
    for payload in [{}, {"required_files": []}, {"required_files": ["artifacts/empty.csv"]}]:
        (artifacts / "manifest.json").write_text(json.dumps(payload))
        result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
        assert result.returncode != 0
