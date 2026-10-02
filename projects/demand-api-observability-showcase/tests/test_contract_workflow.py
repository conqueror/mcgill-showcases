from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("target", ["check", "smoke"])
def test_workflow_rejects_contract_drift_before_export(tmp_path: Path, target: str) -> None:
    project = Path(__file__).resolve().parents[1]
    shutil.copyfile(project / "Makefile", tmp_path / "Makefile")
    contract = tmp_path / "openapi.json"
    contract.write_text('{"drift": true}\n')
    # Substitute only the unavailable uv entrypoint; exercise the real Makefile ordering.
    uv = tmp_path / "uv"
    uv.write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        "  *export_openapi.py*--check*) if test -f drift-fixed; then exit 0; else exit 1; fi ;;\n"
        "  *export_openapi.py*) touch drift-fixed; echo '{}' > openapi.json ;;\n"
        "esac\n"
    )
    uv.chmod(0o755)
    result = subprocess.run(
        ["make", target],
        cwd=tmp_path,
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert contract.read_text() == '{"drift": true}\n'
