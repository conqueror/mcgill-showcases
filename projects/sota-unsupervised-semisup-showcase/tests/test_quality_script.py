import os
import subprocess
from pathlib import Path


def test_quality_script_needs_neither_loan_csv_nor_mermaid(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    script = scripts / "review_quality.sh"
    script.write_bytes((project_root / "scripts/review_quality.sh").read_bytes())
    validator = scripts / "validate_mermaid.sh"
    validator.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    validator.chmod(0o755)
    executable = tmp_path / "uv"
    executable.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALL_LOG"\ncase "$*" in *business*) exit 98;; esac\n',
        encoding="utf-8",
    )
    executable.chmod(0o755)
    call_log = tmp_path / "calls.txt"
    result = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "CALL_LOG": str(call_log)},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert calls[:3] == ["run ruff check src tests", "run ty check src tests", "run pytest"]
    assert len(calls) == 4
    assert "--dataset digits" in calls[-1]
