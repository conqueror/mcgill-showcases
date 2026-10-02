import json
import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("required", "content"),
    [
        ([], "a\n1\n"),
        ([""], "a\n1\n"),
        (["artifacts/data.csv"], ""),
        (["artifacts/data.csv"], "a\n"),
        (["artifacts/data.csv"], "a,b\n1\n"),
        (["artifacts/data.json"], "{"),
        (["artifacts/data.json"], "{}"),
    ],
)
def test_verifier_rejects_empty_or_corrupt_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, required: list[str], content: str
) -> None:
    root = Path(__file__).resolve().parents[1]
    namespace = runpy.run_path(str(root / "scripts/verify_artifacts.py"))
    monkeypatch.setitem(
        namespace["main"].__globals__, "__file__", str(tmp_path / "scripts/verify_artifacts.py")
    )
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "manifest.json").write_text(json.dumps({"required_files": required}))
    if required and required[0]:
        (tmp_path / required[0]).write_text(content)
    with pytest.raises((SystemExit, ValueError)):
        namespace["main"]()


@pytest.mark.parametrize("table", ["metric,value\nf1,0.5\n", ",a,b\nrow,1,2\n"])
def test_verifier_accepts_nonempty_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, table: str
) -> None:
    root = Path(__file__).resolve().parents[1]
    namespace = runpy.run_path(str(root / "scripts/verify_artifacts.py"))
    monkeypatch.setitem(
        namespace["main"].__globals__, "__file__", str(tmp_path / "scripts/verify_artifacts.py")
    )
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "manifest.json").write_text(json.dumps({"required_files": ["artifacts/data.csv"]}))
    (artifacts / "data.csv").write_text(table)
    namespace["main"]()
