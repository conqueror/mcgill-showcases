from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pandas as pd
import pytest
from pytest import MonkeyPatch


@pytest.mark.parametrize(
    "corruption",
    [
        "empty_manifest",
        "empty_file",
        "wrong_csv",
        "broken_csv",
        "empty_batch",
        "empty_parity",
        "fractional_count",
        "wrong_parity_flag",
    ],
)
def test_verifier_rejects_empty_or_corrupt_artifacts(
    tmp_path: Path, monkeypatch: MonkeyPatch, corruption: str
) -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    generate = runpy.run_path(str(scripts / "run_compare_modes.py"))["main"]
    verify = runpy.run_path(str(scripts / "verify_artifacts.py"))["main"]
    for main in (generate, verify):
        monkeypatch.setitem(main.__globals__, "__file__", str(tmp_path / "scripts/run.py"))
    monkeypatch.setattr(sys, "argv", ["run.py", "--quick"])
    generate()
    verify()
    if corruption == "empty_manifest":
        (tmp_path / "artifacts/manifest.json").write_text(json.dumps({"required_files": []}))
    elif corruption in {"empty_batch", "empty_parity", "fractional_count", "wrong_parity_flag"}:
        relative = (
            "artifacts/batch/kpi_output.csv"
            if corruption in {"empty_batch", "fractional_count"}
            else "artifacts/compare/parity_report.csv"
        )
        path = tmp_path / relative
        frame = pd.read_csv(path)
        if corruption.startswith("empty"):
            frame = frame.iloc[:0]
        elif corruption == "fractional_count":
            frame["event_count"] = frame["event_count"].astype(float)
            frame.loc[0, "event_count"] = 0.5
        else:
            frame["within_tolerance"] = ~frame["within_tolerance"]
        frame.to_csv(path, index=False)
    else:
        content = {"empty_file": "", "wrong_csv": "junk\n1\n", "broken_csv": '"unterminated'}
        (tmp_path / "artifacts/events/events.csv").write_text(content[corruption])
    with pytest.raises(SystemExit):
        verify()
