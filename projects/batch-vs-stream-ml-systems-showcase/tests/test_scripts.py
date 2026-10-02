from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pandas as pd
from pytest import MonkeyPatch

from batch_stream_showcase.data_generator import generate_events


def test_stream_run_honors_seed_with_existing_events(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/run_stream.py"
    main = runpy.run_path(str(script))["main"]
    monkeypatch.setitem(main.__globals__, "__file__", str(tmp_path / "scripts/run_stream.py"))
    events_path = tmp_path / "artifacts/events/events.csv"
    events_path.parent.mkdir(parents=True)
    generate_events(n_events=10, seed=42).to_csv(events_path, index=False)
    monkeypatch.setattr(sys, "argv", [str(script), "--quick", "--seed", "7"])

    main()

    events = pd.read_csv(events_path)
    pd.testing.assert_frame_equal(events, generate_events(n_events=400, seed=7))
    metrics = json.loads((tmp_path / "artifacts/stream/stream_metrics.json").read_text())
    assert metrics["seed"] == 7
    assert metrics["events_processed"] == 400
