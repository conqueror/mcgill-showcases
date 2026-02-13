#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from batch_stream_showcase.data_generator import generate_events
from batch_stream_showcase.stream_pipeline import run_stream_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stream KPI pipeline")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allowed-lateness", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    events_path = root / "artifacts/events/events.csv"
    if events_path.exists():
        events = pd.read_csv(events_path)
    else:
        n_events = 400 if args.quick else 1200
        events = generate_events(n_events=n_events, seed=args.seed)
        events_path.parent.mkdir(parents=True, exist_ok=True)
        events.to_csv(events_path, index=False)

    result = run_stream_pipeline(events, allowed_lateness=args.allowed_lateness)

    stream_path = root / "artifacts/stream/kpi_output.csv"
    metrics_path = root / "artifacts/stream/stream_metrics.json"
    stream_path.parent.mkdir(parents=True, exist_ok=True)

    result.window_kpis.to_csv(stream_path, index=False)
    metrics_path.write_text(
        json.dumps({"dropped_late_events": result.dropped_late_events}, indent=2),
        encoding="utf-8",
    )

    manifest_path = root / "artifacts/manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"version": 1, "required_files": []}

    required = set(manifest.get("required_files", []))
    required.update(["artifacts/stream/kpi_output.csv", "artifacts/stream/stream_metrics.json"])
    manifest["required_files"] = sorted(required)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
