#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from batch_stream_showcase.batch_pipeline import run_batch_pipeline
from batch_stream_showcase.data_generator import generate_events

REQUIRED_ARTIFACTS = [
    "artifacts/events/events.csv",
    "artifacts/batch/kpi_output.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch KPI pipeline")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    n_events = 400 if args.quick else 1200
    events = generate_events(n_events=n_events, seed=args.seed)
    batch_output = run_batch_pipeline(events)

    events_path = root / "artifacts/events/events.csv"
    batch_path = root / "artifacts/batch/kpi_output.csv"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    batch_path.parent.mkdir(parents=True, exist_ok=True)

    events.to_csv(events_path, index=False)
    batch_output.to_csv(batch_path, index=False)

    manifest_path = root / "artifacts/manifest.json"
    manifest_path.write_text(
        json.dumps({"version": 1, "required_files": REQUIRED_ARTIFACTS}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
