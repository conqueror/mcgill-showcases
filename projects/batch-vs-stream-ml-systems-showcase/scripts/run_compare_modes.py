#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from batch_stream_showcase.batch_pipeline import run_batch_pipeline
from batch_stream_showcase.data_generator import generate_events
from batch_stream_showcase.reconciliation import compare_batch_stream
from batch_stream_showcase.stream_pipeline import run_stream_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare batch and stream outputs")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    n_events = 400 if args.quick else 1200
    events = generate_events(n_events=n_events, seed=args.seed)

    batch_start = time.perf_counter()
    batch_df = run_batch_pipeline(events)
    batch_latency_ms = (time.perf_counter() - batch_start) * 1000.0

    stream_start = time.perf_counter()
    stream_result = run_stream_pipeline(events)
    stream_latency_ms = (time.perf_counter() - stream_start) * 1000.0

    parity = compare_batch_stream(batch_df, stream_result.window_kpis)

    events_path = root / "artifacts/events/events.csv"
    batch_path = root / "artifacts/batch/kpi_output.csv"
    stream_path = root / "artifacts/stream/kpi_output.csv"
    parity_path = root / "artifacts/compare/parity_report.csv"
    summary_path = root / "artifacts/compare/latency_throughput_summary.md"

    events_path.parent.mkdir(parents=True, exist_ok=True)
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    stream_path.parent.mkdir(parents=True, exist_ok=True)
    parity_path.parent.mkdir(parents=True, exist_ok=True)

    events.to_csv(events_path, index=False)
    batch_df.to_csv(batch_path, index=False)
    stream_result.window_kpis.to_csv(stream_path, index=False)
    parity.to_csv(parity_path, index=False)

    summary_lines = [
        "# Latency and Throughput Summary",
        "",
        f"- events_processed: {len(events)}",
        f"- batch_latency_ms: {batch_latency_ms:.3f}",
        f"- stream_latency_ms: {stream_latency_ms:.3f}",
        f"- dropped_late_events: {stream_result.dropped_late_events}",
        f"- windows_within_tolerance: {int(parity['within_tolerance'].sum())}/{len(parity)}",
        "",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    manifest_path = root / "artifacts/manifest.json"
    manifest = {
        "version": 1,
        "required_files": [
            "artifacts/events/events.csv",
            "artifacts/batch/kpi_output.csv",
            "artifacts/stream/kpi_output.csv",
            "artifacts/compare/parity_report.csv",
            "artifacts/compare/latency_throughput_summary.md",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
