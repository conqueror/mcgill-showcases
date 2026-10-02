#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CSV_COLUMNS = {
    "events.csv": ["event_id", "event_time", "arrival_time", "value"],
    "kpi_output.csv": ["window", "total_value", "event_count"],
    "parity_report.csv": [
        "window",
        "total_value_batch",
        "event_count_batch",
        "total_value_stream",
        "event_count_stream",
        "total_value_abs_diff",
        "event_count_abs_diff",
        "within_tolerance",
    ],
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "artifacts/manifest.json"
    if not manifest_path.exists():
        raise SystemExit("Missing artifacts/manifest.json")

    required = json.loads(manifest_path.read_text(encoding="utf-8")).get("required_files")
    if (
        not isinstance(required, list)
        or not required
        or not all(isinstance(p, str) for p in required)
    ):
        raise SystemExit("Manifest required_files must be a nonempty list of paths")
    missing = [path for path in required if not (root / path).exists()]
    if missing:
        raise SystemExit(f"Missing required artifacts: {missing}")

    try:
        for name in required:
            path = root / name
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Empty or invalid artifact: {name}")
            if path.suffix == ".csv":
                frame = pd.read_csv(path)
                columns = CSV_COLUMNS[path.name]
                if not set(columns).issubset(frame.columns):
                    raise ValueError(f"Invalid CSV columns: {name}")
                if frame.empty and name != "artifacts/stream/kpi_output.csv":
                    raise ValueError(f"CSV artifact must contain rows: {name}")
                for column in columns:
                    if column == "within_tolerance":
                        valid = frame[column].isin([True, False]).all()
                    else:
                        valid = np.isfinite(pd.to_numeric(frame[column])).all()
                    if not valid:
                        raise ValueError(f"Invalid values in {name}: {column}")
                    if column in {"event_id", "event_time", "arrival_time", "window"} or (
                        "event_count" in column
                    ):
                        if not ((frame[column] >= 0) & (frame[column] % 1 == 0)).all():
                            raise ValueError(
                                f"Expected nonnegative integer counts and times: {name}"
                            )
                if path.name == "parity_report.csv":
                    expected = (frame["total_value_abs_diff"] <= 1e-6) & (
                        frame["event_count_abs_diff"] == 0
                    )
                    if not frame["within_tolerance"].equals(expected):
                        raise ValueError("Parity flags do not match the recorded differences")
            elif path.suffix == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict) or not payload:
                    raise ValueError(f"Invalid JSON object: {name}")
    except Exception as exc:
        raise SystemExit(f"Invalid artifact contents: {exc}") from exc

    print("All required artifacts have valid contents.")


if __name__ == "__main__":
    main()
