from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StreamResult:
    window_kpis: pd.DataFrame
    dropped_late_events: int


def run_stream_pipeline(
    events: pd.DataFrame,
    *,
    window_size: int = 20,
    allowed_lateness: int = 3,
) -> StreamResult:
    """Compute windowed KPIs as events arrive, dropping overly late events."""
    aggregates: dict[int, dict[str, float | int]] = {}
    watermark = -1
    dropped = 0

    for row in events.sort_values("arrival_time").itertuples(index=False):
        watermark = max(watermark, int(row.arrival_time) - allowed_lateness)
        if int(row.event_time) < watermark:
            dropped += 1
            continue

        window = int(row.event_time // window_size)
        if window not in aggregates:
            aggregates[window] = {"total_value": 0.0, "event_count": 0}

        bucket = aggregates[window]
        bucket["total_value"] = float(bucket["total_value"]) + float(row.value)
        bucket["event_count"] = int(bucket["event_count"]) + 1

    out = pd.DataFrame(
        [
            {
                "window": window,
                "total_value": values["total_value"],
                "event_count": values["event_count"],
            }
            for window, values in sorted(aggregates.items())
        ]
    )
    return StreamResult(window_kpis=out, dropped_late_events=dropped)
