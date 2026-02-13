from __future__ import annotations

import pandas as pd

from batch_stream_showcase.stream_pipeline import run_stream_pipeline


def test_late_events_get_dropped_with_tight_watermark() -> None:
    events = pd.DataFrame(
        {
            "event_id": [0, 1, 2],
            "event_time": [10, 1, 2],
            "arrival_time": [10, 12, 13],
            "value": [1.0, 1.0, 1.0],
        }
    )
    result = run_stream_pipeline(events, window_size=5, allowed_lateness=0)
    assert result.dropped_late_events >= 1
