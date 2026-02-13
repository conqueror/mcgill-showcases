from __future__ import annotations

import pandas as pd

from batch_stream_showcase.batch_pipeline import run_batch_pipeline
from batch_stream_showcase.reconciliation import compare_batch_stream
from batch_stream_showcase.stream_pipeline import run_stream_pipeline


def test_batch_stream_parity_without_lateness() -> None:
    events = pd.DataFrame(
        {
            "event_id": [0, 1, 2, 3],
            "event_time": [0, 1, 20, 21],
            "arrival_time": [0, 1, 20, 21],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )

    batch = run_batch_pipeline(events, window_size=20)
    stream = run_stream_pipeline(events, window_size=20, allowed_lateness=0).window_kpis
    report = compare_batch_stream(batch, stream)

    assert report["total_value_abs_diff"].sum() == 0.0
