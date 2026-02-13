from __future__ import annotations

import pandas as pd

from batch_stream_showcase.batch_pipeline import run_batch_pipeline


def test_window_assignment_in_batch_pipeline() -> None:
    events = pd.DataFrame(
        {
            "event_id": [0, 1, 2],
            "event_time": [0, 19, 20],
            "arrival_time": [0, 19, 20],
            "value": [10.0, 5.0, 7.0],
        }
    )
    out = run_batch_pipeline(events, window_size=20)
    assert out["window"].tolist() == [0, 1]
    assert out["event_count"].tolist() == [2, 1]
