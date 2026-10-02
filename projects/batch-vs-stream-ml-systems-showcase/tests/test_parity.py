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
    assert report["event_count_abs_diff"].sum() == 0
    assert report["within_tolerance"].all()


def test_equal_values_with_different_counts_fail_parity() -> None:
    # One event worth 3 and two events worth 1 + 2 have equal sums, but counts differ by 1.
    batch = pd.DataFrame({"window": [0], "total_value": [3.0], "event_count": [2]})
    stream = pd.DataFrame({"window": [0], "total_value": [3.0], "event_count": [1]})
    report = compare_batch_stream(batch, stream)
    assert report.loc[0, "total_value_abs_diff"] == 0.0
    assert report.loc[0, "event_count_abs_diff"] == 1
    assert not report.loc[0, "within_tolerance"]


def test_empty_stream_can_be_reconciled() -> None:
    events = pd.DataFrame(columns=["event_id", "event_time", "arrival_time", "value"])
    result = run_stream_pipeline(events)
    assert result.window_kpis.columns.tolist() == ["window", "total_value", "event_count"]
    assert result.dropped_late_events == 0
    batch = pd.DataFrame({"window": [0], "total_value": [3.0], "event_count": [1]})
    report = compare_batch_stream(batch, result.window_kpis)
    assert report.loc[0, "event_count_abs_diff"] == 1
    assert not report.loc[0, "within_tolerance"]


def test_all_dropped_stream_preserves_output_columns() -> None:
    events = pd.DataFrame(
        {"event_id": [0], "event_time": [0], "arrival_time": [10], "value": [3.0]}
    )
    result = run_stream_pipeline(events, allowed_lateness=0)
    assert result.dropped_late_events == 1
    assert result.window_kpis.empty
    report = compare_batch_stream(run_batch_pipeline(events), result.window_kpis)
    assert report.loc[0, "total_value_abs_diff"] == 3.0
    assert not report.loc[0, "within_tolerance"]
