from __future__ import annotations

import pandas as pd

from batch_stream_showcase.kpi import windowed_sum


def run_batch_pipeline(events: pd.DataFrame, *, window_size: int = 20) -> pd.DataFrame:
    return windowed_sum(events, window_size=window_size)
