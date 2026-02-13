from __future__ import annotations

import pandas as pd


def windowed_sum(frame: pd.DataFrame, *, window_size: int) -> pd.DataFrame:
    df = frame.copy()
    df["window"] = (df["event_time"] // window_size).astype(int)
    out = (
        df.groupby("window", as_index=False)
        .agg(total_value=("value", "sum"), event_count=("event_id", "count"))
        .sort_values("window")
        .reset_index(drop=True)
    )
    return out
