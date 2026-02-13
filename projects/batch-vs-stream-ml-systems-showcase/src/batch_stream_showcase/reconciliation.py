from __future__ import annotations

import pandas as pd


def compare_batch_stream(
    batch_df: pd.DataFrame,
    stream_df: pd.DataFrame,
    *,
    value_tolerance: float = 1e-6,
) -> pd.DataFrame:
    merged = batch_df.merge(
        stream_df,
        on="window",
        how="outer",
        suffixes=("_batch", "_stream"),
    ).fillna(0.0)

    merged["total_value_abs_diff"] = (
        merged["total_value_batch"] - merged["total_value_stream"]
    ).abs()
    merged["event_count_abs_diff"] = (
        merged["event_count_batch"] - merged["event_count_stream"]
    ).abs()
    merged["within_tolerance"] = merged["total_value_abs_diff"] <= value_tolerance

    return merged.sort_values("window").reset_index(drop=True)
