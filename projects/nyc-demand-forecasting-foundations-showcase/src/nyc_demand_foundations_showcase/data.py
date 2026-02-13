from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW_REQUIRED_COLUMNS = ["tpep_pickup_datetime", "PULocationID"]


@dataclass(frozen=True)
class GroupedDemandFrame:
    frame: pd.DataFrame
    source: str


def generate_synthetic_grouped_data(
    *,
    n_hours: int = 24 * 60,
    n_zones: int = 35,
    random_state: int = 42,
) -> pd.DataFrame:
    if n_hours < 72:
        raise ValueError("n_hours must be at least 72.")
    if n_zones < 8:
        raise ValueError("n_zones must be at least 8.")

    rng = np.random.default_rng(random_state)
    hours = pd.date_range("2024-01-01", periods=n_hours, freq="h")
    zones = np.arange(1, n_zones + 1)

    rows: list[dict[str, object]] = []
    zone_multiplier = rng.uniform(0.6, 1.7, size=n_zones)

    for zone_index, zone_id in enumerate(zones):
        for pickup_hour in hours:
            hour = pickup_hour.hour
            is_weekend = pickup_hour.weekday() >= 5
            rush_boost = 1.35 if hour in {7, 8, 9, 16, 17, 18} else 1.0
            weekend_factor = 1.12 if is_weekend else 1.0
            seasonal = 1.0 + (pickup_hour.month - 1) * 0.015
            base_rate = 8.0 * zone_multiplier[zone_index] * rush_boost * weekend_factor * seasonal
            expected = max(base_rate + rng.normal(0.0, 1.2), 0.6)
            pickups = int(rng.poisson(expected))

            rows.append(
                {
                    "pickup_zone_id": int(zone_id),
                    "pickup_hour": pickup_hour,
                    "pickups": float(pickups),
                }
            )

    return pd.DataFrame(rows)


def _group_tlc_trips(trips: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in RAW_REQUIRED_COLUMNS if col not in trips.columns]
    if missing:
        raise KeyError(f"Missing raw TLC columns: {missing}")

    work = trips[RAW_REQUIRED_COLUMNS].dropna(subset=RAW_REQUIRED_COLUMNS).copy()
    work["tpep_pickup_datetime"] = pd.to_datetime(work["tpep_pickup_datetime"], errors="coerce")
    work = work.dropna(subset=["tpep_pickup_datetime"])

    work["pickup_hour"] = work["tpep_pickup_datetime"].dt.floor("h")
    grouped = (
        work.groupby(["PULocationID", "pickup_hour"], as_index=False)
        .size()
        .rename(columns={"PULocationID": "pickup_zone_id", "size": "pickups"})
        .sort_values("pickup_hour")
        .reset_index(drop=True)
    )

    grouped["pickup_zone_id"] = grouped["pickup_zone_id"].astype(int)
    grouped["pickups"] = grouped["pickups"].astype(float)
    return grouped


def add_time_features(grouped: pd.DataFrame) -> pd.DataFrame:
    if not {"pickup_zone_id", "pickup_hour", "pickups"}.issubset(set(grouped.columns)):
        raise KeyError("Grouped frame must contain pickup_zone_id, pickup_hour, pickups.")

    frame = grouped.copy()
    frame["pickup_hour"] = pd.to_datetime(frame["pickup_hour"], errors="coerce")
    frame = frame.dropna(subset=["pickup_hour"])

    frame["hour"] = frame["pickup_hour"].dt.hour
    frame["day_of_week"] = frame["pickup_hour"].dt.dayofweek
    frame["month"] = frame["pickup_hour"].dt.month
    frame["is_weekend"] = (frame["day_of_week"] >= 5).astype(int)
    frame["is_peak_hour"] = frame["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)

    frame = frame.sort_values("pickup_hour").reset_index(drop=True)
    return frame


def load_grouped_data(
    *,
    data_path: Path | None,
    quick: bool,
    random_state: int,
) -> GroupedDemandFrame:
    if data_path is None:
        grouped = generate_synthetic_grouped_data(
            n_hours=24 * (14 if quick else 60),
            n_zones=20 if quick else 40,
            random_state=random_state,
        )
        return GroupedDemandFrame(frame=grouped, source="synthetic")

    raw = pd.read_parquet(data_path, columns=RAW_REQUIRED_COLUMNS)
    grouped = _group_tlc_trips(raw)
    return GroupedDemandFrame(frame=grouped, source=f"tlc:{data_path.name}")
