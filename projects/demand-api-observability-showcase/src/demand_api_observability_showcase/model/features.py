from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ModelFeatures:
    pickup_zone_id: int
    hour: int
    day_of_week: int
    month: int


def features_from_datetime(pickup_zone_id: int, pickup_datetime: datetime) -> ModelFeatures:
    return ModelFeatures(
        pickup_zone_id=pickup_zone_id,
        hour=pickup_datetime.hour,
        day_of_week=pickup_datetime.weekday(),
        month=pickup_datetime.month,
    )
