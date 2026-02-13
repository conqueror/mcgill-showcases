from __future__ import annotations

from dataclasses import dataclass

from lightgbm import LGBMRegressor


@dataclass(frozen=True)
class TrainingMetrics:
    mae: float
    rmse: float
    n_train: int
    n_eval: int


@dataclass(frozen=True)
class ModelBundle:
    model: LGBMRegressor
    model_version: str
    trained_at_iso: str
    feature_names: tuple[str, ...]
    metrics: TrainingMetrics
