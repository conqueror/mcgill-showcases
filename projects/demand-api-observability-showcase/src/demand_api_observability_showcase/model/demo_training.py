"""Synthetic model training utilities for demand API observability demos."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from demand_api_observability_showcase.model.bundle import ModelBundle, TrainingMetrics
from demand_api_observability_showcase.model.features import features_from_datetime


def _build_demo_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Build a deterministic synthetic hourly demand dataset."""

    rng = np.random.default_rng(19)
    rows: list[dict[str, int]] = []
    targets: list[float] = []

    for zone in range(1, 35):
        for hour in range(24):
            for day in range(7):
                dt = datetime(2026, 2, day + 1, hour, 0, 0)
                features = features_from_datetime(zone, dt)
                is_peak = hour in {7, 8, 9, 16, 17, 18}
                base = 6.0 + zone * 0.28 + (4.5 if is_peak else 0.0) + (1.2 if day >= 5 else 0.0)
                target = max(base + rng.normal(0.0, 1.3), 0.0)
                rows.append(asdict(features))
                targets.append(float(target))

    return pd.DataFrame(rows), pd.Series(targets, name="pickups")


def train_demo_model(out_dir: Path) -> None:
    """Train and persist a LightGBM demand model bundle plus metrics JSON.

    Side effects:
        Writes ``model.joblib`` and ``metrics.json`` into ``out_dir``.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    frame, target = _build_demo_dataset()
    split = int(frame.shape[0] * 0.8)

    x_train = frame.iloc[:split].copy()
    y_train = target.iloc[:split].copy()
    x_eval = frame.iloc[split:].copy()
    y_eval = target.iloc[split:].copy()

    model = LGBMRegressor(
        objective="regression",
        n_estimators=220,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=1,
    )
    model.fit(x_train, y_train, categorical_feature=["pickup_zone_id"])

    eval_pred = model.predict(x_eval)
    mae = float(mean_absolute_error(y_eval, eval_pred))
    rmse = float(np.sqrt(mean_squared_error(y_eval, eval_pred)))

    bundle = ModelBundle(
        model=model,
        model_version="demo-nyc-demand-v1",
        trained_at_iso=datetime.now(UTC).isoformat(timespec="seconds"),
        feature_names=("pickup_zone_id", "hour", "day_of_week", "month"),
        metrics=TrainingMetrics(
            mae=mae,
            rmse=rmse,
            n_train=int(split),
            n_eval=int(frame.shape[0] - split),
        ),
    )

    model_path = out_dir / "model.joblib"
    metrics_path = out_dir / "metrics.json"
    joblib.dump(bundle, model_path)

    metrics_path.write_text(
        json.dumps(
            {
                "model_version": bundle.model_version,
                "trained_at_iso": bundle.trained_at_iso,
                "mae": bundle.metrics.mae,
                "rmse": bundle.metrics.rmse,
                "n_train": bundle.metrics.n_train,
                "n_eval": bundle.metrics.n_eval,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
