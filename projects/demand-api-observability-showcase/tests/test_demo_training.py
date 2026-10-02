from __future__ import annotations

import importlib
from pathlib import Path

from pytest import MonkeyPatch

from demand_api_observability_showcase.model.demo_training import _build_demo_dataset
from demand_api_observability_showcase.model.store import ModelStore


def test_demo_rows_follow_chronological_order() -> None:
    frame, _ = _build_demo_dataset()
    # February 1, 2026 is Sunday: weekdays 6, 0, ..., 5 are days 0, 1, ..., 6.
    day = frame["day_of_week"].map({6: 0, 0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6})
    hour = day * 24 + frame["hour"]
    assert hour.is_monotonic_increasing


def test_future_rows_cannot_change_the_fitted_model(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    module = importlib.import_module("demand_api_observability_showcase.model.demo_training")
    frame, target = _build_demo_dataset()
    day = frame["day_of_week"].map({6: 0, 0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6})
    # 7 * 24 = 168 hours; floor(168 * 0.8) = 134 whole training hours.
    held_out = day * 24 + frame["hour"] >= 134
    original_dir = tmp_path / "original"
    changed_dir = tmp_path / "changed"
    monkeypatch.setattr(module, "_build_demo_dataset", lambda: (frame, target))
    module.train_demo_model(original_dir)

    changed = frame.copy()
    changed.loc[held_out, "hour"] += 1000
    changed_target = target.copy()
    changed_target.loc[held_out] += 1000.0
    monkeypatch.setattr(module, "_build_demo_dataset", lambda: (changed, changed_target))
    module.train_demo_model(changed_dir)

    original = ModelStore(original_dir / "model.joblib")
    modified = ModelStore(changed_dir / "model.joblib")
    original.load()
    modified.load()
    assert original.bundle is not None and modified.bundle is not None
    assert (
        original.bundle.model.booster_.model_to_string()
        == modified.bundle.model.booster_.model_to_string()
    )
    assert original.bundle.metrics.n_train == 4556  # 134 hours * 34 zones.
    assert original.bundle.metrics.n_eval == 1156  # 34 remaining hours * 34 zones.
