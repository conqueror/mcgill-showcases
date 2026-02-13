from __future__ import annotations

import pandas as pd


def recommend_action(
    drift_report: pd.DataFrame,
    *,
    max_drifted_features_before_retrain: int = 2,
) -> dict[str, int | str | float]:
    """Return a retraining recommendation from drift report signals."""
    drifted_features = int(drift_report["drift_flag"].sum())
    drift_ratio = drifted_features / max(1, len(drift_report))

    if drifted_features > max_drifted_features_before_retrain:
        action = "retrain"
        reason = "drifted_features_exceeded_threshold"
    elif drifted_features > 0:
        action = "monitor"
        reason = "limited_feature_drift_detected"
    else:
        action = "hold"
        reason = "no_material_drift_detected"

    return {
        "action": action,
        "reason": reason,
        "drifted_features": drifted_features,
        "drift_ratio": round(drift_ratio, 4),
        "threshold": max_drifted_features_before_retrain,
    }
