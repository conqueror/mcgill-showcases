"""Loss functions used in the neural network foundations showcase."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mean_squared_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Return the average squared error."""

    return float(np.mean((predictions - targets) ** 2))


def binary_cross_entropy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Return the average binary cross-entropy loss."""

    clipped = np.clip(predictions, 1e-7, 1.0 - 1e-7)
    value = -(targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped))
    return float(np.mean(value))


def hinge_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """Return binary hinge loss for targets encoded as 0/1."""

    signed_targets = np.where(targets > 0.5, 1.0, -1.0)
    signed_logits = np.where(logits > 0.5, 1.0, -1.0) * np.abs(logits)
    margins = 1.0 - signed_targets * signed_logits
    return float(np.mean(np.maximum(0.0, margins)))


def build_loss_comparison_table() -> pd.DataFrame:
    """Create a student-facing comparison table for three prediction scenarios."""

    examples = [
        ("correct_and_confident", 0.9, 1.0),
        ("uncertain_prediction", 0.55, 1.0),
        ("wrong_and_confident", 0.1, 1.0),
    ]
    rows = []
    for scenario, prediction, target in examples:
        prediction_array = np.array([prediction], dtype=np.float64)
        target_array = np.array([target], dtype=np.float64)
        rows.append(
            {
                "scenario": scenario,
                "prediction": prediction,
                "target": target,
                "mean_squared_error": mean_squared_error(
                    prediction_array,
                    target_array,
                ),
                "binary_cross_entropy": binary_cross_entropy(
                    prediction_array,
                    target_array,
                ),
                "hinge_loss": hinge_loss(prediction_array, target_array),
            },
        )
    return pd.DataFrame(rows)
