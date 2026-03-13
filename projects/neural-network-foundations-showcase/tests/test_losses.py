"""Tests for loss helpers."""

from __future__ import annotations

import numpy as np

from neural_network_foundations_showcase import losses


def test_loss_functions_penalize_bad_predictions_more() -> None:
    """Correct, confident predictions should incur less loss."""

    good_prediction = np.array([0.9])
    bad_prediction = np.array([0.1])
    positive_target = np.array([1.0])

    assert losses.binary_cross_entropy(good_prediction, positive_target) < (
        losses.binary_cross_entropy(bad_prediction, positive_target)
    )
    assert losses.mean_squared_error(good_prediction, positive_target) < (
        losses.mean_squared_error(bad_prediction, positive_target)
    )


def test_loss_comparison_table_has_expected_schema() -> None:
    """The loss comparison artifact should expose a stable schema."""

    table = losses.build_loss_comparison_table()

    assert list(table.columns) == [
        "scenario",
        "prediction",
        "target",
        "mean_squared_error",
        "binary_cross_entropy",
        "hinge_loss",
    ]
    assert len(table) >= 3
