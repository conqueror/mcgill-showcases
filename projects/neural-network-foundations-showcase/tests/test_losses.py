"""Tests for loss helpers."""

from __future__ import annotations

import numpy as np
import pytest

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
        "hinge_score",
        "hinge_loss",
    ]
    assert len(table) >= 3


@pytest.mark.parametrize(
    ("score", "target", "expected"),
    [(0.25, 1.0, 0.75), (0.25, 0.0, 1.25), (-0.25, 1.0, 1.25)],
)
def test_hinge_loss_uses_raw_signed_scores(
    score: float,
    target: float,
    expected: float,
) -> None:
    """For y=1 and score=1/4, max(0, 1-y*score) is 3/4."""

    assert losses.hinge_loss(np.array([score]), np.array([target])) == expected


def test_loss_table_defines_hinge_scores_as_log_odds() -> None:
    """Probabilities 0.9, 0.55, 0.1 have odds 9, 11/9, 1/9."""

    table = losses.build_loss_comparison_table()
    scores = np.array([np.log(9.0), np.log(11.0 / 9.0), -np.log(9.0)])
    np.testing.assert_allclose(table["hinge_score"], scores)
    np.testing.assert_allclose(
        table["hinge_loss"],
        [0.0, 1.0 - np.log(11.0 / 9.0), 1.0 + np.log(9.0)],
    )
