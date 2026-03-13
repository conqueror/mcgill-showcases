"""Tests for regularization configuration helpers."""

from __future__ import annotations

from pytorch_training_regularization_showcase import regularization, training


def test_regularization_scenarios_cover_expected_options() -> None:
    """The showcase should compare the main regularization knobs explicitly."""

    scenarios = regularization.build_regularization_scenarios(
        training.TrainingConfig(),
    )

    assert set(scenarios) == {
        "baseline",
        "dropout",
        "batch_norm",
        "weight_decay",
        "all_regularization",
    }
    assert scenarios["dropout"].dropout > 0.0
    assert scenarios["batch_norm"].batch_norm is True
    assert scenarios["weight_decay"].weight_decay > 0.0
