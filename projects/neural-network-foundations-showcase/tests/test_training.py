"""Tests for training helpers."""

from __future__ import annotations

from neural_network_foundations_showcase import data, training


def test_train_network_reduces_training_loss_on_easy_data() -> None:
    """Even a small network should learn a linearly separable toy task."""

    dataset = data.make_toy_dataset(
        "linearly_separable",
        samples_per_class=16,
        random_state=6,
    )
    result = training.train_network(
        dataset,
        training.TrainingConfig(
            layer_sizes=(2, 1),
            epochs=60,
            learning_rate=0.2,
            random_state=6,
        ),
    )

    assert list(result.history.columns) == [
        "epoch",
        "train_loss",
        "validation_loss",
        "train_accuracy",
        "validation_accuracy",
    ]
    assert result.history.iloc[-1]["train_loss"] < result.history.iloc[0]["train_loss"]


def test_underfit_overfit_table_has_expected_regimes() -> None:
    """The summary table should compare the three headline fitting regimes."""

    table = training.build_underfit_overfit_table(random_state=4)

    assert set(table["regime"]) == {"underfit", "well_fit", "overfit"}
    assert {"train_accuracy", "validation_accuracy", "generalization_gap"}.issubset(
        table.columns,
    )
