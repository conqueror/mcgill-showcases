"""Tests for training helpers."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

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


def test_zero_label_noise_keeps_all_labels() -> None:
    """Requesting no corruption must not flip even one label."""

    dataset = data.make_toy_dataset("xor", samples_per_class=8)
    noisy = training._inject_label_noise(dataset, 0.0, 7)
    np.testing.assert_array_equal(noisy.labels, dataset.labels)


def test_overfit_recipe_keeps_validation_labels_clean() -> None:
    """Noise belongs to the training subset, after the deterministic split."""

    clean = data.make_toy_dataset(
        "xor", samples_per_class=14, noise=0.3, random_state=8
    )
    expected = data.train_val_split(clean.features, clean.labels, 0.4, 11)
    result = dict(training._regime_runs(7))["overfit"]
    np.testing.assert_array_equal(
        result.split.validation_labels, expected.validation_labels
    )
    assert np.count_nonzero(result.split.train_labels != expected.train_labels) == 3


def test_held_out_rows_cannot_change_the_fitted_network() -> None:
    """Changing only held-out features and labels must leave every parameter equal."""

    dataset = data.make_toy_dataset("xor", samples_per_class=12)
    config = training.TrainingConfig(epochs=20, label_noise_fraction=0.18)
    reference = training.train_network(dataset, config)
    is_validation = np.array(
        [
            np.any(np.all(row == reference.split.validation_features, axis=1))
            for row in dataset.features
        ]
    )
    features, labels = dataset.features.copy(), dataset.labels.copy()
    features[is_validation] *= 50.0
    labels[is_validation] = 1.0 - labels[is_validation]
    changed = training.train_network(
        replace(dataset, features=features, labels=labels), config
    )
    for original, perturbed in zip(
        reference.network.weights + reference.network.biases,
        changed.network.weights + changed.network.biases,
        strict=True,
    ):
        np.testing.assert_array_equal(original, perturbed)


def test_default_regimes_demonstrate_the_named_fitting_behavior() -> None:
    """Check observed train/validation behavior rather than only the scenario names."""

    table = training.build_underfit_overfit_table().set_index("regime")
    assert table.loc["underfit", "train_accuracy"] < 0.7
    assert table.loc["underfit", "validation_accuracy"] < 0.7
    assert table.loc["well_fit", "train_accuracy"] >= 0.85
    assert table.loc["well_fit", "validation_accuracy"] >= 0.75
    assert table.loc["overfit", "train_accuracy"] >= 0.9
    assert table.loc["overfit", "generalization_gap"] > 0.3
