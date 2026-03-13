"""Tests for toy dataset helpers."""

from __future__ import annotations

import numpy as np

from neural_network_foundations_showcase import data


def test_make_toy_dataset_is_deterministic() -> None:
    """Toy datasets should be reproducible for a fixed random seed."""

    dataset_a = data.make_toy_dataset(
        "linearly_separable",
        samples_per_class=4,
        random_state=11,
    )
    dataset_b = data.make_toy_dataset(
        "linearly_separable",
        samples_per_class=4,
        random_state=11,
    )

    assert dataset_a.features.shape == (8, 2)
    assert dataset_a.labels.shape == (8,)
    assert dataset_a.description.startswith("Linearly separable")
    np.testing.assert_allclose(dataset_a.features, dataset_b.features)
    np.testing.assert_array_equal(dataset_a.labels, dataset_b.labels)


def test_train_val_split_preserves_examples_without_overlap() -> None:
    """Train/validation splitting should preserve every row exactly once."""

    dataset = data.make_toy_dataset("xor", samples_per_class=6, random_state=5)

    split = data.train_val_split(
        dataset.features,
        dataset.labels,
        validation_fraction=0.25,
        random_state=7,
    )

    assert split.train_features.shape == (9, 2)
    assert split.validation_features.shape == (3, 2)
    assert split.train_labels.shape == (9,)
    assert split.validation_labels.shape == (3,)

    original_rows = {tuple(row) for row in dataset.features}
    split_rows = {
        *{tuple(row) for row in split.train_features},
        *{tuple(row) for row in split.validation_features},
    }
    assert original_rows == split_rows
