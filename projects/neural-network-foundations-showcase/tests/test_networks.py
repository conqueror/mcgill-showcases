"""Tests for feed-forward network helpers."""

from __future__ import annotations

import numpy as np

from neural_network_foundations_showcase import data, networks


def test_predict_proba_returns_binary_probabilities() -> None:
    """Forward passes should return one probability per example."""

    dataset = data.make_toy_dataset("linearly_separable", samples_per_class=3)
    network = networks.build_network(
        layer_sizes=(2, 1),
        init_strategy="xavier",
        random_state=4,
    )

    probabilities = networks.predict_proba(network, dataset.features)

    assert probabilities.shape == (6,)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)


def test_initialization_comparison_table_covers_all_strategies() -> None:
    """The initialization artifact should compare the main strategies from class."""

    table = networks.build_initialization_comparison_table(random_state=5)

    assert set(table["strategy"]) == {"zero", "random", "xavier", "he"}
    assert {
        "first_layer_weight_std",
        "hidden_activation_mean",
        "output_probability_mean",
    }.issubset(table.columns)
