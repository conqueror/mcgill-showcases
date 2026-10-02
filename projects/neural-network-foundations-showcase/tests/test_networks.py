"""Tests for feed-forward network helpers."""

from __future__ import annotations

import numpy as np
import pytest

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


def test_xavier_uses_both_fan_in_and_fan_out() -> None:
    """For a 2-to-6 layer, Glorot variance is 2/(2+6)=1/4, std=1/2."""

    weights, biases = networks.initialize_weights(
        2, 6, "xavier", np.random.default_rng(7)
    )
    expected = np.random.default_rng(7).normal(0.0, 0.5, size=(2, 6))
    np.testing.assert_allclose(weights, expected)
    np.testing.assert_array_equal(biases, np.zeros((1, 6)))


@pytest.mark.parametrize(
    ("layer_sizes", "output_activation"),
    [((2, 2), "sigmoid"), ((2, 1), "tanh")],
)
def test_network_rejects_outputs_that_binary_backprop_cannot_train(
    layer_sizes: tuple[int, ...],
    output_activation: str,
) -> None:
    """The output delta p-y requires exactly one sigmoid output."""

    with pytest.raises(ValueError, match="one sigmoid output"):
        networks.build_network(layer_sizes, output_activation=output_activation)
