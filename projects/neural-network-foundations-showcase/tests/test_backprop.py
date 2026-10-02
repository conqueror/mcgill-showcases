"""Tests for manual backpropagation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from neural_network_foundations_showcase import backprop, data, networks, training


def test_compute_gradients_matches_network_shapes() -> None:
    """Backprop gradients should align with each parameter tensor."""

    dataset = data.make_toy_dataset("xor", samples_per_class=4, random_state=3)
    network = networks.build_network(
        layer_sizes=(2, 4, 1),
        init_strategy="xavier",
        random_state=2,
    )

    gradients = backprop.compute_gradients(
        network,
        dataset.features[:5],
        dataset.labels[:5],
    )

    assert len(gradients.weight_gradients) == len(network.weights)
    assert len(gradients.bias_gradients) == len(network.biases)
    for grad, weight in zip(gradients.weight_gradients, network.weights, strict=True):
        assert grad.shape == weight.shape


def test_backprop_gradient_trace_has_one_row_per_layer() -> None:
    """The artifact trace should summarize gradient flow by layer."""

    dataset = data.make_toy_dataset("xor", samples_per_class=4, random_state=9)
    network = networks.build_network(
        layer_sizes=(2, 3, 1),
        init_strategy="he",
        hidden_activation="relu",
        random_state=8,
    )

    trace = backprop.build_backprop_gradient_trace(
        network,
        dataset.features[:6],
        dataset.labels[:6],
    )

    assert list(trace.columns) == [
        "layer",
        "weight_grad_norm",
        "bias_grad_norm",
        "activation_mean",
    ]
    assert len(trace) == 2
    assert (trace["weight_grad_norm"] >= 0.0).all()


def test_all_weights_and_biases_match_central_differences() -> None:
    """Perturb every parameter of a smooth 2-2-1 network independently."""

    network = networks.FeedForwardNetwork(
        weights=[np.array([[0.2, -0.3], [0.4, 0.1]]), np.array([[0.5], [-0.6]])],
        biases=[np.array([[0.1, -0.2]]), np.array([[0.3]])],
    )
    features = np.array([[0.2, -0.4], [0.8, 0.3], [-0.5, 0.7]])
    labels = np.array([0.0, 1.0, 1.0])
    gradients = backprop.compute_gradients(network, features, labels)
    step = 1e-5
    for parameters, expected_gradients in (
        (network.weights, gradients.weight_gradients),
        (network.biases, gradients.bias_gradients),
    ):
        for parameter, gradient in zip(parameters, expected_gradients, strict=True):
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + step
                plus = training.evaluate_binary_classifier(network, features, labels)[
                    "loss"
                ]
                parameter[index] = original - step
                minus = training.evaluate_binary_classifier(network, features, labels)[
                    "loss"
                ]
                parameter[index] = original
                assert gradient[index] == pytest.approx(
                    (plus - minus) / (2 * step), abs=1e-8
                )


def test_saturated_loss_matches_its_backprop_gradient() -> None:
    """Wrong logits +/-100 give mean loss ~100 and dL/dw ~1, not a flat loss."""

    network = networks.FeedForwardNetwork(
        weights=[np.array([[100.0]])],
        biases=[np.array([[0.0]])],
    )
    features, labels = np.array([[1.0], [-1.0]]), np.array([0.0, 1.0])
    assert training.evaluate_binary_classifier(network, features, labels)[
        "loss"
    ] == pytest.approx(100.0)
    gradients = backprop.compute_gradients(network, features, labels)
    assert gradients.weight_gradients[0][0, 0] == pytest.approx(1.0)
    network.weights[0][0, 0] += 1e-4
    plus = training.evaluate_binary_classifier(network, features, labels)["loss"]
    network.weights[0][0, 0] -= 2e-4
    minus = training.evaluate_binary_classifier(network, features, labels)["loss"]
    assert (plus - minus) / 2e-4 == pytest.approx(1.0)
