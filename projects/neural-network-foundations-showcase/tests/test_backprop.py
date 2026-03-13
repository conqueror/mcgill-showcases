"""Tests for manual backpropagation helpers."""

from __future__ import annotations

from neural_network_foundations_showcase import backprop, data, networks


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
