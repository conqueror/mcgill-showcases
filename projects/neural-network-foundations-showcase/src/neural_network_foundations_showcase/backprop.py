"""Manual backpropagation helpers for the showcase."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from neural_network_foundations_showcase import activations, networks


@dataclass(frozen=True)
class GradientBundle:
    """Gradient tensors aligned with a network's parameters."""

    weight_gradients: list[np.ndarray]
    bias_gradients: list[np.ndarray]
    forward_cache: networks.ForwardPass


def compute_gradients(
    network: networks.FeedForwardNetwork,
    features: np.ndarray,
    labels: np.ndarray,
) -> GradientBundle:
    """Compute gradients for binary cross-entropy with a sigmoid output."""

    forward_cache = networks.forward_pass(network, features)
    targets = labels.reshape(-1, 1).astype(np.float64)
    output_activations = forward_cache.activations[-1]
    batch_size = len(features)

    delta = output_activations - targets
    weight_gradients: list[np.ndarray] = [
        np.zeros_like(weight) for weight in network.weights
    ]
    bias_gradients: list[np.ndarray] = [np.zeros_like(bias) for bias in network.biases]

    for layer_index in reversed(range(len(network.weights))):
        previous_activations = forward_cache.activations[layer_index]
        weight_gradients[layer_index] = (previous_activations.T @ delta) / batch_size
        bias_gradients[layer_index] = np.mean(delta, axis=0, keepdims=True)

        if layer_index > 0:
            propagated = delta @ network.weights[layer_index].T
            hidden_outputs = forward_cache.activations[layer_index]
            delta = propagated * activations.activation_derivative(
                network.hidden_activation,
                hidden_outputs,
            )

    return GradientBundle(
        weight_gradients=weight_gradients,
        bias_gradients=bias_gradients,
        forward_cache=forward_cache,
    )


def build_backprop_gradient_trace(
    network: networks.FeedForwardNetwork,
    features: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    """Summarize gradient flow by layer for a single batch."""

    gradients = compute_gradients(network, features, labels)
    rows = []
    for index, (weight_gradient, bias_gradient, layer_outputs) in enumerate(
        zip(
            gradients.weight_gradients,
            gradients.bias_gradients,
            gradients.forward_cache.activations[1:],
            strict=True,
        ),
        start=1,
    ):
        rows.append(
            {
                "layer": f"layer_{index}",
                "weight_grad_norm": float(np.linalg.norm(weight_gradient)),
                "bias_grad_norm": float(np.linalg.norm(bias_gradient)),
                "activation_mean": float(np.mean(layer_outputs)),
            },
        )
    return pd.DataFrame(rows)
