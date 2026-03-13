"""Minimal feed-forward network mechanics for educational experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from neural_network_foundations_showcase import activations, data


@dataclass(frozen=True)
class FeedForwardNetwork:
    """A small fully connected network represented as NumPy arrays."""

    weights: list[np.ndarray]
    biases: list[np.ndarray]
    hidden_activation: str = "tanh"
    output_activation: str = "sigmoid"


@dataclass(frozen=True)
class ForwardPass:
    """Cached activations used for backpropagation and reporting."""

    activations: list[np.ndarray]
    pre_activations: list[np.ndarray]
    output: np.ndarray


def initialize_weights(
    in_features: int,
    out_features: int,
    strategy: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Initialize one layer of parameters using a named strategy."""

    if strategy == "zero":
        weights = np.zeros((in_features, out_features), dtype=np.float64)
    elif strategy == "random":
        weights = rng.normal(0.0, 0.5, size=(in_features, out_features))
    elif strategy == "xavier":
        weights = rng.normal(
            0.0,
            np.sqrt(1.0 / max(1, in_features)),
            size=(in_features, out_features),
        )
    elif strategy == "he":
        weights = rng.normal(
            0.0,
            np.sqrt(2.0 / max(1, in_features)),
            size=(in_features, out_features),
        )
    else:
        raise ValueError(f"Unsupported initialization strategy: {strategy}")

    biases = np.zeros((1, out_features), dtype=np.float64)
    return weights.astype(np.float64), biases


def build_network(
    layer_sizes: tuple[int, ...],
    init_strategy: str = "xavier",
    hidden_activation: str = "tanh",
    output_activation: str = "sigmoid",
    random_state: int = 7,
) -> FeedForwardNetwork:
    """Create a deterministic feed-forward network."""

    if len(layer_sizes) < 2:
        raise ValueError("layer_sizes must include input and output dimensions.")

    rng = np.random.default_rng(random_state)
    weights = []
    biases = []
    for in_features, out_features in zip(
        layer_sizes[:-1],
        layer_sizes[1:],
        strict=True,
    ):
        weight, bias = initialize_weights(
            in_features,
            out_features,
            init_strategy,
            rng,
        )
        weights.append(weight)
        biases.append(bias)

    return FeedForwardNetwork(
        weights=weights,
        biases=biases,
        hidden_activation=hidden_activation,
        output_activation=output_activation,
    )


def forward_pass(
    network: FeedForwardNetwork,
    features: np.ndarray,
) -> ForwardPass:
    """Run a forward pass and keep intermediate values for learning."""

    activations_cache = [features.astype(np.float64)]
    pre_activations: list[np.ndarray] = []
    current = activations_cache[0]

    for layer_index, (weights, bias) in enumerate(
        zip(network.weights, network.biases, strict=True),
    ):
        z_values = current @ weights + bias
        pre_activations.append(z_values)
        activation_name = (
            network.output_activation
            if layer_index == len(network.weights) - 1
            else network.hidden_activation
        )
        current = activations.activation_forward(activation_name, z_values)
        activations_cache.append(current)

    return ForwardPass(
        activations=activations_cache,
        pre_activations=pre_activations,
        output=current.reshape(-1),
    )


def predict_proba(network: FeedForwardNetwork, features: np.ndarray) -> np.ndarray:
    """Return binary probabilities for each example."""

    return forward_pass(network, features).output


def copy_network(network: FeedForwardNetwork) -> FeedForwardNetwork:
    """Create a mutable copy-safe clone of the network parameters."""

    return FeedForwardNetwork(
        weights=[weight.copy() for weight in network.weights],
        biases=[bias.copy() for bias in network.biases],
        hidden_activation=network.hidden_activation,
        output_activation=network.output_activation,
    )


def apply_gradient_step(
    network: FeedForwardNetwork,
    weight_gradients: list[np.ndarray],
    bias_gradients: list[np.ndarray],
    learning_rate: float,
) -> FeedForwardNetwork:
    """Return a new network with one gradient-descent update applied."""

    updated_weights = [
        weight - learning_rate * gradient
        for weight, gradient in zip(network.weights, weight_gradients, strict=True)
    ]
    updated_biases = [
        bias - learning_rate * gradient
        for bias, gradient in zip(network.biases, bias_gradients, strict=True)
    ]
    return FeedForwardNetwork(
        weights=updated_weights,
        biases=updated_biases,
        hidden_activation=network.hidden_activation,
        output_activation=network.output_activation,
    )


def build_initialization_comparison_table(random_state: int = 7) -> pd.DataFrame:
    """Compare how initialization strategies change the first forward pass."""

    sample_dataset = data.make_toy_dataset(
        "xor",
        samples_per_class=12,
        random_state=random_state,
    )
    rows = []
    for strategy in ("zero", "random", "xavier", "he"):
        network = build_network(
            layer_sizes=(2, 6, 1),
            init_strategy=strategy,
            hidden_activation="tanh",
            random_state=random_state,
        )
        pass_result = forward_pass(network, sample_dataset.features)
        rows.append(
            {
                "strategy": strategy,
                "first_layer_weight_std": float(np.std(network.weights[0])),
                "hidden_activation_mean": float(np.mean(pass_result.activations[1])),
                "output_probability_mean": float(np.mean(pass_result.output)),
            },
        )
    return pd.DataFrame(rows)
