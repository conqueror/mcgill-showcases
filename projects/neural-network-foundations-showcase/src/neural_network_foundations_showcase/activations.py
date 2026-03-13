"""Activation functions and comparison-table helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Apply the sigmoid nonlinearity elementwise."""

    return 1.0 / (1.0 + np.exp(-values))


def tanh(values: np.ndarray) -> np.ndarray:
    """Apply hyperbolic tangent elementwise."""

    return np.tanh(values)


def relu(values: np.ndarray) -> np.ndarray:
    """Apply rectified linear activation elementwise."""

    return np.maximum(values, 0.0)


def leaky_relu(values: np.ndarray, slope: float = 0.1) -> np.ndarray:
    """Apply a small negative slope to avoid dead activations."""

    return np.where(values > 0.0, values, slope * values)


def activation_forward(name: str, values: np.ndarray) -> np.ndarray:
    """Dispatch by activation name for the network helpers."""

    if name == "sigmoid":
        return sigmoid(values)
    if name == "tanh":
        return tanh(values)
    if name == "relu":
        return relu(values)
    if name == "leaky_relu":
        return leaky_relu(values)
    raise ValueError(f"Unsupported activation: {name}")


def activation_derivative(name: str, activated_values: np.ndarray) -> np.ndarray:
    """Return the derivative evaluated from activation outputs."""

    if name == "sigmoid":
        return activated_values * (1.0 - activated_values)
    if name == "tanh":
        return 1.0 - activated_values**2
    if name == "relu":
        return (activated_values > 0.0).astype(np.float64)
    if name == "leaky_relu":
        return np.where(activated_values > 0.0, 1.0, 0.1)
    raise ValueError(f"Unsupported activation: {name}")


def build_activation_comparison_table(inputs: np.ndarray | None = None) -> pd.DataFrame:
    """Create a stable artifact table for the showcase README and docs."""

    grid = np.array([-3.0, -1.0, 0.0, 1.0, 3.0]) if inputs is None else inputs
    return pd.DataFrame(
        {
            "input": grid,
            "sigmoid": sigmoid(grid),
            "tanh": tanh(grid),
            "relu": relu(grid),
            "leaky_relu": leaky_relu(grid),
        },
    )
