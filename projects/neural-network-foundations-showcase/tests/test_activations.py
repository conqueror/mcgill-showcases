"""Tests for activation helpers."""

from __future__ import annotations

import numpy as np

from neural_network_foundations_showcase import activations


def test_core_activation_functions_match_expected_values() -> None:
    """Activation helpers should expose standard nonlinearities."""

    inputs = np.array([-2.0, 0.0, 2.0])

    np.testing.assert_allclose(
        activations.sigmoid(inputs),
        np.array([0.11920292, 0.5, 0.88079708]),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        activations.relu(inputs),
        np.array([0.0, 0.0, 2.0]),
    )


def test_activation_comparison_table_has_stable_columns() -> None:
    """The artifact table should keep a readable, deterministic schema."""

    table = activations.build_activation_comparison_table(
        np.array([-1.0, 0.0, 1.0]),
    )

    assert list(table.columns) == [
        "input",
        "sigmoid",
        "tanh",
        "relu",
        "leaky_relu",
    ]
    assert table.shape == (3, 5)


def test_leaky_relu_derivative_uses_the_forward_slope() -> None:
    """A slope of 1/4 gives derivative 1/4 on the negative side."""

    outputs = activations.leaky_relu(np.array([-2.0, 2.0]), slope=0.25)
    np.testing.assert_allclose(
        activations.activation_derivative("leaky_relu", outputs, slope=0.25),
        [0.25, 1.0],
    )


def test_sigmoid_handles_large_negative_inputs_without_overflow() -> None:
    """Extreme logits should still give probabilities in [0, 1]."""

    with np.errstate(over="raise"):
        np.testing.assert_allclose(
            activations.sigmoid(np.array([-1000.0, 0.0, 1000.0])),
            [0.0, 0.5, 1.0],
        )
