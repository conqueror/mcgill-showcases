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
