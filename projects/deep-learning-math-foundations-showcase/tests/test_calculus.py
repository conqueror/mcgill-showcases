"""Tests for calculus teaching helpers."""

from __future__ import annotations

import pandas as pd

from deep_learning_math_foundations_showcase import calculus


def test_derivative_at_point_matches_power_rule_example() -> None:
    """Derivative examples should stay numerically stable."""

    assert calculus.derivative_at_point("x**2", symbol="x", point=2.0) == 4.0


def test_partial_derivatives_match_expected_values() -> None:
    """Partial derivatives should match the symbolic ground truth."""

    result = calculus.partial_derivatives_at_point(
        "x**2 + x*y",
        symbols=("x", "y"),
        point={"x": 2.0, "y": 3.0},
    )

    assert result == {"x": 7.0, "y": 2.0}


def test_derivative_examples_table_has_expected_shape() -> None:
    """Derivative artifact rows should be exposed through a stable table."""

    table = calculus.build_derivative_examples_table()
    assert isinstance(table, pd.DataFrame)
    assert list(table.columns) == [
        "example",
        "expression",
        "evaluation_point",
        "derivative_value",
    ]
