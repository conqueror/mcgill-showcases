"""Tests for optimization teaching helpers."""

from __future__ import annotations

from deep_learning_math_foundations_showcase import optimization


def test_gradient_descent_trace_is_monotonic_on_convex_example() -> None:
    """Loss should decrease monotonically on the simple quadratic example."""

    trace = optimization.run_gradient_descent_trace(
        start_x=8.0,
        learning_rate=0.1,
        steps=8,
    )

    losses = trace["loss"].tolist()
    assert losses == sorted(losses, reverse=True)
    assert round(float(trace.iloc[-1]["x"]), 6) == 1.342177


def test_gradient_trace_has_expected_columns() -> None:
    """The optimization artifact schema should remain stable."""

    trace = optimization.run_gradient_descent_trace()
    assert list(trace.columns) == ["iteration", "x", "gradient", "loss"]
