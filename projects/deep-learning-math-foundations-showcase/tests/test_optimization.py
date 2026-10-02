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
    assert list(trace.columns) == ["iteration", "x_before", "x", "gradient", "loss"]


def test_gradient_trace_records_the_point_where_the_gradient_was_taken() -> None:
    """At x=2, f'=4; a step of 1/4 gives x=1 and f(x)=1."""

    trace = optimization.run_gradient_descent_trace(2.0, 0.25, 2)
    assert trace.iloc[0].to_dict() == {
        "iteration": 1.0,
        "x_before": 2.0,
        "x": 1.0,
        "gradient": 4.0,
        "loss": 1.0,
    }
    assert trace.iloc[1]["x_before"] == 1.0
    assert trace.iloc[1]["gradient"] == 2.0
