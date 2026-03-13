"""Optimization helpers for gradient descent intuition."""

from __future__ import annotations

import pandas as pd


def run_gradient_descent_trace(
    start_x: float = 8.0,
    learning_rate: float = 0.1,
    steps: int = 8,
) -> pd.DataFrame:
    """Run gradient descent on f(x) = x^2 and capture the trace."""

    x_value = float(start_x)
    rows: list[dict[str, float]] = []

    for iteration in range(1, steps + 1):
        gradient = 2.0 * x_value
        x_value = x_value - learning_rate * gradient
        loss = x_value**2
        rows.append(
            {
                "iteration": float(iteration),
                "x": x_value,
                "gradient": gradient,
                "loss": loss,
            },
        )

    return pd.DataFrame(rows)
