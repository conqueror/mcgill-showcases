"""Calculus helpers for gradient-focused deep learning intuition."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import sympy as sp


def derivative_at_point(expression: str, symbol: str, point: float) -> float:
    """Differentiate an expression and evaluate it at a single point."""

    variable = sp.Symbol(symbol)
    expr = sp.sympify(expression)
    derivative = sp.diff(expr, variable)
    return float(derivative.subs(variable, point))


def partial_derivatives_at_point(
    expression: str,
    symbols: Sequence[str],
    point: dict[str, float],
) -> dict[str, float]:
    """Return partial derivatives for each symbol at the given point."""

    expr = sp.sympify(expression)
    subs = {sp.Symbol(name): value for name, value in point.items()}

    partials: dict[str, float] = {}
    for name in symbols:
        variable = sp.Symbol(name)
        derivative = sp.diff(expr, variable)
        partials[name] = float(derivative.subs(subs))
    return partials


def build_derivative_examples_table() -> pd.DataFrame:
    """Build the core derivative examples artifact table."""

    examples = [
        {
            "example": "power_rule_x_squared",
            "expression": "x**2",
            "evaluation_point": "x=2.0",
            "derivative_value": derivative_at_point("x**2", symbol="x", point=2.0),
        },
        {
            "example": "power_rule_x_cubed",
            "expression": "x**3",
            "evaluation_point": "x=2.0",
            "derivative_value": derivative_at_point("x**3", symbol="x", point=2.0),
        },
        {
            "example": "partial_derivative_x_squared_plus_xy",
            "expression": "x**2 + x*y",
            "evaluation_point": "x=2.0, y=3.0",
            "derivative_value": partial_derivatives_at_point(
                "x**2 + x*y",
                symbols=("x", "y"),
                point={"x": 2.0, "y": 3.0},
            )["x"],
        },
    ]
    return pd.DataFrame(examples)
