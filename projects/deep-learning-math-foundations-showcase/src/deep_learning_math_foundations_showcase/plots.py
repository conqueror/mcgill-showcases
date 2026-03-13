"""Plot path helpers for optional showcase visuals."""

from __future__ import annotations

from pathlib import Path


def gradient_descent_plot_path(output_dir: Path) -> Path:
    """Return the conventional path for the gradient descent plot."""

    return output_dir / "gradient_descent_trace.png"
