"""Smoke test for the package scaffold (replaced/expanded by later build phases)."""

from __future__ import annotations

import learning_agents


def test_package_imports() -> None:
    """The package imports and exposes a version string."""
    assert learning_agents.__version__
    assert learning_agents.__doc__
