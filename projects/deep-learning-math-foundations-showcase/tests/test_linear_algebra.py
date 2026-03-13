"""Tests for linear algebra teaching helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from deep_learning_math_foundations_showcase import linear_algebra


def test_vector_operations_match_expected_values() -> None:
    """Basic vector operations should remain deterministic and easy to explain."""

    left = np.array([1.0, 2.0])
    right = np.array([3.0, 4.0])

    np.testing.assert_allclose(linear_algebra.add_vectors(left, right), [4.0, 6.0])
    np.testing.assert_allclose(linear_algebra.scale_vector(left, 2.0), [2.0, 4.0])
    assert linear_algebra.dot_product(left, right) == 11.0


def test_rotation_example_is_correct_to_tolerance() -> None:
    """The rotation example from the source deck should stay stable."""

    rotated = linear_algebra.rotate_vector(np.array([1.0, 2.0]), angle_degrees=45.0)
    expected = np.array([-0.70710678, 2.12132034])
    np.testing.assert_allclose(rotated, expected, atol=1e-6)


def test_linear_algebra_tables_have_expected_columns() -> None:
    """Artifact tables should expose stable columns for docs and reporting."""

    vector_table = linear_algebra.build_vector_operations_table()
    matrix_table = linear_algebra.build_matrix_transformations_table()

    assert isinstance(vector_table, pd.DataFrame)
    assert isinstance(matrix_table, pd.DataFrame)
    assert list(vector_table.columns) == ["operation", "result"]
    assert list(matrix_table.columns) == [
        "transformation",
        "input_vector",
        "output_vector",
    ]
