"""Linear algebra helpers used to build beginner-friendly teaching artifacts."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _to_vector_string(vector: np.ndarray) -> str:
    """Render a vector with stable formatting for CSV artifacts."""

    rounded = [f"{value:.3f}" for value in vector]
    return "[" + ", ".join(rounded) + "]"


def add_vectors(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return element-wise vector addition."""

    return np.asarray(left, dtype=float) + np.asarray(right, dtype=float)


def scale_vector(vector: np.ndarray, scalar: float) -> np.ndarray:
    """Return scalar multiplication for a vector."""

    return np.asarray(vector, dtype=float) * float(scalar)


def dot_product(left: np.ndarray, right: np.ndarray) -> float:
    """Return the dot product between two vectors."""

    return float(np.dot(np.asarray(left, dtype=float), np.asarray(right, dtype=float)))


def rotate_vector(vector: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate a 2D vector by the requested angle in degrees."""

    radians = math.radians(angle_degrees)
    rotation_matrix = np.array(
        [
            [math.cos(radians), -math.sin(radians)],
            [math.sin(radians), math.cos(radians)],
        ],
    )
    return rotation_matrix @ np.asarray(vector, dtype=float)


def build_vector_operations_table() -> pd.DataFrame:
    """Build a compact artifact table covering the main vector examples."""

    left = np.array([1.0, 2.0])
    right = np.array([3.0, 4.0])

    rows = [
        {
            "operation": "addition",
            "result": _to_vector_string(add_vectors(left, right)),
        },
        {
            "operation": "scalar_multiplication",
            "result": _to_vector_string(scale_vector(left, 2.0)),
        },
        {"operation": "dot_product", "result": f"{dot_product(left, right):.3f}"},
    ]
    return pd.DataFrame(rows)


def build_matrix_transformations_table() -> pd.DataFrame:
    """Build a small artifact table for matrix-based transformations."""

    input_vector = np.array([1.0, 2.0])
    rotated = rotate_vector(input_vector, angle_degrees=45.0)
    scaled = scale_vector(input_vector, 1.5)

    rows = [
        {
            "transformation": "rotation_45_degrees",
            "input_vector": _to_vector_string(input_vector),
            "output_vector": _to_vector_string(rotated),
        },
        {
            "transformation": "scaling_1.5x",
            "input_vector": _to_vector_string(input_vector),
            "output_vector": _to_vector_string(scaled),
        },
    ]
    return pd.DataFrame(rows)
