from __future__ import annotations

GRID_SPACE: dict[str, list[int]] = {
    "n_estimators": [40, 80, 120],
    "max_depth": [3, 5, 8],
    "min_samples_split": [2, 4, 8],
}

RANDOM_SPACE_BOUNDS: dict[str, tuple[int, int]] = {
    "n_estimators": (30, 150),
    "max_depth": (2, 12),
    "min_samples_split": (2, 12),
}
