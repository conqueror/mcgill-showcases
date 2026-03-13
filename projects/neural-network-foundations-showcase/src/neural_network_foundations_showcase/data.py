"""Deterministic toy datasets used throughout the showcase."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ToyDataset:
    """A small binary classification dataset with explanatory metadata."""

    name: str
    features: np.ndarray
    labels: np.ndarray
    description: str


@dataclass(frozen=True)
class DataSplit:
    """Train/validation split used by the training helpers."""

    train_features: np.ndarray
    train_labels: np.ndarray
    validation_features: np.ndarray
    validation_labels: np.ndarray


def make_toy_dataset(
    name: str,
    samples_per_class: int = 80,
    noise: float = 0.2,
    random_state: int = 7,
) -> ToyDataset:
    """Build a deterministic binary classification dataset."""

    rng = np.random.default_rng(random_state)
    if name == "linearly_separable":
        class_zero = rng.normal(
            loc=(-1.1, -0.9),
            scale=noise,
            size=(samples_per_class, 2),
        )
        class_one = rng.normal(
            loc=(1.1, 0.9),
            scale=noise,
            size=(samples_per_class, 2),
        )
        description = (
            "Linearly separable clusters show when a single weighted sum is enough."
        )
    elif name == "xor":
        base = rng.uniform(-1.2, 1.2, size=(samples_per_class * 2, 2))
        base += rng.normal(0.0, noise * 0.35, size=base.shape)
        labels = (base[:, 0] * base[:, 1] > 0).astype(float)
        return ToyDataset(
            name=name,
            features=base.astype(np.float64),
            labels=labels.astype(np.float64),
            description=(
                "XOR requires a hidden layer because no single line can separate it."
            ),
        )
    else:
        raise ValueError(f"Unsupported dataset name: {name}")

    features = np.vstack([class_zero, class_one]).astype(np.float64)
    labels = np.concatenate(
        [
            np.zeros(samples_per_class, dtype=np.float64),
            np.ones(samples_per_class, dtype=np.float64),
        ],
    )
    return ToyDataset(
        name=name,
        features=features,
        labels=labels,
        description=description,
    )


def train_val_split(
    features: np.ndarray,
    labels: np.ndarray,
    validation_fraction: float = 0.25,
    random_state: int = 7,
) -> DataSplit:
    """Shuffle and split examples into train and validation subsets."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    rng = np.random.default_rng(random_state)
    indices = np.arange(len(features))
    rng.shuffle(indices)

    validation_size = max(1, int(round(len(features) * validation_fraction)))
    validation_indices = indices[:validation_size]
    train_indices = indices[validation_size:]

    return DataSplit(
        train_features=features[train_indices],
        train_labels=labels[train_indices],
        validation_features=features[validation_indices],
        validation_labels=labels[validation_indices],
    )
