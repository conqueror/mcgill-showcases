"""Dataset helpers for the PyTorch showcase."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.datasets import load_digits, make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from pytorch_training_regularization_showcase import config


@dataclass(frozen=True)
class DatasetBundle:
    """A self-contained train/validation/test dataset bundle."""

    dataset_name: str
    input_dim: int
    num_classes: int
    class_names: list[str]
    train_loader: DataLoader
    validation_loader: DataLoader
    test_loader: DataLoader


def _split_arrays(
    features: np.ndarray,
    targets: np.ndarray,
    random_state: int,
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """Split arrays into deterministic train, validation, and test datasets."""

    train_features, test_features, train_targets, test_targets = train_test_split(
        features,
        targets,
        test_size=0.2,
        stratify=targets,
        random_state=random_state,
    )
    train_features, validation_features, train_targets, validation_targets = (
        train_test_split(
            train_features,
            train_targets,
            test_size=0.25,
            stratify=train_targets,
            random_state=random_state,
        )
    )

    def to_dataset(
        input_features: np.ndarray,
        input_targets: np.ndarray,
    ) -> TensorDataset:
        return TensorDataset(
            torch.tensor(input_features, dtype=torch.float32),
            torch.tensor(input_targets, dtype=torch.long),
        )

    return (
        to_dataset(train_features, train_targets),
        to_dataset(validation_features, validation_targets),
        to_dataset(test_features, test_targets),
    )


def _synthetic_arrays(
    random_state: int,
    quick: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build a deterministic multiclass synthetic classification problem."""

    sample_count = 180 if quick else 480
    features, targets = make_classification(
        n_samples=sample_count,
        n_features=16,
        n_informative=12,
        n_redundant=2,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.7,
        random_state=random_state,
    )
    return (
        features.astype(np.float32),
        targets.astype(np.int64),
        [
            "class_0",
            "class_1",
            "class_2",
        ],
    )


def _digits_arrays(
    quick: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the built-in sklearn digits dataset and flatten it for an MLP."""

    digits = load_digits()
    features = (digits.data / 16.0).astype(np.float32)
    targets = digits.target.astype(np.int64)
    if quick:
        features = features[:900]
        targets = targets[:900]
    class_names = [str(index) for index in range(10)]
    return features, targets, class_names


def _fashion_mnist_arrays(
    quick: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load FashionMNIST and flatten it for a feed-forward classifier."""

    train_dataset = datasets.FashionMNIST(
        root=config.DATA_DIR,
        train=True,
        download=True,
    )
    test_dataset = datasets.FashionMNIST(
        root=config.DATA_DIR,
        train=False,
        download=True,
    )
    features = torch.cat(
        [train_dataset.data.float(), test_dataset.data.float()],
        dim=0,
    ).reshape(-1, 28 * 28)
    targets = torch.cat([train_dataset.targets, test_dataset.targets], dim=0)

    features = (features / 255.0).numpy().astype(np.float32)
    targets = targets.numpy().astype(np.int64)
    if quick:
        features = features[:3000]
        targets = targets[:3000]

    class_names = list(train_dataset.classes)
    return features, targets, class_names


def build_dataset_bundle(
    dataset_name: str = "digits",
    batch_size: int = 64,
    random_state: int = 7,
    quick: bool = False,
) -> DatasetBundle:
    """Build a named dataset bundle with deterministic splits and loaders."""

    if dataset_name == "synthetic":
        features, targets, class_names = _synthetic_arrays(random_state, quick)
    elif dataset_name == "digits":
        features, targets, class_names = _digits_arrays(quick)
    elif dataset_name == "fashion_mnist":
        features, targets, class_names = _fashion_mnist_arrays(quick)
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    train_dataset, validation_dataset, test_dataset = _split_arrays(
        features,
        targets,
        random_state,
    )
    generator = torch.Generator().manual_seed(random_state)

    return DatasetBundle(
        dataset_name=dataset_name,
        input_dim=features.shape[1],
        num_classes=len(class_names),
        class_names=class_names,
        train_loader=DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        ),
        validation_loader=DataLoader(validation_dataset, batch_size=batch_size),
        test_loader=DataLoader(test_dataset, batch_size=batch_size),
    )
