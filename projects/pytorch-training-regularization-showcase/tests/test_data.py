"""Tests for dataset helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pytorch_training_regularization_showcase import data, training


def test_build_dataset_bundle_for_synthetic_has_expected_shapes() -> None:
    """Synthetic dataset bundles should expose stable loader shapes."""

    bundle = data.build_dataset_bundle(
        dataset_name="synthetic",
        batch_size=16,
        random_state=5,
    )

    features, targets = next(iter(bundle.train_loader))
    assert features.shape[1] == bundle.input_dim
    assert bundle.num_classes == 3
    assert targets.dtype == torch.long


def test_synthetic_dataset_bundle_is_deterministic() -> None:
    """Fixed seeds should generate the same first training batch."""

    bundle_a = data.build_dataset_bundle(
        dataset_name="synthetic",
        batch_size=8,
        random_state=9,
    )
    bundle_b = data.build_dataset_bundle(
        dataset_name="synthetic",
        batch_size=8,
        random_state=9,
    )

    batch_a = next(iter(bundle_a.train_loader))
    batch_b = next(iter(bundle_b.train_loader))
    assert torch.allclose(batch_a[0], batch_b[0])
    assert torch.equal(batch_a[1], batch_b[1])


@pytest.mark.parametrize("quick", [False, True])
def test_fashion_mnist_preserves_official_test_boundary(
    monkeypatch: pytest.MonkeyPatch,
    quick: bool,
) -> None:
    """Official test rows stay held out, even if their pixels or labels change."""

    train_pixels = torch.arange(40).reshape(-1, 1, 1).expand(-1, 28, 28).clone()
    test_pixels = torch.arange(100, 112).reshape(-1, 1, 1).expand(-1, 28, 28).clone()
    train_targets = torch.arange(40) % 2
    test_targets = torch.arange(12) % 2

    def local_fashion_mnist(
        *, root: Path, train: bool, download: bool
    ) -> SimpleNamespace:
        return SimpleNamespace(
            data=train_pixels if train else test_pixels,
            targets=train_targets if train else test_targets,
            classes=["zero", "one"],
        )

    monkeypatch.setattr(data.datasets, "FashionMNIST", local_fashion_mnist)
    original = data.build_dataset_bundle("fashion_mnist", random_state=7, quick=quick)
    features, targets = original.test_loader.dataset.tensors
    torch.testing.assert_close(features, test_pixels.reshape(-1, 784).float() / 255.0)
    torch.testing.assert_close(targets, test_targets)
    config = training.TrainingConfig(hidden_dims=(4,), epochs=1, dropout=0.0)
    original_fit = training.train_classifier(original, config)

    test_pixels = test_pixels + 20
    test_targets = 1 - test_targets
    changed = data.build_dataset_bundle("fashion_mnist", random_state=7, quick=quick)
    for loader_name in ("train_loader", "validation_loader"):
        for before, after in zip(
            getattr(original, loader_name).dataset.tensors,
            getattr(changed, loader_name).dataset.tensors,
            strict=True,
        ):
            torch.testing.assert_close(before, after, rtol=0.0, atol=0.0)
    changed_fit = training.train_classifier(changed, config)
    for name, parameter in original_fit.model.state_dict().items():
        torch.testing.assert_close(
            parameter, changed_fit.model.state_dict()[name], rtol=0.0, atol=0.0
        )
