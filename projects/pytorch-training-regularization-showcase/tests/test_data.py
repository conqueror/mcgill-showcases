"""Tests for dataset helpers."""

from __future__ import annotations

import torch

from pytorch_training_regularization_showcase import data


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
