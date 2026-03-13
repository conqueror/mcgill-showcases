"""Tests for model helpers."""

from __future__ import annotations

import torch

from pytorch_training_regularization_showcase import models


def test_feedforward_classifier_produces_expected_logit_shape() -> None:
    """The classifier should emit one logit vector per example."""

    model = models.build_classifier(
        input_dim=16,
        num_classes=3,
        hidden_dims=(12, 6),
        dropout=0.2,
        batch_norm=True,
    )
    batch = torch.randn(5, 16)

    logits = model(batch)

    assert logits.shape == (5, 3)
    assert models.count_trainable_parameters(model) > 0
