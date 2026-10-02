"""Tests for training helpers."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import torch

from pytorch_training_regularization_showcase import data, models, training


def test_train_classifier_records_history_with_expected_schema() -> None:
    """Training should produce a compact, readable history table."""

    bundle = data.build_dataset_bundle(
        dataset_name="synthetic",
        batch_size=24,
        random_state=4,
        quick=True,
    )
    result = training.train_classifier(
        bundle,
        training.TrainingConfig(
            hidden_dims=(24,),
            epochs=4,
            learning_rate=0.02,
            optimizer_name="adam",
            early_stopping_patience=2,
            random_state=4,
        ),
    )

    assert list(result.history.columns) == [
        "epoch",
        "train_loss",
        "validation_loss",
        "train_accuracy",
        "validation_accuracy",
        "learning_rate",
    ]
    assert 1 <= len(result.history) <= 4
    assert 0.0 <= result.best_validation_accuracy <= 1.0


def test_repeated_training_is_independent_of_previous_loader_use() -> None:
    """A dedicated shuffle generator must restart for each seeded training run."""

    bundle = data.build_dataset_bundle("synthetic", batch_size=16, quick=True)
    config = training.TrainingConfig(hidden_dims=(8,), epochs=3, random_state=7)
    original = training.train_classifier(bundle, config)
    training.train_classifier(bundle, replace(config, optimizer_name="sgd"))
    repeated = training.train_classifier(bundle, config)
    pd.testing.assert_frame_equal(original.history, repeated.history, check_exact=True)
    assert original.test_metrics == repeated.test_metrics
    for name, parameter in original.model.state_dict().items():
        torch.testing.assert_close(
            parameter, repeated.model.state_dict()[name], rtol=0.0, atol=0.0
        )


def test_gradient_diagnostic_preserves_model_state_and_mode() -> None:
    """Measuring gradients must not update BatchNorm, parameters, gradients, or mode."""

    bundle = data.build_dataset_bundle("synthetic", batch_size=16, quick=True)
    model = models.build_classifier(
        bundle.input_dim, bundle.num_classes, (8,), dropout=0.2, batch_norm=True
    )
    model.eval()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    state = {name: value.clone() for name, value in model.state_dict().items()}
    gradients = [parameter.grad.clone() for parameter in model.parameters()]
    rng_state = torch.get_rng_state().clone()
    loader_state = bundle.train_loader.generator.get_state().clone()
    report = training.measure_gradient_health(model, bundle.train_loader)
    assert not report.empty
    assert not model.training
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0.0, atol=0.0)
    for parameter, gradient in zip(model.parameters(), gradients, strict=True):
        torch.testing.assert_close(parameter.grad, gradient)
    torch.testing.assert_close(torch.get_rng_state(), rng_state)
    torch.testing.assert_close(bundle.train_loader.generator.get_state(), loader_state)
