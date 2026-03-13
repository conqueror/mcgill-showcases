"""Tests for training helpers."""

from __future__ import annotations

from pytorch_training_regularization_showcase import data, training


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
