"""Default runtime settings for the PyTorch showcase."""

from __future__ import annotations

from pytorch_training_regularization_showcase import training


def default_training_config(
    dataset_name: str,
    quick: bool,
    random_state: int = 7,
) -> training.TrainingConfig:
    """Return a practical default config for each supported dataset."""

    if dataset_name == "synthetic":
        return training.TrainingConfig(
            hidden_dims=(24, 12),
            epochs=4 if quick else 6,
            learning_rate=0.02,
            optimizer_name="adam",
            dropout=0.15,
            early_stopping_patience=2,
            random_state=random_state,
        )
    if dataset_name == "fashion_mnist":
        return training.TrainingConfig(
            hidden_dims=(128, 64),
            epochs=5 if quick else 8,
            learning_rate=0.001,
            optimizer_name="adam",
            dropout=0.2,
            batch_norm=True,
            early_stopping_patience=3,
            random_state=random_state,
        )
    return training.TrainingConfig(
        hidden_dims=(64, 32),
        epochs=5 if quick else 8,
        learning_rate=0.01,
        optimizer_name="adam",
        dropout=0.15,
        early_stopping_patience=3,
        random_state=random_state,
    )
