"""Training helpers for the neural network foundations showcase."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from neural_network_foundations_showcase import backprop, data, losses, networks


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for a simple full-batch gradient descent loop."""

    layer_sizes: tuple[int, ...] = (2, 6, 1)
    epochs: int = 120
    learning_rate: float = 0.2
    hidden_activation: str = "tanh"
    init_strategy: str = "xavier"
    validation_fraction: float = 0.25
    random_state: int = 7


@dataclass(frozen=True)
class TrainingResult:
    """Outputs from a deterministic training run."""

    network: networks.FeedForwardNetwork
    history: pd.DataFrame
    split: data.DataSplit
    config: TrainingConfig


def evaluate_binary_classifier(
    network: networks.FeedForwardNetwork,
    features: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Compute loss and accuracy for a binary classifier."""

    probabilities = networks.predict_proba(network, features)
    predictions = (probabilities >= 0.5).astype(np.float64)
    return {
        "loss": losses.binary_cross_entropy(probabilities, labels),
        "accuracy": float(np.mean(predictions == labels)),
    }


def train_network(
    dataset: data.ToyDataset,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Train a small network with full-batch gradient descent."""

    effective_config = config or TrainingConfig()
    split = data.train_val_split(
        dataset.features,
        dataset.labels,
        validation_fraction=effective_config.validation_fraction,
        random_state=effective_config.random_state,
    )
    network = networks.build_network(
        layer_sizes=effective_config.layer_sizes,
        init_strategy=effective_config.init_strategy,
        hidden_activation=effective_config.hidden_activation,
        random_state=effective_config.random_state,
    )

    rows = []
    for epoch in range(effective_config.epochs + 1):
        train_metrics = evaluate_binary_classifier(
            network,
            split.train_features,
            split.train_labels,
        )
        validation_metrics = evaluate_binary_classifier(
            network,
            split.validation_features,
            split.validation_labels,
        )
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "validation_loss": validation_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "validation_accuracy": validation_metrics["accuracy"],
            },
        )

        if epoch == effective_config.epochs:
            break

        gradients = backprop.compute_gradients(
            network,
            split.train_features,
            split.train_labels,
        )
        network = networks.apply_gradient_step(
            network,
            gradients.weight_gradients,
            gradients.bias_gradients,
            effective_config.learning_rate,
        )

    return TrainingResult(
        network=network,
        history=pd.DataFrame(rows),
        split=split,
        config=effective_config,
    )


def _inject_label_noise(
    dataset: data.ToyDataset,
    fraction: float,
    random_state: int,
) -> data.ToyDataset:
    """Flip a deterministic subset of labels to provoke overfitting."""

    rng = np.random.default_rng(random_state)
    noisy_labels = dataset.labels.copy()
    flips = max(1, int(round(len(noisy_labels) * fraction)))
    indices = rng.choice(len(noisy_labels), size=flips, replace=False)
    noisy_labels[indices] = 1.0 - noisy_labels[indices]
    return data.ToyDataset(
        name=f"{dataset.name}_with_label_noise",
        features=dataset.features,
        labels=noisy_labels,
        description=(
            dataset.description + " Training labels include a small noisy subset."
        ),
    )


def _regime_runs(random_state: int) -> list[tuple[str, TrainingResult]]:
    """Build the three headline fitting regimes used in docs and artifacts."""

    xor_dataset = data.make_toy_dataset(
        "xor",
        samples_per_class=40,
        noise=0.22,
        random_state=random_state,
    )
    noisy_small_dataset = _inject_label_noise(
        data.make_toy_dataset(
            "xor",
            samples_per_class=14,
            noise=0.3,
            random_state=random_state + 1,
        ),
        fraction=0.18,
        random_state=random_state + 2,
    )
    scenarios = [
        (
            "underfit",
            xor_dataset,
            TrainingConfig(
                layer_sizes=(2, 1),
                epochs=80,
                learning_rate=0.25,
                random_state=random_state,
            ),
        ),
        (
            "well_fit",
            xor_dataset,
            TrainingConfig(
                layer_sizes=(2, 8, 1),
                epochs=200,
                learning_rate=0.25,
                random_state=random_state + 3,
            ),
        ),
        (
            "overfit",
            noisy_small_dataset,
            TrainingConfig(
                layer_sizes=(2, 16, 16, 1),
                epochs=320,
                learning_rate=0.18,
                hidden_activation="relu",
                init_strategy="he",
                validation_fraction=0.4,
                random_state=random_state + 4,
            ),
        ),
    ]
    return [
        (regime, train_network(dataset, config))
        for regime, dataset, config in scenarios
    ]


def build_underfit_overfit_table(random_state: int = 7) -> pd.DataFrame:
    """Summarize final train/validation behavior for the three fit regimes."""

    rows = []
    for regime, result in _regime_runs(random_state):
        final_row = result.history.iloc[-1]
        rows.append(
            {
                "regime": regime,
                "train_accuracy": float(final_row["train_accuracy"]),
                "validation_accuracy": float(final_row["validation_accuracy"]),
                "generalization_gap": float(
                    final_row["train_accuracy"] - final_row["validation_accuracy"]
                ),
                "epochs": result.config.epochs,
                "architecture": "-".join(
                    str(size) for size in result.config.layer_sizes
                ),
            },
        )
    return pd.DataFrame(rows)


def build_training_curve_table(random_state: int = 7) -> pd.DataFrame:
    """Return a long-form training curve table for all three fit regimes."""

    tables = []
    for regime, result in _regime_runs(random_state):
        history = result.history.copy()
        history.insert(0, "scenario", regime)
        tables.append(history)
    return pd.concat(tables, ignore_index=True)
