"""Model definitions for the PyTorch training showcase."""

from __future__ import annotations

import torch
from torch import nn


class FeedForwardClassifier(nn.Module):
    """A small multilayer perceptron for flat vector inputs."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.1,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Produce class logits for a batch of features."""

        flat_features = features.view(features.size(0), -1)
        return self.network(flat_features)


def build_classifier(
    input_dim: int,
    num_classes: int,
    hidden_dims: tuple[int, ...] = (64, 32),
    dropout: float = 0.1,
    batch_norm: bool = False,
) -> FeedForwardClassifier:
    """Build the default classifier used across experiments."""

    return FeedForwardClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout=dropout,
        batch_norm=batch_norm,
    )


def count_trainable_parameters(model: nn.Module) -> int:
    """Count the number of parameters that receive gradient updates."""

    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
