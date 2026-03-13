"""Training loop helpers for the PyTorch showcase."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import pandas as pd
import torch
from torch import nn

from pytorch_training_regularization_showcase import data, evaluation, models


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for a compact PyTorch training run."""

    hidden_dims: tuple[int, ...] = (64, 32)
    dropout: float = 0.1
    batch_norm: bool = False
    epochs: int = 10
    learning_rate: float = 0.01
    optimizer_name: str = "adam"
    scheduler_name: str = "none"
    weight_decay: float = 0.0
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-4
    random_state: int = 7


@dataclass(frozen=True)
class TrainingResult:
    """Outputs from one training run."""

    model: nn.Module
    history: pd.DataFrame
    best_epoch: int
    best_validation_accuracy: float
    test_metrics: dict[str, float]


def build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """Construct a named optimizer for the classifier."""

    parameters = model.parameters()
    if config.optimizer_name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    if config.optimizer_name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    if config.optimizer_name == "rmsprop":
        return torch.optim.RMSprop(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer_name: {config.optimizer_name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Construct an optional scheduler for the run."""

    if config.scheduler_name == "none":
        return None
    if config.scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, config.epochs // 2),
            gamma=0.5,
        )
    if config.scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, config.epochs),
        )
    raise ValueError(f"Unsupported scheduler_name: {config.scheduler_name}")


def _run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    """Run one epoch in training or evaluation mode."""

    training_mode = optimizer is not None
    model.train(training_mode)
    total_loss = 0.0
    total_examples = 0
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for features, targets in loader:
        if training_mode:
            optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, targets)

        if training_mode:
            loss.backward()
            optimizer.step()

        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        all_logits.append(logits.detach())
        all_targets.append(targets.detach())

    stacked_logits = torch.cat(all_logits, dim=0)
    stacked_targets = torch.cat(all_targets, dim=0)
    return {
        "loss": total_loss / total_examples,
        "accuracy": evaluation.compute_accuracy(stacked_logits, stacked_targets),
    }


def evaluate_loader(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    """Evaluate a model on a loader and return metrics plus raw outputs."""

    criterion = nn.CrossEntropyLoss()
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for features, targets in loader:
            logits = model(features)
            loss = criterion(logits, targets)
            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            all_logits.append(logits)
            all_targets.append(targets)

    stacked_logits = torch.cat(all_logits, dim=0)
    stacked_targets = torch.cat(all_targets, dim=0)
    metrics = {
        "loss": total_loss / total_examples,
        "accuracy": evaluation.compute_accuracy(stacked_logits, stacked_targets),
    }
    return metrics, stacked_logits, stacked_targets


def train_classifier(
    bundle: data.DatasetBundle,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Train a small PyTorch classifier and record its learning curve."""

    effective_config = config or TrainingConfig()
    torch.manual_seed(effective_config.random_state)
    model = models.build_classifier(
        input_dim=bundle.input_dim,
        num_classes=bundle.num_classes,
        hidden_dims=effective_config.hidden_dims,
        dropout=effective_config.dropout,
        batch_norm=effective_config.batch_norm,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, effective_config)
    scheduler = build_scheduler(optimizer, effective_config)

    best_validation_accuracy = -1.0
    best_epoch = 1
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    rows = []

    for epoch in range(1, effective_config.epochs + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        train_metrics = _run_epoch(model, bundle.train_loader, criterion, optimizer)
        with torch.no_grad():
            validation_metrics = _run_epoch(
                model,
                bundle.validation_loader,
                criterion,
                optimizer=None,
            )
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "validation_loss": validation_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "validation_accuracy": validation_metrics["accuracy"],
                "learning_rate": learning_rate,
            },
        )

        if validation_metrics["accuracy"] > (
            best_validation_accuracy + effective_config.early_stopping_min_delta
        ):
            best_validation_accuracy = validation_metrics["accuracy"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if scheduler is not None:
            scheduler.step()
        if patience_counter >= effective_config.early_stopping_patience:
            break

    model.load_state_dict(best_state)
    test_metrics, _, _ = evaluate_loader(model, bundle.test_loader)
    return TrainingResult(
        model=model,
        history=pd.DataFrame(rows),
        best_epoch=best_epoch,
        best_validation_accuracy=best_validation_accuracy,
        test_metrics={
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
        },
    )


def measure_gradient_health(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
) -> pd.DataFrame:
    """Collect gradient norms for one batch to explain training stability."""

    criterion = nn.CrossEntropyLoss()
    model.train()
    features, targets = next(iter(loader))
    model.zero_grad()
    logits = model(features)
    loss = criterion(logits, targets)
    loss.backward()

    rows = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        rows.append(
            {
                "parameter": name,
                "gradient_norm": float(parameter.grad.norm().item()),
                "weight_norm": float(parameter.data.norm().item()),
            },
        )
    return pd.DataFrame(rows)
