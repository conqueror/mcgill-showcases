"""Experiment runners for the PyTorch training showcase."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from pytorch_training_regularization_showcase import (
    data,
    evaluation,
    regularization,
    training,
)


def _result_row(
    label: str,
    run: training.TrainingResult,
    key_name: str,
) -> dict[str, float | int | str]:
    """Convert one training run into a summary row."""

    final_history = run.history.iloc[-1]
    return {
        key_name: label,
        "best_epoch": int(run.best_epoch),
        "best_validation_accuracy": float(run.best_validation_accuracy),
        "final_train_loss": float(final_history["train_loss"]),
        "final_validation_loss": float(final_history["validation_loss"]),
        "test_accuracy": float(run.test_metrics["test_accuracy"]),
    }


def run_optimizer_comparison(
    bundle: data.DatasetBundle,
    base_config: training.TrainingConfig,
) -> pd.DataFrame:
    """Compare the main optimizer choices for the same model and data."""

    rows = []
    for optimizer_name in ("sgd", "adam", "rmsprop"):
        config = replace(base_config, optimizer_name=optimizer_name)
        result = training.train_classifier(bundle, config)
        rows.append(_result_row(optimizer_name, result, "optimizer"))
    return pd.DataFrame(rows)


def run_scheduler_comparison(
    bundle: data.DatasetBundle,
    base_config: training.TrainingConfig,
) -> pd.DataFrame:
    """Compare a small set of learning-rate schedules."""

    rows = []
    for scheduler_name in ("none", "step", "cosine"):
        config = replace(base_config, scheduler_name=scheduler_name)
        result = training.train_classifier(bundle, config)
        rows.append(_result_row(scheduler_name, result, "scheduler"))
    return pd.DataFrame(rows)


def run_regularization_ablation(
    bundle: data.DatasetBundle,
    base_config: training.TrainingConfig,
) -> pd.DataFrame:
    """Compare the effect of dropout, batch norm, and weight decay."""

    rows = []
    for label, config in regularization.build_regularization_scenarios(
        base_config,
    ).items():
        result = training.train_classifier(bundle, config)
        rows.append(_result_row(label, result, "experiment"))
    return pd.DataFrame(rows)


def build_error_analysis_table(
    run: training.TrainingResult,
    bundle: data.DatasetBundle,
) -> pd.DataFrame:
    """Collect example-level error analysis on the test split."""

    _, logits, targets = training.evaluate_loader(run.model, bundle.test_loader)
    return evaluation.build_error_analysis_table(logits, targets, bundle.class_names)
