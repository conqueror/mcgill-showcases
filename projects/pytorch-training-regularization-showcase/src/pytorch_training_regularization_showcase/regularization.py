"""Regularization scenario helpers."""

from __future__ import annotations

from dataclasses import replace

from pytorch_training_regularization_showcase import training


def build_regularization_scenarios(
    base_config: training.TrainingConfig,
) -> dict[str, training.TrainingConfig]:
    """Return the named regularization scenarios used in the showcase."""

    baseline = replace(
        base_config,
        dropout=0.0,
        batch_norm=False,
        weight_decay=0.0,
    )
    return {
        "baseline": baseline,
        "dropout": replace(baseline, dropout=max(0.25, base_config.dropout)),
        "batch_norm": replace(baseline, batch_norm=True),
        "weight_decay": replace(
            baseline,
            weight_decay=max(1e-3, base_config.weight_decay),
        ),
        "all_regularization": replace(
            baseline,
            dropout=max(0.25, base_config.dropout),
            batch_norm=True,
            weight_decay=max(1e-3, base_config.weight_decay),
        ),
    }
