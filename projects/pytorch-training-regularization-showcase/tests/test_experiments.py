"""Tests for experiment orchestration helpers."""

from __future__ import annotations

from pytorch_training_regularization_showcase import data, experiments, training


def test_optimizer_and_regularization_experiments_return_stable_tables() -> None:
    """Experiment runners should return predictable summary schemas."""

    bundle = data.build_dataset_bundle(
        dataset_name="synthetic",
        batch_size=24,
        random_state=8,
        quick=True,
    )
    base_config = training.TrainingConfig(
        hidden_dims=(24,),
        epochs=3,
        learning_rate=0.02,
        optimizer_name="adam",
        random_state=8,
    )

    optimizer_table = experiments.run_optimizer_comparison(bundle, base_config)
    scheduler_table = experiments.run_scheduler_comparison(bundle, base_config)
    regularization_table = experiments.run_regularization_ablation(bundle, base_config)

    assert set(optimizer_table["optimizer"]) == {"sgd", "adam", "rmsprop"}
    assert {
        "best_validation_accuracy",
        "test_accuracy",
    }.issubset(optimizer_table.columns)
    assert set(scheduler_table["scheduler"]) == {"none", "step", "cosine"}
    assert set(regularization_table["experiment"]) == {
        "baseline",
        "dropout",
        "batch_norm",
        "weight_decay",
        "all_regularization",
    }
