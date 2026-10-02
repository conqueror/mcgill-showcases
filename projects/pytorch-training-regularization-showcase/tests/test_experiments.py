"""Tests for experiment orchestration helpers."""

from __future__ import annotations

import pandas as pd

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
    for table in (optimizer_table, scheduler_table, regularization_table):
        assert (table["max_epochs"] == 3).all()
        assert table["epochs_run"].between(1, 3).all()
        assert (table["random_state"] == 8).all()
        assert (table["initial_learning_rate"] == 0.02).all()


def test_optimizer_comparison_is_independent_of_other_comparisons() -> None:
    """A focused optimizer table must match the same table after scheduler runs."""

    bundle = data.build_dataset_bundle("synthetic", batch_size=24, quick=True)
    config = training.TrainingConfig(hidden_dims=(8,), epochs=2)
    focused = experiments.run_optimizer_comparison(bundle, config)
    experiments.run_scheduler_comparison(bundle, config)
    full = experiments.run_optimizer_comparison(bundle, config)
    pd.testing.assert_frame_equal(focused, full, check_exact=True)
