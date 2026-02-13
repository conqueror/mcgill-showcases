from __future__ import annotations

from sota_showcase.active_learning import run_active_learning_simulation
from sota_showcase.data import load_digits_dataset, make_train_test_split


def test_active_learning_outputs_expected_columns() -> None:
    dataset = load_digits_dataset(scale=True)
    split = make_train_test_split(
        X=dataset.X,
        y=dataset.y,
        test_size=0.3,
        labeled_fraction=0.08,
        random_state=42,
    )

    output = run_active_learning_simulation(
        X_train=split.X_train,
        y_train=split.y_train,
        y_train_masked=split.y_train_masked,
        X_test=split.X_test,
        y_test=split.y_test,
        random_state=42,
        rounds=3,
        query_size=20,
    )

    expected_columns = {"strategy", "round", "labeled_budget", "accuracy", "f1_macro"}
    assert expected_columns.issubset(set(output.metrics.columns))
    assert {"random", "uncertainty"}.issubset(set(output.metrics["strategy"].unique()))


def test_active_learning_budget_grows_over_rounds() -> None:
    dataset = load_digits_dataset(scale=True)
    split = make_train_test_split(
        X=dataset.X,
        y=dataset.y,
        test_size=0.3,
        labeled_fraction=0.08,
        random_state=42,
    )

    output = run_active_learning_simulation(
        X_train=split.X_train,
        y_train=split.y_train,
        y_train_masked=split.y_train_masked,
        X_test=split.X_test,
        y_test=split.y_test,
        random_state=42,
        rounds=4,
        query_size=15,
    )

    for _, group in output.metrics.groupby("strategy"):
        budgets = group.sort_values("round")["labeled_budget"].tolist()
        assert budgets == sorted(budgets)
