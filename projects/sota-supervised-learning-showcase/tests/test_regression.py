from sota_supervised_showcase.data import load_regression_split
from sota_supervised_showcase.regression import (
    evaluate_regression_models,
    manual_gradient_boosting_example,
)


def test_regression_benchmark_contains_required_models() -> None:
    split = load_regression_split()
    result = evaluate_regression_models(split)
    names = set(result["model"])
    assert names == {
        "baseline_dummy_mean",
        "linear_regression",
        "gradient_boosting_regression",
    }
    assert (result["rmse"] > 0).all()


def test_manual_gradient_boosting_example_predicts_expected_shape() -> None:
    split = load_regression_split()
    predictions = manual_gradient_boosting_example(
        x_train=split.x_train[:120],
        y_train=split.y_train[:120],
        x_new=split.x_test[:8],
    )
    assert predictions.shape == (8,)
