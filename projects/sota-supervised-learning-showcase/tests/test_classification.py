from sota_supervised_showcase.classification import (
    build_classification_benchmark,
    evaluate_binary_classification,
    evaluate_multiclass_strategies,
    evaluate_multilabel_classification,
    evaluate_multioutput_denoising,
)
from sota_supervised_showcase.data import ClassificationSplit, load_digits_split


def _small_split() -> ClassificationSplit:
    split = load_digits_split()
    return ClassificationSplit(
        x_train=split.x_train[:600],
        x_test=split.x_test[:200],
        y_train=split.y_train[:600],
        y_test=split.y_test[:200],
        feature_names=split.feature_names,
        target_names=split.target_names,
    )


def test_evaluate_binary_classification_outputs_expected_strategies() -> None:
    split = _small_split()
    result = evaluate_binary_classification(split)
    assert set(result.metrics["strategy"]) == {
        "none",
        "upsample_minority",
        "downsample_majority",
    }
    assert result.metrics["f1"].between(0.0, 1.0).all()


def test_multiclass_multilabel_multioutput_reports_are_nonempty() -> None:
    split = _small_split()
    multiclass = evaluate_multiclass_strategies(split)
    multilabel = evaluate_multilabel_classification(split)
    multioutput = evaluate_multioutput_denoising(split)

    assert not multiclass.empty
    assert set(multiclass["model"]) == {"ovr_logistic", "ovo_svc"}
    assert not multilabel.empty
    assert "macro_average" in set(multilabel["label"])
    assert not multioutput.empty
    assert set(multioutput["metric"]) == {"mae_pixels", "mse_pixels"}


def test_classification_benchmark_has_baseline_and_ensembles() -> None:
    split = _small_split()
    benchmark = build_classification_benchmark(split)
    model_names = set(benchmark["model"])
    assert "baseline_dummy" in model_names
    assert "decision_tree" in model_names
    assert "random_forest" in model_names
    assert "stacking" in model_names
