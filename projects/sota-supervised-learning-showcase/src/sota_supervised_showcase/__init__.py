"""Utilities for supervised learning tutorial demos."""

from .classification import (
    build_classification_benchmark,
    build_model_selection_summary,
    evaluate_binary_classification,
    evaluate_multiclass_strategies,
    evaluate_multilabel_classification,
    evaluate_multioutput_denoising,
)
from .data import (
    ClassificationSplit,
    RegressionSplit,
    build_binary_target,
    build_multilabel_targets,
    load_digits_split,
    load_regression_split,
    rebalance_binary_training_data,
)
from .regression import evaluate_regression_models

__all__ = [
    "ClassificationSplit",
    "RegressionSplit",
    "build_binary_target",
    "build_classification_benchmark",
    "build_model_selection_summary",
    "build_multilabel_targets",
    "evaluate_binary_classification",
    "evaluate_multiclass_strategies",
    "evaluate_multilabel_classification",
    "evaluate_multioutput_denoising",
    "evaluate_regression_models",
    "load_digits_split",
    "load_regression_split",
    "rebalance_binary_training_data",
]
