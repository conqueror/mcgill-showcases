"""Dataset loaders and target transformations for supervised learning demos."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_diabetes, load_digits
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from .config import RANDOM_STATE, TEST_SIZE


@dataclass(frozen=True)
class ClassificationSplit:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    target_names: list[str]


@dataclass(frozen=True)
class RegressionSplit:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    target_name: str


def load_digits_split(
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> ClassificationSplit:
    """Load the scikit-learn digits dataset and produce a train/test split."""
    dataset = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset.target,
    )
    target_names = [str(label) for label in dataset.target_names]
    feature_names = [f"pixel_{index}" for index in range(dataset.data.shape[1])]
    return ClassificationSplit(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        target_names=target_names,
    )


def load_regression_split(
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> RegressionSplit:
    """
    Load a regression dataset that ships with scikit-learn.

    `load_diabetes` is fully local, so it avoids network calls in beginner setups.
    """
    dataset = load_diabetes()
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=test_size,
        random_state=random_state,
    )
    feature_names = list(dataset.feature_names)
    return RegressionSplit(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        target_name="disease_progression",
    )


def build_binary_target(y: np.ndarray, positive_digit: int = 0) -> np.ndarray:
    """Convert multiclass digit labels into a binary problem (digit vs non-digit)."""
    return (y == positive_digit).astype(int)


def build_multilabel_targets(y: np.ndarray) -> np.ndarray:
    """
    Build two binary labels per sample:
    - is_large_digit: class >= 5
    - is_odd_digit: class % 2 == 1
    """
    is_large_digit = (y >= 5).astype(int)
    is_odd_digit = (y % 2 == 1).astype(int)
    return np.c_[is_large_digit, is_odd_digit]


def rebalance_binary_training_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    strategy: str,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rebalance binary targets using upsampling or downsampling.

    Supported strategies: "none", "upsample_minority", "downsample_majority".
    """
    if strategy == "none":
        return x_train, y_train

    positive_mask = y_train == 1
    x_positive = x_train[positive_mask]
    y_positive = y_train[positive_mask]
    x_negative = x_train[~positive_mask]
    y_negative = y_train[~positive_mask]

    if strategy == "upsample_minority":
        x_positive_up, y_positive_up = resample(
            x_positive,
            y_positive,
            replace=True,
            n_samples=len(y_negative),
            random_state=random_state,
        )
        x_balanced = np.vstack([x_negative, x_positive_up])
        y_balanced = np.concatenate([y_negative, y_positive_up])
        return x_balanced, y_balanced

    if strategy == "downsample_majority":
        x_negative_down, y_negative_down = resample(
            x_negative,
            y_negative,
            replace=False,
            n_samples=len(y_positive),
            random_state=random_state,
        )
        x_balanced = np.vstack([x_positive, x_negative_down])
        y_balanced = np.concatenate([y_positive, y_negative_down])
        return x_balanced, y_balanced

    if strategy in {"smote", "smotetomek", "smoteenn"}:
        try:
            from imblearn.combine import SMOTEENN, SMOTETomek
            from imblearn.over_sampling import SMOTE
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ValueError(
                "SMOTE-based strategies require `imbalanced-learn` optional dependency."
            ) from exc

        if strategy == "smote":
            sampler = SMOTE(random_state=random_state)
        elif strategy == "smotetomek":
            sampler = SMOTETomek(random_state=random_state)
        else:
            sampler = SMOTEENN(random_state=random_state)

        x_res, y_res = sampler.fit_resample(x_train, y_train)
        return np.asarray(x_res), np.asarray(y_res)

    raise ValueError(
        "Unknown strategy. Use one of: "
        "'none', 'upsample_minority', 'downsample_majority', "
        "'smote', 'smotetomek', 'smoteenn'."
    )


def list_rebalance_strategies() -> list[str]:
    strategies = ["none", "upsample_minority", "downsample_majority"]
    try:
        import imblearn  # noqa: F401

        strategies.extend(["smote", "smotetomek", "smoteenn"])
    except Exception:  # pragma: no cover - optional dependency
        pass
    return strategies
