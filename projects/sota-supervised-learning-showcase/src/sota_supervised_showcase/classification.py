"""Classification workflows for a self-guided supervised learning tutorial."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_STATE
from .data import (
    ClassificationSplit,
    build_binary_target,
    build_multilabel_targets,
    list_rebalance_strategies,
    rebalance_binary_training_data,
)

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - optional dependency
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover - optional dependency
    CatBoostClassifier = None


@dataclass(frozen=True)
class BinaryEvaluationResult:
    metrics: pd.DataFrame
    pr_curves: pd.DataFrame
    roc_curves: pd.DataFrame


@dataclass(frozen=True)
class ModelSelectionResult:
    summary: dict[str, float]
    validation_curve: pd.DataFrame
    learning_curve: pd.DataFrame


def evaluate_binary_classification(
    split: ClassificationSplit,
    positive_digit: int = 0,
) -> BinaryEvaluationResult:
    """
    Evaluate binary classification across three imbalance strategies.

    The binary task is "digit == positive_digit" vs all other digits.
    """
    y_train_binary = build_binary_target(split.y_train, positive_digit=positive_digit)
    y_test_binary = build_binary_target(split.y_test, positive_digit=positive_digit)

    strategies = list_rebalance_strategies()
    metric_rows: list[dict[str, float | str]] = []
    pr_rows: list[dict[str, float | str]] = []
    roc_rows: list[dict[str, float | str]] = []

    for strategy in strategies:
        x_balanced, y_balanced = rebalance_binary_training_data(
            split.x_train,
            y_train_binary,
            strategy=strategy,
            random_state=RANDOM_STATE,
        )
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2_000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        )
        model.fit(x_balanced, y_balanced)
        y_pred = model.predict(split.x_test)
        y_score = model.decision_function(split.x_test)

        precision, recall, pr_thresholds = precision_recall_curve(
            y_test_binary, y_score
        )
        fpr, tpr, roc_thresholds = roc_curve(y_test_binary, y_score)

        metric_rows.append(
            {
                "strategy": strategy,
                "accuracy": accuracy_score(y_test_binary, y_pred),
                "precision": precision_score(y_test_binary, y_pred, zero_division=0),
                "recall": recall_score(y_test_binary, y_pred, zero_division=0),
                "f1": f1_score(y_test_binary, y_pred, zero_division=0),
                "pr_auc": auc(recall, precision),
                "roc_auc": roc_auc_score(y_test_binary, y_score),
            }
        )

        pr_rows.extend(
            {
                "strategy": strategy,
                "precision": float(precision_value),
                "recall": float(recall_value),
                "threshold": (
                    float(pr_thresholds[index])
                    if index < len(pr_thresholds)
                    else np.nan
                ),
            }
            for index, (precision_value, recall_value) in enumerate(
                zip(precision, recall, strict=False)
            )
        )
        roc_rows.extend(
            {
                "strategy": strategy,
                "fpr": float(fpr_value),
                "tpr": float(tpr_value),
                "threshold": (
                    float(roc_thresholds[index])
                    if index < len(roc_thresholds)
                    else np.nan
                ),
            }
            for index, (fpr_value, tpr_value) in enumerate(zip(fpr, tpr, strict=False))
        )

    return BinaryEvaluationResult(
        metrics=pd.DataFrame(metric_rows).sort_values("f1", ascending=False),
        pr_curves=pd.DataFrame(pr_rows),
        roc_curves=pd.DataFrame(roc_rows),
    )


def evaluate_multiclass_strategies(split: ClassificationSplit) -> pd.DataFrame:
    """Compare One-vs-Rest and One-vs-One strategies on the digits task."""
    ovr_model = OneVsRestClassifier(
        LogisticRegression(max_iter=2_000, random_state=RANDOM_STATE)
    )
    ovo_model = OneVsOneClassifier(
        make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma="scale"))
    )

    rows: list[dict[str, float | str]] = []
    for model_name, model in (("ovr_logistic", ovr_model), ("ovo_svc", ovo_model)):
        model.fit(split.x_train, split.y_train)
        y_pred = model.predict(split.x_test)
        rows.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(split.y_test, y_pred),
                "f1_micro": f1_score(split.y_test, y_pred, average="micro"),
                "f1_macro": f1_score(split.y_test, y_pred, average="macro"),
                "f1_weighted": f1_score(split.y_test, y_pred, average="weighted"),
            }
        )
    return pd.DataFrame(rows).sort_values("f1_macro", ascending=False)


def evaluate_multilabel_classification(split: ClassificationSplit) -> pd.DataFrame:
    """Train a multi-label classifier and compute macro/weighted F1 metrics."""
    y_train_multilabel = build_multilabel_targets(split.y_train)
    y_test_multilabel = build_multilabel_targets(split.y_test)

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(split.x_train, y_train_multilabel)
    y_pred = model.predict(split.x_test)

    label_names = ["is_large_digit", "is_odd_digit"]
    rows: list[dict[str, float | str]] = []
    for label_index, label_name in enumerate(label_names):
        rows.append(
            {
                "label": label_name,
                "f1": f1_score(
                    y_test_multilabel[:, label_index], y_pred[:, label_index]
                ),
            }
        )
    rows.append(
        {
            "label": "macro_average",
            "f1": f1_score(y_test_multilabel, y_pred, average="macro"),
        }
    )
    rows.append(
        {
            "label": "weighted_average",
            "f1": f1_score(y_test_multilabel, y_pred, average="weighted"),
        }
    )
    return pd.DataFrame(rows)


def evaluate_multioutput_denoising(split: ClassificationSplit) -> pd.DataFrame:
    """
    Build a multi-output denoising task by predicting clean pixel intensities.

    Each target dimension is one pixel value in the original image.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    noise_train = rng.integers(0, 5, size=split.x_train.shape)
    noise_test = rng.integers(0, 5, size=split.x_test.shape)

    x_train_noisy = split.x_train + noise_train
    x_test_noisy = split.x_test + noise_test
    y_train_clean = split.x_train
    y_test_clean = split.x_test

    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(x_train_noisy, y_train_clean)
    y_pred_clean = model.predict(x_test_noisy)

    rows = [
        {
            "metric": "mae_pixels",
            "value": mean_absolute_error(y_test_clean, y_pred_clean),
        },
        {
            "metric": "mse_pixels",
            "value": mean_squared_error(y_test_clean, y_pred_clean),
        },
    ]
    return pd.DataFrame(rows)


def _build_optional_boosting_models() -> list[tuple[str, object]]:
    optional_models: list[tuple[str, object]] = []
    if XGBClassifier is not None:
        optional_models.append(
            (
                "xgboost",
                XGBClassifier(
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    n_estimators=120,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            )
        )
    if LGBMClassifier is not None:
        optional_models.append(
            (
                "lightgbm",
                LGBMClassifier(
                    n_estimators=120,
                    learning_rate=0.05,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    verbose=-1,
                ),
            )
        )
    if CatBoostClassifier is not None:
        optional_models.append(
            (
                "catboost",
                CatBoostClassifier(
                    iterations=120,
                    learning_rate=0.05,
                    depth=6,
                    random_seed=RANDOM_STATE,
                    verbose=False,
                ),
            )
        )
    return optional_models


def build_classification_benchmark(split: ClassificationSplit) -> pd.DataFrame:
    """Benchmark tree and ensemble algorithms used in the tutorial."""
    # Keep explicit feature names to avoid warning noise in optional libraries
    # (for example LightGBM) and to improve consistency across estimators.
    x_train = pd.DataFrame(split.x_train, columns=split.feature_names)
    x_test = pd.DataFrame(split.x_test, columns=split.feature_names)

    base_models: list[tuple[str, object]] = [
        ("baseline_dummy", DummyClassifier(strategy="most_frequent")),
        (
            "decision_tree",
            DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
        ),
        (
            "bagging_tree",
            BaggingClassifier(
                estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
                n_estimators=150,
                bootstrap=True,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "adaboost",
            AdaBoostClassifier(
                estimator=DecisionTreeClassifier(
                    max_depth=1,
                    random_state=RANDOM_STATE,
                ),
                n_estimators=120,
                learning_rate=0.8,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "gradient_boosting",
            GradientBoostingClassifier(
                n_estimators=120,
                learning_rate=0.05,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=220,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "mlp_classifier",
            make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    max_iter=300,
                    random_state=RANDOM_STATE,
                ),
            ),
        ),
    ]

    voting_model = VotingClassifier(
        estimators=[
            (
                "lr",
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=2_000, random_state=RANDOM_STATE),
                ),
            ),
            ("rf", RandomForestClassifier(n_estimators=180, random_state=RANDOM_STATE)),
            ("svc", make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma="scale"))),
        ],
        voting="hard",
    )
    stacking_model = StackingClassifier(
        estimators=[
            (
                "lr",
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=2_000, random_state=RANDOM_STATE),
                ),
            ),
            ("rf", RandomForestClassifier(n_estimators=180, random_state=RANDOM_STATE)),
            ("svc", make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma="scale"))),
            ("knn", KNeighborsClassifier(n_neighbors=5)),
        ],
        final_estimator=LogisticRegression(max_iter=2_000, random_state=RANDOM_STATE),
        n_jobs=-1,
    )
    base_models.extend([("voting_hard", voting_model), ("stacking", stacking_model)])
    base_models.extend(_build_optional_boosting_models())

    rows: list[dict[str, float | str]] = []
    for model_name, model in base_models:
        model.fit(x_train, split.y_train)
        y_pred = model.predict(x_test)
        rows.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(split.y_test, y_pred),
                "f1_macro": f1_score(split.y_test, y_pred, average="macro"),
                "f1_weighted": f1_score(split.y_test, y_pred, average="weighted"),
            }
        )
    return pd.DataFrame(rows).sort_values("f1_macro", ascending=False)


def build_model_selection_summary(split: ClassificationSplit) -> ModelSelectionResult:
    """
    Compute validation and learning curves for model-selection discussions.

    Uses RandomForest to keep alignment with the ensemble section.
    """
    model = RandomForestClassifier(
        n_estimators=180,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    param_range = np.arange(1, 16)
    train_scores, validation_scores = validation_curve(
        model,
        split.x_train,
        split.y_train,
        param_name="max_depth",
        param_range=param_range,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
    )
    validation_df = pd.DataFrame(
        {
            "max_depth": param_range,
            "train_mean": train_scores.mean(axis=1),
            "train_std": train_scores.std(axis=1),
            "validation_mean": validation_scores.mean(axis=1),
            "validation_std": validation_scores.std(axis=1),
        }
    )

    train_sizes, train_curve_scores, val_curve_scores = learning_curve(
        model,
        split.x_train,
        split.y_train,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
    )
    learning_df = pd.DataFrame(
        {
            "train_size": train_sizes,
            "train_mean": train_curve_scores.mean(axis=1),
            "train_std": train_curve_scores.std(axis=1),
            "validation_mean": val_curve_scores.mean(axis=1),
            "validation_std": val_curve_scores.std(axis=1),
        }
    )

    best_index = validation_df["validation_mean"].idxmax()
    summary = {
        "best_max_depth": float(validation_df.loc[best_index, "max_depth"]),
        "best_validation_f1_macro": float(
            validation_df.loc[best_index, "validation_mean"]
        ),
        "largest_train_size": float(learning_df["train_size"].max()),
        "largest_train_size_validation_f1_macro": float(
            learning_df.loc[learning_df["train_size"].idxmax(), "validation_mean"]
        ),
    }
    return ModelSelectionResult(
        summary=summary,
        validation_curve=validation_df,
        learning_curve=learning_df,
    )
