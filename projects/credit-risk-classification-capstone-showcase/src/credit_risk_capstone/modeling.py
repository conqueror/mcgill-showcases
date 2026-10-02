from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from ml_core.splits import SplitBundle3


class SupportsBinaryProba(Protocol):
    def fit(self, x: pd.DataFrame, y: pd.Series) -> object: ...

    def predict_proba(self, x: pd.DataFrame) -> NDArray[Any]: ...


def list_imbalance_methods() -> list[str]:
    return [
        "none",
        "upsample_minority",
        "downsample_majority",
        "smote",
        "smotetomek",
        "smoteenn",
    ]


def fit_candidate_model(
    name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    imbalance_strategy: str = "none",
    random_state: int = 42,
) -> SupportsBinaryProba:
    from ml_core.imbalance import resample_binary

    if name == "logistic_regression":
        model: SupportsBinaryProba = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced", max_iter=1200, random_state=random_state
                    ),
                ),
            ]
        )
    elif name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=320,
            max_depth=8,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=1,
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    x_resampled, y_resampled = resample_binary(
        x_train, y_train, method=imbalance_strategy, random_state=random_state
    )
    model.fit(x_resampled, y_resampled)
    return model


def evaluate_imbalance_strategies(
    split: SplitBundle3,
    *,
    random_state: int = 42,
) -> tuple[pd.DataFrame, NDArray[np.float64], str]:
    rows: list[dict[str, float | str]] = []
    best_scores: NDArray[np.float64] | None = None
    best_strategy = "none"
    best_f1 = -1.0

    for method in list_imbalance_methods():
        try:
            model = fit_candidate_model(
                "logistic_regression",
                split.x_train,
                split.y_train,
                imbalance_strategy=method,
                random_state=random_state,
            )
        except RuntimeError:
            continue

        scores = np.asarray(model.predict_proba(split.x_val)[:, 1], dtype=np.float64)
        preds = (scores >= 0.5).astype(int)

        f1 = float(f1_score(split.y_val, preds, zero_division=0))
        roc = float(roc_auc_score(split.y_val, scores))
        pr = float(average_precision_score(split.y_val, scores))

        rows.append(
            {
                "strategy": method,
                "val_f1": f1,
                "val_roc_auc": roc,
                "val_average_precision": pr,
            }
        )

        if f1 > best_f1:
            best_f1 = f1
            best_scores = scores
            best_strategy = method

    if best_scores is None:
        raise RuntimeError("No imbalance strategy could be evaluated.")

    result = pd.DataFrame(rows).sort_values("val_f1", ascending=False).reset_index(drop=True)
    return result, best_scores, best_strategy


def model_benchmark(
    split: SplitBundle3, *, random_state: int = 42, imbalance_strategy: str = "none"
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for name in ["logistic_regression", "random_forest"]:
        model = fit_candidate_model(
            name,
            split.x_train,
            split.y_train,
            imbalance_strategy=imbalance_strategy,
            random_state=random_state,
        )
        scores = np.asarray(model.predict_proba(split.x_val)[:, 1], dtype=np.float64)
        preds = (scores >= 0.5).astype(int)
        rows.append(
            {
                "model": name,
                "val_f1": float(f1_score(split.y_val, preds, zero_division=0)),
                "val_roc_auc": float(roc_auc_score(split.y_val, scores)),
                "val_average_precision": float(average_precision_score(split.y_val, scores)),
            }
        )

    return pd.DataFrame(rows).sort_values("val_f1", ascending=False).reset_index(drop=True)


def best_model_summary(benchmark: pd.DataFrame) -> dict[str, float | str]:
    best = benchmark.loc[benchmark["val_f1"].idxmax()]
    return {
        "best_model": str(best["model"]),
        "val_f1": float(best["val_f1"]),
        "val_roc_auc": float(best["val_roc_auc"]),
        "val_average_precision": float(best["val_average_precision"]),
    }
