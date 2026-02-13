from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_logistic(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    sample_weight: npt.NDArray[np.float64] | None = None,
    class_weight: str | None = None,
    random_state: int = 42,
) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=400,
                    random_state=random_state,
                    class_weight=class_weight,
                ),
            ),
        ]
    )
    fit_kwargs: dict[str, object] = {}
    if sample_weight is not None:
        fit_kwargs["clf__sample_weight"] = sample_weight
    model.fit(x_train, y_train, **fit_kwargs)
    return model


def predict_probabilities(model: Pipeline, x_values: pd.DataFrame) -> npt.NDArray[np.float64]:
    probs = model.predict_proba(x_values)[:, 1]
    return np.asarray(probs, dtype=float)


def evaluate_binary(
    y_true: pd.Series,
    probas: npt.NDArray[np.float64],
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    preds = (probas >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probas)),
        "accuracy": float(accuracy_score(y_true, preds)),
    }
