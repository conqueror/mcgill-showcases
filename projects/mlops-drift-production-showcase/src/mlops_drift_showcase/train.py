from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_and_evaluate(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_features: pd.DataFrame,
    test_target: pd.Series,
    random_state: int = 42,
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame]:
    """Fit the supplied training partition and evaluate the supplied held-out partition."""

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=400, random_state=random_state)),
        ]
    )
    model.fit(features, target)

    probs = model.predict_proba(test_features)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = pd.DataFrame(
        [
            {
                "metric": "roc_auc",
                "value": float(roc_auc_score(test_target, probs)),
            },
            {
                "metric": "accuracy",
                "value": float(accuracy_score(test_target, preds)),
            },
            {
                "metric": "n_train",
                "value": float(len(features)),
            },
            {
                "metric": "n_test",
                "value": float(len(test_features)),
            },
        ]
    )

    holdout = test_features.copy()
    holdout["y_true"] = test_target.to_numpy()
    holdout["y_pred_proba"] = probs
    holdout["y_pred"] = preds
    return model, metrics, holdout


def save_model(model: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def load_model(model_path: Path) -> Any:
    return joblib.load(model_path)


def as_float_array(values: list[float]) -> npt.NDArray[np.float64]:
    return np.asarray(values, dtype=float)
