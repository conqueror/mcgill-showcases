from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_and_evaluate(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    random_state: int = 42,
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame]:
    """Train a baseline model and return metrics plus holdout predictions."""
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=random_state,
        stratify=target,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=400, random_state=random_state)),
        ]
    )
    model.fit(x_train, y_train)

    probs = model.predict_proba(x_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = pd.DataFrame(
        [
            {
                "metric": "roc_auc",
                "value": float(roc_auc_score(y_test, probs)),
            },
            {
                "metric": "accuracy",
                "value": float(accuracy_score(y_test, preds)),
            },
            {
                "metric": "n_train",
                "value": float(len(x_train)),
            },
            {
                "metric": "n_test",
                "value": float(len(x_test)),
            },
        ]
    )

    holdout = x_test.copy()
    holdout["y_true"] = y_test.to_numpy()
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
