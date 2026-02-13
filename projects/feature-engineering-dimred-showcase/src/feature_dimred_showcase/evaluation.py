from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def evaluate_classifier(
    x_train: npt.NDArray[np.float64],
    x_test: npt.NDArray[np.float64],
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, float]:
    model = LogisticRegression(max_iter=300, random_state=42)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds, average="macro")),
    }
