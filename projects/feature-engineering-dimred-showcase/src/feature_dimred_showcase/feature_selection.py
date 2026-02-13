from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression


def compute_selection_scores(
    x_train: npt.NDArray[np.float64],
    y_train: pd.Series,
    feature_names: list[str],
) -> pd.DataFrame:
    mutual = mutual_info_classif(x_train, y_train, random_state=42)

    l1_model = LogisticRegression(
        penalty="elasticnet",
        l1_ratio=1.0,
        solver="saga",
        random_state=42,
        max_iter=2000,
    )
    l1_model.fit(x_train, y_train)
    coefs = np.abs(l1_model.coef_[0])

    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "mutual_info": mutual,
            "abs_l1_coef": coefs,
        }
    )
    frame["combined_score"] = frame["mutual_info"] + frame["abs_l1_coef"]
    return frame.sort_values(by="combined_score", ascending=False).reset_index(drop=True)
