from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


@dataclass(frozen=True)
class DatasetBundle:
    features: pd.DataFrame
    target: pd.Series


def generate_reference_data(
    *,
    n_samples: int = 1200,
    n_features: int = 8,
    random_state: int = 42,
) -> DatasetBundle:
    """Create a synthetic binary-classification reference dataset."""
    x_values, y_values = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=6,
        n_redundant=1,
        weights=[0.62, 0.38],
        random_state=random_state,
    )
    feature_names = [f"f_{idx}" for idx in range(n_features)]
    features = pd.DataFrame(x_values, columns=feature_names)
    target = pd.Series(y_values, name="target")
    return DatasetBundle(features=features, target=target)


def generate_incoming_data(
    reference: pd.DataFrame,
    *,
    shift_strength: float = 0.65,
    random_state: int = 99,
) -> pd.DataFrame:
    """Generate incoming data with controlled covariate drift."""
    rng = np.random.default_rng(random_state)
    incoming = reference.copy()
    col_count = incoming.shape[1]
    shift_cols = incoming.columns[: max(1, col_count // 2)]

    for column in shift_cols:
        noise = rng.normal(loc=shift_strength, scale=0.25, size=len(incoming))
        incoming[column] = incoming[column] + noise

    return incoming
