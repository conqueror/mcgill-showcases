from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_iris

from feature_dimred_showcase.preprocessing import make_split, transform_split


def test_transformed_arrays_have_no_nans() -> None:
    raw = load_iris(as_frame=True)
    x_df = raw.data.copy()
    y = raw.target
    x_df["cat"] = pd.cut(
        x_df["petal length (cm)"],
        bins=[0.0, 2.5, 4.5, 8.0],
        labels=["short", "medium", "long"],
        include_lowest=True,
    ).astype(str)
    x_df.loc[x_df.index[::13], "sepal width (cm)"] = float("nan")

    split = make_split(x_df, y)
    x_train, x_test, _, _ = transform_split(
        split,
        numeric_features=[c for c in x_df.columns if c != "cat"],
        categorical_features=["cat"],
        encoding="onehot",
    )

    assert not pd.isna(x_train).any()
    assert not pd.isna(x_test).any()
