from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


@dataclass(frozen=True)
class SplitBundle:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def make_split(
    frame: pd.DataFrame,
    target: pd.Series,
    *,
    random_state: int = 42,
) -> SplitBundle:
    x_train, x_test, y_train, y_test = train_test_split(
        frame,
        target,
        test_size=0.3,
        random_state=random_state,
        stratify=target,
    )
    return SplitBundle(
        x_train=x_train.reset_index(drop=True),
        x_test=x_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )


def build_preprocessor(
    *,
    numeric_features: list[str],
    categorical_features: list[str],
    encoding: str,
) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    if encoding == "onehot":
        cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    elif encoding == "ordinal":
        cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", cat_encoder),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ]
    )


def transform_split(
    split: SplitBundle,
    *,
    numeric_features: list[str],
    categorical_features: list[str],
    encoding: str,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], list[str], ColumnTransformer]:
    pre = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        encoding=encoding,
    )
    x_train = pre.fit_transform(split.x_train)
    x_test = pre.transform(split.x_test)

    feature_names = list(pre.get_feature_names_out())
    return (
        np.asarray(x_train, dtype=float),
        np.asarray(x_test, dtype=float),
        feature_names,
        pre,
    )
