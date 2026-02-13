from __future__ import annotations

from sklearn.datasets import load_iris

from feature_dimred_showcase.feature_selection import compute_selection_scores
from feature_dimred_showcase.preprocessing import make_split, transform_split


def test_selection_scores_columns_exist() -> None:
    raw = load_iris(as_frame=True)
    x_df = raw.data.copy()
    y = raw.target
    x_df["cat"] = "a"

    split = make_split(x_df, y)
    x_train, _, feature_names, _ = transform_split(
        split,
        numeric_features=[c for c in x_df.columns if c != "cat"],
        categorical_features=["cat"],
        encoding="onehot",
    )
    scores = compute_selection_scores(x_train, split.y_train, feature_names)

    assert {"feature", "mutual_info", "abs_l1_coef", "combined_score"}.issubset(scores.columns)
