from __future__ import annotations

from pathlib import Path

import pandas as pd

from causal_showcase.data import load_marketing_ab_data, train_test_split_prepared


def test_load_marketing_ab_data_normalizes_columns(tmp_path: Path) -> None:
    sample_df = pd.DataFrame(
        {
            "test group": ["ad", "psa", "ad", "psa"],
            "converted": [1, 0, 1, 0],
            "total ads": [4, 2, 10, 1],
            "most ads day": ["Monday", "Tuesday", "Monday", "Friday"],
            "most ads hour": [18, 9, 20, 7],
        }
    )
    csv_path = tmp_path / "marketing_ab.csv"
    sample_df.to_csv(csv_path, index=False)

    prepared = load_marketing_ab_data(csv_path)

    assert prepared.X.shape[0] == 4
    assert prepared.treatment.tolist() == [1, 0, 1, 0]
    assert prepared.outcome.tolist() == [1, 0, 1, 0]
    assert "total_ads" in prepared.feature_names


def test_train_test_split_prepared_preserves_total_rows(tmp_path: Path) -> None:
    sample_df = pd.DataFrame(
        {
            "test group": ["ad", "psa", "ad", "psa", "ad", "psa", "ad", "psa"],
            "converted": [1, 0, 1, 0, 0, 1, 0, 1],
            "total ads": [4, 2, 10, 1, 3, 2, 5, 2],
            "most ads day": [
                "Monday",
                "Tuesday",
                "Monday",
                "Friday",
                "Friday",
                "Monday",
                "Tuesday",
                "Friday",
            ],
            "most ads hour": [18, 9, 20, 7, 12, 21, 8, 13],
        }
    )
    csv_path = tmp_path / "marketing_ab.csv"
    sample_df.to_csv(csv_path, index=False)

    prepared = load_marketing_ab_data(csv_path)
    train_data, test_data = train_test_split_prepared(prepared, test_size=0.25, random_state=7)

    assert train_data.X.shape[0] + test_data.X.shape[0] == prepared.X.shape[0]
    assert len(train_data.feature_names) == len(prepared.feature_names)
