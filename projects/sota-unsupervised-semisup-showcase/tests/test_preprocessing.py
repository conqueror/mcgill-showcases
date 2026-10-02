from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from sota_showcase import data
from sota_showcase.config import ShowcaseConfig
from sota_showcase.data import LOAN_FEATURE_COLUMNS, make_train_test_split
from sota_showcase.pipeline import _load_dataset


def test_digits_held_out_rows_cannot_change_training_preprocessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    y = np.array([0, 1] * 10, dtype=np.int64)
    raw = np.arange(60, dtype=np.float64).reshape(20, 3)
    _, held_out = train_test_split(np.arange(20), test_size=0.3, stratify=y, random_state=7)
    config = ShowcaseConfig(test_size=0.3, random_state=7)

    def load() -> data.SplitDataset:
        monkeypatch.setattr(
            data,
            "load_digits",
            lambda: SimpleNamespace(data=raw.copy(), target=y, target_names=np.array([0, 1])),
        )
        dataset, _ = _load_dataset(config)
        return make_train_test_split(dataset.X, y, 0.3, 0.1, 7)

    original = load()
    raw[held_out] += 100_000
    changed = load()
    np.testing.assert_allclose(original.X_train, changed.X_train, atol=1e-12)
    assert not np.allclose(original.X_test, changed.X_test)
    np.testing.assert_allclose(original.X_train.mean(axis=0), 0.0, atol=1e-12)


def test_business_held_out_rows_cannot_change_medians_categories_or_scaling(tmp_path: Path) -> None:
    frame = pd.DataFrame([{column: 1 for column in LOAN_FEATURE_COLUMNS}] * 20)
    frame["loan_status"] = ["Fully Paid", "Charged Off"] * 10
    frame["loan_amnt"] = np.arange(20, dtype=float) + 100
    frame["term"] = "36 months"
    frame["int_rate"] = "10%"
    frame["revol_util"] = "20%"
    frame["emp_length"] = "2 years"
    frame["grade"] = "A"
    y = np.array([0, 1] * 10, dtype=np.int64)
    train, held_out = train_test_split(np.arange(20), test_size=0.3, stratify=y, random_state=7)
    frame.loc[train[:3], "loan_amnt"] = np.nan
    csv_path = tmp_path / "loans.csv"
    config = ShowcaseConfig(
        dataset="business",
        business_csv_path=csv_path,
        business_sample_size=500,
        test_size=0.3,
        random_state=7,
    )

    def load() -> data.SplitDataset:
        frame.to_csv(csv_path, index=False)
        dataset, _ = _load_dataset(config)
        return make_train_test_split(dataset.X, y, 0.3, 0.1, 7)

    original = load()
    frame.loc[held_out, "loan_amnt"] = 100_000
    frame.loc[held_out, "grade"] = "ONLY_IN_TEST"
    changed = load()
    np.testing.assert_allclose(original.X_train, changed.X_train, atol=1e-12)
    assert not np.allclose(original.X_test, changed.X_test)
