"""Dataset utilities shared across unsupervised/semi/self-supervised demos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ArrayF = NDArray[np.float64]
ArrayI = NDArray[np.int64]

LOAN_FEATURE_COLUMNS = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "addr_state",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "application_type",
]

LOAN_RESOLVED_STATUS_MAP = {
    "Fully Paid": 0,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Charged Off": 1,
    "Does not meet the credit policy. Status:Charged Off": 1,
}


@dataclass(frozen=True)
class LabeledDataset:
    """Container for feature matrix, labels, and metadata."""

    X: ArrayF
    y: ArrayI
    feature_names: list[str]
    target_names: list[str]
    dataset_name: str


@dataclass(frozen=True)
class SplitDataset:
    """Train/test split plus partially labeled training labels for semi-supervision."""

    X_train: ArrayF
    X_test: ArrayF
    y_train: ArrayI
    y_test: ArrayI
    y_train_masked: ArrayI


def load_digits_dataset(scale: bool = True) -> LabeledDataset:
    """Load the sklearn digits dataset as standardized floating point features."""

    bunch = load_digits()
    X = bunch.data.astype(np.float64)
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    y = bunch.target.astype(np.int64)
    feature_names = [f"pixel_{i}" for i in range(X.shape[1])]
    target_names = [str(label) for label in bunch.target_names]
    return LabeledDataset(
        X=X,
        y=y,
        feature_names=feature_names,
        target_names=target_names,
        dataset_name="sklearn_digits",
    )


def load_digits_dataframe(scale: bool = True) -> pd.DataFrame:
    """Return a DataFrame representation used for learner-friendly inspection."""

    dataset = load_digits_dataset(scale=scale)
    frame = pd.DataFrame(dataset.X, columns=dataset.feature_names)
    frame["target"] = dataset.y
    return frame


def parse_term_months(series: pd.Series) -> pd.Series:
    """Convert loan term strings like '36 months' to numeric months."""

    extracted = series.astype(str).str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def parse_percentage(series: pd.Series) -> pd.Series:
    """Convert strings like '13.56%' to numeric percentages."""

    cleaned = series.astype(str).str.replace("%", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def parse_emp_length_years(series: pd.Series) -> pd.Series:
    """Convert employment length strings to approximate numeric years."""

    normalized = series.fillna("Unknown").astype(str).str.strip()
    years = normalized.str.extract(r"(\d+)", expand=False)
    out = pd.to_numeric(years, errors="coerce")
    out = out.where(~normalized.str.contains("<", regex=False), other=0.5)
    return out


def _default_loan_csv_path() -> Path:
    return Path(__file__).resolve().parents[4] / "loan.csv"


def load_business_loan_dataset(
    csv_path: Path | None = None,
    read_rows: int = 60_000,
    sample_size: int = 3_000,
    scale: bool = True,
    random_state: int = 42,
) -> LabeledDataset:
    """Load and preprocess a learner-friendly Lending Club credit-risk dataset."""

    path = csv_path if csv_path is not None else _default_loan_csv_path()
    if not path.exists():
        msg = (
            "loan.csv not found. Checklist: provide --business-csv-path, "
            "or place loan.csv at repository root."
        )
        raise FileNotFoundError(msg)

    required_columns = sorted(set(LOAN_FEATURE_COLUMNS + ["loan_status"]))
    raw = pd.read_csv(path, usecols=required_columns, nrows=read_rows)
    raw = raw[raw["loan_status"].isin(LOAN_RESOLVED_STATUS_MAP)].copy()

    if raw.empty:
        msg = "No resolved loan statuses found after filtering for Fully Paid/Charged Off records."
        raise ValueError(msg)

    raw["target"] = raw["loan_status"].map(LOAN_RESOLVED_STATUS_MAP).astype(np.int64)
    y = raw["target"].to_numpy(dtype=np.int64)

    if sample_size > 0 and sample_size < len(raw):
        all_idx = np.arange(len(raw))
        sampled_idx, _ = train_test_split(
            all_idx,
            train_size=sample_size,
            stratify=y,
            random_state=random_state,
        )
        raw = raw.iloc[sampled_idx].reset_index(drop=True)
        y = raw["target"].to_numpy(dtype=np.int64)

    features = raw[LOAN_FEATURE_COLUMNS].copy()
    features["term"] = parse_term_months(features["term"])
    features["int_rate"] = parse_percentage(features["int_rate"])
    features["revol_util"] = parse_percentage(features["revol_util"])
    features["emp_length"] = parse_emp_length_years(features["emp_length"])

    numeric_cols = [
        "loan_amnt",
        "term",
        "int_rate",
        "installment",
        "emp_length",
        "annual_inc",
        "dti",
        "delinq_2yrs",
        "inq_last_6mths",
        "open_acc",
        "pub_rec",
        "revol_bal",
        "revol_util",
        "total_acc",
    ]
    categorical_cols = [col for col in features.columns if col not in numeric_cols]

    for col in numeric_cols:
        features[col] = pd.to_numeric(features[col], errors="coerce")

    numeric_fill_values = features[numeric_cols].median(numeric_only=True)
    features[numeric_cols] = features[numeric_cols].fillna(numeric_fill_values)

    features[categorical_cols] = features[categorical_cols].fillna("Unknown").astype(str)
    encoded = pd.get_dummies(features, columns=categorical_cols, drop_first=False)

    X = encoded.to_numpy(dtype=np.float64)
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return LabeledDataset(
        X=X,
        y=y,
        feature_names=encoded.columns.tolist(),
        target_names=["non_risky", "risky"],
        dataset_name="lendingclub_credit_risk",
    )


def make_semisupervised_labels(
    y: ArrayI,
    labeled_fraction: float,
    random_state: int,
) -> ArrayI:
    """Mask a stratified subset with -1 to emulate unlabeled data."""

    if not 0.0 < labeled_fraction < 1.0:
        msg = "labeled_fraction must be in (0, 1)."
        raise ValueError(msg)

    rng = np.random.default_rng(random_state)
    y_masked = np.full_like(y, fill_value=-1)

    for label in np.unique(y):
        label_indices = np.flatnonzero(y == label)
        rng.shuffle(label_indices)
        n_labeled = max(1, int(round(label_indices.size * labeled_fraction)))
        labeled_indices = label_indices[:n_labeled]
        y_masked[labeled_indices] = y[labeled_indices]

    return y_masked


def make_train_test_split(
    X: ArrayF,
    y: ArrayI,
    test_size: float,
    labeled_fraction: float,
    random_state: int,
) -> SplitDataset:
    """Create stratified train/test split and masked labels for semi-supervised tasks."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    y_train_masked = make_semisupervised_labels(
        y=y_train,
        labeled_fraction=labeled_fraction,
        random_state=random_state,
    )

    return SplitDataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_train_masked=y_train_masked,
    )


def split_two_views(X: ArrayF) -> tuple[ArrayF, ArrayF]:
    """Split features into two complementary views for co-training experiments."""

    midpoint = X.shape[1] // 2
    left = X[:, :midpoint]
    right = X[:, midpoint:]
    return left, right
