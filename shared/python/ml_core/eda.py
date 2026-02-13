"""EDA helpers shared across supervised showcase pipelines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def infer_feature_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Infer logical feature categories from pandas dtypes."""

    rows: list[dict[str, str]] = []
    for col in frame.columns:
        series = frame[col]
        dtype_name = str(series.dtype)
        if pd.api.types.is_numeric_dtype(series):
            logical = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            logical = "datetime"
        else:
            logical = "categorical"
        rows.append({"feature": str(col), "dtype": dtype_name, "logical_type": logical})
    return pd.DataFrame(rows)


def univariate_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute per-feature summary statistics with missingness details."""

    rows: list[dict[str, float | int | str]] = []
    for col in frame.columns:
        series = frame[col]
        missing_ratio = float(series.isna().mean())
        row: dict[str, float | int | str] = {
            "feature": str(col),
            "missing_ratio": missing_ratio,
            "n_unique": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            row.update(
                {
                    "mean": float(series.mean(skipna=True)),
                    "std": float(series.std(skipna=True) or 0.0),
                    "min": float(series.min(skipna=True)),
                    "p50": float(series.median(skipna=True)),
                    "max": float(series.max(skipna=True)),
                    "mode": "",
                }
            )
        else:
            mode_value = series.mode(dropna=True)
            row.update(
                {
                    "mean": float("nan"),
                    "std": float("nan"),
                    "min": float("nan"),
                    "p50": float("nan"),
                    "max": float("nan"),
                    "mode": str(mode_value.iloc[0]) if len(mode_value) > 0 else "",
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def bivariate_vs_target(frame: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """Compute feature-to-target signal summaries for mixed data types.

    Numeric features use Pearson correlation when the target is numeric.
    Non-numeric features are summarized via top category target mean.
    """

    y = target
    rows: list[dict[str, float | str | int]] = []
    y_is_numeric = pd.api.types.is_numeric_dtype(y)

    for col in frame.columns:
        series = frame[col]
        if pd.api.types.is_numeric_dtype(series) and y_is_numeric:
            aligned = pd.concat([series, y], axis=1).dropna()
            if aligned.shape[0] < 3:
                corr = 0.0
            else:
                corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
            rows.append(
                {
                    "feature": str(col),
                    "analysis_type": "numeric_corr",
                    "stat": "pearson_corr",
                    "value": corr,
                }
            )
        else:
            joined = pd.DataFrame({"x": series.astype(str), "y": y})
            grouped = joined.groupby("x", dropna=False)["y"].mean().sort_values(ascending=False)
            top_cat = str(grouped.index[0]) if not grouped.empty else ""
            top_mean = float(grouped.iloc[0]) if not grouped.empty else float("nan")
            rows.append(
                {
                    "feature": str(col),
                    "analysis_type": "category_target_mean",
                    "stat": f"top_category:{top_cat}",
                    "value": top_mean,
                }
            )

    return pd.DataFrame(rows)


def missingness_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Report missing value counts and ratios for each feature."""

    rows: list[dict[str, float | int | str]] = []
    n_rows = max(1, len(frame))
    for col in frame.columns:
        missing_count = int(frame[col].isna().sum())
        rows.append(
            {
                "feature": str(col),
                "missing_count": missing_count,
                "missing_ratio": float(missing_count / n_rows),
            }
        )
    return pd.DataFrame(rows).sort_values("missing_ratio", ascending=False).reset_index(drop=True)


def correlation_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Return numeric correlation matrix or an empty frame when unavailable."""

    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty:
        return pd.DataFrame()
    return numeric.corr(numeric_only=True)


def maybe_write_profile_report(frame: pd.DataFrame, output_path: Path) -> str:
    """Write a ydata-profiling HTML report when optional deps are installed.

    Returns:
        ``written`` when the report is created,
        ``skipped_missing_dependency`` when profiling packages are unavailable.
    """

    try:
        from ydata_profiling import ProfileReport
    except Exception:
        return "skipped_missing_dependency"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = ProfileReport(frame, title="Data Profiling Report", minimal=True)
    profile.to_file(output_path=output_path)
    return "written"


def maybe_write_missingness_plot(frame: pd.DataFrame, output_path: Path) -> str:
    """Write a missingness matrix image when optional plotting deps are installed.

    Returns:
        ``written`` when the image is created,
        ``skipped_missing_dependency`` when plotting packages are unavailable.
    """

    try:
        import matplotlib.pyplot as plt
        import missingno as msno
    except Exception:
        return "skipped_missing_dependency"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = msno.matrix(frame)
    fig.figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig.figure)
    return "written"
