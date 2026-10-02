from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import ks_2samp


def _psi(
    expected: npt.NDArray[np.float64],
    actual: npt.NDArray[np.float64],
    *,
    bins: int = 10,
) -> float:
    """Population Stability Index (PSI) for one feature."""
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    raw_breaks = np.quantile(expected, quantiles)
    interior_breaks = np.unique(raw_breaks[1:-1])
    breaks = np.concatenate((np.array([-np.inf]), interior_breaks, np.array([np.inf])))

    expected_counts, _ = np.histogram(expected, bins=breaks)
    actual_counts, _ = np.histogram(actual, bins=breaks)

    expected_dist = np.clip(expected_counts / max(1, expected_counts.sum()), 1e-6, None)
    actual_dist = np.clip(actual_counts / max(1, actual_counts.sum()), 1e-6, None)
    return float(np.sum((actual_dist - expected_dist) * np.log(actual_dist / expected_dist)))


def compute_drift_report(
    reference: pd.DataFrame,
    incoming: pd.DataFrame,
    *,
    ks_alpha: float = 0.05,
    psi_threshold: float = 0.2,
) -> pd.DataFrame:
    """Compare matching numeric features with at least two finite samples per dataset."""
    if reference.shape[1] == 0:
        raise ValueError("Drift evidence must contain features")
    if (
        not reference.columns.is_unique
        or not incoming.columns.is_unique
        or set(reference.columns) != set(incoming.columns)
    ):
        raise ValueError("Reference and incoming feature schemas must match without duplicates")
    if len(reference) < 2 or len(incoming) < 2:
        raise ValueError("Drift evidence requires at least two samples per dataset")
    for frame in (reference, incoming):
        if len(frame.select_dtypes(include="number").columns) != len(frame.columns):
            raise ValueError("Drift evidence must contain finite numeric values")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError("Drift evidence must contain finite numeric values")
    rows: list[dict[str, float | int | str]] = []

    for column in reference.columns:
        ref_col = reference[column].to_numpy()
        inc_col = incoming[column].to_numpy()
        ks_stat, ks_pvalue = ks_2samp(ref_col, inc_col)
        psi_value = _psi(ref_col, inc_col)
        drift_flag = int((ks_pvalue < ks_alpha) or (psi_value >= psi_threshold))

        rows.append(
            {
                "feature": column,
                "ks_stat": float(ks_stat),
                "ks_pvalue": float(ks_pvalue),
                "psi": float(psi_value),
                "drift_flag": drift_flag,
            }
        )

    return pd.DataFrame(rows).sort_values(by="drift_flag", ascending=False).reset_index(drop=True)
