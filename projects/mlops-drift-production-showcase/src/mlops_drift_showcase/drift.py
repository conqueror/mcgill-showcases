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
    breaks = np.quantile(expected, quantiles)
    breaks[0] = -np.inf
    breaks[-1] = np.inf

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
    """Return per-feature KS and PSI drift signals."""
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
