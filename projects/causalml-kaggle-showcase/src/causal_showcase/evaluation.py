from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_empirical_ate(y: np.ndarray, treatment: np.ndarray) -> float:
    """Difference in means between treated and control groups."""
    treated_mask = treatment == 1
    control_mask = treatment == 0
    if treated_mask.sum() == 0 or control_mask.sum() == 0:
        raise ValueError("Both treatment and control groups must have observations.")
    return float(y[treated_mask].mean() - y[control_mask].mean())


def uplift_at_k(
    y: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    *,
    top_fraction: float = 0.3,
) -> float:
    """Observed uplift among top-k ranked users by predicted uplift score."""
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1].")

    n_select = max(1, int(len(uplift_scores) * top_fraction))
    top_idx = np.argsort(-uplift_scores)[:n_select]

    y_top = y[top_idx]
    w_top = treatment[top_idx]
    return estimate_empirical_ate(y_top, w_top)


def qini_curve(
    y: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    *,
    n_bins: int = 20,
) -> pd.DataFrame:
    """
    Build a Qini-style curve using incremental gains at ranked prefixes.

    At each ranked prefix, we compare observed treated outcomes against
    expected treated outcomes under control response rates.
    """
    if len(y) != len(treatment) or len(y) != len(uplift_scores):
        raise ValueError("y, treatment, and uplift_scores must have equal length.")

    order = np.argsort(-uplift_scores)
    y_sorted = y[order]
    w_sorted = treatment[order]

    cum_treated = np.cumsum(w_sorted)
    cum_control = np.cumsum(1 - w_sorted)
    cum_outcome_treated = np.cumsum(y_sorted * w_sorted)
    cum_outcome_control = np.cumsum(y_sorted * (1 - w_sorted))

    control_rate = np.divide(
        cum_outcome_control,
        np.maximum(cum_control, 1),
        out=np.zeros_like(cum_outcome_control, dtype=float),
        where=np.maximum(cum_control, 1) > 0,
    )
    expected_treated_outcome = control_rate * cum_treated
    incremental_gain = cum_outcome_treated - expected_treated_outcome

    population_fraction = np.arange(1, len(y_sorted) + 1) / len(y_sorted)

    bin_edges = np.linspace(0, len(y_sorted) - 1, n_bins, dtype=int)
    curve = pd.DataFrame(
        {
            "fraction": population_fraction[bin_edges],
            "incremental_gain": incremental_gain[bin_edges],
        }
    )
    return curve


def qini_auc(curve: pd.DataFrame) -> float:
    """Area under the Qini curve."""
    return float(np.trapezoid(curve["incremental_gain"], x=curve["fraction"]))
