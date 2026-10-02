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
    if n_bins < 2 or not len(y) or not np.isin(treatment, [0, 1]).all():
        raise ValueError("Qini curves need nonempty binary treatment data and at least two bins.")
    if not (treatment == 1).any() or not (treatment == 0).any():
        raise ValueError("Qini curves require both treatment arms.")
    if not np.isfinite(y).all() or not np.isfinite(uplift_scores).all():
        raise ValueError("Outcomes and uplift scores must be finite.")

    order = np.argsort(-uplift_scores)
    y_sorted = y[order]
    w_sorted = treatment[order]

    cum_treated = np.cumsum(w_sorted)
    cum_control = np.cumsum(1 - w_sorted)
    cum_outcome_treated = np.cumsum(y_sorted * w_sorted)
    cum_outcome_control = np.cumsum(y_sorted * (1 - w_sorted))

    control_rate = np.divide(
        cum_outcome_control,
        cum_control,
        out=np.full_like(cum_outcome_control, np.nan, dtype=float),
        where=cum_control > 0,
    )
    expected_treated_outcome = control_rate * cum_treated
    incremental_gain = cum_outcome_treated - expected_treated_outcome

    population_fraction = np.arange(1, len(y_sorted) + 1) / len(y_sorted)

    first_supported = int(np.flatnonzero((cum_treated > 0) & (cum_control > 0))[0])
    bin_edges = np.unique(np.linspace(first_supported, len(y_sorted) - 1, n_bins - 1, dtype=int))
    bin_edges[-1] = len(y_sorted) - 1
    curve = pd.DataFrame(
        {
            "fraction": np.r_[0.0, population_fraction[bin_edges]],
            "incremental_gain": np.r_[0.0, incremental_gain[bin_edges]],
        }
    )
    return curve


def qini_auc(curve: pd.DataFrame) -> float:
    """Raw incremental-gain area; this does not subtract the random-targeting line."""
    if len(curve) < 2 or not np.isfinite(curve[["fraction", "incremental_gain"]]).all().all():
        raise ValueError("Qini area requires at least two finite curve points.")
    return float(np.trapezoid(curve["incremental_gain"], x=curve["fraction"]))
