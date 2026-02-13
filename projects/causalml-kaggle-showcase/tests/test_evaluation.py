from __future__ import annotations

import numpy as np

from causal_showcase.evaluation import estimate_empirical_ate, qini_auc, qini_curve, uplift_at_k


def test_estimate_empirical_ate_matches_manual_difference() -> None:
    y = np.array([1, 1, 0, 0, 1, 0])
    treatment = np.array([1, 1, 0, 0, 1, 0])

    ate = estimate_empirical_ate(y, treatment)

    assert ate == 1.0


def test_uplift_at_k_returns_float() -> None:
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    treatment = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    uplift_scores = np.array([0.8, 0.2, 0.6, 0.1, 0.7, 0.4, 0.5, 0.3])

    score = uplift_at_k(y, treatment, uplift_scores, top_fraction=0.5)

    assert isinstance(score, float)


def test_qini_curve_and_auc_shape() -> None:
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 0, 1])
    treatment = np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 1])
    uplift_scores = np.linspace(1.0, 0.0, num=10)

    curve = qini_curve(y, treatment, uplift_scores, n_bins=5)
    auc = qini_auc(curve)

    assert list(curve.columns) == ["fraction", "incremental_gain"]
    assert curve.shape[0] == 5
    assert isinstance(auc, float)
