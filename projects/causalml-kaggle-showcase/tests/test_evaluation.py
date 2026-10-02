from __future__ import annotations

import numpy as np
import pytest

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


def test_qini_omits_unsupported_prefixes_and_includes_origin() -> None:
    curve = qini_curve(np.array([1, 0, 1, 0]), np.array([1, 1, 0, 0]),
                       np.array([0.9, 0.8, 0.7, 0.6]), n_bins=4)
    # k=1,2 have no controls. At k=3: 1 - 2*(1/1) = -1; k=4: 1 - 2*(1/2) = 0.
    assert curve["fraction"].tolist() == [0.0, 0.75, 1.0]
    assert curve["incremental_gain"].tolist() == [0.0, -1.0, 0.0]
    # Raw gain area: (0-1)/2 * .75 + (-1+0)/2 * .25 = -0.5.
    assert qini_auc(curve) == -0.5


@pytest.mark.parametrize("treatment", [np.array([], dtype=int), np.array([1, 1])])
def test_qini_rejects_empty_or_one_arm_input(treatment: np.ndarray) -> None:
    with pytest.raises(ValueError):
        qini_curve(np.zeros(len(treatment)), treatment, np.ones(len(treatment)))


def test_two_bin_qini_curve_reaches_the_full_population() -> None:
    curve = qini_curve(np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]),
                       np.array([0.9, 0.8, 0.7, 0.6]), n_bins=2)
    # The two treated rows convert; neither control does. End gain = 2 - 2*(0/2) = 2.
    assert curve["fraction"].tolist() == [0.0, 1.0]
    assert curve["incremental_gain"].tolist() == [0.0, 2.0]
    assert qini_auc(curve) == 1.0  # Triangle: base 1, height 2.
