from __future__ import annotations

import math

import pandas as pd

from mlops_drift_showcase.drift import compute_drift_report


def test_shifted_feature_is_flagged() -> None:
    reference = pd.DataFrame({"f_0": [0.0, 0.1, 0.2, 0.3], "f_1": [1.0, 1.0, 1.0, 1.0]})
    incoming = pd.DataFrame({"f_0": [1.5, 1.6, 1.7, 1.8], "f_1": [1.0, 1.0, 1.0, 1.0]})
    report = compute_drift_report(reference, incoming, ks_alpha=0.2, psi_threshold=0.1)
    drift_map = dict(zip(report["feature"], report["drift_flag"], strict=True))
    assert drift_map["f_0"] == 1


def test_constant_feature_does_not_break_psi_binning() -> None:
    reference = pd.DataFrame({"constant": [1.0, 1.0, 1.0, 1.0]})
    incoming = pd.DataFrame({"constant": [1.0, 1.0, 1.0, 1.0]})

    report = compute_drift_report(reference, incoming, ks_alpha=0.2, psi_threshold=0.1)

    assert report.shape[0] == 1
    psi_value = float(report.loc[0, "psi"])
    assert math.isfinite(psi_value)
