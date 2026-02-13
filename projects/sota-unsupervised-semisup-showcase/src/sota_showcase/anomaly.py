"""Anomaly and novelty detection demos for tutorial workflows."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

ArrayF = NDArray[np.float64]
ArrayI = NDArray[np.int64]


def build_anomaly_target(y: ArrayI, anomaly_label: int | None = None) -> ArrayI:
    """Define anomalies as one class (defaults to the minority class)."""

    if anomaly_label is None:
        labels, counts = np.unique(y, return_counts=True)
        anomaly_label = int(labels[np.argmin(counts)])
    return (y == anomaly_label).astype(np.int64)


def _evaluate_detector(
    name: str,
    y_true: ArrayI,
    y_pred_anomaly: ArrayI,
    anomaly_scores: ArrayF,
) -> dict[str, float | str]:
    return {
        "algorithm": name,
        "precision": float(precision_score(y_true, y_pred_anomaly, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_anomaly, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred_anomaly, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, anomaly_scores)),
    }


def _threshold_scores(scores: ArrayF, contamination: float) -> ArrayI:
    threshold = np.quantile(scores, 1.0 - contamination)
    return (scores >= threshold).astype(np.int64)


def _pca_reconstruction_detector(X: ArrayF, contamination: float) -> tuple[ArrayI, ArrayF]:
    model = PCA(n_components=0.95, random_state=42)
    latent = model.fit_transform(X)
    reconstructed = model.inverse_transform(latent)
    reconstruction_error = np.mean((X - reconstructed) ** 2, axis=1)
    preds = _threshold_scores(reconstruction_error, contamination=contamination)
    return preds.astype(np.int64), reconstruction_error.astype(np.float64)


def _gmm_density_detector(
    X: ArrayF,
    contamination: float,
    random_state: int,
) -> tuple[ArrayI, ArrayF]:
    model = GaussianMixture(n_components=10, covariance_type="full", random_state=random_state)
    model.fit(X)
    neg_log_likelihood = -model.score_samples(X)
    preds = _threshold_scores(neg_log_likelihood, contamination=contamination)
    return preds.astype(np.int64), neg_log_likelihood.astype(np.float64)


def _isolation_forest_detector(
    X: ArrayF,
    contamination: float,
    random_state: int,
) -> tuple[ArrayI, ArrayF]:
    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(X)
    preds = (model.predict(X) == -1).astype(np.int64)
    scores = -model.decision_function(X)
    return preds, scores.astype(np.float64)


def _lof_detector(X: ArrayF, contamination: float) -> tuple[ArrayI, ArrayF]:
    model = LocalOutlierFactor(contamination=contamination)
    preds = (model.fit_predict(X) == -1).astype(np.int64)
    scores = -model.negative_outlier_factor_
    return preds, scores.astype(np.float64)


def _one_class_svm_detector(X: ArrayF, contamination: float) -> tuple[ArrayI, ArrayF]:
    model = OneClassSVM(nu=contamination, gamma="scale")
    model.fit(X)
    preds = (model.predict(X) == -1).astype(np.int64)
    scores = -model.decision_function(X)
    return preds, scores.astype(np.float64)


def _fast_mcd_detector(X: ArrayF, contamination: float, random_state: int) -> tuple[ArrayI, ArrayF]:
    # Robust covariance can be unstable in high dimensions if the matrix is near-singular.
    # Reduce dimensionality first to keep the tutorial run numerically stable.
    reduced = PCA(n_components=min(20, X.shape[1]), random_state=random_state).fit_transform(X)
    model = EllipticEnvelope(contamination=contamination, random_state=random_state)
    model.fit(reduced)
    preds = (model.predict(reduced) == -1).astype(np.int64)
    scores = -model.decision_function(reduced)
    return preds, scores.astype(np.float64)


def run_anomaly_benchmark(
    X: ArrayF,
    y: ArrayI,
    random_state: int,
    anomaly_label: int | None = None,
) -> pd.DataFrame:
    """Compare classical anomaly detectors covered in the lecture notes."""

    y_true = build_anomaly_target(y=y, anomaly_label=anomaly_label)
    contamination = float(np.mean(y_true))
    contamination = min(0.4, max(0.01, contamination))
    rows: list[dict[str, float | str]] = []

    detectors = [
        (
            "IsolationForest",
            _isolation_forest_detector(X=X, contamination=contamination, random_state=random_state),
        ),
        ("LocalOutlierFactor", _lof_detector(X=X, contamination=contamination)),
        ("OneClassSVM", _one_class_svm_detector(X=X, contamination=contamination)),
        (
            "FastMCD_EllipticEnvelope",
            _fast_mcd_detector(X=X, contamination=contamination, random_state=random_state),
        ),
    ]

    for name, (y_pred, scores) in detectors:
        rows.append(
            _evaluate_detector(
                name=name,
                y_true=y_true,
                y_pred_anomaly=y_pred,
                anomaly_scores=scores,
            ),
        )

    pca_pred, pca_scores = _pca_reconstruction_detector(X=X, contamination=contamination)
    rows.append(
        _evaluate_detector(
            name="PCA_Reconstruction",
            y_true=y_true,
            y_pred_anomaly=pca_pred,
            anomaly_scores=pca_scores,
        ),
    )

    gmm_pred, gmm_scores = _gmm_density_detector(
        X=X,
        contamination=contamination,
        random_state=random_state,
    )
    rows.append(
        _evaluate_detector(
            name="GaussianMixture_Density",
            y_true=y_true,
            y_pred_anomaly=gmm_pred,
            anomaly_scores=gmm_scores,
        ),
    )

    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
