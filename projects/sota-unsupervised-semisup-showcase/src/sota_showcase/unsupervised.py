"""Unsupervised learning benchmark for tutorial workflows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.cluster import (
    DBSCAN,
    AffinityPropagation,
    AgglomerativeClustering,
    Birch,
    KMeans,
    MeanShift,
    MiniBatchKMeans,
    SpectralClustering,
)
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

ArrayF = NDArray[np.float64]
ArrayI = NDArray[np.int64]


@dataclass(frozen=True)
class ClusteringOutput:
    """Named output container for tabular reporting."""

    metrics: pd.DataFrame
    kmeans_selection: pd.DataFrame


def _safe_silhouette_score(X: ArrayF, labels: ArrayI) -> float:
    n_unique = len(set(labels.tolist()))
    if n_unique < 2:
        return float("nan")
    try:
        return float(silhouette_score(X, labels))
    except ValueError:
        return float("nan")


def _evaluate_labels(
    name: str,
    X: ArrayF,
    y_true: ArrayI,
    labels: ArrayI,
) -> dict[str, float | str | int]:
    n_clusters = len(set(labels.tolist())) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    return {
        "algorithm": name,
        "n_clusters": n_clusters,
        "n_noise_points": n_noise,
        "silhouette": _safe_silhouette_score(X, labels),
        "ari": float(adjusted_rand_score(y_true, labels)),
        "nmi": float(normalized_mutual_info_score(y_true, labels)),
    }


def _get_clustering_algorithms(
    random_state: int,
    n_clusters: int,
    n_samples: int,
) -> list[tuple[str, Callable[[ArrayF], ArrayI]]]:
    """Build algorithm roster and skip expensive O(n^2+) methods on large sample sizes."""

    gmm_components = max(n_clusters, 2)
    bayesian_components = max(n_clusters * 2, n_clusters + 2)

    algorithms: list[tuple[str, Callable[[ArrayF], ArrayI]]] = [
        (
            "KMeans",
            lambda X: KMeans(
                n_clusters=n_clusters,
                n_init="auto",
                random_state=random_state,
            ).fit_predict(X),
        ),
        (
            "MiniBatchKMeans",
            lambda X: MiniBatchKMeans(
                n_clusters=n_clusters,
                n_init="auto",
                random_state=random_state,
                batch_size=256,
            ).fit_predict(X),
        ),
        ("DBSCAN", lambda X: DBSCAN(eps=2.4, min_samples=8).fit_predict(X)),
        (
            "AgglomerativeClustering",
            lambda X: AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage="ward",
            ).fit_predict(X),
        ),
        (
            "BIRCH",
            lambda X: Birch(n_clusters=n_clusters, threshold=1.2).fit_predict(X),
        ),
        (
            "GaussianMixture",
            lambda X: GaussianMixture(
                n_components=gmm_components,
                covariance_type="full",
                random_state=random_state,
            ).fit(X).predict(X),
        ),
        (
            "BayesianGaussianMixture",
            lambda X: BayesianGaussianMixture(
                n_components=bayesian_components,
                covariance_type="full",
                random_state=random_state,
                weight_concentration_prior_type="dirichlet_process",
            ).fit(X).predict(X),
        ),
    ]

    if n_samples <= 3_000:
        algorithms.extend(
            [
                (
                    "MeanShift",
                    lambda X: MeanShift(bin_seeding=True, cluster_all=False).fit_predict(X),
                ),
                (
                    "AffinityPropagation",
                    lambda X: AffinityPropagation(random_state=random_state).fit_predict(X),
                ),
                (
                    "SpectralClustering",
                    lambda X: SpectralClustering(
                        n_clusters=n_clusters,
                        assign_labels="kmeans",
                        random_state=random_state,
                        affinity="nearest_neighbors",
                    ).fit_predict(X),
                ),
            ],
        )

    try:
        import hdbscan  # type: ignore[import-not-found]

        algorithms.append(
            (
                "HDBSCAN",
                lambda X: hdbscan.HDBSCAN(min_cluster_size=max(20, n_clusters * 2)).fit_predict(X),
            ),
        )
    except Exception:
        # Optional dependency: this keeps the showcase runnable with base deps.
        pass

    return algorithms


def run_kmeans_model_selection(
    X: ArrayF,
    random_state: int,
    n_clusters: int,
    k_min: int = 2,
    k_max: int | None = None,
) -> pd.DataFrame:
    """Evaluate inertia/silhouette over a K range to teach model selection heuristics."""

    default_k_max = min(20, max(6, n_clusters * 3))
    max_k = default_k_max if k_max is None else k_max
    max_k = min(max_k, max(2, X.shape[0] - 1))

    rows: list[dict[str, float | int]] = []
    for k in range(k_min, max_k + 1):
        model = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        labels = model.fit_predict(X)
        inertia = float(cast(float, model.inertia_))
        rows.append(
            {
                "k": k,
                "inertia": inertia,
                "silhouette": _safe_silhouette_score(X, labels),
            },
        )

    return pd.DataFrame(rows)


def run_clustering_benchmark(
    X: ArrayF,
    y: ArrayI,
    random_state: int,
    n_clusters: int | None = None,
) -> ClusteringOutput:
    """Benchmark core clustering algorithms used in this project."""

    resolved_clusters = max(2, len(np.unique(y))) if n_clusters is None else max(2, n_clusters)

    rows: list[dict[str, float | str | int]] = []
    for name, runner in _get_clustering_algorithms(
        random_state=random_state,
        n_clusters=resolved_clusters,
        n_samples=X.shape[0],
    ):
        labels = runner(X)
        rows.append(_evaluate_labels(name=name, X=X, y_true=y, labels=labels.astype(np.int64)))

    metrics = pd.DataFrame(rows).sort_values("ari", ascending=False).reset_index(drop=True)
    kmeans_selection = run_kmeans_model_selection(
        X=X,
        random_state=random_state,
        n_clusters=resolved_clusters,
    )
    return ClusteringOutput(metrics=metrics, kmeans_selection=kmeans_selection)
