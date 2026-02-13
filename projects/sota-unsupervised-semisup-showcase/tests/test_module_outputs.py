from __future__ import annotations

from sota_showcase.anomaly import run_anomaly_benchmark
from sota_showcase.data import load_digits_dataset
from sota_showcase.unsupervised import run_clustering_benchmark


def test_clustering_benchmark_contains_key_algorithms() -> None:
    dataset = load_digits_dataset(scale=True)
    subset_X = dataset.X[:500]
    subset_y = dataset.y[:500]

    output = run_clustering_benchmark(subset_X, subset_y, random_state=42)
    algorithms = set(output.metrics["algorithm"].tolist())

    assert "KMeans" in algorithms
    assert "GaussianMixture" in algorithms
    assert output.kmeans_selection["k"].min() == 2


def test_anomaly_benchmark_has_required_columns() -> None:
    dataset = load_digits_dataset(scale=True)
    output = run_anomaly_benchmark(dataset.X[:600], dataset.y[:600], random_state=42)

    expected_columns = {"algorithm", "precision", "recall", "f1", "roc_auc"}
    assert expected_columns.issubset(set(output.columns))
    assert not output.empty
