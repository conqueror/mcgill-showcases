"""Orchestrates tutorial demos and exports learning artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .active_learning import run_active_learning_simulation
from .anomaly import run_anomaly_benchmark
from .config import PathsConfig, ShowcaseConfig
from .data import (
    LabeledDataset,
    load_business_loan_dataset,
    load_digits_dataset,
    make_train_test_split,
)
from .dec import run_dec_benchmark
from .plots import (
    plot_active_learning_curve,
    plot_embedding_projection,
    plot_kmeans_selection,
    plot_metric_bar,
)
from .self_supervised import run_self_supervised_benchmark
from .semi_supervised import run_semi_supervised_benchmark
from .unsupervised import run_clustering_benchmark


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _load_dataset(config: ShowcaseConfig) -> tuple[LabeledDataset, int | None]:
    if config.dataset == "business":
        dataset = load_business_loan_dataset(
            csv_path=config.business_csv_path,
            read_rows=config.business_read_rows,
            sample_size=config.business_sample_size,
            scale=True,
            random_state=config.random_state,
        )
        return dataset, 1

    dataset = load_digits_dataset(scale=True)
    return dataset, 9


def run_full_showcase(config: ShowcaseConfig, paths: PathsConfig) -> dict[str, pd.DataFrame]:
    """Execute all showcase modules and persist metrics/figures."""

    paths.ensure()
    dataset, anomaly_label = _load_dataset(config)
    split = make_train_test_split(
        X=dataset.X,
        y=dataset.y,
        test_size=config.test_size,
        labeled_fraction=config.labeled_fraction,
        random_state=config.random_state,
    )

    n_classes = len(np.unique(dataset.y))
    semisup_clusters = min(
        config.semisup_kmeans_clusters,
        max(2, n_classes * 8),
        max(2, split.X_train.shape[0] // 2),
    )

    clustering_out = run_clustering_benchmark(
        X=dataset.X,
        y=dataset.y,
        random_state=config.random_state,
        n_clusters=n_classes,
    )
    anomaly_df = run_anomaly_benchmark(
        X=dataset.X,
        y=dataset.y,
        random_state=config.random_state,
        anomaly_label=anomaly_label,
    )
    semi_df = run_semi_supervised_benchmark(
        X_train=split.X_train,
        X_test=split.X_test,
        y_train=split.y_train,
        y_test=split.y_test,
        y_train_masked=split.y_train_masked,
        random_state=config.random_state,
        semisup_kmeans_clusters=semisup_clusters,
        self_training_threshold=config.self_training_threshold,
    )
    selfsup_out = run_self_supervised_benchmark(
        X_train=split.X_train,
        X_test=split.X_test,
        y_train=split.y_train,
        y_test=split.y_test,
        y_train_masked=split.y_train_masked,
        random_state=config.random_state,
        embedding_dim=config.contrastive_embedding_dim,
        epochs=config.contrastive_epochs,
        batch_size=config.contrastive_batch_size,
    )
    dec_out = run_dec_benchmark(
        X=dataset.X,
        y=dataset.y,
        random_state=config.random_state,
        pretrain_epochs=config.dec_pretrain_epochs,
        finetune_epochs=config.dec_finetune_epochs,
        n_clusters=n_classes,
    )
    active_learning_out = run_active_learning_simulation(
        X_train=split.X_train,
        y_train=split.y_train,
        y_train_masked=split.y_train_masked,
        X_test=split.X_test,
        y_test=split.y_test,
        random_state=config.random_state,
        rounds=config.active_learning_rounds,
        query_size=config.active_learning_query_size,
    )

    prefix = config.dataset

    _write_csv(clustering_out.metrics, paths.reports_dir / f"{prefix}_clustering_metrics.csv")
    _write_csv(
        clustering_out.kmeans_selection,
        paths.reports_dir / f"{prefix}_kmeans_model_selection.csv",
    )
    _write_csv(anomaly_df, paths.reports_dir / f"{prefix}_anomaly_metrics.csv")
    _write_csv(semi_df, paths.reports_dir / f"{prefix}_semi_supervised_metrics.csv")
    _write_csv(selfsup_out.metrics, paths.reports_dir / f"{prefix}_self_supervised_metrics.csv")
    _write_csv(dec_out.metrics, paths.reports_dir / f"{prefix}_dec_metrics.csv")
    _write_csv(
        active_learning_out.metrics,
        paths.reports_dir / f"{prefix}_active_learning_metrics.csv",
    )

    pretty_name = "Digits" if config.dataset == "digits" else "Business Loan"
    plot_kmeans_selection(
        selection_df=clustering_out.kmeans_selection,
        output_path=paths.figures_dir / f"{prefix}_kmeans_selection.png",
    )
    plot_metric_bar(
        df=clustering_out.metrics,
        method_col="algorithm",
        metric_col="ari",
        title=f"{pretty_name}: Clustering ARI Comparison",
        output_path=paths.figures_dir / f"{prefix}_clustering_ari.png",
    )
    plot_metric_bar(
        df=semi_df,
        method_col="method",
        metric_col="accuracy",
        title=f"{pretty_name}: Semi-Supervised Accuracy",
        output_path=paths.figures_dir / f"{prefix}_semi_supervised_accuracy.png",
    )
    plot_metric_bar(
        df=selfsup_out.metrics,
        method_col="method",
        metric_col="accuracy",
        title=f"{pretty_name}: Self-Supervised Transfer",
        output_path=paths.figures_dir / f"{prefix}_self_supervised_accuracy.png",
    )
    plot_embedding_projection(
        embeddings=selfsup_out.train_embeddings,
        y=split.y_train,
        output_path=paths.figures_dir / f"{prefix}_contrastive_embeddings_train_pca.png",
        title=f"{pretty_name}: Contrastive Embedding (Train)",
    )
    plot_active_learning_curve(
        df=active_learning_out.metrics,
        output_path=paths.figures_dir / f"{prefix}_active_learning_curve.png",
        title=f"{pretty_name}: Active Learning Progress",
    )

    final_round = active_learning_out.metrics["round"].max()
    final_active = active_learning_out.metrics[active_learning_out.metrics["round"] == final_round]
    active_top = final_active.sort_values("accuracy", ascending=False).iloc[0]["strategy"]
    final_scores: dict[str, float] = {}
    for strategy, score in zip(final_active["strategy"], final_active["accuracy"], strict=False):
        final_scores[str(strategy)] = float(score)
    active_gain = float(
        final_scores.get("uncertainty", np.nan) - final_scores.get("random", np.nan),
    )

    summary = {
        "dataset": dataset.dataset_name,
        "mode": config.dataset,
        "n_samples": int(dataset.X.shape[0]),
        "n_features": int(dataset.X.shape[1]),
        "n_classes": n_classes,
        "labeled_fraction_train": config.labeled_fraction,
        "best_clustering_algorithm": str(clustering_out.metrics.iloc[0]["algorithm"]),
        "best_semisup_method": str(semi_df.iloc[0]["method"]),
        "best_selfsup_method": str(selfsup_out.metrics.iloc[0]["method"]),
        "best_active_learning_strategy": str(active_top),
        "active_learning_gain_vs_random_at_final_round": active_gain,
    }

    with (paths.reports_dir / f"{prefix}_run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return {
        "clustering": clustering_out.metrics,
        "anomaly": anomaly_df,
        "semi_supervised": semi_df,
        "self_supervised": selfsup_out.metrics,
        "dec": dec_out.metrics,
        "active_learning": active_learning_out.metrics,
    }
