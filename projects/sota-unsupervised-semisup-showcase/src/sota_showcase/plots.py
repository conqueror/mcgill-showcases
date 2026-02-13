"""Plotting helpers for learner-friendly artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from numpy.typing import NDArray
from sklearn.decomposition import PCA


def plot_kmeans_selection(selection_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(selection_df["k"], selection_df["inertia"], marker="o")
    axes[0].set_title("KMeans Elbow (Inertia)")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(selection_df["k"], selection_df["silhouette"], marker="o", color="#ff7f0e")
    axes[1].set_title("KMeans Silhouette")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_metric_bar(
    df: pd.DataFrame,
    method_col: str,
    metric_col: str,
    title: str,
    output_path: Path,
) -> None:
    ranked = df.sort_values(metric_col, ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(ranked[method_col], ranked[metric_col], color="#4C78A8")
    ax.set_title(title)
    ax.set_ylabel(metric_col)
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_embedding_projection(
    embeddings: NDArray,
    y: NDArray,
    output_path: Path,
    title: str,
) -> None:
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=y, cmap="tab10", s=12, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    legend = ax.legend(*scatter.legend_elements(), title="Class", loc="best", fontsize=8)
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_active_learning_curve(df: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for strategy, group in df.groupby("strategy"):
        ordered = group.sort_values("labeled_budget")
        ax.plot(
            ordered["labeled_budget"],
            ordered["accuracy"],
            marker="o",
            linewidth=2,
            label=strategy,
        )

    ax.set_title(title)
    ax.set_xlabel("Number of Labeled Training Examples")
    ax.set_ylabel("Accuracy")
    ax.legend(title="Sampling Strategy")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
