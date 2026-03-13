"""Plotting helpers for decision-boundary visualizations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from neural_network_foundations_showcase import data, networks


@dataclass(frozen=True)
class DecisionBoundaryExperiment:
    """A trained model paired with the dataset it should visualize."""

    title: str
    dataset: data.ToyDataset
    network: networks.FeedForwardNetwork


def plot_decision_boundaries(
    experiments: list[DecisionBoundaryExperiment],
    output_path: Path,
) -> None:
    """Plot one decision-boundary panel per experiment."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, len(experiments), figsize=(5 * len(experiments), 4))
    axes_array = np.atleast_1d(axes)

    for axis, experiment in zip(axes_array, experiments, strict=True):
        features = experiment.dataset.features
        labels = experiment.dataset.labels
        x_min, x_max = features[:, 0].min() - 0.5, features[:, 0].max() + 0.5
        y_min, y_max = features[:, 1].min() - 0.5, features[:, 1].max() + 0.5
        x_values, y_values = np.meshgrid(
            np.linspace(x_min, x_max, 180),
            np.linspace(y_min, y_max, 180),
        )
        grid = np.column_stack([x_values.ravel(), y_values.ravel()])
        probabilities = networks.predict_proba(experiment.network, grid).reshape(
            x_values.shape,
        )

        axis.contourf(
            x_values,
            y_values,
            probabilities,
            levels=np.linspace(0.0, 1.0, 11),
            cmap="coolwarm",
            alpha=0.35,
        )
        axis.contour(
            x_values,
            y_values,
            probabilities,
            levels=[0.5],
            colors="black",
            linewidths=1.0,
        )
        axis.scatter(
            features[:, 0],
            features[:, 1],
            c=labels,
            cmap="coolwarm",
            edgecolors="black",
            s=28,
        )
        axis.set_title(experiment.title)
        axis.set_xlabel("feature_1")
        axis.set_ylabel("feature_2")

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
