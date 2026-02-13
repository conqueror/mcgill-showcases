from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_qini_curves(curves: dict[str, pd.DataFrame], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, curve in curves.items():
        ax.plot(curve["fraction"], curve["incremental_gain"], label=name, linewidth=2)

    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Population fraction targeted")
    ax.set_ylabel("Incremental gain")
    ax.set_title("Qini-Style Curves by Causal Learner")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_uplift_distribution(
    uplift_scores: np.ndarray,
    learner_name: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(uplift_scores, bins=40, kde=True, ax=ax, color="#2b8cbe")
    ax.set_title(f"Predicted Uplift Distribution - {learner_name}")
    ax.set_xlabel("Predicted uplift")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_propensity_overlap(
    propensity_scores: np.ndarray,
    treatment: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.kdeplot(
        propensity_scores[treatment == 0],
        fill=True,
        common_norm=False,
        alpha=0.4,
        label="Control",
        ax=ax,
    )
    sns.kdeplot(
        propensity_scores[treatment == 1],
        fill=True,
        common_norm=False,
        alpha=0.4,
        label="Treatment",
        ax=ax,
    )
    ax.set_xlabel("Estimated propensity score P(T=1|X)")
    ax.set_ylabel("Density")
    ax.set_title("Propensity Score Overlap")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
