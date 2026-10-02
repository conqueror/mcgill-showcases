"""Information theory helpers for beginner-friendly loss intuition."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _normalize(probabilities: Sequence[float]) -> np.ndarray:
    """Normalize a sequence into a proper probability vector."""

    values = np.asarray(probabilities, dtype=float)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.isfinite(values).all()
        or (values < 0).any()
    ):
        raise ValueError("Probabilities must be a finite, nonnegative 1-D vector.")
    if values.max() == 0:
        raise ValueError("Probability values must sum to a positive number.")
    values = values / values.max()
    total = values.sum()
    return values / total


def entropy(probabilities: Sequence[float]) -> float:
    """Return Shannon entropy in bits."""

    probs = _normalize(probabilities)
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def cross_entropy(
    target_probabilities: Sequence[float],
    predicted_probabilities: Sequence[float],
) -> float:
    """Return cross-entropy in bits."""

    target = _normalize(target_probabilities)
    predicted = _normalize(predicted_probabilities)
    if target.shape != predicted.shape:
        raise ValueError("Probability vectors must have matching lengths.")
    support = target > 0
    if (predicted[support] == 0).any():
        return float("inf")
    return float(-(target[support] * np.log2(predicted[support])).sum())


def kl_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Return KL divergence in bits."""

    return cross_entropy(p, q) - entropy(p)


def build_information_theory_summary_markdown() -> str:
    """Build a short teaching summary for entropy-related concepts."""

    balanced_entropy = entropy([0.5, 0.5])
    binary_cross_entropy = cross_entropy([1.0, 0.0], [0.8, 0.2])
    divergence = kl_divergence([0.5, 0.5], [0.75, 0.25])

    return "\n".join(
        [
            "# Information Theory Summary",
            "",
            "## Entropy",
            f"- Balanced binary entropy: {balanced_entropy:.6f} bits",
            "",
            "## Cross-Entropy",
            f"- One-hot vs predicted distribution: {binary_cross_entropy:.6f} bits",
            "",
            "## KL Divergence",
            (
                "- Divergence between balanced and skewed distributions: "
                f"{divergence:.6f} bits"
            ),
        ],
    )
