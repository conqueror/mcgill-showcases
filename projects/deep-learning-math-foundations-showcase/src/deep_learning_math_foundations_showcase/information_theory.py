"""Information theory helpers for beginner-friendly loss intuition."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

EPSILON = 1e-12


def _normalize(probabilities: Sequence[float]) -> np.ndarray:
    """Normalize a sequence into a proper probability vector."""

    values = np.asarray(probabilities, dtype=float)
    total = values.sum()
    if total <= 0:
        raise ValueError("Probability values must sum to a positive number.")
    return values / total


def entropy(probabilities: Sequence[float]) -> float:
    """Return Shannon entropy in bits."""

    probs = np.clip(_normalize(probabilities), EPSILON, 1.0)
    return float(-(probs * np.log2(probs)).sum())


def cross_entropy(
    target_probabilities: Sequence[float],
    predicted_probabilities: Sequence[float],
) -> float:
    """Return cross-entropy in bits."""

    target = np.clip(_normalize(target_probabilities), EPSILON, 1.0)
    predicted = np.clip(_normalize(predicted_probabilities), EPSILON, 1.0)
    return float(-(target * np.log2(predicted)).sum())


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
