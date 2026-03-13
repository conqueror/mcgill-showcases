"""Probability and statistics helpers for deep learning prerequisites."""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_gaussian_samples(seed: int, sample_size: int) -> dict[str, float]:
    """Generate a seeded Gaussian sample summary."""

    rng = np.random.default_rng(seed)
    samples = rng.normal(loc=0.0, scale=1.0, size=sample_size)
    return {
        "sample_mean": float(samples.mean()),
        "sample_variance": float(samples.var(ddof=1)),
    }


def estimate_bernoulli_probability(seed: int, trials: int, p: float) -> float:
    """Estimate a Bernoulli probability by sample frequency."""

    rng = np.random.default_rng(seed)
    samples = rng.binomial(1, p, size=trials)
    return float(samples.mean())


def propagate_uncertainty(seed: int, samples: int) -> dict[str, float]:
    """Propagate uncertainty through a simple linear prediction rule."""

    rng = np.random.default_rng(seed)
    uncertain_input = rng.normal(loc=2.0, scale=0.2, size=samples)
    predictions = 1.5 * uncertain_input + 0.5
    return {
        "prediction_mean": float(predictions.mean()),
        "prediction_std": float(predictions.std(ddof=1)),
    }


def build_probability_simulations_table() -> pd.DataFrame:
    """Build a single table summarizing the probability examples."""

    gaussian = summarize_gaussian_samples(seed=7, sample_size=2_000)
    uncertainty = propagate_uncertainty(seed=7, samples=2_000)

    rows = [
        {
            "category": "gaussian",
            "metric": "sample_mean",
            "value": gaussian["sample_mean"],
        },
        {
            "category": "gaussian",
            "metric": "sample_variance",
            "value": gaussian["sample_variance"],
        },
        {
            "category": "bernoulli",
            "metric": "estimated_probability",
            "value": estimate_bernoulli_probability(seed=7, trials=1_000, p=0.3),
        },
        {
            "category": "uncertainty",
            "metric": "prediction_mean",
            "value": uncertainty["prediction_mean"],
        },
        {
            "category": "uncertainty",
            "metric": "prediction_std",
            "value": uncertainty["prediction_std"],
        },
    ]
    return pd.DataFrame(rows)
