"""Tests for probability and statistics helpers."""

from __future__ import annotations

import pandas as pd

from deep_learning_math_foundations_showcase import probability


def test_gaussian_summary_is_reasonable_and_seeded() -> None:
    """Seeded Gaussian summaries should be reproducible."""

    summary = probability.summarize_gaussian_samples(seed=7, sample_size=2_000)

    assert round(summary["sample_mean"], 3) == -0.04
    assert round(summary["sample_variance"], 3) == 0.97


def test_bernoulli_probability_estimate_is_reproducible() -> None:
    """Bernoulli estimation should be deterministic for a fixed seed."""

    estimate = probability.estimate_bernoulli_probability(seed=7, trials=1_000, p=0.3)
    assert round(estimate, 3) == 0.297


def test_probability_table_has_expected_columns() -> None:
    """Probability simulations should be surfaced through a stable artifact schema."""

    table = probability.build_probability_simulations_table()
    assert isinstance(table, pd.DataFrame)
    assert list(table.columns) == ["category", "metric", "value"]
