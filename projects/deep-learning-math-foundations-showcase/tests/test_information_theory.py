"""Tests for information theory helpers."""

from __future__ import annotations

from deep_learning_math_foundations_showcase import information_theory


def test_entropy_uses_log_base_two() -> None:
    """Entropy should be reported in bits for beginner-friendly interpretation."""

    assert information_theory.entropy([0.5, 0.5]) == 1.0


def test_cross_entropy_matches_expected_binary_example() -> None:
    """Cross-entropy should match a simple one-hot example."""

    value = information_theory.cross_entropy([1.0, 0.0], [0.8, 0.2])
    assert round(value, 6) == 0.321928


def test_kl_divergence_is_positive_for_mismatched_distributions() -> None:
    """KL divergence should be positive when two distributions differ."""

    value = information_theory.kl_divergence([0.5, 0.5], [0.75, 0.25])
    assert round(value, 6) == 0.207519


def test_information_theory_markdown_mentions_core_terms() -> None:
    """The teaching summary should name the key information-theory concepts."""

    summary = information_theory.build_information_theory_summary_markdown()
    assert "Entropy" in summary
    assert "Cross-Entropy" in summary
    assert "KL Divergence" in summary
