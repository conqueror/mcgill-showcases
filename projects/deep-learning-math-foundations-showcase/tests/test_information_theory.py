"""Tests for information theory helpers."""

from __future__ import annotations

import math

import pytest

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


def test_zero_mass_terms_and_support_mismatch() -> None:
    """0 log 0 contributes zero; positive mass times -log2(0) is infinite."""

    assert information_theory.entropy([1.0, 0.0]) == 0.0
    assert information_theory.entropy([0.0, 0.5, 0.5]) == 1.0
    assert information_theory.cross_entropy([1.0, 0.0], [1.0, 0.0]) == 0.0
    assert information_theory.kl_divergence([1.0, 0.0], [1.0, 0.0]) == 0.0
    assert math.isinf(information_theory.cross_entropy([0.5, 0.5], [1.0, 0.0]))
    assert math.isinf(information_theory.kl_divergence([0.5, 0.5], [1.0, 0.0]))


def test_one_hot_kl_is_one_bit() -> None:
    """H([1, 0]) = 0 and -log2(1/2) = 1, so KL is exactly one bit."""

    assert information_theory.kl_divergence([1.0, 0.0], [0.5, 0.5]) == 1.0
    assert information_theory.entropy([1e308, 1e308]) == 1.0


@pytest.mark.parametrize(
    "probabilities",
    [[-1.0, 2.0], [float("nan"), 1.0], [float("inf"), 1.0], [[1.0, 1.0]]],
)
def test_invalid_probability_vectors_are_rejected(probabilities: list) -> None:
    """Positive totals must not hide negative, nonfinite, or multidimensional data."""

    with pytest.raises(ValueError):
        information_theory.entropy(probabilities)
    with pytest.raises(ValueError):
        information_theory.cross_entropy(probabilities, [0.5, 0.5])
    with pytest.raises(ValueError):
        information_theory.cross_entropy([0.5, 0.5], probabilities)


def test_cross_entropy_rejects_mismatched_supports() -> None:
    """A length-one prediction must not broadcast across a two-class target."""

    with pytest.raises(ValueError):
        information_theory.cross_entropy([1.0, 0.0], [1.0])
    with pytest.raises(ValueError):
        information_theory.kl_divergence([1.0], [0.5, 0.5])
