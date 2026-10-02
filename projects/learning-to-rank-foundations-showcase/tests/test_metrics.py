import math

import numpy as np
import pytest

from ltr_foundations_showcase.metrics import grouped_ndcg


def test_ndcg_uses_exponential_gain_and_query_average() -> None:
    relevance = np.array([3.0, 2.0, 0.0, 1.0])
    scores = np.array([0.8, 0.9, 0.1, 0.2])
    # Query a ranks grades 2,3,0: DCG@2=3+7/log2(3), IDCG@2=7+3/log2(3).
    # Query b has one positive item, so DCG/IDCG=1. Average the two queries.
    query_a = (3 + 7 / math.log2(3)) / (7 + 3 / math.log2(3))
    assert grouped_ndcg(relevance, scores, ["a", "a", "a", "b"], k=2) == pytest.approx(
        (query_a + 1) / 2
    )


def test_positive_singletons_have_perfect_ndcg() -> None:
    assert grouped_ndcg(np.array([3.0, 1.0]), np.array([0.0, 0.0]), ["a", "b"], k=5) == 1.0


def test_zero_idcg_matches_lightgbm_convention() -> None:
    assert grouped_ndcg(np.zeros(2), np.array([0.8, 0.2]), ["a", "a"], k=5) == 1.0


@pytest.mark.parametrize("k", [0, -1])
def test_ndcg_rejects_invalid_cutoff(k: int) -> None:
    with pytest.raises(ValueError, match="k"):
        grouped_ndcg(np.array([1.0, 0.0]), np.array([0.8, 0.2]), ["a", "a"], k=k)


def test_ndcg_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        grouped_ndcg(np.array([]), np.array([]), [], k=5)
