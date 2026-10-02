import numpy as np
import pandas as pd
import pytest

from modern_nlp_pipeline_showcase.data import build_chunks, load_corpus, load_queries
from modern_nlp_pipeline_showcase.lexical import fit_tfidf_index
from modern_nlp_pipeline_showcase.models import HashingSentenceEncoder
from modern_nlp_pipeline_showcase.retrieval import dense_search, evaluate_retrieval


def test_evaluate_retrieval_compares_lexical_and_dense() -> None:
    corpus = load_corpus()
    chunks = build_chunks(corpus)
    queries = load_queries()
    lexical_index = fit_tfidf_index(chunks["chunk_text"].tolist())

    metrics, examples = evaluate_retrieval(
        chunks=chunks,
        queries=queries,
        lexical_index=lexical_index,
        encoder=HashingSentenceEncoder(),
        top_k=3,
    )

    assert {"strategy", "recall_at_k", "mrr_at_k", "top_k"} <= set(metrics.columns)
    assert set(metrics["strategy"]) == {"lexical_tfidf", "dense_hashing"}
    assert len(examples) == len(queries) * 2


@pytest.mark.parametrize("top_k", [0, -1])
def test_evaluation_rejects_nonpositive_top_k(top_k: int) -> None:
    chunks = build_chunks(load_corpus())
    index = fit_tfidf_index(chunks["chunk_text"].tolist())
    with pytest.raises(ValueError, match="top_k"):
        evaluate_retrieval(chunks, load_queries(), index, HashingSentenceEncoder(), top_k)


def test_evaluation_handles_empty_chunks() -> None:
    chunks = build_chunks(load_corpus())
    index = fit_tfidf_index(chunks["chunk_text"].tolist())
    metrics, examples = evaluate_retrieval(
        chunks.iloc[:0], load_queries()[:1], index, HashingSentenceEncoder()
    )
    assert metrics["recall_at_k"].tolist() == [0.0, 0.0]
    assert metrics["mrr_at_k"].tolist() == [0.0, 0.0]
    assert all(example["retrieved_paper_ids"] == [] for example in examples)
    assert all(example["top_chunk_id"] is None for example in examples)


def test_dense_search_cosine_values_and_stable_ties() -> None:
    chunks = pd.DataFrame({"chunk_id": ["a", "b", "c"]})
    # (3, 4) and (6, 8) have lengths 5 and 10, so both cosines with (1, 0) are 3/5.
    results = dense_search(
        "q", chunks, np.array([[3.0, 4.0], [6.0, 8.0], [0.0, 5.0]]), np.array([1.0, 0.0]), top_k=3
    )
    assert results["chunk_id"].tolist() == ["a", "b", "c"]
    assert results["score"].tolist() == pytest.approx([0.6, 0.6, 0.0])


@pytest.mark.parametrize("top_k", [0, -1])
def test_dense_search_rejects_nonpositive_top_k(top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k"):
        dense_search(
            "q",
            pd.DataFrame({"chunk_id": ["a"]}),
            np.array([[1.0, 0.0]]),
            np.array([1.0, 0.0]),
            top_k,
        )


def test_retrieval_metrics_match_hand_calculation() -> None:
    chunks = pd.DataFrame(
        {"chunk_id": ["a", "b"], "paper_id": ["A", "B"], "chunk_text": ["apple", "banana"]}
    )
    queries = [
        {"query_id": paper, "query": "apple", "relevant_paper_id": paper}
        for paper in ("A", "B", "C")
    ]
    metrics, examples = evaluate_retrieval(
        chunks, queries, fit_tfidf_index(["apple", "banana"]), HashingSentenceEncoder(), 2
    )
    # First-hit ranks are 1, 2, and absent. Recall = 2/3; MRR = (1 + 1/2 + 0)/3.
    assert metrics["recall_at_k"].tolist() == [0.6667, 0.6667]
    assert metrics["mrr_at_k"].tolist() == [0.5, 0.5]
    assert [example["hit_rank"] for example in examples] == [1, 1, 2, 2, None, None]
