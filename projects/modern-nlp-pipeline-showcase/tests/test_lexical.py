import pandas as pd
import pytest

from modern_nlp_pipeline_showcase.data import build_chunks, load_corpus
from modern_nlp_pipeline_showcase.lexical import fit_tfidf_index, lexical_search


def test_fit_tfidf_index_matches_chunk_count() -> None:
    chunks = build_chunks(load_corpus().iloc[:4])
    index = fit_tfidf_index(chunks["chunk_text"].tolist())

    assert index.matrix.shape[0] == len(chunks)
    assert len(index.texts) == len(chunks)


def test_lexical_search_returns_ranked_results() -> None:
    chunks = build_chunks(load_corpus())
    index = fit_tfidf_index(chunks["chunk_text"].tolist())

    results = lexical_search(
        query="dense retrieval for scientific literature",
        chunks=chunks,
        index=index,
        top_k=3,
    )

    assert len(results) == 3
    assert results["score"].iloc[0] >= results["score"].iloc[-1]


def test_lexical_search_preserves_input_order_on_ties() -> None:
    chunks = pd.DataFrame({"chunk_id": ["a", "b", "c"], "chunk_text": ["cats"] * 3})
    index = fit_tfidf_index(chunks["chunk_text"].tolist())
    assert lexical_search("cats", chunks, index, 2)["chunk_id"].tolist() == ["a", "b"]


@pytest.mark.parametrize("top_k", [0, -1])
def test_lexical_search_rejects_nonpositive_top_k(top_k: int) -> None:
    chunks = build_chunks(load_corpus())
    index = fit_tfidf_index(chunks["chunk_text"].tolist())
    with pytest.raises(ValueError, match="top_k"):
        lexical_search("cats", chunks, index, top_k)
