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
