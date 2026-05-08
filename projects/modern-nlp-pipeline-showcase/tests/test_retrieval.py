from modern_nlp_pipeline_showcase.data import build_chunks, load_corpus, load_queries
from modern_nlp_pipeline_showcase.lexical import fit_tfidf_index
from modern_nlp_pipeline_showcase.models import HashingSentenceEncoder
from modern_nlp_pipeline_showcase.retrieval import evaluate_retrieval


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
