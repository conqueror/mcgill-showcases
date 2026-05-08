from modern_nlp_pipeline_showcase.data import build_chunks, load_corpus, load_queries
from modern_nlp_pipeline_showcase.generation import generate_grounded_outputs
from modern_nlp_pipeline_showcase.lexical import fit_tfidf_index
from modern_nlp_pipeline_showcase.models import (
    HashingSentenceEncoder,
    HeuristicQABackend,
    HeuristicSummarizerBackend,
)
from modern_nlp_pipeline_showcase.retrieval import evaluate_retrieval


def test_generate_grounded_outputs_emits_qa_and_summary_records() -> None:
    corpus = load_corpus()
    chunks = build_chunks(corpus)
    queries = load_queries()[:2]
    lexical_index = fit_tfidf_index(chunks["chunk_text"].tolist())
    _, retrieval_examples = evaluate_retrieval(
        chunks=chunks,
        queries=queries,
        lexical_index=lexical_index,
        encoder=HashingSentenceEncoder(),
        top_k=2,
    )

    qa_outputs, summaries = generate_grounded_outputs(
        queries=queries,
        retrieval_examples=retrieval_examples,
        qa_backend=HeuristicQABackend(),
        summarizer_backend=HeuristicSummarizerBackend(),
    )

    assert len(qa_outputs) == 2
    assert len(summaries) == 2
    assert {"query_id", "predicted_answer", "backend"} <= set(qa_outputs.columns)
    assert all(summary["summary_text"] for summary in summaries)
