import pytest

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


def test_generation_uses_lexical_hit_instead_of_dense_miss() -> None:
    queries = [
        {
            "query_id": "q",
            "query": "cats",
            "qa_question": "What do cats eat?",
            "expected_answer": "Cats eat fish.",
        }
    ]
    examples: list[dict[str, object]] = [
        {
            "query_id": "q",
            "strategy": "lexical_tfidf",
            "hit_rank": 1,
            "top_passages": ["Cats eat fish."],
        },
        {
            "query_id": "q",
            "strategy": "dense_hashing",
            "hit_rank": None,
            "top_passages": ["Dogs eat meat."],
        },
    ]
    answers, summaries = generate_grounded_outputs(
        queries, examples, HeuristicQABackend(), HeuristicSummarizerBackend()
    )
    assert answers.iloc[0]["source_strategy"] == "lexical_tfidf"
    assert answers.iloc[0]["predicted_answer"] == "Cats eat fish."
    assert summaries[0]["summary_text"] == "Cats eat fish."


def test_heuristic_qa_abstains_without_question_overlap() -> None:
    backend = HeuristicQABackend()
    assert backend.answer("Where do penguins live?", "Cats eat fish.") == ""
    assert backend.answer("Where do penguins live?", "") == ""


def test_generation_handles_no_retrieved_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    queries = [
        {
            "query_id": "q",
            "query": "cats",
            "qa_question": "What do cats eat?",
            "expected_answer": "Cats eat fish.",
        }
    ]
    monkeypatch.setattr(HeuristicQABackend, "answer", lambda *args: "Unsupported answer")
    monkeypatch.setattr(
        HeuristicSummarizerBackend, "summarize", lambda *args: "Unsupported summary"
    )
    answers, summaries = generate_grounded_outputs(
        queries, [], HeuristicQABackend(), HeuristicSummarizerBackend()
    )
    assert answers.iloc[0]["predicted_answer"] == ""
    assert summaries[0]["summary_text"] == ""
