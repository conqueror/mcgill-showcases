"""Retrieval-grounded QA and summarization helpers."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from modern_nlp_pipeline_showcase.models import QABackend, SummarizerBackend


def generate_grounded_outputs(
    queries: list[dict[str, str]],
    retrieval_examples: list[dict[str, object]],
    qa_backend: QABackend,
    summarizer_backend: SummarizerBackend,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Generate QA answers and query summaries from retrieval results."""
    grouped_examples = defaultdict(list)
    for example in retrieval_examples:
        grouped_examples[str(example["query_id"])].append(example)

    qa_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, str]] = []
    for query in queries:
        chosen = _select_best_example(grouped_examples[query["query_id"]])
        passages = chosen.get("top_passages", [])
        if not isinstance(passages, list):
            passages = [str(passages)]
        context = "\n".join(str(item) for item in passages)
        predicted_answer = qa_backend.answer(query["qa_question"], context)
        summary_text = summarizer_backend.summarize(query["query"], context)
        qa_rows.append(
            {
                "query_id": query["query_id"],
                "backend": qa_backend.backend_name,
                "source_strategy": chosen["strategy"],
                "predicted_answer": predicted_answer,
                "expected_answer": query["expected_answer"],
            }
        )
        summary_rows.append(
            {
                "query_id": query["query_id"],
                "backend": summarizer_backend.backend_name,
                "source_strategy": str(chosen["strategy"]),
                "summary_text": summary_text,
            }
        )
    return pd.DataFrame(qa_rows), summary_rows


def _select_best_example(examples: list[dict[str, object]]) -> dict[str, object]:
    dense_like = [item for item in examples if str(item["strategy"]) != "lexical_tfidf"]
    ranked = dense_like or examples
    ranked.sort(key=lambda item: (item["hit_rank"] is None, item["hit_rank"] or 999))
    return ranked[0]
