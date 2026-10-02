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
        predicted_answer = (
            qa_backend.answer(query["qa_question"], context) if context.strip() else ""
        )
        summary_text = (
            summarizer_backend.summarize(query["query"], context) if context.strip() else ""
        )
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
    """Choose the best known hit for this labelled teaching example."""
    if not examples:
        return {"strategy": "none", "top_passages": []}
    return min(examples, key=lambda item: (item["hit_rank"] is None, item["hit_rank"] or 999))
