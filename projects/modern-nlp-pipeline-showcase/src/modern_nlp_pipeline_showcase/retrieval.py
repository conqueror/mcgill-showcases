"""Retrieval evaluation helpers."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from modern_nlp_pipeline_showcase.lexical import TfidfIndex, lexical_search
from modern_nlp_pipeline_showcase.models import SentenceEncoder


def evaluate_retrieval(
    chunks: pd.DataFrame,
    queries: list[dict[str, str]],
    lexical_index: TfidfIndex,
    encoder: SentenceEncoder,
    top_k: int = 5,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Evaluate lexical and dense retrieval on the query set."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    examples: list[dict[str, object]] = []
    dense_chunk_vectors = (
        encoder.encode(chunks["chunk_text"].tolist()) if not chunks.empty else np.empty((0, 0))
    )
    dense_query_vectors = encoder.encode([item["query"] for item in queries])

    metrics_store: dict[str, list[float]] = defaultdict(list)

    for query, dense_query_vector in zip(queries, dense_query_vectors, strict=True):
        lexical_results = lexical_search(query["query"], chunks, lexical_index, top_k=top_k)
        dense_results = dense_search(
            query["query"],
            chunks,
            dense_chunk_vectors,
            dense_query_vector,
            top_k=top_k,
        )

        for strategy, results in (
            ("lexical_tfidf", lexical_results),
            (encoder.backend_name, dense_results),
        ):
            rank = _find_relevant_rank(results, query["relevant_paper_id"])
            metrics_store[f"{strategy}:recall"].append(1.0 if rank is not None else 0.0)
            metrics_store[f"{strategy}:mrr"].append(0.0 if rank is None else 1.0 / rank)
            examples.append(
                {
                    "query_id": query["query_id"],
                    "query": query["query"],
                    "strategy": strategy,
                    "relevant_paper_id": query["relevant_paper_id"],
                    "hit_rank": rank,
                    "retrieved_paper_ids": results["paper_id"].tolist(),
                    "top_chunk_id": results["chunk_id"].iloc[0] if not results.empty else None,
                    "top_chunk_text": results["chunk_text"].iloc[0] if not results.empty else None,
                    "top_passages": results["chunk_text"].tolist(),
                }
            )

    rows = []
    for strategy in ("lexical_tfidf", encoder.backend_name):
        rows.append(
            {
                "strategy": strategy,
                "recall_at_k": round(float(np.mean(metrics_store[f"{strategy}:recall"])), 4),
                "mrr_at_k": round(float(np.mean(metrics_store[f"{strategy}:mrr"])), 4),
                "top_k": top_k,
            }
        )
    return pd.DataFrame(rows), examples


def dense_search(
    query: str,
    chunks: pd.DataFrame,
    chunk_vectors: np.ndarray,
    query_vector: np.ndarray,
    top_k: int = 5,
) -> pd.DataFrame:
    """Rank chunks with cosine similarity over dense vectors."""
    _ = query
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if chunks.empty:
        return chunks.assign(score=pd.Series(dtype=float))
    scores = cosine_similarity(query_vector.reshape(1, -1), chunk_vectors).ravel()
    top_indices = np.argsort(-scores, kind="stable")[:top_k]
    results = chunks.iloc[top_indices].copy()
    results["score"] = scores[top_indices]
    return results.reset_index(drop=True)


def _find_relevant_rank(results: pd.DataFrame, relevant_paper_id: str) -> int | None:
    for index, row in enumerate(results.itertuples(index=False), start=1):
        if row.paper_id == relevant_paper_id:
            return index
    return None
