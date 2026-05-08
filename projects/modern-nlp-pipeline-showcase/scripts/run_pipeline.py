"""Run the end-to-end NLP showcase pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if TYPE_CHECKING:
    import pandas as pd

    from modern_nlp_pipeline_showcase.models import QABackend, SentenceEncoder, SummarizerBackend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use lightweight fallback backends only.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Top-k depth for retrieval metrics.")
    return parser.parse_args()


def main() -> None:
    from modern_nlp_pipeline_showcase.classification import run_classification_comparison
    from modern_nlp_pipeline_showcase.config import ARTIFACTS_DIR
    from modern_nlp_pipeline_showcase.data import build_chunks, load_corpus, load_queries
    from modern_nlp_pipeline_showcase.generation import generate_grounded_outputs
    from modern_nlp_pipeline_showcase.lexical import fit_tfidf_index
    from modern_nlp_pipeline_showcase.reporting import (
        build_manifest,
        write_dataframe,
        write_json,
        write_markdown,
    )
    from modern_nlp_pipeline_showcase.retrieval import evaluate_retrieval

    args = parse_args()
    artifact_root = ARTIFACTS_DIR
    corpus = load_corpus()
    queries = load_queries()
    chunks = build_chunks(corpus)

    lexical_index = fit_tfidf_index(chunks["chunk_text"].tolist())
    encoder = _resolve_encoder(args.quick)
    qa_backend = _resolve_qa_backend(args.quick)
    summarizer_backend = _resolve_summarizer_backend(args.quick)

    corpus_overview = corpus.assign(
        abstract_words=corpus["abstract"].str.split().str.len(),
        summary_words=corpus["summary"].str.split().str.len(),
    )[["paper_id", "title", "topic", "abstract_words", "summary_words"]]
    topic_distribution = corpus.groupby("topic", as_index=False).agg(
        paper_count=("paper_id", "count")
    )
    classification_metrics = run_classification_comparison(corpus, encoder=encoder)
    retrieval_metrics, retrieval_examples = evaluate_retrieval(
        chunks=chunks,
        queries=queries,
        lexical_index=lexical_index,
        encoder=encoder,
        top_k=args.top_k,
    )
    qa_outputs, summaries = generate_grounded_outputs(
        queries=queries,
        retrieval_examples=retrieval_examples,
        qa_backend=qa_backend,
        summarizer_backend=summarizer_backend,
    )

    write_dataframe(corpus_overview, artifact_root / "data/corpus_overview.csv")
    write_dataframe(topic_distribution, artifact_root / "data/topic_distribution.csv")
    write_dataframe(classification_metrics, artifact_root / "classification/metrics_summary.csv")
    write_dataframe(retrieval_metrics, artifact_root / "retrieval/retrieval_metrics.csv")
    write_json(retrieval_examples, artifact_root / "retrieval/retrieval_examples.json")
    write_dataframe(qa_outputs, artifact_root / "generation/qa_outputs.csv")
    write_json(summaries, artifact_root / "generation/query_summaries.json")
    write_json(build_manifest(), artifact_root / "manifest.json")
    write_markdown(
        build_summary_markdown(
            classification_metrics=classification_metrics,
            retrieval_metrics=retrieval_metrics,
            qa_outputs=qa_outputs,
            encoder_backend=encoder.backend_name,
            qa_backend=qa_backend.backend_name,
            summarizer_backend=summarizer_backend.backend_name,
            quick=args.quick,
        ),
        artifact_root / "summary.md",
    )


def build_summary_markdown(
    classification_metrics: pd.DataFrame,
    retrieval_metrics: pd.DataFrame,
    qa_outputs: pd.DataFrame,
    encoder_backend: str,
    qa_backend: str,
    summarizer_backend: str,
    quick: bool,
) -> str:
    best_classifier = classification_metrics.sort_values("macro_f1", ascending=False).iloc[0]
    best_retriever = retrieval_metrics.sort_values("mrr_at_k", ascending=False).iloc[0]
    qa_preview = qa_outputs.iloc[0]["predicted_answer"]

    return f"""
# Modern NLP Pipeline Showcase Summary

- Run mode: {"quick" if quick else "full"}
- Dense encoder backend: `{encoder_backend}`
- QA backend: `{qa_backend}`
- Summarizer backend: `{summarizer_backend}`
- Best classifier: `{best_classifier["model"]}` with macro F1 `{best_classifier["macro_f1"]}`
- Best retriever: `{best_retriever["strategy"]}` with MRR@k `{best_retriever["mrr_at_k"]}`
- Example QA output: {qa_preview}
"""


def _resolve_encoder(quick: bool) -> SentenceEncoder:
    from modern_nlp_pipeline_showcase.models import HashingSentenceEncoder, load_dense_encoder

    if quick:
        return HashingSentenceEncoder()
    encoder = load_dense_encoder(prefer_transformer=True)
    try:
        encoder.encode(["healthcheck"])
        return encoder
    except Exception:
        return HashingSentenceEncoder()


def _resolve_qa_backend(quick: bool) -> QABackend:
    from modern_nlp_pipeline_showcase.models import HeuristicQABackend, load_qa_backend

    if quick:
        return HeuristicQABackend()
    backend = load_qa_backend(prefer_transformer=True)
    try:
        backend.answer("What is being studied?", "This work studies retrieval quality.")
        return backend
    except Exception:
        return HeuristicQABackend()


def _resolve_summarizer_backend(quick: bool) -> SummarizerBackend:
    from modern_nlp_pipeline_showcase.models import (
        HeuristicSummarizerBackend,
        load_summarizer_backend,
    )

    if quick:
        return HeuristicSummarizerBackend()
    backend = load_summarizer_backend(prefer_transformer=True)
    try:
        backend.summarize("retrieval quality", "This work studies retrieval quality.")
        return backend
    except Exception:
        return HeuristicSummarizerBackend()


if __name__ == "__main__":
    main()
