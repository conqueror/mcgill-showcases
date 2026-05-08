"""Corpus loading and chunk preparation utilities."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from importlib import resources

import pandas as pd

from modern_nlp_pipeline_showcase.config import CORPUS_PATH, QUERY_PATH

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_REQUIRED_COLUMNS = ("paper_id", "title", "topic", "abstract", "summary")


def load_corpus() -> pd.DataFrame:
    """Load the local research corpus and enforce the expected schema."""
    if CORPUS_PATH.exists():
        corpus = pd.read_csv(CORPUS_PATH)
    else:
        corpus = _load_sample_corpus()
    missing = set(_REQUIRED_COLUMNS) - set(corpus.columns)
    if missing:
        raise ValueError(f"Corpus is missing columns: {sorted(missing)}")
    return corpus.loc[:, list(_REQUIRED_COLUMNS)].copy()


def load_queries() -> list[dict[str, str]]:
    """Load retrieval and QA prompts for evaluation."""
    if QUERY_PATH.exists():
        with QUERY_PATH.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    else:
        raw = _load_sample_queries()
    if not isinstance(raw, list):
        raise ValueError("Query file must contain a list of query objects.")
    return [dict(item) for item in raw]


def split_into_sentences(text: str) -> list[str]:
    """Split text into simple sentence-like spans."""
    parts = [part.strip() for part in _SENTENCE_SPLIT.split(text.strip()) if part.strip()]
    return parts if parts else [text.strip()]


def build_chunks(corpus: pd.DataFrame) -> pd.DataFrame:
    """Create chunk-level rows from abstracts and summaries."""
    rows: list[dict[str, object]] = []
    for paper in corpus.to_dict(orient="records"):
        segments = _paper_segments(paper["abstract"], paper["summary"])
        for position, chunk_text in enumerate(segments):
            rows.append(
                {
                    "chunk_id": f"{paper['paper_id']}_C{position:02d}",
                    "paper_id": paper["paper_id"],
                    "title": paper["title"],
                    "topic": paper["topic"],
                    "position": position,
                    "chunk_text": chunk_text,
                }
            )
    return pd.DataFrame(rows)


def _paper_segments(abstract: str, summary: str) -> Sequence[str]:
    abstract_sentences = split_into_sentences(abstract)
    summary_sentences = split_into_sentences(summary)
    return [*abstract_sentences, *summary_sentences]


def _load_sample_corpus() -> pd.DataFrame:
    sample_path = resources.files("modern_nlp_pipeline_showcase").joinpath(
        "sample_data/research_corpus.csv"
    )
    with sample_path.open("r", encoding="utf-8") as handle:
        return pd.read_csv(handle)


def _load_sample_queries() -> object:
    sample_path = resources.files("modern_nlp_pipeline_showcase").joinpath(
        "sample_data/research_queries.json"
    )
    with sample_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
