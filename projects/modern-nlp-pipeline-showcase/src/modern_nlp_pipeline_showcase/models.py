"""Model backends with lightweight fallbacks for tests and offline runs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

from modern_nlp_pipeline_showcase.config import (
    DEFAULT_DENSE_MODEL,
    DEFAULT_QA_MODEL,
    DEFAULT_SUMMARY_MODEL,
)

_TOKEN_RE = re.compile(r"\b[a-zA-Z][a-zA-Z\-]+\b")


class SentenceEncoder(Protocol):
    """Minimal sentence encoder interface."""

    @property
    def backend_name(self) -> str:
        """Backend identifier."""

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into dense vectors."""


class QABackend(Protocol):
    """Question-answering backend interface."""

    @property
    def backend_name(self) -> str:
        """Backend identifier."""

    def answer(self, question: str, context: str) -> str:
        """Generate an answer from a question and grounded context."""


class SummarizerBackend(Protocol):
    """Summarization backend interface."""

    @property
    def backend_name(self) -> str:
        """Backend identifier."""

    def summarize(self, query: str, context: str) -> str:
        """Generate a query-focused summary from grounded context."""


@dataclass
class HashingSentenceEncoder:
    """Deterministic dense-like fallback based on hashing features."""

    backend_name: str = "dense_hashing"
    n_features: int = 512

    def encode(self, texts: list[str]) -> np.ndarray:
        vectorizer = HashingVectorizer(
            n_features=self.n_features,
            alternate_sign=False,
            stop_words="english",
            ngram_range=(1, 2),
        )
        matrix = vectorizer.transform(texts)
        return np.asarray(normalize(matrix).toarray(), dtype=np.float64)


@dataclass
class TransformerSentenceEncoder:
    """Sentence-transformer encoder wrapper."""

    model_name: str
    _model: Any | None = field(default=None, init=False, repr=False)

    @property
    def backend_name(self) -> str:
        return self.model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer

        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(embeddings, dtype=float)


@dataclass
class HeuristicQABackend:
    """Sentence-overlap QA fallback."""

    backend_name: str = "heuristic_qa"

    def answer(self, question: str, context: str) -> str:
        sentences = _split_sentences(context)
        ranked = sorted(
            sentences,
            key=lambda sentence: _overlap_score(question, sentence),
            reverse=True,
        )
        return ranked[0] if ranked else context.strip()


@dataclass
class TransformersQABackend:
    """Transformers pipeline QA backend."""

    model_name: str
    _pipeline: Any | None = field(default=None, init=False, repr=False)

    @property
    def backend_name(self) -> str:
        return self.model_name

    def answer(self, question: str, context: str) -> str:
        from transformers import pipeline

        if self._pipeline is None:
            pipeline_fn: Any = pipeline
            self._pipeline = pipeline_fn(
                task="question-answering",
                model=self.model_name,
                tokenizer=self.model_name,
            )
        response = self._pipeline(question=question, context=context)
        return str(response["answer"]).strip()


@dataclass
class HeuristicSummarizerBackend:
    """Simple extractive summarizer fallback."""

    backend_name: str = "heuristic_summary"

    def summarize(self, query: str, context: str) -> str:
        sentences = _split_sentences(context)
        ranked = sorted(
            sentences,
            key=lambda sentence: _overlap_score(query, sentence),
            reverse=True,
        )
        top_sentences = ranked[:2] if ranked else [context.strip()]
        return " ".join(sentence.strip() for sentence in top_sentences if sentence.strip())


@dataclass
class TransformersSummarizerBackend:
    """Instruction-style summarization backend."""

    model_name: str
    _pipeline: Any | None = field(default=None, init=False, repr=False)

    @property
    def backend_name(self) -> str:
        return self.model_name

    def summarize(self, query: str, context: str) -> str:
        from transformers import pipeline

        if self._pipeline is None:
            pipeline_fn: Any = pipeline
            self._pipeline = pipeline_fn(
                task="text2text-generation",
                model=self.model_name,
                tokenizer=self.model_name,
            )
        prompt = (
            "Summarize the following research evidence for the query.\n"
            f"Query: {query}\n"
            f"Evidence: {context}\n"
            "Answer in 2 concise sentences."
        )
        response = self._pipeline(prompt, max_new_tokens=96, do_sample=False)[0]["generated_text"]
        return str(response).strip()


def load_dense_encoder(
    prefer_transformer: bool = True,
    model_name: str = DEFAULT_DENSE_MODEL,
) -> SentenceEncoder:
    """Load the preferred dense encoder, falling back when unavailable."""
    if prefer_transformer:
        try:
            return TransformerSentenceEncoder(model_name=model_name)
        except Exception:
            pass
    return HashingSentenceEncoder()


def load_qa_backend(
    prefer_transformer: bool = True,
    model_name: str = DEFAULT_QA_MODEL,
) -> QABackend:
    """Load the preferred QA backend, falling back when unavailable."""
    if prefer_transformer:
        try:
            return TransformersQABackend(model_name=model_name)
        except Exception:
            pass
    return HeuristicQABackend()


def load_summarizer_backend(
    prefer_transformer: bool = True, model_name: str = DEFAULT_SUMMARY_MODEL
) -> SummarizerBackend:
    """Load the preferred summarization backend, falling back when unavailable."""
    if prefer_transformer:
        try:
            return TransformersSummarizerBackend(model_name=model_name)
        except Exception:
            pass
    return HeuristicSummarizerBackend()


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]


def _overlap_score(query: str, text: str) -> int:
    query_terms = set(_TOKEN_RE.findall(query.lower()))
    text_terms = set(_TOKEN_RE.findall(text.lower()))
    return len(query_terms & text_terms)
