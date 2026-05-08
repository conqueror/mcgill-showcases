"""Lexical baseline helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


@dataclass(frozen=True)
class TfidfIndex:
    """Container for TF-IDF features and source texts."""

    vectorizer: TfidfVectorizer
    matrix: spmatrix
    texts: list[str]


def fit_tfidf_index(texts: list[str]) -> TfidfIndex:
    """Fit a TF-IDF vectorizer on chunk texts."""
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    return TfidfIndex(vectorizer=vectorizer, matrix=matrix, texts=texts)


def lexical_search(
    query: str,
    chunks: pd.DataFrame,
    index: TfidfIndex,
    top_k: int = 5,
) -> pd.DataFrame:
    """Rank chunks for a query with cosine similarity over TF-IDF features."""
    query_vector = index.vectorizer.transform([query])
    scores = linear_kernel(query_vector, index.matrix).ravel()
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = chunks.iloc[top_indices].copy()
    results["score"] = scores[top_indices]
    return results.sort_values("score", ascending=False).reset_index(drop=True)
