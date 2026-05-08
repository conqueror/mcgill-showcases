"""Topic classification utilities."""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from modern_nlp_pipeline_showcase.config import DEFAULT_RANDOM_STATE
from modern_nlp_pipeline_showcase.models import SentenceEncoder


def run_classification_comparison(corpus: pd.DataFrame, encoder: SentenceEncoder) -> pd.DataFrame:
    """Compare lexical and dense document classification baselines."""
    texts = _combine_text_fields(corpus)
    labels = corpus["topic"]
    train_texts, test_texts, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.33,
        random_state=DEFAULT_RANDOM_STATE,
        stratify=labels,
    )

    lexical_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    x_train_tfidf = lexical_vectorizer.fit_transform(train_texts)
    x_test_tfidf = lexical_vectorizer.transform(test_texts)
    tfidf_model = LogisticRegression(max_iter=2000)
    tfidf_model.fit(x_train_tfidf, y_train)
    tfidf_predictions = tfidf_model.predict(x_test_tfidf)

    x_train_dense = encoder.encode(train_texts.tolist())
    x_test_dense = encoder.encode(test_texts.tolist())
    dense_model = LogisticRegression(max_iter=2000)
    dense_model.fit(x_train_dense, y_train)
    dense_predictions = dense_model.predict(x_test_dense)

    return pd.DataFrame(
        [
            _metric_row(
                "tfidf_logreg",
                y_test,
                tfidf_predictions,
                len(train_texts),
                len(test_texts),
            ),
            _metric_row(
                "dense_logreg",
                y_test,
                dense_predictions,
                len(train_texts),
                len(test_texts),
            ),
        ]
    )


def _combine_text_fields(corpus: pd.DataFrame) -> pd.Series:
    return corpus["title"] + ". " + corpus["summary"] + " " + corpus["abstract"]


def _metric_row(
    model_name: str, y_true: pd.Series, y_pred: pd.Series, train_rows: int, test_rows: int
) -> dict[str, object]:
    return {
        "model": model_name,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "train_rows": train_rows,
        "test_rows": test_rows,
    }
