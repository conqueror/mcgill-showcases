from modern_nlp_pipeline_showcase.classification import run_classification_comparison
from modern_nlp_pipeline_showcase.data import load_corpus
from modern_nlp_pipeline_showcase.models import HashingSentenceEncoder


def test_run_classification_comparison_returns_baseline_and_dense_rows() -> None:
    metrics = run_classification_comparison(load_corpus(), encoder=HashingSentenceEncoder())

    assert {"model", "accuracy", "macro_f1", "train_rows", "test_rows"} <= set(metrics.columns)
    assert set(metrics["model"]) == {"tfidf_logreg", "dense_logreg"}
