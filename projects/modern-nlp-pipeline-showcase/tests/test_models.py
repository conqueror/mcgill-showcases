import pytest

from modern_nlp_pipeline_showcase.models import (
    TransformerSentenceEncoder,
    TransformersQABackend,
    TransformersSummarizerBackend,
    load_dense_encoder,
    load_qa_backend,
    load_summarizer_backend,
)


def test_dense_loader_falls_back_when_lazy_loading_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(self: TransformerSentenceEncoder, texts: list[str]) -> None:
        raise OSError("Model is not cached")

    monkeypatch.setattr(TransformerSentenceEncoder, "encode", unavailable)
    encoder = load_dense_encoder()
    assert encoder.backend_name == "dense_hashing"
    assert encoder.encode(["cats"])[0].sum() > 0


def test_qa_loader_falls_back_when_lazy_loading_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(self: TransformersQABackend, question: str, context: str) -> None:
        raise OSError("Model is not cached")

    monkeypatch.setattr(TransformersQABackend, "answer", unavailable)
    backend = load_qa_backend()
    assert backend.answer("What do cats eat?", "Cats eat fish.") == "Cats eat fish."


def test_summary_loader_falls_back_when_lazy_loading_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(self: TransformersSummarizerBackend, query: str, context: str) -> None:
        raise OSError("Model is not cached")

    monkeypatch.setattr(TransformersSummarizerBackend, "summarize", unavailable)
    backend = load_summarizer_backend()
    assert backend.summarize("cats", "Cats eat fish.") == "Cats eat fish."
