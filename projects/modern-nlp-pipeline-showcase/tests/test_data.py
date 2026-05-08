from modern_nlp_pipeline_showcase.data import build_chunks, load_corpus, load_queries


def test_load_corpus_has_expected_schema() -> None:
    corpus = load_corpus()

    assert not corpus.empty
    assert set(corpus.columns) == {"paper_id", "title", "topic", "abstract", "summary"}
    assert corpus["paper_id"].nunique() == len(corpus)


def test_load_queries_has_expected_keys() -> None:
    queries = load_queries()

    assert len(queries) >= 5
    assert {
        "query_id",
        "query",
        "topic_label",
        "relevant_paper_id",
        "qa_question",
        "expected_answer",
    } <= set(queries[0])


def test_build_chunks_produces_multiple_rows_per_document() -> None:
    corpus = load_corpus().iloc[:2]
    chunks = build_chunks(corpus)

    assert len(chunks) >= len(corpus) * 2
    assert {"chunk_id", "paper_id", "chunk_text", "position"} <= set(chunks.columns)
