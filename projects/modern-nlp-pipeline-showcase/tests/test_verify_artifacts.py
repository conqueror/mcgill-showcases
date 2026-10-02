from pathlib import Path

import pytest

from modern_nlp_pipeline_showcase.reporting import (
    required_artifact_paths,
    verify_required_artifacts,
)


def test_verify_required_artifacts_detects_missing_files(tmp_path: Path) -> None:
    missing = verify_required_artifacts(tmp_path, required_artifact_paths())

    assert "artifacts/manifest.json" in missing


@pytest.mark.parametrize(
    ("relative_path", "content"),
    [
        ("artifacts/summary.md", ""),
        ("artifacts/summary.md", "   \n"),
        ("artifacts/manifest.json", "not json"),
        ("artifacts/manifest.json", "{}"),
        ("artifacts/generation/query_summaries.json", "[]"),
        ("artifacts/generation/query_summaries.json", '[{"wrong": "schema"}]'),
        (
            "artifacts/generation/query_summaries.json",
            '[{"query_id":"q","backend":null,"source_strategy":false,"summary_text":[]}]',
        ),
        (
            "artifacts/generation/qa_outputs.csv",
            "query_id,backend,source_strategy,predicted_answer,expected_answer\n,,,,\n",
        ),
        ("artifacts/retrieval/retrieval_metrics.csv", "strategy,recall_at_k,mrr_at_k,top_k\n"),
        ("artifacts/retrieval/retrieval_metrics.csv", "wrong\nvalue\n"),
        (
            "artifacts/retrieval/retrieval_metrics.csv",
            "strategy,recall_at_k,mrr_at_k,top_k\nlexical_tfidf,nan,0.5,3\n",
        ),
    ],
)
def test_verifier_rejects_empty_or_corrupt_content(
    tmp_path: Path, relative_path: str, content: str
) -> None:
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    assert verify_required_artifacts(tmp_path, [relative_path]) == [relative_path]


def test_verifier_rejects_directory_instead_of_file(tmp_path: Path) -> None:
    relative_path = "artifacts/summary.md"
    (tmp_path / relative_path).mkdir(parents=True)
    assert verify_required_artifacts(tmp_path, [relative_path]) == [relative_path]
