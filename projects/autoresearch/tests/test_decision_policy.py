from __future__ import annotations

from autoresearch_showcase.decision_policy import recommend_status


def test_keep_for_clear_improvement() -> None:
    status, rationale = recommend_status(0.9989, 0.9975, complexity_delta=0)
    assert status == "keep"
    assert "improved" in rationale.lower() or "meaningful" in rationale.lower()


def test_keep_for_flat_but_simpler_code() -> None:
    status, rationale = recommend_status(0.9962, 0.9962, complexity_delta=-2)
    assert status == "keep"
    assert "simpler" in rationale.lower()


def test_crash_is_logged_but_not_kept() -> None:
    status, rationale = recommend_status(0.9962, 0.0, complexity_delta=4, crashed=True)
    assert status == "crash"
    assert "crashed" in rationale.lower()
