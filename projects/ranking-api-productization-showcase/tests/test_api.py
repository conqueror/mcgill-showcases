from __future__ import annotations

import asyncio
import importlib
import io
import json
import logging
import runpy
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest
from fastapi.testclient import TestClient
from numpy.typing import NDArray
from pytest import MonkeyPatch

from ranking_api_showcase.api.app import create_app
from ranking_api_showcase.config import Settings, load_settings
from ranking_api_showcase.logging import JsonFormatter
from ranking_api_showcase.model.artifacts import load_artifacts
from ranking_api_showcase.model.scoring import score as original_score


def _write_test_artifacts(tmp_path: Path, feature_names: list[str] | None = None) -> Settings:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "model.txt"
    feature_names_path = artifacts_dir / "feature_names.json"
    meta_path = artifacts_dir / "model_meta.json"

    feature_names = feature_names or ["x"]
    matrix = np.asarray([[float(i)] * len(feature_names) for i in range(5)], dtype=np.float64)
    target = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    train_set = lgb.Dataset(matrix, label=target, feature_name=feature_names)
    booster = lgb.train(
        params={
            "objective": "regression",
            "metric": "l2",
            "verbosity": -1,
            "seed": 0,
            "monotone_constraints": [1] * len(feature_names),
            "min_data_in_leaf": 1,
        },
        train_set=train_set,
        num_boost_round=20,
    )
    booster.save_model(str(model_path))

    feature_names_path.write_text(json.dumps(feature_names), encoding="utf-8")
    meta_path.write_text(json.dumps({"kind": "test"}), encoding="utf-8")

    return Settings(
        log_level="CRITICAL",
        model_path=model_path,
        feature_names_path=feature_names_path,
        model_meta_path=meta_path,
        fail_on_model_load=True,
    )


def test_health_reports_model_loaded(tmp_path: Path) -> None:
    settings = _write_test_artifacts(tmp_path)
    client = TestClient(create_app(settings=settings))

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_rank_sorts_by_score(tmp_path: Path) -> None:
    settings = _write_test_artifacts(tmp_path)
    client = TestClient(create_app(settings=settings))

    response = client.post(
        "/rank",
        json={
            "records": [
                {"player_id": "p1", "features": {"x": 0.0}},
                {"player_id": "p2", "features": {"x": 4.0}},
            ]
        },
    )
    assert response.status_code == 200
    payload = response.json()

    assert [row["player_id"] for row in payload["rankings"]] == ["p2", "p1"]
    assert [row["rank"] for row in payload["rankings"]] == [1, 2]


def test_unexpected_feature_rejected(tmp_path: Path) -> None:
    settings = _write_test_artifacts(tmp_path)
    client = TestClient(create_app(settings=settings))

    response = client.post(
        "/score",
        json={"records": [{"player_id": "p1", "features": {"x": 1.0, "y": 2.0}}]},
    )
    assert response.status_code == 400
    assert "unexpected feature keys" in response.json()["detail"]


def test_schema_endpoint_exposes_expected_features(tmp_path: Path) -> None:
    settings = _write_test_artifacts(tmp_path)
    client = TestClient(create_app(settings=settings))

    response = client.get("/model/schema")
    assert response.status_code == 200
    assert response.json()["feature_names"] == ["x"]


def test_score_success(tmp_path: Path) -> None:
    settings = _write_test_artifacts(tmp_path)
    client = TestClient(create_app(settings=settings))

    response = client.post(
        "/score",
        json={"records": [{"player_id": "p1", "features": {"x": 1.0}}]},
        headers={"x-trace-id": "t-123"},
    )
    assert response.status_code == 200
    assert response.headers["x-trace-id"] == "t-123"
    assert response.json()["scores"][0]["player_id"] == "p1"


def test_predict_alias(tmp_path: Path) -> None:
    settings = _write_test_artifacts(tmp_path)
    client = TestClient(create_app(settings=settings))

    response = client.post(
        "/predict",
        json={"records": [{"player_id": "p1", "features": {"x": 1.0}}]},
    )
    assert response.status_code == 200


def test_model_not_loaded_returns_503() -> None:
    settings = Settings(log_level="CRITICAL", fail_on_model_load=False)
    client = TestClient(create_app(settings=settings, load_model=False))

    response = client.post(
        "/score",
        json={"records": [{"player_id": "p1", "features": {"x": 1.0}}]},
    )
    assert response.status_code == 503


def test_middleware_exception_path(tmp_path: Path) -> None:
    settings = _write_test_artifacts(tmp_path)
    app = create_app(settings=settings)

    @app.get("/boom")
    async def boom() -> dict[str, str]:
        raise RuntimeError("boom")

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/boom")
    assert response.status_code == 500
    assert "x-trace-id" in response.headers


def test_json_formatter_serializes_extras() -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())

    test_logger = logging.getLogger("test_json_formatter")
    test_logger.setLevel(logging.INFO)
    test_logger.handlers = [handler]
    test_logger.propagate = False

    test_logger.info(
        "hello",
        extra={
            "trace_id": "t1",
            "payload": {"a": 1},
            "items": [1, 2, 3],
            "_skip_me": "nope",
            "obj": object(),
        },
    )

    data = json.loads(stream.getvalue())
    assert data["trace_id"] == "t1"
    assert data["payload"]["a"] == 1
    assert data["items"] == [1, 2, 3]
    assert "_skip_me" not in data


def test_json_formatter_includes_exc_info() -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())

    test_logger = logging.getLogger("test_json_formatter_exc")
    test_logger.setLevel(logging.INFO)
    test_logger.handlers = [handler]
    test_logger.propagate = False

    try:
        raise ValueError("boom")
    except ValueError:
        test_logger.exception("caught", extra={"trace_id": "t2"})

    data = json.loads(stream.getvalue().splitlines()[-1])
    assert data["trace_id"] == "t2"
    assert "exc_info" in data


def test_load_artifacts_handles_missing_meta(tmp_path: Path) -> None:
    settings = _write_test_artifacts(tmp_path)
    settings.model_meta_path.unlink()

    artifacts = load_artifacts(
        model_path=settings.model_path,
        feature_names_path=settings.feature_names_path,
        meta_path=settings.model_meta_path,
    )
    assert artifacts.meta is None


def test_load_settings_defaults() -> None:
    settings = load_settings()
    assert settings.model_path.name == "model.txt"


def test_api_main_module_creates_app(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    settings = _write_test_artifacts(tmp_path)
    monkeypatch.setenv("RANK_API_LOG_LEVEL", "CRITICAL")
    monkeypatch.setenv("RANK_API_MODEL_PATH", str(settings.model_path))
    monkeypatch.setenv("RANK_API_FEATURE_NAMES_PATH", str(settings.feature_names_path))
    monkeypatch.setenv("RANK_API_MODEL_META_PATH", str(settings.model_meta_path))
    monkeypatch.setenv("RANK_API_FAIL_ON_MODEL_LOAD", "true")

    sys.modules.pop("ranking_api_showcase.api.main", None)
    module = importlib.import_module("ranking_api_showcase.api.main")

    assert hasattr(module, "app")
    assert getattr(module.app.state, "artifacts", None) is not None


def test_load_artifacts_rejects_invalid_feature_names(tmp_path: Path) -> None:
    settings = _write_test_artifacts(tmp_path)
    settings.feature_names_path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")

    try:
        load_artifacts(
            model_path=settings.model_path,
            feature_names_path=settings.feature_names_path,
            meta_path=settings.model_meta_path,
        )
    except ValueError as exc:
        assert "Invalid feature schema" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_load_artifacts_requires_files(tmp_path: Path) -> None:
    settings = _write_test_artifacts(tmp_path)
    settings.model_path.unlink()

    try:
        load_artifacts(
            model_path=settings.model_path,
            feature_names_path=settings.feature_names_path,
            meta_path=settings.model_meta_path,
        )
    except FileNotFoundError as exc:
        assert "Model file not found" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")


@pytest.mark.parametrize("endpoint", ["/score", "/predict", "/rank"])
def test_missing_feature_is_rejected(tmp_path: Path, endpoint: str) -> None:
    client = TestClient(create_app(settings=_write_test_artifacts(tmp_path)))
    response = client.post(endpoint, json={"records": [{"player_id": "p1", "features": {}}]})
    assert response.status_code == 400
    assert "missing feature keys" in response.json()["detail"]


def test_omitted_features_are_rejected() -> None:
    client = TestClient(create_app(load_model=False))
    response = client.post("/score", json={"records": [{"player_id": "p1"}]})
    assert response.status_code == 422


@pytest.mark.parametrize("names", [["y", "x"], ["x", "x"], ["x"], ["x", "y", "z"]])
def test_artifact_schema_must_match_booster(tmp_path: Path, names: list[str]) -> None:
    settings = _write_test_artifacts(tmp_path, ["x", "y"])
    settings.feature_names_path.write_text(json.dumps(names))
    with pytest.raises(ValueError, match="schema|duplicate"):
        load_artifacts(settings.model_path, settings.feature_names_path, settings.model_meta_path)


def test_request_batch_size_is_bounded() -> None:
    client = TestClient(create_app(load_model=False))
    response = client.post(
        "/score", json={"records": [{"player_id": "p", "features": {"x": 1.0}}] * 1001}
    )
    assert response.status_code == 422


@pytest.mark.parametrize("endpoint", ["/score", "/predict", "/rank"])
def test_prediction_runs_outside_the_event_loop(
    tmp_path: Path, monkeypatch: MonkeyPatch, endpoint: str
) -> None:
    module = importlib.import_module("ranking_api_showcase.api.app")

    def checked_score(booster: lgb.Booster, matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        return original_score(booster, matrix)

    monkeypatch.setattr(module, "score", checked_score)
    client = TestClient(create_app(settings=_write_test_artifacts(tmp_path)))
    response = client.post(endpoint, json={"records": [{"player_id": "p", "features": {"x": 1.0}}]})
    assert response.status_code == 200


def test_openapi_declares_error_responses() -> None:
    spec = create_app(load_model=False).openapi()
    for endpoint in ("/score", "/predict", "/rank"):
        assert {"400", "503"} <= set(spec["paths"][endpoint]["post"]["responses"])
    assert "503" in spec["paths"]["/model/schema"]["get"]["responses"]


@pytest.mark.parametrize("filename", ["model.txt", "feature_names.json", "model_meta.json"])
def test_verifier_rejects_corrupt_artifacts(
    tmp_path: Path, monkeypatch: MonkeyPatch, filename: str
) -> None:
    _write_test_artifacts(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts/verify_artifacts.py"
    verify = runpy.run_path(str(script))["main"]
    monkeypatch.setitem(
        verify.__globals__, "__file__", str(tmp_path / "scripts/verify_artifacts.py")
    )
    verify()
    (tmp_path / "artifacts" / filename).write_text("broken artifact")
    with pytest.raises(SystemExit):
        verify()
