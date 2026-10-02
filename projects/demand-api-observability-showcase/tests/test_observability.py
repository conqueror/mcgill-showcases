from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from prometheus_client import REGISTRY
from pytest import LogCaptureFixture

from demand_api_observability_showcase.api.app import create_app
from demand_api_observability_showcase.settings import Settings


def test_failed_request_is_counted_logged_and_correlated(
    tmp_path: Path, caplog: LogCaptureFixture
) -> None:
    app = create_app(Settings(model_path=tmp_path / "missing.joblib"))

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("inference failed")

    labels = {"method": "GET", "route": "/boom", "status": "500"}
    before = REGISTRY.get_sample_value("http_requests_total", labels) or 0
    latency_labels = {"method": "GET", "route": "/boom"}
    latency_before = (
        REGISTRY.get_sample_value("http_request_latency_seconds_count", latency_labels) or 0
    )
    with caplog.at_level(logging.INFO), TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/boom", headers={"x-trace-id": "failed-request"})
    assert response.status_code == 500
    assert response.headers["x-trace-id"] == "failed-request"
    assert REGISTRY.get_sample_value("http_requests_total", labels) == before + 1
    assert (
        REGISTRY.get_sample_value("http_request_latency_seconds_count", latency_labels)
        == latency_before + 1
    )
    events = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == "demand_api_observability_showcase" and record.message.startswith("{")
    ]
    assert any(
        event.get("status") == 500 and event["trace_id"] == "failed-request" for event in events
    )


def test_metrics_use_route_templates_and_one_unmatched_label(tmp_path: Path) -> None:
    app = create_app(Settings(model_path=tmp_path / "missing.joblib"))

    @app.get("/items/{item_id}")
    def item(item_id: str) -> dict[str, str]:
        return {"item_id": item_id}

    matched = {"method": "GET", "route": "/items/{item_id}", "status": "200"}
    unmatched = {"method": "GET", "route": "unmatched", "status": "404"}
    before_matched = REGISTRY.get_sample_value("http_requests_total", matched) or 0
    before_unmatched = REGISTRY.get_sample_value("http_requests_total", unmatched) or 0
    with TestClient(app) as client:
        for path in ("/items/one", "/items/two", "/unknown-one", "/unknown-two"):
            client.get(path)
    assert REGISTRY.get_sample_value("http_requests_total", matched) == before_matched + 2
    assert REGISTRY.get_sample_value("http_requests_total", unmatched) == before_unmatched + 2


def test_request_correlation_is_linked_to_active_otel_span(
    tmp_path: Path, caplog: LogCaptureFixture
) -> None:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    span = provider.get_tracer("test").start_span("incoming")
    app = create_app(Settings(model_path=tmp_path / "missing.joblib"))
    with (
        caplog.at_level(logging.INFO),
        trace.use_span(span, end_on_exit=True),
        TestClient(app) as client,
    ):
        response = client.get("/health", headers={"x-trace-id": "client-correlation"})
    assert response.headers["x-trace-id"] == "client-correlation"
    recorded = exporter.get_finished_spans()[0]
    assert recorded.attributes is not None
    assert recorded.attributes["request.correlation_id"] == "client-correlation"
    expected_trace = f"{span.get_span_context().trace_id:032x}"
    events = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == "demand_api_observability_showcase" and record.message.startswith("{")
    ]
    assert any(event.get("otel_trace_id") == expected_trace for event in events)


def test_openapi_declares_missing_model_response() -> None:
    spec = create_app().openapi()
    assert "503" in spec["paths"]["/predict"]["post"]["responses"]
