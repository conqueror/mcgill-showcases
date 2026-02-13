from __future__ import annotations

from dataclasses import dataclass

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


@dataclass(frozen=True)
class OTelConfig:
    enabled: bool
    service_name: str
    otlp_endpoint: str | None


def init_otel(app: FastAPI, config: OTelConfig) -> None:
    if not config.enabled:
        return

    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.requests import RequestsInstrumentor
    except ModuleNotFoundError:
        return

    resource = Resource.create({"service.name": config.service_name})
    provider = TracerProvider(resource=resource)

    if config.otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter(endpoint=config.otlp_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    RequestsInstrumentor().instrument()
    FastAPIInstrumentor.instrument_app(app)
