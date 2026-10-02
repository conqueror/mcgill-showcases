from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI
from fastapi import Request as FastAPIRequest
from opentelemetry import trace
from starlette.responses import JSONResponse, Response

from demand_api_observability_showcase.api.routes.health import router as health_router
from demand_api_observability_showcase.api.routes.predict import router as predict_router
from demand_api_observability_showcase.model.store import ModelStore
from demand_api_observability_showcase.observability.logging import init_logging
from demand_api_observability_showcase.observability.metrics import (
    REQUEST_LATENCY_SECONDS,
    REQUESTS_TOTAL,
    metrics_endpoint,
)
from demand_api_observability_showcase.observability.otel import OTelConfig, init_otel
from demand_api_observability_showcase.settings import Settings, get_settings

# FastAPI requires the concrete Request class when evaluating route annotations.
if TYPE_CHECKING:
    Request = FastAPIRequest[Any]
else:
    Request = FastAPIRequest

logger = logging.getLogger("demand_api_observability_showcase")
NextCall = Callable[[Request], Awaitable[Response]]


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    model_store: ModelStore = app.state.model_store
    model_store.load()
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    init_logging(settings.log_level)

    app = FastAPI(
        title="Demand API Observability Showcase",
        version="0.1.0",
        lifespan=_lifespan,
    )

    app.state.model_store = ModelStore(settings.model_path)

    if settings.prometheus_enabled:
        app.add_api_route("/metrics", metrics_endpoint, methods=["GET"], include_in_schema=False)

    init_otel(
        app,
        OTelConfig(
            enabled=settings.otel_enabled,
            service_name=settings.otel_service_name,
            otlp_endpoint=settings.otel_exporter_otlp_endpoint,
        ),
    )

    @app.middleware("http")
    async def request_observability(request: Request, call_next: NextCall) -> Response:
        start = time.perf_counter()
        span = trace.get_current_span()
        context = span.get_span_context()
        otel_trace_id = f"{context.trace_id:032x}" if context.is_valid else None
        trace_id = request.headers.get("x-trace-id") or otel_trace_id or uuid.uuid4().hex
        if context.is_valid:
            span.set_attribute("request.correlation_id", trace_id)
        request.state.trace_id = trace_id

        try:
            response = await call_next(request)
        except Exception:
            logger.exception(json.dumps({"event": "request_failed", "trace_id": trace_id}))
            response = JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

        latency_ms = (time.perf_counter() - start) * 1000.0
        route = getattr(request.scope.get("route"), "path", "unmatched")
        status = str(response.status_code)
        method = request.method

        REQUESTS_TOTAL.labels(method=method, route=route, status=status).inc()
        REQUEST_LATENCY_SECONDS.labels(method=method, route=route).observe(latency_ms / 1000.0)

        logger.info(
            json.dumps(
                {
                    "trace_id": trace_id,
                    "otel_trace_id": otel_trace_id,
                    "tenant_id": None,
                    "route": route,
                    "method": method,
                    "status": response.status_code,
                    "latency_ms": round(latency_ms, 3),
                }
            )
        )

        response.headers["x-trace-id"] = trace_id
        return response

    app.include_router(health_router)
    app.include_router(predict_router)
    return app


app = create_app()
