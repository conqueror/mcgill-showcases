from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    labelnames=("method", "route", "status"),
)

REQUEST_LATENCY_SECONDS = Histogram(
    "http_request_latency_seconds",
    "HTTP request latency in seconds",
    labelnames=("method", "route"),
)


def metrics_endpoint() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
