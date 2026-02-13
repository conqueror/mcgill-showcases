# Demand API

Project: `projects/demand-api-observability-showcase`

## Summary

This API serves demand predictions from the demo model bundle used in the observability showcase.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Service health check |
| `POST` | `/predict` | Demand inference endpoint |

## OpenAPI

- Download schema: <a id="demand-openapi-link" href="../assets/openapi/demand-api.json">demand-api.json</a>
- Source schema in repo: `docs/api/assets/openapi/demand-api.json`

## Local Commands

```bash
cd projects/demand-api-observability-showcase
make export-openapi
make dev
```

## Example Requests And Responses

Base URL for local runs:

```text
http://127.0.0.1:8000
```

### `GET /health`

```bash
curl -s http://127.0.0.1:8000/health
```

```json
{
  "status": "ok",
  "version": "0.1.0"
}
```

### `POST /predict`

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "content-type: application/json" \
  -d '{
    "pickup_zone_id": 132,
    "pickup_datetime": "2026-02-13T09:00:00Z"
  }'
```

```json
{
  "pickup_zone_id": 132,
  "pickup_datetime": "2026-02-13T09:00:00Z",
  "predicted_pickups": 41.7,
  "model_version": "demo-v1"
}
```

### `GET /metrics`

This endpoint is part of the observability workflow and exposes Prometheus-formatted metrics.

```bash
curl -s http://127.0.0.1:8000/metrics | head -n 12
```

```text
# HELP http_requests_total Total HTTP requests by method and route.
# TYPE http_requests_total counter
http_requests_total{method="GET",route="/health"} 3.0
http_requests_total{method="POST",route="/predict"} 2.0
# HELP http_request_latency_seconds Request latency in seconds by method and route.
# TYPE http_request_latency_seconds histogram
```

## ReDoc Viewer

<div id="redoc-demand"></div>
<script src="https://cdn.jsdelivr.net/npm/redoc@2.1.3/bundles/redoc.standalone.js"></script>
<script>
  (function () {
    const target = document.getElementById("redoc-demand");
    const specAnchor = document.getElementById("demand-openapi-link");

    if (!target || !specAnchor) {
      return;
    }
    if (!window.Redoc || typeof window.Redoc.init !== "function") {
      target.innerHTML =
        "<p><strong>Unable to load embedded ReDoc.</strong> Use the OpenAPI JSON download link above.</p>";
      return;
    }

    try {
      window.Redoc.init(specAnchor.href, { hideDownloadButton: false }, target);
    } catch (_error) {
      target.innerHTML =
        "<p><strong>Unable to load embedded ReDoc.</strong> Use the OpenAPI JSON download link above.</p>";
    }
  })();
</script>
