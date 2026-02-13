# Demand API Observability Showcase

FastAPI demand-serving showcase focused on observability and contract-first API patterns.

## Learning outcomes
- Serve demand predictions from a model bundle.
- Add request metrics with Prometheus (`/metrics`).
- Add request-level trace IDs and structured JSON logs.
- Enable optional OpenTelemetry instrumentation.
- Enforce contract drift checks via exported OpenAPI.

## Quickstart
```bash
cd projects/demand-api-observability-showcase
make sync
make train-demo
make test
make export-openapi
make verify
```

Run API:
```bash
make dev
```

## Key outputs
- `artifacts/model.joblib`
- `artifacts/metrics.json`
- `openapi.json`
