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

- Download schema: <a id="demand-openapi-link" href="assets/openapi/demand-api.json">demand-api.json</a>
- Source schema in repo: `docs/api/assets/openapi/demand-api.json`

## Local Commands

```bash
cd projects/demand-api-observability-showcase
make export-openapi
make dev
```

## ReDoc Viewer

<div id="redoc-demand"></div>
<script src="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js"></script>
<script>
  (function () {
    const specUrl = document.getElementById("demand-openapi-link").href;
    Redoc.init(specUrl, { hideDownloadButton: false }, document.getElementById("redoc-demand"));
  })();
</script>
