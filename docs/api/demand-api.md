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
