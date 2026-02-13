# Ranking API

Project: `projects/ranking-api-productization-showcase`

## Summary

This API serves ranking-related inference for grouped candidate items.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Service health check |
| `GET` | `/model/schema` | Model feature/schema metadata |
| `POST` | `/score` | Score individual candidate rows |
| `POST` | `/rank` | Rank a batch of candidates |
| `POST` | `/predict` | Compatibility alias for ranked predictions |

## OpenAPI

- Download schema: <a id="ranking-openapi-link" href="assets/openapi/ranking-api.json">ranking-api.json</a>
- Source schema in repo: `docs/api/assets/openapi/ranking-api.json`

## Local Commands

```bash
cd projects/ranking-api-productization-showcase
make export-openapi
make dev
```

## ReDoc Viewer

<div id="redoc-ranking"></div>
<script src="https://cdn.jsdelivr.net/npm/redoc@2.1.3/bundles/redoc.standalone.js"></script>
<script>
  (function () {
    const target = document.getElementById("redoc-ranking");
    const specAnchor = document.getElementById("ranking-openapi-link");

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
