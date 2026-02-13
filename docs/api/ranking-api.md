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
<script src="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js"></script>
<script>
  (function () {
    const specUrl = document.getElementById("ranking-openapi-link").href;
    Redoc.init(specUrl, { hideDownloadButton: false }, document.getElementById("redoc-ranking"));
  })();
</script>
