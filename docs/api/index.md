# API Reference

This section documents the two FastAPI showcase services in this repository.

## APIs

- Ranking API: `projects/ranking-api-productization-showcase`
- Demand API: `projects/demand-api-observability-showcase`

## OpenAPI Assets Used By This Site

- `docs/api/assets/openapi/ranking-api.json`
- `docs/api/assets/openapi/demand-api.json`

These files are refreshed in the GitHub Pages workflow before docs are built.

## Refresh Locally

```bash
make -C projects/ranking-api-productization-showcase export-openapi
make -C projects/demand-api-observability-showcase export-openapi
cp projects/ranking-api-productization-showcase/openapi.json docs/api/assets/openapi/ranking-api.json
cp projects/demand-api-observability-showcase/openapi.json docs/api/assets/openapi/demand-api.json
make docs-check
```

## Pages

- [Ranking API reference](ranking-api.md)
- [Demand API reference](demand-api.md)
