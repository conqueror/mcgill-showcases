# Troubleshooting

## `uv: command not found`
Install `uv` and reopen your terminal.

## Dependency resolution fails
From project directory:
```bash
uv lock
uv sync --extra dev
```

## Kaggle auth error (causal project)
Check:
```bash
ls -l ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

## Notebook doesn't open or validate
Reinstall notebook dependencies and validate JSON format:
```bash
python -m pip install nbformat
```

## MkDocs build fails
Run strict docs build to surface the first error:
```bash
make docs-check
```
If dependency resolution fails, retry:
```bash
uv run --with-requirements docs/requirements-mkdocs.txt mkdocs build --strict
```

## OpenAPI check fails in API showcases
If `openapi-check` reports drift, regenerate and re-check:
```bash
cd projects/<api-showcase>
make export-openapi
make openapi-check
```

## API docs page shows stale schema
Refresh project OpenAPI files and copy them into docs assets:
```bash
make -C projects/ranking-api-productization-showcase export-openapi
make -C projects/demand-api-observability-showcase export-openapi
cp projects/ranking-api-productization-showcase/openapi.json docs/api/assets/openapi/ranking-api.json
cp projects/demand-api-observability-showcase/openapi.json docs/api/assets/openapi/demand-api.json
make docs-check
```

## GitHub Pages site is not updating
Check that:

- GitHub Pages source is set to `GitHub Actions` in repository settings.
- The `Docs Pages` workflow completed successfully on `main`.
- `make docs-check` passes locally before pushing.

## TLC demand run fails due to missing data
For `nyc-demand-forecasting-foundations-showcase`, download sample TLC files first:
```bash
cd projects/nyc-demand-forecasting-foundations-showcase
make download-data
make run-real
```
If network access is unavailable, use synthetic mode:
```bash
make run
```

## Tests fail only on one project
Run commands inside that project to isolate:
```bash
cd projects/<project-name>
make check
```

## `make check-contracts` takes longer than expected
This command may generate missing artifacts before validating contracts. This is expected on a fresh checkout.

To isolate one supervised project:
```bash
cd projects/<project-name>
make smoke
make verify
```

## Contract verification fails in one supervised project
Regenerate local artifacts and retry:
```bash
cd projects/<project-name>
make run
make verify
```

Then rerun at root:
```bash
cd /path/to/mcgill-showcases
make check-contracts
```
