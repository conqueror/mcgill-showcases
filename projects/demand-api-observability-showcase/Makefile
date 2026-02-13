SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help sync train-demo dev export-openapi openapi-check test ruff ty check smoke verify

help:
	@echo "Demand API Observability Showcase"
	@echo "  make sync          Install dependencies"
	@echo "  make train-demo    Train and write demo model bundle"
	@echo "  make dev           Run FastAPI service"
	@echo "  make export-openapi Export OpenAPI to openapi.json"
	@echo "  make openapi-check Check OpenAPI drift"
	@echo "  make smoke         Quick integration smoke"
	@echo "  make verify        Verify required artifacts"

sync:
	uv sync --extra dev

train-demo:
	uv run python scripts/train_demo_model.py

dev:
	uv run uvicorn demand_api_observability_showcase.api.main:app --host 127.0.0.1 --port 8000 --reload

export-openapi:
	uv run python scripts/export_openapi.py

openapi-check:
	uv run python scripts/export_openapi.py --check

test:
	uv run pytest

ruff:
	uv run ruff check src tests scripts

ty:
	uv run mypy src tests scripts

check: ruff ty test

smoke:
	uv run python scripts/train_demo_model.py
	uv run pytest tests/test_predict_with_model.py tests/test_predict_no_model.py tests/test_health.py
	uv run python scripts/export_openapi.py
	uv run python scripts/export_openapi.py --check

verify:
	uv run python scripts/verify_artifacts.py
