SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help sync train-demo dev test ruff ty check export-openapi smoke verify

help:
	@echo "Ranking API Productization Showcase"
	@echo "  make sync          Install dependencies"
	@echo "  make train-demo    Train synthetic ranking model artifacts"
	@echo "  make dev           Run FastAPI service"
	@echo "  make export-openapi Export OpenAPI JSON"
	@echo "  make smoke         Train + API tests"
	@echo "  make verify        Verify artifacts"

sync:
	uv sync --extra dev

train-demo:
	uv run python scripts/train_demo_model.py

dev:
	uv run uvicorn ranking_api_showcase.api.main:app --host 127.0.0.1 --port 8000 --reload

export-openapi:
	uv run python scripts/export_openapi.py --out openapi.json

test:
	uv run pytest

ruff:
	uv run ruff check src tests scripts

ty:
	uv run mypy src tests scripts

check: ruff ty test

smoke:
	uv run python scripts/train_demo_model.py
	uv run pytest tests/test_api.py

verify:
	uv run python scripts/verify_artifacts.py
