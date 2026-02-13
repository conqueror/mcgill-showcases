SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help sync sync-tracking run run-tracking run-drift serve test ruff ty check smoke verify

help:
	@echo "MLOps Drift Production Showcase"
	@echo "  make sync       Install dependencies"
	@echo "  make sync-tracking Install optional MLflow tracking extras"
	@echo "  make run        Run training pipeline"
	@echo "  make run-tracking Run training pipeline with MLflow tracking"
	@echo "  make run-drift  Run drift monitor"
	@echo "  make serve      Start local API service"
	@echo "  make test       Run tests"
	@echo "  make ruff       Run Ruff lint"
	@echo "  make ty         Run mypy type checks"
	@echo "  make check      Run lint + type + tests"
	@echo "  make smoke      Run quick pipeline and drift checks"
	@echo "  make verify     Verify artifacts"

sync:
	uv sync --extra dev

sync-tracking:
	uv sync --extra dev --extra tracking

run:
	uv run python scripts/run_pipeline.py

run-tracking:
	uv run python scripts/run_pipeline.py --with-mlflow

run-drift:
	uv run python scripts/run_drift_monitor.py

serve:
	bash scripts/run_service_local.sh

test:
	uv run pytest

ruff:
	uv run ruff check src tests scripts

ty:
	uv run mypy src scripts tests

check: ruff ty test

smoke:
	uv run python scripts/run_pipeline.py --quick
	uv run python scripts/run_drift_monitor.py --quick

verify:
	uv run python scripts/verify_artifacts.py
