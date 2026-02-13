SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help sync sync-advanced run run-advanced run-budget test ruff ty check smoke verify

help:
	@echo "AutoML HPO Showcase"
	@echo "  make sync        Install dependencies"
	@echo "  make sync-advanced Install optional Hyperopt + MLflow extras"
	@echo "  make run         Run strategy benchmark"
	@echo "  make run-advanced Run benchmark with Hyperopt + MLflow logging"
	@echo "  make run-budget  Run budget sensitivity experiment"
	@echo "  make test        Run tests"
	@echo "  make ruff        Run Ruff"
	@echo "  make ty          Run mypy"
	@echo "  make check       Run lint + type + tests"
	@echo "  make smoke       Run quick benchmark"
	@echo "  make verify      Verify artifacts"

sync:
	uv sync --extra dev

sync-advanced:
	uv sync --extra dev --extra advanced

run:
	uv run python scripts/run_hpo_benchmark.py

run-advanced:
	uv run python scripts/run_hpo_benchmark.py --with-hyperopt --with-mlflow

run-budget:
	uv run python scripts/run_budget_sensitivity.py

test:
	uv run pytest

ruff:
	uv run ruff check src tests scripts

ty:
	uv run mypy src tests scripts

check: ruff ty test

smoke:
	uv run python scripts/run_hpo_benchmark.py --quick
	uv run python scripts/run_budget_sensitivity.py --quick

verify:
	uv run python scripts/verify_artifacts.py
