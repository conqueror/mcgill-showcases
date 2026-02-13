SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help sync sync-boosting lock run test ruff-check ruff-format ruff-fix ty-check lint format check quality ci

help:
	@echo "Available targets:"
	@echo "  make sync           - Install project + dev dependencies with uv"
	@echo "  make sync-boosting  - Install project + dev + boosting dependencies"
	@echo "  make lock           - Refresh uv.lock"
	@echo "  make run            - Run the tutorial pipeline"
	@echo "  make test           - Run pytest"
	@echo "  make ruff-check     - Run Ruff lint checks"
	@echo "  make ruff-format    - Check Ruff formatting"
	@echo "  make ruff-fix       - Auto-fix Ruff lint/format issues"
	@echo "  make ty-check       - Run type checks with ty"
	@echo "  make lint           - Run Ruff + ty"
	@echo "  make format         - Run formatter fixes"
	@echo "  make quality        - Run full local quality gate"
	@echo "  make ci             - Alias for quality checks"
	@echo "  make check          - lint + tests + pipeline"

sync:
	uv sync --extra dev

sync-boosting:
	uv sync --extra dev --extra boosting

lock:
	uv lock

run:
	uv run python scripts/run_showcase.py

test:
	uv run pytest

ruff-check:
	uv run ruff check src tests scripts/run_showcase.py

ruff-format:
	uv run ruff format --check src tests scripts/run_showcase.py

ruff-fix:
	uv run ruff check --fix src tests scripts/run_showcase.py
	uv run ruff format src tests scripts/run_showcase.py

ty-check:
	uv run ty check src tests scripts/run_showcase.py \
		--ignore unresolved-import \
		--ignore unresolved-attribute

lint: ruff-check ruff-format ty-check

format: ruff-fix

quality:
	./scripts/run_quality_checks.sh

ci: quality

check: lint test run
