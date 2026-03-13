SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help sync lock run smoke verify test ruff ruff-format ruff-fix ty lint format quality check

help:
	@echo "Available targets:"
	@echo "  make sync         - Install project + dev dependencies with uv"
	@echo "  make lock         - Refresh uv.lock"
	@echo "  make run          - Run the showcase pipeline"
	@echo "  make smoke        - Alias for run"
	@echo "  make verify       - Verify generated artifacts"
	@echo "  make test         - Run pytest"
	@echo "  make ruff         - Run Ruff lint checks"
	@echo "  make ruff-format  - Check Ruff formatting"
	@echo "  make ruff-fix     - Auto-fix Ruff issues"
	@echo "  make ty           - Run type checks with ty"
	@echo "  make lint         - Run Ruff + ty"
	@echo "  make format       - Apply formatter fixes"
	@echo "  make quality      - Run the local quality gate"
	@echo "  make check        - Run lint + tests + verify"

sync:
	uv sync --extra dev

lock:
	uv lock

run:
	uv run python scripts/run_showcase.py

smoke: run

verify:
	uv run python scripts/verify_artifacts.py

test:
	uv run pytest

ruff:
	uv run ruff check src tests scripts

ruff-format:
	uv run ruff format --check src tests scripts

ruff-fix:
	uv run ruff check --fix src tests scripts
	uv run ruff format src tests scripts

ty:
	uv run ty check src tests scripts --ignore unresolved-import --ignore unresolved-attribute

lint: ruff ruff-format ty

format: ruff-fix

quality:
	./scripts/run_quality_checks.sh

check: lint test verify
