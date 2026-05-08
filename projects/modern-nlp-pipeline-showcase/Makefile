SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help sync lock run smoke verify test ruff ty check

help:
	@echo "Modern NLP Pipeline Showcase"
	@echo "  make sync    Install dependencies"
	@echo "  make run     Run the full pipeline"
	@echo "  make smoke   Run the quick pipeline"
	@echo "  make verify  Verify artifacts"
	@echo "  make test    Run pytest"
	@echo "  make ruff    Run Ruff"
	@echo "  make ty      Run mypy"
	@echo "  make check   Run Ruff + mypy + pytest"

sync:
	uv sync --extra dev

lock:
	uv lock

run:
	uv run python scripts/run_pipeline.py

smoke:
	uv run python scripts/run_pipeline.py --quick

verify:
	uv run python scripts/verify_artifacts.py

test:
	uv run pytest

ruff:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

ty:
	uv run mypy src tests scripts

check: ruff ty test verify
