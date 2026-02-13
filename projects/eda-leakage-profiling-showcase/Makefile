SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help sync sync-profiling run test ruff ty check smoke verify

help:
	@echo "EDA Leakage Profiling Showcase"
	@echo "  make sync           Install dependencies"
	@echo "  make sync-profiling Install optional profiling extras"
	@echo "  make run            Run EDA + leakage workflow"
	@echo "  make test           Run tests"
	@echo "  make check          Run lint + type + tests"
	@echo "  make smoke          Run quick workflow"
	@echo "  make verify         Verify artifacts"

sync:
	uv sync --extra dev

sync-profiling:
	uv sync --extra dev --extra profiling

run:
	uv run python scripts/run_pipeline.py

test:
	uv run pytest

ruff:
	uv run ruff check src tests scripts

ty:
	uv run mypy src tests scripts

check: ruff ty test

smoke:
	uv run python scripts/run_pipeline.py --quick

verify:
	uv run python scripts/verify_artifacts.py
