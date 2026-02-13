SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help sync run test ruff ty check smoke verify

help:
	@echo "Learning-to-Rank Foundations Showcase"
	@echo "  make sync    Install dependencies"
	@echo "  make run     Run grouped ranking pipeline"
	@echo "  make smoke   Run quick ranking pipeline"
	@echo "  make verify  Verify artifacts"

sync:
	uv sync --extra dev

run:
	uv run python scripts/run_pipeline.py

smoke:
	uv run python scripts/run_pipeline.py --quick

test:
	uv run pytest

ruff:
	uv run ruff check src tests scripts

ty:
	uv run mypy src tests scripts

check: ruff ty test

verify:
	uv run python scripts/verify_artifacts.py
