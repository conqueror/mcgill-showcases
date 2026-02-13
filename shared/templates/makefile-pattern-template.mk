SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help sync run test ruff ty check smoke verify

help:
	@echo "Project commands"

sync:
	uv sync --extra dev

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

