SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help sync run test ruff ty check smoke verify

help:
	@echo "Model Release Rollout Showcase"
	@echo "  make sync    Install dependencies"
	@echo "  make run     Run rollout simulation"
	@echo "  make test    Run tests"
	@echo "  make check   Run lint + type + tests"
	@echo "  make smoke   Run quick simulation"
	@echo "  make verify  Verify artifacts"

sync:
	uv sync --extra dev

run:
	uv run python scripts/run_rollout.py

test:
	uv run pytest

ruff:
	uv run ruff check src tests scripts

ty:
	uv run mypy src tests scripts

check: ruff ty test

smoke:
	uv run python scripts/run_rollout.py --quick

verify:
	uv run python scripts/verify_artifacts.py
