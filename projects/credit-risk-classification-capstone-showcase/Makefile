SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help sync sync-profiling sync-boosters run test ruff ty check smoke verify

help:
	@echo "Credit Risk Classification Capstone Showcase"
	@echo "  make sync          Install dependencies"
	@echo "  make sync-profiling Install optional profiling dependencies"
	@echo "  make sync-boosters Install optional boosting dependencies"
	@echo "  make run           Run end-to-end credit risk pipeline"
	@echo "  make smoke         Run quick pipeline"
	@echo "  make verify        Verify artifacts"

sync:
	uv sync --extra dev

sync-profiling:
	uv sync --extra dev --extra profiling

sync-boosters:
	uv sync --extra dev --extra boosters

run:
	PYTHONPATH=../../shared/python uv run python scripts/run_pipeline.py

test:
	uv run pytest

ruff:
	uv run ruff check src tests scripts

ty:
	uv run mypy src tests scripts

check: ruff ty test

smoke:
	PYTHONPATH=../../shared/python uv run python scripts/run_pipeline.py --quick

verify:
	uv run python scripts/verify_artifacts.py
