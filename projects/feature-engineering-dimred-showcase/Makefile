SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help sync sync-advanced run run-dimred run-advanced test ruff ty check smoke verify

help:
	@echo "Feature Engineering + Dimensionality Reduction Showcase"
	@echo "  make sync          Install dependencies"
	@echo "  make sync-advanced Install optional UMAP extras"
	@echo "  make run           Run feature engineering pipeline"
	@echo "  make run-dimred    Run dimensionality reduction comparison"
	@echo "  make run-advanced  Run advanced FE libs and embedding artifacts"
	@echo "  make smoke         Run quick full flow"
	@echo "  make verify        Verify artifacts"

sync:
	uv sync --extra dev

sync-advanced:
	uv sync --extra dev --extra advanced

run:
	uv run python scripts/run_feature_pipeline.py

run-dimred:
	uv run python scripts/run_dimred_comparison.py

run-advanced:
	uv run python scripts/run_advanced_features.py

test:
	uv run pytest

ruff:
	uv run ruff check src tests scripts

ty:
	uv run mypy src tests scripts

check: ruff ty test

smoke:
	uv run python scripts/run_feature_pipeline.py --quick
	uv run python scripts/run_dimred_comparison.py --quick

verify:
	uv run python scripts/verify_artifacts.py
