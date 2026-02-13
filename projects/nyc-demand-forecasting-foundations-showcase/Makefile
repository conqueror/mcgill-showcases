SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help sync run run-real download-data test ruff ty check smoke verify

help:
	@echo "NYC Demand Forecasting Foundations Showcase"
	@echo "  make sync          Install dependencies"
	@echo "  make run           Run full synthetic pipeline"
	@echo "  make smoke         Run quick synthetic pipeline"
	@echo "  make download-data Download optional TLC sample"
	@echo "  make run-real      Run using local TLC parquet"
	@echo "  make verify        Verify artifacts"

sync:
	uv sync --extra dev

run:
	uv run python scripts/run_pipeline.py

smoke:
	uv run python scripts/run_pipeline.py --quick

download-data:
	uv run python scripts/download_tlc.py --year 2024 --month 1

run-real:
	uv run python scripts/run_pipeline.py --data-path data/raw/yellow_tripdata_2024-01.parquet

test:
	uv run pytest

ruff:
	uv run ruff check src tests scripts

ty:
	uv run mypy src tests scripts

check: ruff ty test

verify:
	uv run python scripts/verify_artifacts.py
