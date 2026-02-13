SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help sync run-batch run-stream run-compare test ruff ty check smoke verify

help:
	@echo "Batch vs Stream ML Systems Showcase"
	@echo "  make sync         Install dependencies"
	@echo "  make run-batch    Run batch KPI pipeline"
	@echo "  make run-stream   Run stream KPI pipeline"
	@echo "  make run-compare  Compare parity between modes"
	@echo "  make smoke        Run quick full flow"
	@echo "  make verify       Verify artifacts"

sync:
	uv sync --extra dev

run-batch:
	uv run python scripts/run_batch.py

run-stream:
	uv run python scripts/run_stream.py

run-compare:
	uv run python scripts/run_compare_modes.py

test:
	uv run pytest

ruff:
	uv run ruff check src tests scripts

ty:
	uv run mypy src tests scripts

check: ruff ty test

smoke:
	uv run python scripts/run_compare_modes.py --quick

verify:
	uv run python scripts/verify_artifacts.py

