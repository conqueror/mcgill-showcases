SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help sync run run-compare test ruff ty check smoke verify

help:
	@echo "RL Bandits Policy Showcase"
	@echo "  make sync         Install dependencies"
	@echo "  make run          Run round-by-round simulations"
	@echo "  make run-compare  Build policy comparison report"
	@echo "  make smoke        Run quick simulation"
	@echo "  make verify       Verify artifacts"

sync:
	uv sync --extra dev

run:
	uv run python scripts/run_simulation.py

run-compare:
	uv run python scripts/run_policy_comparison.py

test:
	uv run pytest

ruff:
	uv run ruff check src tests scripts

ty:
	uv run mypy src tests scripts

check: ruff ty test

smoke:
	uv run python scripts/run_simulation.py --quick
	uv run python scripts/run_policy_comparison.py --quick

verify:
	uv run python scripts/verify_artifacts.py

