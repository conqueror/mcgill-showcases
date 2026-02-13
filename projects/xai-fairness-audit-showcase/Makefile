SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help sync sync-explainability run run-explainability run-mitigations test ruff ty check smoke verify

help:
	@echo "XAI Fairness Audit Showcase"
	@echo "  make sync            Install dependencies"
	@echo "  make sync-explainability Install optional SHAP + LIME dependencies"
	@echo "  make run             Run explainability + fairness audit"
	@echo "  make run-explainability Run audit with SHAP and LIME outputs"
	@echo "  make run-mitigations Run mitigation strategy comparison"
	@echo "  make test            Run tests"
	@echo "  make ruff            Run Ruff"
	@echo "  make ty              Run mypy"
	@echo "  make check           Run lint + type + tests"
	@echo "  make smoke           Run quick audit flow"
	@echo "  make verify          Verify artifact manifest"

sync:
	uv sync --extra dev

sync-explainability:
	uv sync --extra dev --extra explainability

run:
	uv run python scripts/run_audit.py

run-explainability:
	uv run python scripts/run_audit.py --with-shap --with-lime

run-mitigations:
	uv run python scripts/run_mitigation_compare.py

test:
	uv run pytest

ruff:
	uv run ruff check src tests scripts

ty:
	uv run mypy src scripts tests

check: ruff ty test

smoke:
	uv run python scripts/run_audit.py --quick
	uv run python scripts/run_mitigation_compare.py --quick

verify:
	uv run python scripts/verify_artifacts.py
