SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help sync download pipeline policy confounding contract verify verify-strict ruff ruff-fix format type ty test pytest check all clean

help:
	@echo "CausalML Kaggle Showcase - Commands"
	@echo ""
	@echo "Environment"
	@echo "  make sync            Install dependencies with uv (including dev tools)."
	@echo ""
	@echo "Data + Pipeline"
	@echo "  make download        Download Kaggle dataset into data/raw."
	@echo "  make pipeline        Run main training and evaluation pipeline."
	@echo "  make policy          Run policy simulation by budget levels."
	@echo "  make confounding     Run confounding diagnostics."
	@echo "  make contract        Run synthetic supervised contract baseline artifacts."
	@echo "  make verify          Validate required artifacts are present and well-formed."
	@echo "  make verify-strict   Same as verify, plus notebook-generated outputs."
	@echo ""
	@echo "Quality"
	@echo "  make ruff            Lint with Ruff."
	@echo "  make ruff-fix        Auto-fix lint issues with Ruff when possible."
	@echo "  make format          Format code with Ruff formatter."
	@echo "  make ty              Type check with mypy (ty alias)."
	@echo "  make pytest          Run tests."
	@echo "  make check           Run ruff + ty + pytest."
	@echo "  make all             Run sync + check."
	@echo ""
	@echo "Cleanup"
	@echo "  make clean           Remove local tool caches."

sync:
	uv sync --extra dev

download:
	uv run python scripts/download_kaggle_dataset.py

pipeline:
	uv run python scripts/run_pipeline.py

policy:
	uv run python scripts/policy_simulator.py

confounding:
	uv run python scripts/confounding_checks.py

contract:
	uv run python scripts/run_contract_baseline.py

verify:
	uv run python scripts/verify_learning_artifacts.py

verify-strict:
	uv run python scripts/verify_learning_artifacts.py --require-notebook-artifacts

ruff:
	uv run ruff check .

ruff-fix:
	uv run ruff check . --fix

format:
	uv run ruff format .

type ty:
	uv run mypy src scripts

test pytest:
	uv run pytest

check:
	$(MAKE) ruff
	$(MAKE) ty
	$(MAKE) pytest

all:
	$(MAKE) sync
	$(MAKE) check

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache
