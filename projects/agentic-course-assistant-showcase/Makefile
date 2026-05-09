SHELL := /bin/bash
.DEFAULT_GOAL := help

QUESTION ?= I understand train test split basics, but I am confused about data leakage in feature engineering.

.PHONY: help sync sync-openai sync-adk sync-live run eval trace-check harness-report test ruff ty check smoke verify

help:
	@echo "Agentic Course Assistant Showcase"
	@echo "  make sync    Install dependencies"
	@echo "  make sync-openai  Install OpenAI Agents SDK optional extra"
	@echo "  make sync-adk     Install Google ADK optional extra"
	@echo "  make sync-live    Install both live SDK optional extras"
	@echo "  make run     Generate assistant artifacts for QUESTION"
	@echo "  make eval    Generate harness eval and judge artifacts"
	@echo "  make trace-check     Verify generated trace and harness artifacts"
	@echo "  make harness-report  Generate harness report artifacts"
	@echo "  make test    Run pytest"
	@echo "  make ruff    Run Ruff"
	@echo "  make ty      Run mypy"
	@echo "  make check   Run lint + type + tests"
	@echo "  make smoke   Run the lightweight assistant path"
	@echo "  make verify  Verify generated artifacts"

sync:
	uv sync --extra dev

sync-openai:
	uv sync --extra dev --extra openai

sync-adk:
	uv sync --extra dev --extra adk

sync-live:
	uv sync --extra dev --extra live

run:
	uv run python scripts/run_showcase.py --question "$(QUESTION)"

eval:
	uv run python scripts/run_harness_lab.py --question "$(QUESTION)"

trace-check:
	uv run python scripts/verify_artifacts.py

harness-report:
	uv run python scripts/run_harness_lab.py --question "$(QUESTION)"

test:
	uv run pytest

ruff:
	uv run ruff check src tests scripts

ty:
	uv run mypy src tests scripts

check: ruff ty test

smoke:
	uv run python scripts/run_showcase.py --question "Help me debug why my validation score is too good."

verify:
	uv run python scripts/verify_artifacts.py
