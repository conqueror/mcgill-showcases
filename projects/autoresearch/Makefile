SHELL := /bin/bash
.DEFAULT_GOAL := help

AGENT ?= codex

.PHONY: help sync run render-macos render-unix test ruff ty check smoke verify

help:
	@echo "Autoresearch Showcase"
	@echo "  make sync         Install dependencies"
	@echo "  make run          Generate learning artifacts and launch briefs"
	@echo "  make render-macos Render one macOS brief (set AGENT=codex|claude)"
	@echo "  make render-unix  Render one Unix brief (set AGENT=codex|claude)"
	@echo "  make test         Run pytest"
	@echo "  make ruff         Run Ruff"
	@echo "  make ty           Run mypy"
	@echo "  make check        Run lint + type + tests"
	@echo "  make smoke        Run the lightweight artifact path"
	@echo "  make verify       Verify generated artifacts"

sync:
	uv sync --extra dev

run:
	uv run python scripts/run_showcase.py

render-macos:
	uv run python scripts/render_agent_brief.py --platform macos --agent $(AGENT) --output artifacts/agent/$(AGENT)_macos.md

render-unix:
	uv run python scripts/render_agent_brief.py --platform unix --agent $(AGENT) --output artifacts/agent/$(AGENT)_unix.md

test:
	uv run pytest

ruff:
	uv run ruff check src tests scripts

ty:
	uv run mypy src tests scripts

check: ruff ty test

smoke:
	uv run python scripts/run_showcase.py

verify:
	uv run python scripts/verify_artifacts.py
