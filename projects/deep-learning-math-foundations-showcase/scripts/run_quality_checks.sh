#!/usr/bin/env bash
set -euo pipefail

uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run ty check src tests scripts --ignore unresolved-import --ignore unresolved-attribute
uv run pytest
