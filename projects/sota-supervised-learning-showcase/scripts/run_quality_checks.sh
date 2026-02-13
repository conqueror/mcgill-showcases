#!/usr/bin/env bash
set -euo pipefail

echo "Running tutorial quality checks..."
echo "1) Unit tests"
uv run pytest

echo "2) Mermaid render validation"
./scripts/validate_mermaid.sh

echo "3) Markdown link validation"
./scripts/validate_markdown_links.sh

echo "4) Notebook validation"
./scripts/validate_notebooks.sh

echo "5) End-to-end artifact generation"
uv run python scripts/run_showcase.py

echo "All checks passed."
