#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/4] Lint"
uv run ruff check src tests

echo "[2/4] Type checks"
uv run ty check src tests

echo "[3/4] Unit tests"
uv run pytest

echo "[4/4] Digits smoke run"
uv run sota-showcase \
  --dataset digits \
  --contrastive-epochs 1 \
  --dec-pretrain-epochs 1 \
  --dec-finetune-epochs 1 \
  --active-learning-rounds 2 \
  --active-learning-query-size 15

echo "Quality review complete."
