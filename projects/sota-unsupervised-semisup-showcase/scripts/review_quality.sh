#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/6] Lint"
uv run ruff check src tests

echo "[2/6] Type checks"
uv run ty check src tests

echo "[3/6] Unit tests"
uv run pytest

echo "[4/6] Mermaid validation"
./scripts/validate_mermaid.sh

echo "[5/6] Digits smoke run"
uv run sota-showcase \
  --dataset digits \
  --contrastive-epochs 1 \
  --dec-pretrain-epochs 1 \
  --dec-finetune-epochs 1 \
  --active-learning-rounds 2 \
  --active-learning-query-size 15

echo "[6/6] Business smoke run"
uv run sota-showcase \
  --dataset business \
  --business-read-rows 20000 \
  --business-sample-size 1200 \
  --contrastive-epochs 1 \
  --dec-pretrain-epochs 1 \
  --dec-finetune-epochs 1 \
  --active-learning-rounds 2 \
  --active-learning-query-size 15

echo "Quality review complete."
