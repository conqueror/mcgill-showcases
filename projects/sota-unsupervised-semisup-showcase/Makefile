.PHONY: help sync sync-advanced lock ruff ty test check diagrams smoke-digits smoke-business quality

help:
	@echo "Available targets:"
	@echo "  make sync            # install dependencies with uv (dev extras)"
	@echo "  make sync-advanced   # install dependencies with advanced extras"
	@echo "  make lock            # refresh uv.lock"
	@echo "  make ruff            # run Ruff lint checks"
	@echo "  make ty              # run ty type checks"
	@echo "  make test            # run pytest"
	@echo "  make check           # run ruff + ty + pytest"
	@echo "  make diagrams        # validate/render Mermaid diagrams"
	@echo "  make smoke-digits    # quick digits mode run"
	@echo "  make smoke-business  # quick business mode run"
	@echo "  make quality         # full review pipeline"

sync:
	uv sync --extra dev

sync-advanced:
	uv sync --extra dev --extra advanced

lock:
	uv lock

ruff:
	uv run ruff check src tests

ty:
	uv run ty check src tests

test:
	uv run pytest

check: ruff ty test

diagrams:
	./scripts/validate_mermaid.sh

smoke-digits:
	uv run sota-showcase \
	  --dataset digits \
	  --contrastive-epochs 1 \
	  --dec-pretrain-epochs 1 \
	  --dec-finetune-epochs 1 \
	  --active-learning-rounds 2 \
	  --active-learning-query-size 15

smoke-business:
	uv run sota-showcase \
	  --dataset business \
	  --business-read-rows 20000 \
	  --business-sample-size 1200 \
	  --contrastive-epochs 1 \
	  --dec-pretrain-epochs 1 \
	  --dec-finetune-epochs 1 \
	  --active-learning-rounds 2 \
	  --active-learning-query-size 15

quality:
	./scripts/review_quality.sh
