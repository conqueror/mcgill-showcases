SHELL := /bin/bash
.DEFAULT_GOAL := help

CAUSAL_DIR := projects/causalml-kaggle-showcase
SUPERVISED_DIR := projects/sota-supervised-learning-showcase
UNSUP_DIR := projects/sota-unsupervised-semisup-showcase
MLOPS_DIR := projects/mlops-drift-production-showcase
XAI_DIR := projects/xai-fairness-audit-showcase
AUTOML_DIR := projects/automl-hpo-showcase
FE_DIR := projects/feature-engineering-dimred-showcase
RL_DIR := projects/rl-bandits-policy-showcase
SYSTEMS_DIR := projects/batch-vs-stream-ml-systems-showcase
ROLLOUT_DIR := projects/model-release-rollout-showcase
EDA_DIR := projects/eda-leakage-profiling-showcase
CREDIT_DIR := projects/credit-risk-classification-capstone-showcase
LTR_DIR := projects/learning-to-rank-foundations-showcase
RANK_API_DIR := projects/ranking-api-productization-showcase
NYC_DEMAND_DIR := projects/nyc-demand-forecasting-foundations-showcase
DEMAND_API_OBS_DIR := projects/demand-api-observability-showcase
DOCS_REQUIREMENTS := docs/requirements-mkdocs.txt

.PHONY: help sync lint type ty test check check-contracts check-supervised verify smoke docs-serve docs-build docs-check

help:
	@echo "mcgill-showcases root commands"
	@echo ""
	@echo "  make sync    Install dev dependencies for all projects"
	@echo "  make lint    Run lint checks across all projects"
	@echo "  make ty      Run type checks across all projects"
	@echo "  make test    Run test suites across all projects"
	@echo "  make check   Run lint + ty + test"
	@echo "  make check-contracts  Validate supervised artifact contracts"
	@echo "  make check-supervised Alias for check-contracts"
	@echo "  make verify  Run artifact/schema verification where available"
	@echo "  make smoke   Run lightweight runtime smoke commands"
	@echo "  make docs-serve  Run MkDocs Material site locally"
	@echo "  make docs-build  Build MkDocs site into site/"
	@echo "  make docs-check  Strict MkDocs build for CI/pre-PR checks"

docs-serve:
	uv run --with-requirements $(DOCS_REQUIREMENTS) mkdocs serve

docs-build:
	uv run --with-requirements $(DOCS_REQUIREMENTS) mkdocs build

docs-check:
	uv run --with-requirements $(DOCS_REQUIREMENTS) mkdocs build --strict

sync:
	$(MAKE) -C $(CAUSAL_DIR) sync
	$(MAKE) -C $(SUPERVISED_DIR) sync
	$(MAKE) -C $(UNSUP_DIR) sync
	$(MAKE) -C $(MLOPS_DIR) sync
	$(MAKE) -C $(XAI_DIR) sync
	$(MAKE) -C $(AUTOML_DIR) sync
	$(MAKE) -C $(FE_DIR) sync
	$(MAKE) -C $(RL_DIR) sync
	$(MAKE) -C $(SYSTEMS_DIR) sync
	$(MAKE) -C $(ROLLOUT_DIR) sync
	$(MAKE) -C $(EDA_DIR) sync
	$(MAKE) -C $(CREDIT_DIR) sync
	$(MAKE) -C $(LTR_DIR) sync
	$(MAKE) -C $(RANK_API_DIR) sync
	$(MAKE) -C $(NYC_DEMAND_DIR) sync
	$(MAKE) -C $(DEMAND_API_OBS_DIR) sync

lint:
	$(MAKE) -C $(CAUSAL_DIR) ruff
	$(MAKE) -C $(SUPERVISED_DIR) ruff-check
	$(MAKE) -C $(SUPERVISED_DIR) ruff-format
	$(MAKE) -C $(UNSUP_DIR) ruff
	$(MAKE) -C $(MLOPS_DIR) ruff
	$(MAKE) -C $(XAI_DIR) ruff
	$(MAKE) -C $(AUTOML_DIR) ruff
	$(MAKE) -C $(FE_DIR) ruff
	$(MAKE) -C $(RL_DIR) ruff
	$(MAKE) -C $(SYSTEMS_DIR) ruff
	$(MAKE) -C $(ROLLOUT_DIR) ruff
	$(MAKE) -C $(EDA_DIR) ruff
	$(MAKE) -C $(CREDIT_DIR) ruff
	$(MAKE) -C $(LTR_DIR) ruff
	$(MAKE) -C $(RANK_API_DIR) ruff
	$(MAKE) -C $(NYC_DEMAND_DIR) ruff
	$(MAKE) -C $(DEMAND_API_OBS_DIR) ruff

type ty:
	$(MAKE) -C $(CAUSAL_DIR) ty
	$(MAKE) -C $(SUPERVISED_DIR) ty-check
	$(MAKE) -C $(UNSUP_DIR) ty
	$(MAKE) -C $(MLOPS_DIR) ty
	$(MAKE) -C $(XAI_DIR) ty
	$(MAKE) -C $(AUTOML_DIR) ty
	$(MAKE) -C $(FE_DIR) ty
	$(MAKE) -C $(RL_DIR) ty
	$(MAKE) -C $(SYSTEMS_DIR) ty
	$(MAKE) -C $(ROLLOUT_DIR) ty
	$(MAKE) -C $(EDA_DIR) ty
	$(MAKE) -C $(CREDIT_DIR) ty
	$(MAKE) -C $(LTR_DIR) ty
	$(MAKE) -C $(RANK_API_DIR) ty
	$(MAKE) -C $(NYC_DEMAND_DIR) ty
	$(MAKE) -C $(DEMAND_API_OBS_DIR) ty

test:
	$(MAKE) -C $(CAUSAL_DIR) pytest
	$(MAKE) -C $(SUPERVISED_DIR) test
	$(MAKE) -C $(UNSUP_DIR) test
	$(MAKE) -C $(MLOPS_DIR) test
	$(MAKE) -C $(XAI_DIR) test
	$(MAKE) -C $(AUTOML_DIR) test
	$(MAKE) -C $(FE_DIR) test
	$(MAKE) -C $(RL_DIR) test
	$(MAKE) -C $(SYSTEMS_DIR) test
	$(MAKE) -C $(ROLLOUT_DIR) test
	$(MAKE) -C $(EDA_DIR) test
	$(MAKE) -C $(CREDIT_DIR) test
	$(MAKE) -C $(LTR_DIR) test
	$(MAKE) -C $(RANK_API_DIR) test
	$(MAKE) -C $(NYC_DEMAND_DIR) test
	$(MAKE) -C $(DEMAND_API_OBS_DIR) test

check: lint ty test check-contracts

check-supervised check-contracts:
	python3 shared/scripts/verify_supervised_contract.py --bootstrap-missing

verify:
	@if [ -f "$(CAUSAL_DIR)/artifacts/metrics_summary.csv" ]; then \
		$(MAKE) -C $(CAUSAL_DIR) verify; \
	else \
		echo "Skipping causal artifact verification: run pipeline first in $(CAUSAL_DIR)"; \
	fi
	@if [ -f "$(SUPERVISED_DIR)/artifacts/summary.md" ]; then \
		echo "Supervised artifacts detected in $(SUPERVISED_DIR)"; \
	else \
		echo "No shared artifact verifier is defined for $(SUPERVISED_DIR)"; \
	fi
	@if [ -f "$(UNSUP_DIR)/artifacts/reports/digits_run_summary.json" ]; then \
		echo "Unsupervised artifacts detected in $(UNSUP_DIR)"; \
	else \
		echo "No shared artifact verifier is defined for $(UNSUP_DIR)"; \
	fi
	@if [ -f "$(MLOPS_DIR)/artifacts/manifest.json" ]; then $(MAKE) -C $(MLOPS_DIR) verify; else echo "Skipping $(MLOPS_DIR) verify: run pipeline first"; fi
	@if [ -f "$(XAI_DIR)/artifacts/manifest.json" ]; then $(MAKE) -C $(XAI_DIR) verify; else echo "Skipping $(XAI_DIR) verify: run pipeline first"; fi
	@if [ -f "$(AUTOML_DIR)/artifacts/manifest.json" ]; then $(MAKE) -C $(AUTOML_DIR) verify; else echo "Skipping $(AUTOML_DIR) verify: run pipeline first"; fi
	@if [ -f "$(FE_DIR)/artifacts/manifest.json" ]; then $(MAKE) -C $(FE_DIR) verify; else echo "Skipping $(FE_DIR) verify: run pipeline first"; fi
	@if [ -f "$(RL_DIR)/artifacts/manifest.json" ]; then $(MAKE) -C $(RL_DIR) verify; else echo "Skipping $(RL_DIR) verify: run pipeline first"; fi
	@if [ -f "$(SYSTEMS_DIR)/artifacts/manifest.json" ]; then $(MAKE) -C $(SYSTEMS_DIR) verify; else echo "Skipping $(SYSTEMS_DIR) verify: run pipeline first"; fi
	@if [ -f "$(ROLLOUT_DIR)/artifacts/manifest.json" ]; then $(MAKE) -C $(ROLLOUT_DIR) verify; else echo "Skipping $(ROLLOUT_DIR) verify: run pipeline first"; fi
	@if [ -f "$(EDA_DIR)/artifacts/manifest.json" ]; then $(MAKE) -C $(EDA_DIR) verify; else echo "Skipping $(EDA_DIR) verify: run pipeline first"; fi
	@if [ -f "$(CREDIT_DIR)/artifacts/manifest.json" ]; then $(MAKE) -C $(CREDIT_DIR) verify; else echo "Skipping $(CREDIT_DIR) verify: run pipeline first"; fi
	@if [ -f "$(LTR_DIR)/artifacts/manifest.json" ]; then $(MAKE) -C $(LTR_DIR) verify; else echo "Skipping $(LTR_DIR) verify: run pipeline first"; fi
	@if [ -f "$(RANK_API_DIR)/artifacts/model.txt" ]; then $(MAKE) -C $(RANK_API_DIR) verify; else echo "Skipping $(RANK_API_DIR) verify: run train-demo first"; fi
	@if [ -f "$(NYC_DEMAND_DIR)/artifacts/manifest.json" ]; then $(MAKE) -C $(NYC_DEMAND_DIR) verify; else echo "Skipping $(NYC_DEMAND_DIR) verify: run pipeline first"; fi
	@if [ -f "$(DEMAND_API_OBS_DIR)/artifacts/model.joblib" ]; then $(MAKE) -C $(DEMAND_API_OBS_DIR) verify; else echo "Skipping $(DEMAND_API_OBS_DIR) verify: run train-demo first"; fi

smoke:
	$(MAKE) -C $(SUPERVISED_DIR) run
	$(MAKE) -C $(UNSUP_DIR) smoke-digits
	$(MAKE) -C $(MLOPS_DIR) smoke
	$(MAKE) -C $(XAI_DIR) smoke
	$(MAKE) -C $(AUTOML_DIR) smoke
	$(MAKE) -C $(FE_DIR) smoke
	$(MAKE) -C $(RL_DIR) smoke
	$(MAKE) -C $(SYSTEMS_DIR) smoke
	$(MAKE) -C $(ROLLOUT_DIR) smoke
	$(MAKE) -C $(EDA_DIR) smoke
	$(MAKE) -C $(CREDIT_DIR) smoke
	$(MAKE) -C $(LTR_DIR) smoke
	$(MAKE) -C $(RANK_API_DIR) smoke
	$(MAKE) -C $(NYC_DEMAND_DIR) smoke
	$(MAKE) -C $(DEMAND_API_OBS_DIR) smoke
