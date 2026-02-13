SHELL := /bin/bash
.DEFAULT_GOAL := help

CAUSAL_DIR := projects/causalml-kaggle-showcase
SUPERVISED_DIR := projects/sota-supervised-learning-showcase
UNSUP_DIR := projects/sota-unsupervised-semisup-showcase

.PHONY: help sync lint type ty test check verify smoke

help:
	@echo "mcgill-showcases root commands"
	@echo ""
	@echo "  make sync    Install dev dependencies for all projects"
	@echo "  make lint    Run lint checks across all projects"
	@echo "  make ty      Run type checks across all projects"
	@echo "  make test    Run test suites across all projects"
	@echo "  make check   Run lint + ty + test"
	@echo "  make verify  Run artifact/schema verification where available"
	@echo "  make smoke   Run lightweight runtime smoke commands"

sync:
	$(MAKE) -C $(CAUSAL_DIR) sync
	$(MAKE) -C $(SUPERVISED_DIR) sync
	$(MAKE) -C $(UNSUP_DIR) sync

lint:
	$(MAKE) -C $(CAUSAL_DIR) ruff
	$(MAKE) -C $(SUPERVISED_DIR) ruff-check
	$(MAKE) -C $(SUPERVISED_DIR) ruff-format
	$(MAKE) -C $(UNSUP_DIR) ruff

type ty:
	$(MAKE) -C $(CAUSAL_DIR) ty
	$(MAKE) -C $(SUPERVISED_DIR) ty-check
	$(MAKE) -C $(UNSUP_DIR) ty

test:
	$(MAKE) -C $(CAUSAL_DIR) pytest
	$(MAKE) -C $(SUPERVISED_DIR) test
	$(MAKE) -C $(UNSUP_DIR) test

check: lint ty test

verify:
	@if [ -f "$(CAUSAL_DIR)/artifacts/metrics_summary.csv" ]; then \
		$(MAKE) -C $(CAUSAL_DIR) verify; \
	else \
		echo "Skipping causal artifact verification: run pipeline first in $(CAUSAL_DIR)"; \
	fi
	@echo "No shared artifact verifier is defined for $(SUPERVISED_DIR)"
	@echo "No shared artifact verifier is defined for $(UNSUP_DIR)"

smoke:
	$(MAKE) -C $(SUPERVISED_DIR) run
	$(MAKE) -C $(UNSUP_DIR) smoke-digits
