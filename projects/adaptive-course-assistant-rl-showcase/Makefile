SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help sync sync-drl run run-all run-bandit run-rule run-mdp run-q-learning run-sarsa run-reinforce run-compare run-reward-audit run-learning-agent-story run-policy-export run-drl-optional test ruff ty check smoke verify verify-core verify-full

help:
	@echo "Adaptive Course Assistant RL Showcase"
	@echo "  make sync              Install core dev dependencies"
	@echo "  make sync-drl          Install the optional Gymnasium/SB3 extras"
	@echo "  make run               Generate the full core artifact set"
	@echo "  make run-all           Alias for the full core run"
	@echo "  make run-bandit        Run the contextual bandit path"
	@echo "  make run-rule          Evaluate the fixed rule-based baseline"
	@echo "  make run-mdp           Write the MDP and assistant-context artifacts"
	@echo "  make run-q-learning    Train the tabular Q-learning policy"
	@echo "  make run-sarsa         Train the tabular SARSA policy"
	@echo "  make run-reinforce     Train the small REINFORCE path"
	@echo "  make run-compare       Compare random, rule-based, and learned policies"
	@echo "  make run-reward-audit  Run the reward-hacking audit"
	@echo "  make run-learning-agent-story  Write the student-facing learning-agent bridge artifact"
	@echo "  make run-policy-export Write the deterministic assistant bridge artifacts"
	@echo "  make run-drl-optional  Run the optional DQN/PPO bridge"
	@echo "  make smoke             Run the quick end-to-end path"
	@echo "  make verify            Verify the core artifact contract"
	@echo "  make verify-core       Alias for the core artifact contract"
	@echo "  make verify-full       Verify the full artifact contract, including optional DRL"

sync:
	uv sync --extra dev

sync-drl:
	uv sync --extra dev --extra drl

run:
	uv run python scripts/run_showcase.py

run-all: run

run-bandit:
	uv run python scripts/run_contextual_bandit.py

run-rule:
	uv run python scripts/run_rule_policy.py

run-mdp:
	uv run python scripts/run_mdp_policy.py

run-q-learning:
	uv run python scripts/run_q_learning.py

run-sarsa:
	uv run python scripts/run_sarsa.py

run-reinforce:
	uv run python scripts/run_policy_gradient.py

run-compare:
	uv run python scripts/run_rl_family_comparison.py

run-reward-audit:
	uv run python scripts/run_reward_audit.py

run-learning-agent-story:
	uv run python scripts/run_learning_agent_story.py

run-policy-export:
	uv run python scripts/run_policy_export.py

run-drl-optional:
	uv run --extra drl python scripts/run_drl_optional.py

test:
	uv run pytest

ruff:
	uv run ruff check src tests scripts

ty:
	uv run mypy src tests scripts

check: ruff ty test

smoke:
	uv run python scripts/run_showcase.py --quick

verify:
	uv run python scripts/verify_artifacts.py

verify-core:
	uv run python scripts/verify_artifacts.py

verify-full:
	uv run python scripts/verify_artifacts.py --require-optional-drl
