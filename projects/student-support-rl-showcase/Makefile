SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help sync sync-drl run run-all run-bandit run-mdp run-q-learning run-dp run-sarsa run-reinforce run-reward-check run-eval run-drl-optional run-business test ruff ty check smoke verify

help:
	@echo "Student Support RL Showcase"
	@echo "  make sync              Install dependencies"
	@echo "  make sync-drl          Install optional Gymnasium/SB3 dependencies"
	@echo "  make run               Run the full core showcase (one script per concept)"
	@echo "  make run-all           Run the full core showcase in a single process"
	@echo "  make run-bandit        Run the contextual bandit warm-up"
	@echo "  make run-mdp           Write the MDP concept artifacts"
	@echo "  make run-q-learning    Train the tabular Q-learning policy"
	@echo "  make run-dp            Compute exact optimal Q* and the Q-learning gap"
	@echo "  make run-sarsa         Train on-policy SARSA (contrast to Q-learning)"
	@echo "  make run-reinforce     Train tabular REINFORCE (policy gradient)"
	@echo "  make run-reward-check  Compare bad and better reward designs"
	@echo "  make run-eval          Evaluate baseline and learned policies"
	@echo "  make run-drl-optional  Run the optional Gymnasium/SB3 bridge"
	@echo "  make run-business      Write the deploy-shadow-reject memo"
	@echo "  make smoke             Run the quick end-to-end path"
	@echo "  make verify            Verify the artifact contract"

sync:
	uv sync --extra dev

sync-drl:
	uv sync --extra dev --extra drl

run:
	uv run python scripts/run_mdp_simulation.py
	uv run python scripts/run_bandit.py
	uv run python scripts/run_q_learning.py
	uv run python scripts/run_dynamic_programming.py
	uv run python scripts/run_sarsa.py
	uv run python scripts/run_policy_gradient.py
	uv run python scripts/run_reward_hacking_check.py
	uv run python scripts/run_policy_evaluation.py
	uv run python scripts/run_drl_optional.py
	uv run python scripts/write_business_memo.py

run-all:
	uv run python scripts/run_showcase.py

run-bandit:
	uv run python scripts/run_bandit.py

run-mdp:
	uv run python scripts/run_mdp_simulation.py

run-q-learning:
	uv run python scripts/run_q_learning.py

run-dp:
	uv run python scripts/run_dynamic_programming.py

run-sarsa:
	uv run python scripts/run_sarsa.py

run-reinforce:
	uv run python scripts/run_policy_gradient.py

run-reward-check:
	uv run python scripts/run_reward_hacking_check.py

run-eval:
	uv run python scripts/run_policy_evaluation.py

run-drl-optional:
	uv run --extra drl python scripts/run_drl_optional.py

run-business:
	uv run python scripts/write_business_memo.py

test:
	uv run pytest

ruff:
	uv run ruff check src tests scripts

ty:
	uv run mypy src tests scripts

check: ruff ty test

smoke:
	uv run python scripts/run_mdp_simulation.py --quick
	uv run python scripts/run_bandit.py --quick
	uv run python scripts/run_q_learning.py --quick
	uv run python scripts/run_dynamic_programming.py --quick
	uv run python scripts/run_sarsa.py --quick
	uv run python scripts/run_policy_gradient.py --quick
	uv run python scripts/run_reward_hacking_check.py --quick
	uv run python scripts/run_policy_evaluation.py --quick
	uv run python scripts/run_drl_optional.py --quick
	uv run python scripts/write_business_memo.py --quick

verify:
	uv run python scripts/verify_artifacts.py
