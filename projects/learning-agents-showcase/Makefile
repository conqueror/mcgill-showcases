SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help sync sync-sdk run run-all run-bandit run-mdp run-q-learning run-dp run-sarsa run-reinforce run-offline run-ope run-cascade run-sdk-bridge run-preference run-marl run-drl run-reward-check run-eval run-business test ruff ty check smoke verify

help:
	@echo "Learning Agents Showcase"
	@echo "  make sync              Install dependencies (dev)"
	@echo "  make sync-sdk          Install optional OpenAI Agents SDK extra (Lane A)"
	@echo "  make run               Run the full core showcase (one script per concept)"
	@echo "  make run-all           Run the full core showcase in a single process"
	@echo "  make run-bandit        Run the contextual bandit routing warm-up"
	@echo "  make run-mdp           Write the MDP concept artifacts and a sample episode"
	@echo "  make run-q-learning    Train the tabular Q-learning orchestration policy"
	@echo "  make run-dp            Compute exact optimal Q* and the Q-learning gap"
	@echo "  make run-sarsa         Train on-policy SARSA (contrast to Q-learning)"
	@echo "  make run-reinforce     Train tabular REINFORCE (policy gradient)"
	@echo "  make run-offline       Learn offline from a behaviour log (Fitted-Q Iteration)"
	@echo "  make run-ope           Off-policy evaluation of targets, graded vs truth"
	@echo "  make run-cascade       Sweep the cost-aware cascade's cost/quality frontier"
	@echo "  make run-sdk-bridge    Drive an agent loop with the learned policy (SDK bridge)"
	@echo "  make run-preference    Preference-tune a toy LM (RLHF/DPO/GRPO/RLVR)"
	@echo "  make run-marl          Multi-agent coordination (independent vs joint-action)"
	@echo "  make run-drl           Optional deep-RL lane: DQN vs PPO vs tabular Q-learning"
	@echo "  make run-reward-check  Contrast the judge rubric with a hackable reward"
	@echo "  make run-eval          Evaluate baseline and learned policies offline"
	@echo "  make run-business      Write the deploy-shadow-reject memo"
	@echo "  make check             Lint + type-check + tests"
	@echo "  make smoke             Run the quick end-to-end path"
	@echo "  make verify            Verify the artifact contract"

sync:
	uv sync --extra dev

sync-sdk:
	uv sync --extra dev --extra sdk

run:
	uv run python scripts/run_mdp_simulation.py
	uv run python scripts/run_bandit.py
	uv run python scripts/run_q_learning.py
	uv run python scripts/run_dynamic_programming.py
	uv run python scripts/run_sarsa.py
	uv run python scripts/run_policy_gradient.py
	uv run python scripts/run_offline_rl.py
	uv run python scripts/run_ope.py
	uv run python scripts/run_cost_cascade.py
	uv run python scripts/run_sdk_bridge.py
	uv run python scripts/run_preference_optimization.py
	uv run python scripts/run_marl.py
	uv run python scripts/run_reward_hacking_check.py
	uv run python scripts/run_policy_evaluation.py
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

run-offline:
	uv run python scripts/run_offline_rl.py

run-ope:
	uv run python scripts/run_ope.py

run-cascade:
	uv run python scripts/run_cost_cascade.py

run-sdk-bridge:
	uv run python scripts/run_sdk_bridge.py

run-preference:
	uv run python scripts/run_preference_optimization.py

run-marl:
	uv run python scripts/run_marl.py

run-drl:
	uv run python scripts/run_deep_rl.py

run-reward-check:
	uv run python scripts/run_reward_hacking_check.py

run-eval:
	uv run python scripts/run_policy_evaluation.py

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
	uv run python scripts/run_offline_rl.py --quick
	uv run python scripts/run_ope.py --quick
	uv run python scripts/run_cost_cascade.py --quick
	uv run python scripts/run_sdk_bridge.py --quick
	uv run python scripts/run_preference_optimization.py --quick
	uv run python scripts/run_marl.py --quick
	uv run python scripts/run_reward_hacking_check.py --quick
	uv run python scripts/run_policy_evaluation.py --quick
	uv run python scripts/write_business_memo.py --quick

verify:
	uv run python scripts/verify_artifacts.py
