from __future__ import annotations

from rl_bandits_showcase.evaluation import run_policy_suite


def test_policy_suite_deterministic_with_fixed_seed() -> None:
    _, summary_a = run_policy_suite(arm_probs=[0.2, 0.4, 0.5], horizon=120, seed=13)
    _, summary_b = run_policy_suite(arm_probs=[0.2, 0.4, 0.5], horizon=120, seed=13)
    assert summary_a.equals(summary_b)
