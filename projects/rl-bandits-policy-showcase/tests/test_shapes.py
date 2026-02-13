from __future__ import annotations

from rl_bandits_showcase.evaluation import run_policy_suite


def test_trace_columns_present() -> None:
    trace, summary = run_policy_suite(arm_probs=[0.2, 0.4, 0.5], horizon=60, seed=2)
    assert {"round", "strategy", "reward", "cumulative_reward", "cumulative_regret"}.issubset(
        trace.columns
    )
    assert {"strategy", "cumulative_reward", "cumulative_regret"}.issubset(summary.columns)
