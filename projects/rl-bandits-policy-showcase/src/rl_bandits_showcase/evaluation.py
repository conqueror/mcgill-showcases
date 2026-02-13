from __future__ import annotations

import pandas as pd

from rl_bandits_showcase.bandits import UCB1, BanditPolicy, EpsilonGreedy, ThompsonSampling
from rl_bandits_showcase.environment import BernoulliBanditEnvironment
from rl_bandits_showcase.simulation import run_policy_simulation


def run_policy_suite(
    *,
    arm_probs: list[float],
    horizon: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    policies: list[BanditPolicy] = [
        EpsilonGreedy(n_arms=len(arm_probs), epsilon=0.1, seed=seed),
        UCB1(n_arms=len(arm_probs)),
        ThompsonSampling(n_arms=len(arm_probs), seed=seed),
    ]

    traces: list[pd.DataFrame] = []
    for idx, policy in enumerate(policies):
        env = BernoulliBanditEnvironment(arm_probs=arm_probs, seed=seed + idx)
        traces.append(run_policy_simulation(policy, env, horizon=horizon))

    trace_df = pd.concat(traces, ignore_index=True)
    summary = (
        trace_df.sort_values("round")
        .groupby("strategy", as_index=False)
        .tail(1)
        .loc[:, ["strategy", "cumulative_reward", "cumulative_regret"]]
        .sort_values(by="cumulative_reward", ascending=False)
        .reset_index(drop=True)
    )
    return trace_df, summary
