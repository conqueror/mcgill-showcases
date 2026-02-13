from __future__ import annotations

import pandas as pd

from rl_bandits_showcase.bandits import BanditPolicy
from rl_bandits_showcase.environment import BernoulliBanditEnvironment


def run_policy_simulation(
    policy: BanditPolicy,
    environment: BernoulliBanditEnvironment,
    *,
    horizon: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    cumulative_reward = 0.0
    cumulative_regret = 0.0

    for step in range(1, horizon + 1):
        arm = policy.select_arm()
        reward = environment.pull(arm)
        policy.update(arm, reward)

        cumulative_reward += reward
        instant_regret = environment.optimal_mean_reward - reward
        cumulative_regret += instant_regret

        rows.append(
            {
                "round": step,
                "strategy": policy.name,
                "arm": arm,
                "reward": reward,
                "cumulative_reward": cumulative_reward,
                "instant_regret": instant_regret,
                "cumulative_regret": cumulative_regret,
            }
        )

    return pd.DataFrame(rows)
