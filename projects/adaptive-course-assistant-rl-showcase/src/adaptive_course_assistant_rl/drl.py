"""Optional DQN and PPO bridge on the same tutoring environment family."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, SupportsInt, cast

from adaptive_course_assistant_rl.environment import (
    ACTION_LABELS,
    AssistantInterventionEnvironment,
    AssistantState,
)
from adaptive_course_assistant_rl.evaluation import evaluate_policies
from adaptive_course_assistant_rl.policies import ModelPolicy
from adaptive_course_assistant_rl.q_learning import train_q_learning


class OptionalDRLError(RuntimeError):
    """Raised when Gymnasium or Stable-Baselines3 extras are not installed."""


@dataclass(frozen=True)
class DRLComparisonResult:
    """Returned by the optional deep-RL comparison path."""

    comparison_rows: list[dict[str, int | float | str]]
    scenario_rows: list[dict[str, int | float | str]]
    dqn_training_rows: list[dict[str, int | float | str]]
    ppo_training_rows: list[dict[str, int | float | str]]
    policy_gradient_notes: str


class TrainablePredictModel(Protocol):
    """Tiny protocol shared by SB3 DQN and PPO agents."""

    def learn(
        self,
        *,
        total_timesteps: int,
        reset_num_timesteps: bool = False,
        progress_bar: bool = False,
    ) -> object: ...

    def predict(
        self,
        observation: Sequence[float],
        deterministic: bool = True,
    ) -> tuple[SupportsInt, object]: ...


def run_drl_comparison(
    *,
    timesteps: int,
    seed: int = 7,
    horizon: int = 5,
    scenario_ids: tuple[int, ...] = (0, 1, 2, 3, 4),
    quick: bool = False,
) -> DRLComparisonResult:
    """Train DQN and PPO and compare them to tabular Q-learning."""
    gym, np, spaces, algorithms = _load_drl_dependencies()
    random.seed(seed)
    np.random.seed(seed)
    _seed_torch_if_available(seed)

    observation_dim = len(
        AssistantInterventionEnvironment(horizon=horizon).reset(scenario_id=0).as_normalized_vector(horizon=horizon)
    )

    class GymAssistantEnv(gym.Env):  # type: ignore[misc,name-defined]
        metadata: dict[str, list[str]] = {"render_modes": []}

        def __init__(self) -> None:
            self._seed = seed
            self._reset_count = 0
            self._scenario_ids = scenario_ids
            self._env = AssistantInterventionEnvironment(horizon=horizon)
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(observation_dim,), dtype=np.float32)
            self.action_space = spaces.Discrete(len(ACTION_LABELS))

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, object] | None = None,
        ) -> tuple[object, dict[str, object]]:
            if seed is not None:
                self._seed = seed
            scenario_id = (
                self._scenario_ids[self._reset_count % len(self._scenario_ids)]
                if options is None or "scenario_id" not in options
                else int(cast(int, options["scenario_id"]))
            )
            state = self._env.reset(seed=self._seed + self._reset_count, scenario_id=scenario_id)
            self._reset_count += 1
            return np.array(state.as_normalized_vector(horizon=horizon), dtype=np.float32), {"scenario_id": scenario_id}

        def step(
            self,
            action: int,
        ) -> tuple[object, float, bool, bool, dict[str, int | float | str]]:
            transition = self._env.step(int(action))
            return (
                np.array(transition.state.as_normalized_vector(horizon=horizon), dtype=np.float32),
                float(transition.reward),
                bool(transition.done),
                False,
                transition.info,
            )

    def observation_fn(state: AssistantState) -> Sequence[float]:
        return state.as_normalized_vector(horizon=horizon)

    def build_model(name: str) -> tuple[GymAssistantEnv, TrainablePredictModel]:
        env = GymAssistantEnv()
        if name == "dqn":
            model = algorithms["dqn"](
                "MlpPolicy",
                env,
                verbose=0,
                seed=seed,
                learning_starts=64,
                buffer_size=512,
                batch_size=32,
                train_freq=4,
                gradient_steps=1,
                gamma=0.95,
                exploration_fraction=0.35,
                exploration_final_eps=0.05,
            )
        else:
            model = algorithms["ppo"](
                "MlpPolicy",
                env,
                verbose=0,
                seed=seed,
                n_steps=64 if quick else 96,
                batch_size=32,
                gamma=0.95,
            )
        return env, model

    q_result = train_q_learning(episodes=160 if quick else 420, seed=seed, horizon=horizon, scenario_ids=scenario_ids)
    dqn_env, dqn_model = build_model("dqn")
    ppo_env, ppo_model = build_model("ppo")

    dqn_training_rows = _chunked_training_rows(
        name="dqn",
        model=dqn_model,
        timesteps=timesteps,
        chunk=max(200, timesteps // 4),
        seed=seed,
        scenario_ids=scenario_ids,
        horizon=horizon,
        observation_fn=observation_fn,
    )
    ppo_training_rows = _chunked_training_rows(
        name="ppo",
        model=ppo_model,
        timesteps=timesteps,
        chunk=max(200, timesteps // 4),
        seed=seed,
        scenario_ids=scenario_ids,
        horizon=horizon,
        observation_fn=observation_fn,
    )

    comparison_rows, scenario_rows = evaluate_policies(
        policies=[
            q_result.greedy_policy(),
            ModelPolicy(model=dqn_model, observation_fn=observation_fn, name="dqn"),
            ModelPolicy(model=ppo_model, observation_fn=observation_fn, name="ppo"),
        ],
        scenario_ids=scenario_ids,
        episodes_per_scenario=3 if quick else 6,
        base_seed=seed,
        horizon=horizon,
    )
    for row in comparison_rows:
        row["family"] = _policy_family(str(row["policy"]))
    dqn_env.close()
    ppo_env.close()
    return DRLComparisonResult(
        comparison_rows=comparison_rows,
        scenario_rows=scenario_rows,
        dqn_training_rows=dqn_training_rows,
        ppo_training_rows=ppo_training_rows,
        policy_gradient_notes=(
            "# Policy-Gradient Bridge Notes\n\n"
            "Q-learning and DQN learn action values. REINFORCE, actor-critic methods, and PPO learn a policy more directly.\n"
            "PPO is the practical actor-critic anchor in this project's ladder.\n"
        ),
    )


def _chunked_training_rows(
    *,
    name: str,
    model: TrainablePredictModel,
    timesteps: int,
    chunk: int,
    seed: int,
    scenario_ids: tuple[int, ...],
    horizon: int,
    observation_fn: Callable[[AssistantState], Sequence[float]],
) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []
    elapsed = 0
    while elapsed < timesteps:
        step_size = min(chunk, timesteps - elapsed)
        model.learn(total_timesteps=step_size, reset_num_timesteps=False, progress_bar=False)
        elapsed += step_size
        summary_rows, _ = evaluate_policies(
            policies=[ModelPolicy(model=model, observation_fn=observation_fn, name=name)],
            scenario_ids=scenario_ids,
            episodes_per_scenario=2,
            base_seed=seed,
            horizon=horizon,
        )
        row = summary_rows[0]
        rows.append(
            {
                "policy": name,
                "timesteps": elapsed,
                "avg_reward": row["avg_reward"],
                "solved_rate": row["solved_rate"],
                "avg_final_safety_risk": row["avg_final_safety_risk"],
            }
        )
    return rows


def _policy_family(policy_name: str) -> str:
    if policy_name == "dqn":
        return "deep_value_based"
    if policy_name == "ppo":
        return "actor_critic_policy_gradient"
    return "tabular_value_based"


def _load_drl_dependencies() -> tuple[Any, Any, Any, dict[str, Any]]:
    try:
        import gymnasium as gym
        import numpy as np
        from gymnasium import spaces
        from stable_baselines3 import DQN, PPO
    except ImportError as exc:  # pragma: no cover - exercised in optional-dependency test
        raise OptionalDRLError(str(exc)) from exc
    return gym, np, spaces, {"dqn": DQN, "ppo": PPO}


def _seed_torch_if_available(seed: int) -> None:
    try:
        import torch
    except ImportError:  # pragma: no cover - optional dependency
        return
    torch.manual_seed(seed)
