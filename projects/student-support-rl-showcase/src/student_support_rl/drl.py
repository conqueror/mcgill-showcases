"""Bridge the from-scratch RL ladder to deep RL via Gymnasium and Stable-Baselines3.

This module is the showcase's OPTIONAL deep-RL stage and is intentionally a black box: the
mechanisms it relies on (replay buffers, target networks, neural value/policy approximators,
clipped surrogate objectives) live inside Stable-Baselines3, whereas the *from-scratch* versions
of the same ideas live in ``q_learning.py`` (tabular value-based control) and
``policy_gradient.py`` (tabular REINFORCE). The point of the bridge is comparison, not
re-derivation: it wraps the student-support environment as a ``gymnasium.Env``, trains a DQN and a
PPO agent, and evaluates both against the tabular Q-learning baseline on the SAME scenarios,
horizon, and seed family so the numbers are apples-to-apples.

Where this sits on the ladder: contextual bandit -> MDP -> Q-learning -> DQN -> policy gradient ->
actor-critic -> PPO. DQN is the deep generalization of Q-learning (a neural net replaces the
Q-table); PPO is the actor-critic, policy-gradient end of the ladder. Both are reached only after
the tabular methods have made the underlying ideas explicit.

The two DQN stabilization tricks (explained in detail on the relevant docstrings below):
    - Experience replay: store transitions in a buffer and train on random minibatches, which
      decorrelates the otherwise highly sequential samples and reuses data.
    - Target network: bootstrap the TD target from a slowly-updated copy of the network so the
      regression target does not chase the parameters it is being fit to.

Because the deep-RL extras (Gymnasium, Stable-Baselines3, PyTorch) may not be installed on every
machine, the loaders raise :class:`OptionalDRLError` rather than crashing with a bare ImportError,
giving callers a clean signal to fall back to the tabular path.

RL concept:
    Deep value-based control (DQN) and deep actor-critic policy gradients (PPO); cross-link
    docs/deep-rl.md and docs/policy-gradient-and-actor-critic.md, with the tabular counterparts
    in docs/value-based-learning.md.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, SupportsInt

from student_support_rl.environment import StudentState
from student_support_rl.evaluation import evaluate_policies
from student_support_rl.policies import ModelPolicy
from student_support_rl.q_learning import train_q_learning


class OptionalDRLError(RuntimeError):
    """Signal that the optional deep-RL stack (Gymnasium / Stable-Baselines3) is unavailable.

    The deep-RL bridge is opt-in: its dependencies are heavy and not required to run the
    from-scratch tabular ladder. Raising a dedicated error type (instead of letting a raw
    ``ImportError`` propagate) lets callers detect the missing-extras case precisely and fall
    back to the tabular path, while preserving the original import failure message.

    RL concept:
        Optional deep-RL tooling boundary; see docs/deep-rl.md.
    """


@dataclass(frozen=True)
class DRLComparisonResult:
    """Hold the artifacts of a DQN-vs-PPO-vs-tabular-Q-learning comparison.

    This is the immutable return value of :func:`run_drl_comparison`. It bundles the head-to-head
    evaluation of the deep agents (DQN, PPO) against the tabular Q-learning baseline together with
    the deep agents' learning-progress trace and two human-readable narrative reports. Keeping it
    frozen makes the comparison a stable, hashable record suitable for reporting and tests.

    Attributes:
        comparison_rows: One summary row per policy (Q-learning, DQN, PPO) aggregated over all
            scenarios; each row also carries a ``family`` tag (see :func:`_policy_family`).
        scenario_rows: Per-policy, per-scenario evaluation rows (finer granularity than the
            aggregated comparison rows).
        training_rows: Learning-progress trace for the deep agents only, sampled at each training
            chunk boundary (policy name, cumulative step count, mean reward, mean final risk).
        policy_gradient_notes: Markdown narrative positioning PPO as the policy-gradient /
            actor-critic reference point relative to value-based methods.
        bridge_report: Markdown narrative summarizing the run configuration and how DQN/PPO relate
            to the tabular baseline.

    RL concept:
        Cross-method evaluation record (tabular vs deep); see docs/evaluation-and-governance.md.
    """

    comparison_rows: list[dict[str, int | float | str]]
    scenario_rows: list[dict[str, int | float | str]]
    training_rows: list[dict[str, int | float | str]]
    policy_gradient_notes: str
    bridge_report: str


class TrainablePredictModel(Protocol):
    """Structural type for a Stable-Baselines3 agent that can be trained and queried.

    This Protocol captures only the two methods the bridge actually uses from an SB3 model -- the
    ``learn``/``predict`` pair shared by DQN and PPO -- so the rest of the module can stay agnostic
    to the concrete algorithm class and avoid a hard import of Stable-Baselines3 at type-check
    time. ``predict`` returns an ``(action, state)`` pair following the SB3 convention; only the
    action is consumed here (the recurrent-state slot is ignored for these feedforward policies).

    RL concept:
        Agent training/inference interface (deep value-based and actor-critic); see
        docs/deep-rl.md.
    """

    def learn(
        self,
        *,
        total_timesteps: int,
        reset_num_timesteps: bool = False,
        progress_bar: bool = False,
    ) -> object:
        """Run gradient-based training for ``total_timesteps`` environment steps."""
        ...

    def predict(
        self,
        observation: Sequence[float],
        deterministic: bool = True,
    ) -> tuple[SupportsInt, object]:
        """Map an observation to an ``(action, recurrent_state)`` pair, greedy by default."""
        ...


def run_drl_comparison(
    *,
    timesteps: int,
    output_dir: Path,
    quick: bool,
    seed: int = 7,
    horizon: int = 6,
    scenario_ids: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> DRLComparisonResult:
    """Train DQN and PPO on the student-support env and compare them to tabular Q-learning.

    Intuition: the tabular ladder (Q-learning, REINFORCE) makes each RL idea explicit on a small
    discrete problem; this bridge then asks "does the *deep* version of the same idea agree?" It
    wraps the student-support MDP as a ``gymnasium.Env``, trains a DQN agent (deep value-based) and
    a PPO agent (deep actor-critic policy gradient) for ``timesteps`` steps each, and evaluates all
    three policies on the SAME scenario set, horizon, and seed family so differences reflect the
    learning method rather than the test conditions. All RNGs (Python, NumPy, and -- if installed
    -- PyTorch) are seeded up front for reproducibility.

    DQN here learns ``Q(s,a)`` with a neural approximator, stabilized by experience replay
    (``buffer_size``/``batch_size``/``learning_starts``) and an implicit target network inside
    Stable-Baselines3, with epsilon-greedy exploration annealed via ``exploration_fraction`` and
    ``exploration_final_eps``. PPO is on-policy and optimizes the policy directly using a clipped
    surrogate objective with advantage estimates (collecting ``n_steps`` of experience per update).
    Training is run in chunks with ``reset_num_timesteps=False`` so the agent's step counter and
    schedules persist across chunks while an evaluation snapshot is recorded after each chunk.

    Args:
        timesteps: Total environment steps to train EACH deep agent (DQN and PPO).
        output_dir: Accepted for interface symmetry with other runners but unused (see ``del``
            below); this function returns its artifacts in memory.
        quick: When True, use lighter evaluation/training budgets for a fast smoke run.
        seed: Master seed for Python, NumPy, PyTorch, the gym env, and the tabular baseline.
        horizon: Episode length (number of decision weeks) for every environment instance.
        scenario_ids: Student scenarios cycled through for training resets and evaluation.

    Returns:
        DRLComparisonResult: Aggregated comparison rows (with a ``family`` tag per policy),
        per-scenario rows, the deep agents' training trace, and two markdown narrative reports.

    Raises:
        OptionalDRLError: If Gymnasium or Stable-Baselines3 is not installed (raised by
            :func:`_load_drl_dependencies`).

    RL concept:
        Deep value-based control (DQN) vs deep actor-critic policy gradient (PPO) vs tabular
        Q-learning, on a shared MDP; see docs/deep-rl.md and
        docs/policy-gradient-and-actor-critic.md.

    Math:
        DQN regresses toward the Bellman-optimality target
        Q*(s,a)=E[R_{t+1}+gamma*max_a' Q*(s',a')]; PPO ascends grad J = E[grad log pi(A|s)
        (G_t - b)] through a clipped surrogate, where (G_t - b) is the advantage estimate.
    """
    del output_dir  # interface-only argument; artifacts are returned in memory, not written
    gym, np, spaces, algorithms = _load_drl_dependencies()
    # Seed every RNG up front (Python, NumPy, and torch if present) for a reproducible comparison.
    random.seed(seed)
    np.random.seed(seed)
    _seed_torch_if_available(seed)

    class GymStudentSupportEnv(gym.Env):  # type: ignore[misc,name-defined]
        """Adapt the student-support MDP to the Gymnasium ``Env`` interface for SB3.

        Stable-Baselines3 consumes environments through the Gymnasium API, so this thin adapter
        exposes the in-house :class:`StudentSupportEnvironment` as a ``gym.Env``: a 6-dim
        normalized ``Box`` observation space (the state features scaled to [0, 1]) and a
        ``Discrete(4)`` action space (the four support actions). It is the seam where the from-
        scratch MDP meets the deep-RL library; the dynamics themselves are unchanged.

        RL concept:
            MDP-to-Gymnasium environment adapter (states, actions, transitions, rewards); see
            docs/mdp-and-environment.md.
        """

        metadata: dict[str, list[str]] = {"render_modes": []}

        def __init__(self) -> None:
            """Build the wrapped environment and declare the observation/action spaces."""
            from student_support_rl.environment import StudentSupportEnvironment

            self._seed = seed
            self._reset_count = 0
            self._scenario_ids = scenario_ids
            self._env = StudentSupportEnvironment(horizon=horizon)
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(6,),
                dtype=np.float32,
            )
            self.action_space = spaces.Discrete(4)

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, object] | None = None,
        ) -> tuple[object, dict[str, object]]:
            """Start a new episode and return the initial normalized observation.

            Scenarios are cycled deterministically by reset count (unless an explicit
            ``scenario_id`` is passed via ``options``), and each episode gets a distinct
            ``seed + reset_count`` so the comparison is reproducible yet not degenerate.

            Args:
                seed: Optional override for the episode seed family.
                options: Optional dict; an ``"scenario_id"`` entry forces a specific scenario.

            Returns:
                A ``(observation, info)`` pair following the Gymnasium reset contract.
            """
            if seed is not None:
                self._seed = seed
            # Cycle scenarios by reset count unless the caller pins one via options.
            scenario_value = (
                self._scenario_ids[self._reset_count % len(self._scenario_ids)]
                if options is None or "scenario_id" not in options
                else options["scenario_id"]
            )
            scenario_id = _coerce_int(scenario_value)
            scenario_seed = self._seed + self._reset_count
            self._reset_count += 1
            state = self._env.reset(seed=scenario_seed, scenario_id=scenario_id)
            return np.array(state.as_normalized_vector(horizon=horizon), dtype=np.float32), {
                "scenario_id": scenario_id
            }

        def step(
            self,
            action: int,
        ) -> tuple[object, float, bool, bool, dict[str, int | float | str]]:
            """Apply one action and return the Gymnasium ``step`` 5-tuple.

            Delegates the dynamics to the wrapped environment, then repackages the resulting
            transition as ``(observation, reward, terminated, truncated, info)``. ``truncated`` is
            always False here because the episode ends only via the environment's own horizon-based
            ``done`` flag (reported as ``terminated``), not via an external time limit.

            Args:
                action: Discrete support action index in ``[0, 3]``.

            Returns:
                The ``(obs, reward, terminated, truncated, info)`` tuple; ``reward`` is R_{t+1}.
            """
            transition = self._env.step(int(action))
            return (
                np.array(
                    transition.state.as_normalized_vector(horizon=horizon),
                    dtype=np.float32,
                ),
                float(transition.reward),
                bool(transition.done),
                False,
                transition.info,
            )

    def observation_fn(state: StudentState) -> Sequence[float]:
        """Encode a discrete student state as the normalized vector the deep models expect."""
        return state.as_normalized_vector(horizon=horizon)

    def build_model(name: str) -> tuple[GymStudentSupportEnv, TrainablePredictModel]:
        """Construct a fresh env and the requested SB3 agent (DQN or PPO).

        The hyperparameters are deliberately small (tiny replay buffer, short rollouts) so the
        bridge runs quickly as a teaching demo rather than a tuned benchmark. DQN is configured
        as off-policy value learning with experience replay and epsilon-greedy annealing; PPO is
        configured as on-policy actor-critic with short rollouts. Any name other than ``"dqn"``
        builds PPO.

        Args:
            name: Either ``"dqn"`` or ``"ppo"``.

        Returns:
            A ``(env, model)`` pair; the model satisfies :class:`TrainablePredictModel`.
        """
        env = GymStudentSupportEnv()
        if name == "dqn":
            # DQN: deep value-based control. buffer_size/batch_size/learning_starts configure
            # EXPERIENCE REPLAY (sample random minibatches to decorrelate sequential transitions);
            # SB3 maintains a TARGET NETWORK internally to stabilize the bootstrap target.
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
            # PPO: on-policy actor-critic policy gradient. n_steps sets the rollout length
            # collected per update before the clipped-surrogate objective is optimized.
            model = algorithms["ppo"](
                "MlpPolicy",
                env,
                verbose=0,
                seed=seed,
                n_steps=32,
                batch_size=32,
                gamma=0.95,
            )
        return env, model

    training_rows: list[dict[str, int | float | str]] = []
    learned_policies: list[ModelPolicy] = []
    chunk = max(128, timesteps // 4)  # cap snapshot granularity at ~4 evaluation points

    for policy_name in ("dqn", "ppo"):
        env, model = build_model(policy_name)
        learned_steps = 0
        while learned_steps < timesteps:
            learn_chunk = min(chunk, timesteps - learned_steps)
            # reset_num_timesteps=False keeps the step counter and exploration schedule across
            # chunks; we snapshot evaluation performance after each chunk to build a learning curve.
            model.learn(total_timesteps=learn_chunk, reset_num_timesteps=False, progress_bar=False)
            learned_steps += learn_chunk
            evaluation_rows, _ = evaluate_policies(
                policies=[
                    ModelPolicy(
                        model=model,
                        observation_fn=observation_fn,
                        name=policy_name,
                    )
                ],
                scenario_ids=scenario_ids,
                episodes_per_scenario=1 if quick else 2,
                base_seed=seed,
                horizon=horizon,
            )
            training_rows.append(
                {
                    "policy": policy_name,
                    "step": learned_steps,
                    "mean_reward": evaluation_rows[0]["avg_reward"],
                    "mean_final_risk": evaluation_rows[0]["avg_final_risk"],
                }
            )
        learned_policies.append(
            ModelPolicy(
                model=model,
                observation_fn=observation_fn,
                name=policy_name,
            )
        )
        env.close()

    # Tabular Q-learning baseline trained on the SAME scenarios/horizon/seed for a fair comparison.
    q_learning = train_q_learning(
        episodes=240 if quick else 900,
        seed=seed,
        scenario_ids=scenario_ids,
        horizon=horizon,
    ).greedy_policy()
    # Evaluate all three policies (tabular baseline + the two deep agents) on identical conditions.
    comparison_rows, scenario_rows = evaluate_policies(
        policies=[q_learning, *learned_policies],
        scenario_ids=scenario_ids,
        episodes_per_scenario=2 if quick else 4,
        base_seed=seed,
        horizon=horizon,
    )
    for row in comparison_rows:
        row["family"] = _policy_family(str(row["policy"]))

    policy_gradient_notes = _policy_gradient_notes(comparison_rows)
    bridge_report = (
        "# Optional DRL Bridge\n\n"
        f"Ran DQN and PPO for {timesteps} timesteps on the same student-support environment family "
        f"used by tabular Q-learning, with seed {seed}.\n\n"
        "- DQN extends value-based control from a Q-table to a neural approximator.\n"
        "- PPO provides an actor-critic, policy-gradient reference point.\n"
        "- The comparison artifacts reuse the same scenario set, horizon, and seed family.\n"
    )
    return DRLComparisonResult(
        comparison_rows=comparison_rows,
        scenario_rows=scenario_rows,
        training_rows=training_rows,
        policy_gradient_notes=policy_gradient_notes,
        bridge_report=bridge_report,
    )


def _load_drl_dependencies() -> tuple[Any, Any, Any, dict[str, Any]]:
    """Import the optional deep-RL stack, converting any failure into OptionalDRLError.

    Imports are performed lazily here (rather than at module load) so the tabular ladder can be
    used without the heavy deep-RL extras installed. Any import failure is re-raised as
    :class:`OptionalDRLError` so callers can detect the missing-extras case and fall back.

    Returns:
        A tuple ``(gymnasium, numpy, gymnasium.spaces, {"dqn": DQN, "ppo": PPO})``.

    Raises:
        OptionalDRLError: If Gymnasium, NumPy, or Stable-Baselines3 cannot be imported.

    RL concept:
        Optional deep-RL dependency boundary; see docs/deep-rl.md.
    """
    try:
        import gymnasium as gym
        import numpy as np
        from gymnasium import spaces
        from stable_baselines3 import DQN, PPO
    except Exception as exc:  # pragma: no cover - import availability varies by machine
        raise OptionalDRLError(str(exc)) from exc
    return gym, np, spaces, {"dqn": DQN, "ppo": PPO}


def _seed_torch_if_available(seed: int) -> None:
    """Seed PyTorch (and request deterministic kernels) when it is installed.

    Stable-Baselines3 uses PyTorch under the hood, so seeding torch is necessary for reproducible
    deep-agent training. Because torch is optional, a missing import is silently ignored -- the
    rest of the comparison still runs deterministically for everything that does not depend on it.

    Args:
        seed: Master seed to apply to the torch global RNG.

    RL concept:
        Reproducibility of deep-RL training; see docs/evaluation-and-governance.md.
    """
    try:  # pragma: no cover - torch is optional on the execution machine
        import torch
    except Exception:
        return

    torch.manual_seed(seed)  # seed torch's global RNG so SB3 network init/sampling is reproducible
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)


def _policy_family(policy_name: str) -> str:
    """Map a policy name to its position on the RL method ladder.

    Tags each compared policy with the family of algorithm it represents so reports can group
    tabular vs deep and value-based vs policy-gradient methods. This is the labeling that makes
    the ladder (Q-learning -> DQN -> PPO) explicit in the comparison artifacts.

    Args:
        policy_name: One of ``"q_learning"``, ``"dqn"``, ``"ppo"`` (others map to ``"other"``).

    Returns:
        A family tag: ``"tabular_value_based"``, ``"deep_value_based"``,
        ``"actor_critic_policy_gradient"``, or ``"other"``.

    RL concept:
        Taxonomy of RL methods (value-based vs actor-critic policy gradient); see
        docs/policy-gradient-and-actor-critic.md.
    """
    if policy_name == "q_learning":
        return "tabular_value_based"  # tabular Q-learning: value-based control with a Q-table
    if policy_name == "dqn":
        return "deep_value_based"  # DQN: value-based control with a neural Q-approximator
    if policy_name == "ppo":
        return "actor_critic_policy_gradient"  # PPO: actor-critic, direct policy optimization
    return "other"


def _policy_gradient_notes(summary_rows: Sequence[dict[str, int | float | str]]) -> str:
    """Render a markdown note positioning PPO as the policy-gradient / actor-critic reference.

    Builds the narrative that ties the three compared methods together on the ladder: Q-learning
    learns values via a Bellman backup, DQN keeps that objective but uses a neural net, and PPO
    instead optimizes the policy directly (actor) while a critic estimates value. It then quotes
    each method's average reward/risk from the supplied summary rows for a same-environment
    comparison.

    Args:
        summary_rows: Aggregated comparison rows; must contain entries for the policy names
            ``"ppo"``, ``"dqn"``, and ``"q_learning"``.

    Returns:
        A markdown string contrasting on-policy actor-critic PPO with off-policy value learning.

    Raises:
        KeyError: If any of the required policy rows is missing from ``summary_rows``.

    RL concept:
        Policy-gradient and actor-critic framing vs value-based control; see
        docs/policy-gradient-and-actor-critic.md.

    Math:
        Policy-gradient ascent uses grad J = E[grad log pi(A|s) (G_t - b)]; the critic supplies the
        baseline b so that (G_t - b) is an advantage estimate.
    """
    # Index the aggregated rows by policy name so the narrative can quote each method's numbers.
    by_policy = {str(row["policy"]): row for row in summary_rows}
    ppo = by_policy["ppo"]
    dqn = by_policy["dqn"]
    q_learning = by_policy["q_learning"]
    return (
        "# Policy Gradient Coverage\n\n"
        "PPO is the showcase's policy-gradient and actor-critic reference point.\n\n"
        "## How the ideas connect\n\n"
        "- Q-learning learns action values with a Bellman backup.\n"
        "- DQN keeps that value-learning objective but swaps the Q-table for a neural network.\n"
        "- Policy-gradient methods optimize the policy directly instead of improving "
        "a value table first.\n"
        "- Actor-critic methods pair a policy learner with a value estimator.\n"
        "- PPO is the actor-critic implementation used in this optional bridge.\n\n"
        "## Same-environment comparison\n\n"
        f"- PPO average reward: {ppo['avg_reward']}\n"
        f"- PPO average final risk: {ppo['avg_final_risk']}\n"
        f"- DQN average reward: {dqn['avg_reward']}\n"
        f"- Tabular Q-learning average reward: {q_learning['avg_reward']}\n\n"
        "Interpretation: compare PPO's on-policy actor-critic behavior against DQN's "
        "off-policy value learning and the tabular control baseline.\n"
    )


def _coerce_int(value: object) -> int:
    """Coerce a scenario-id-like value (bool/int/float/str) into a plain int.

    The Gymnasium ``reset`` ``options`` dict carries values of unknown static type, so this helper
    normalizes a scenario id passed by a caller into the ``int`` the environment expects, rejecting
    incompatible types. The ``bool`` case is checked first because ``bool`` is a subclass of
    ``int`` in Python.

    Args:
        value: A scenario id supplied as a bool, int, float, or numeric string.

    Returns:
        The value as an ``int``.

    Raises:
        TypeError: If ``value`` is not a bool, int, float, or string.
    """
    if isinstance(value, bool):  # bool is a subclass of int; handle it before the int branch
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"Expected scenario id compatible value, got {type(value)!r}")
