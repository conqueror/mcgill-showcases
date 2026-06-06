"""Tabular softmax REINFORCE (Monte-Carlo policy gradient) for student support.

What + why: this module demystifies "policy gradient" with no neural network. Instead of
learning action values and acting greedily (Q-learning), it optimizes the policy pi_theta
*directly* by gradient ascent on expected return, using one logit per (state, action) in a
lookup table. The update is exactly the score-function estimator that PPO scales up, so a
learner can read the whole algorithm in a few lines here before meeting it behind a deep network.

RL concept:
    Monte-Carlo policy gradient -- the policy-gradient rung of the ladder (contextual bandit ->
    MDP -> Q-learning -> DQN -> policy gradient -> actor-critic -> PPO). See
    docs/policy-gradient-and-actor-critic.md, docs/value-based-learning.md, docs/math-notes.md,
    and docs/glossary.md.

Math:
    pi(a|s) = exp(theta_{s,a}) / sum_{a'} exp(theta_{s,a'})
    return G_t = sum_k gamma^k R_{t+k+1}        (reward after acting = R_{t+1})
    theta_{s,a'} <- theta_{s,a'} + alpha * (G_t - b) * (1[a'=A_t] - pi(a'|s))
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from student_support_rl.environment import (
    ACTION_LABELS,
    StudentState,
    StudentSupportEnvironment,
    default_reward,
)
from student_support_rl.policies import greedy_action

StateKey = tuple[int, int, int, int, int, int]
"""Discrete six-int state signature (see ``StudentState.as_tuple``)."""


def softmax(logits: list[float]) -> list[float]:
    """Map per-action logits to a valid probability distribution.

    What + WHY: a softmax policy pi(a|s) turns unconstrained per-action logits
    (``theta[s]``) into action probabilities, so REINFORCE can optimize the policy
    directly without ever learning a value table first. We subtract the max logit
    before exponentiating so very large logits cannot overflow ``math.exp`` -- the
    shift cancels in the ratio and leaves the distribution unchanged.

    RL concept: this is the policy parameterization rung of the ladder
    (contextual bandit -> MDP -> Q-learning -> DQN -> *policy gradient* ->
    actor-critic -> PPO); the same softmax-over-logits idea is what PPO scales up
    with a neural network in place of this lookup table.
    See docs/policy-gradient-and-actor-critic.md and docs/math-notes.md.

    Math:
        pi(a|s) = exp(theta_{s,a}) / sum_{a'} exp(theta_{s,a'})

    Args:
        logits: Per-action logits ``theta[s]``; any finite real values.

    Returns:
        Probabilities in ``[0, 1]`` that sum to ``1.0`` (one entry per logit).

    Raises:
        ValueError: If ``logits`` is empty.
    """
    if not logits:
        raise ValueError("logits must be non-empty")
    shift = max(logits)  # numerical stability: subtract max so exp() cannot overflow
    exponentials = [math.exp(logit - shift) for logit in logits]
    normalizer = sum(exponentials)
    return [value / normalizer for value in exponentials]


@dataclass
class ReinforcePolicy:
    """Deterministic greedy policy read off learned softmax logits.

    What + WHY: after training we deploy the *mode* of pi_theta -- the highest-logit
    action per state -- because evaluation wants a single committed decision rather
    than a stochastic sample. Unseen states fall back to action ``0``
    (``no_intervention``), the conservative do-nothing baseline, so the policy stays
    total over the whole state space.

    RL concept: greedy readout of a policy-gradient policy; the policy-gradient rung
    of the ladder (contextual bandit -> MDP -> Q-learning -> DQN -> *policy gradient*
    -> actor-critic -> PPO). See docs/policy-gradient-and-actor-critic.md.

    Attributes:
        theta: Learned per-state logits, one list of length ``len(ACTION_LABELS)``.
        name: Policy identifier surfaced in evaluation tables.
    """

    theta: dict[StateKey, list[float]]
    name: str = "reinforce"

    def reset(self) -> None:
        """Reset per-episode state (none here; satisfies the Policy protocol)."""
        return None

    def select_action(self, state: StudentState) -> int:
        """Return argmax_a theta[s], or action 0 for an unseen state.

        Args:
            state: Current environment observation.

        Returns:
            The greedy action index under the learned logits.
        """
        logits = self.theta.get(state.as_tuple())
        if logits is None:
            return 0  # safe fallback: do-nothing baseline for never-visited states
        return greedy_action(logits)


@dataclass
class ReinforceResult:
    """Container for trained softmax logits and the per-episode learning curve.

    What + WHY: bundles the learned policy parameters with the training trace so a
    learner can both deploy a greedy policy and plot how episode return improved as
    the policy gradient pushed probability mass toward better actions.

    RL concept: output of Monte-Carlo policy gradient (REINFORCE), the
    *policy gradient* rung of the ladder. See docs/policy-gradient-and-actor-critic.md.

    Attributes:
        theta: Learned per-state logits ``theta[s]`` (length ``len(ACTION_LABELS)``).
        training_curve: One row per episode with keys ``episode``, ``scenario_id``,
            ``total_reward``, ``baseline`` and ``steps``.
    """

    theta: dict[StateKey, list[float]]
    training_curve: list[dict[str, int | float]]

    def greedy_policy(self) -> ReinforcePolicy:
        """Build the deterministic greedy policy from the learned logits.

        Returns:
            A ``ReinforcePolicy`` (``name='reinforce'``) choosing argmax theta[s].
        """
        return ReinforcePolicy(theta=self.theta)


def train_reinforce(
    *,
    episodes: int = 400,
    seed: int = 7,
    scenario_ids: tuple[int, ...] = (0, 1, 2, 3, 4),
    alpha: float = 0.1,
    gamma: float = 0.9,
    horizon: int = 6,
    use_baseline: bool = True,
) -> ReinforceResult:
    """Train a tabular softmax policy with Monte-Carlo policy gradient (REINFORCE).

    What + WHY: REINFORCE optimizes the policy pi_theta *directly* by gradient ascent
    on expected return, instead of first learning action values and acting greedily as
    Q-learning does. We keep it tabular (per-state logits, no neural net) to demystify
    "policy gradient": the update below is exactly the idea PPO scales up. Each episode
    is rolled out on-policy, then every visited state-action pair is nudged by the
    score-function estimator weighted by the (baseline-subtracted) Monte-Carlo return.

    RL concept: Monte-Carlo policy gradient, the *policy gradient* rung of the ladder
    (contextual bandit -> MDP -> Q-learning -> DQN -> *policy gradient* -> actor-critic
    -> PPO). The mean-return baseline is the simplest variance reduction and the seed of
    the "critic" that actor-critic methods learn. See docs/policy-gradient-and-actor-critic.md,
    docs/mdp-and-environment.md and docs/glossary.md.

    Math:
        return G_t = sum_k gamma^k R_{t+k+1}
        grad_theta J = E[ grad_theta log pi(A_t|s_t) * (G_t - b) ]
        d/dtheta_{s,a'} log pi(A_t|s_t) = 1[a'=A_t] - pi(a'|s_t)
        theta_{s,a'} <- theta_{s,a'} + alpha * (G_t - b) * (1[a'=A_t] - pi(a'|s_t))
    where reward after acting = R_{t+1} and the baseline b is the episode mean of G_t.

    Args:
        episodes: Number of on-policy training episodes (must be positive).
        seed: Seed for the action-sampling RNG and per-episode environment resets.
        scenario_ids: Scenario ids cycled through, one per episode.
        alpha: Policy-gradient step size (learning rate).
        gamma: Discount factor used to form the returns ``G_t``.
        horizon: Episode length passed to the environment.
        use_baseline: If true, subtract the episode-mean return as a baseline ``b``;
            otherwise ``b = 0``.

    Returns:
        A ``ReinforceResult`` with the learned logits and the per-episode curve.

    Raises:
        ValueError: If ``episodes`` is not positive.
    """
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    rng = random.Random(seed)
    action_count = len(ACTION_LABELS)
    theta: dict[StateKey, list[float]] = {}
    training_curve: list[dict[str, int | float]] = []

    for episode in range(1, episodes + 1):
        scenario_id = scenario_ids[(episode - 1) % len(scenario_ids)]
        environment = StudentSupportEnvironment(horizon=horizon, reward_fn=default_reward)
        state = environment.reset(seed=seed + episode, scenario_id=scenario_id)
        trajectory: list[tuple[StateKey, int, float]] = []
        total_reward = 0.0

        # Roll out on-policy: sample A ~ softmax(theta[s]) and record (s, A, r).
        while not environment.is_done():
            state_key = state.as_tuple()
            theta.setdefault(state_key, [0.0] * action_count)
            action = _sample_action(theta[state_key], rng)  # A ~ pi_theta(.|s)
            transition = environment.step(action)
            trajectory.append((state_key, action, transition.reward))
            total_reward += transition.reward
            state = transition.state

        # Monte-Carlo returns G_t = sum_{k>=t} gamma^{k-t} r_k (backward accumulation).
        returns = [0.0] * len(trajectory)
        running_return = 0.0
        for step_index in range(len(trajectory) - 1, -1, -1):
            running_return = trajectory[step_index][2] + gamma * running_return
            returns[step_index] = running_return

        # Baseline b: episode-mean return for variance reduction (else 0).
        baseline = (sum(returns) / len(returns)) if use_baseline else 0.0

        # Policy-gradient ascent on every visited (s_t, A_t) and every action a'.
        for step_index, (state_key, action, _reward) in enumerate(trajectory):
            probabilities = softmax(theta[state_key])  # pi(.|s_t) at current theta
            advantage = returns[step_index] - baseline  # (G_t - b) scales the gradient
            for candidate in range(action_count):
                indicator = 1.0 if candidate == action else 0.0
                grad_log = indicator - probabilities[candidate]  # d log pi(A_t|s_t)/dtheta_{s,a'}
                theta[state_key][candidate] += alpha * advantage * grad_log  # REINFORCE update

        training_curve.append(
            {
                "episode": episode,
                "scenario_id": scenario_id,
                "total_reward": round(total_reward, 4),
                "baseline": round(baseline, 4),
                "steps": len(trajectory),
            }
        )

    return ReinforceResult(theta=theta, training_curve=training_curve)


def _sample_action(logits: list[float], rng: random.Random) -> int:
    """Sample an action index A ~ softmax(logits) via inverse-CDF.

    What + WHY: REINFORCE is on-policy, so exploration must come from the stochastic
    policy itself; we draw from pi_theta(.|s) rather than using epsilon-greedy.

    Args:
        logits: Per-action logits ``theta[s]``.
        rng: Seeded RNG so sampling is deterministic for a given seed.

    Returns:
        The sampled action index.
    """
    probabilities = softmax(logits)
    threshold = rng.random()
    cumulative = 0.0
    for action, probability in enumerate(probabilities):
        cumulative += probability
        if threshold <= cumulative:
            return action
    return len(probabilities) - 1  # guard against float rounding at the tail
