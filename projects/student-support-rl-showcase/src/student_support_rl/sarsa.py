"""Tabular SARSA -- on-policy temporal-difference control.

What + why: SARSA is the on-policy sibling of Q-learning. It bootstraps from the value of the
action it *actually takes next* under its epsilon-greedy behaviour, so it learns the value of the
policy it follows (exploration included) rather than of the greedy policy. Its only structural
difference from Q-learning is the backup target.

RL concept:
    On-policy TD control -- the value-based rung of the ladder, beside off-policy Q-learning.
    See docs/value-based-learning.md and docs/math-notes.md.

Math:
    TD target uses the next CHOSEN action A':  target = R_{t+1} + gamma * Q(s', A')  (0 if terminal)
    TD error: delta = target - Q(s, A);  update: Q(s, A) <- Q(s, A) + alpha * delta
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from student_support_rl.environment import ACTION_LABELS, StudentSupportEnvironment, default_reward
from student_support_rl.policies import QLearningPolicy, greedy_action

StateKey = tuple[int, int, int, int, int, int]
"""Six-int tabular state key produced by ``StudentState.as_tuple()``."""


@dataclass
class SarsaResult:
    """Hold the learned action-value table and the per-episode training curve.

    What + why: SARSA is *on-policy* temporal-difference control; this container mirrors
    ``QLearningResult`` so the two TD methods are directly comparable in notebooks and
    artifact pipelines. RL concept: it sits on the value-based rung of the ladder
    (contextual bandit -> MDP -> Q-learning -> DQN -> policy gradient -> actor-critic -> PPO),
    one step beside off-policy Q-learning.

    Attributes:
        q_table: Action values Q(s,a), keyed by the six-int state tuple; each value is a
            list of four floats, one per action in ``ACTION_LABELS``.
        training_curve: One dict per episode with columns
            ``episode, scenario_id, total_reward, epsilon, steps`` (matches Q-learning).

    RL concept: see docs/value-based-learning.md.
    """

    q_table: dict[StateKey, list[float]]
    training_curve: list[dict[str, int | float]]

    def greedy_policy(self) -> QLearningPolicy:
        """Wrap the learned table in a greedy, deterministic evaluation policy.

        What + why: training is exploratory (epsilon-greedy) but deployment is greedy; we
        reuse ``QLearningPolicy`` because greedy action selection over a tabular Q is
        identical regardless of how Q was estimated. RL concept: greedy action
        a* = argmax_a Q(s,a) recovers the policy implied by the value function.

        Returns:
            A ``QLearningPolicy`` selecting argmax_a Q(s,a) over this table.

        RL concept: see docs/value-based-learning.md.
        """
        return QLearningPolicy(q_table=self.q_table)


def train_sarsa(
    *,
    episodes: int = 120,
    seed: int = 7,
    scenario_ids: tuple[int, ...] = (0, 1, 2, 3, 4),
    alpha: float = 0.35,
    gamma: float = 0.9,
    epsilon: float = 0.4,
    epsilon_decay: float = 0.97,
    epsilon_min: float = 0.05,
    horizon: int = 6,
) -> SarsaResult:
    """Train tabular SARSA, the canonical on-policy contrast to off-policy Q-learning.

    What + why: SARSA bootstraps from the value of the action it *actually takes next*
    under its epsilon-greedy behaviour, so during exploration it learns the value of the
    policy it follows rather than of a hypothetical greedy policy. Because the target folds in
    the cost of exploratory moves, SARSA *can* learn a more conservative value function than
    Q-learning when those exploratory mistakes are expensive (the textbook Cliff-Walking
    result); whether that gap appears, and its sign, depends on the environment's reward
    structure and the seed, so treat it as something to observe rather than assume. RL
    concept: SARSA is on-policy TD control on the value-based rung of the ladder
    (contextual bandit -> MDP -> Q-learning -> DQN -> policy gradient -> actor-critic -> PPO);
    its only structural difference from Q-learning is the backup target.

    Math:
        TD target uses the next *chosen* action A' (not the max):
            target = R_{t+1} + gamma * Q(s', A')        (0 if terminal)
        TD error: delta = target - Q(s, A)
        Update:   Q(s, A) <- Q(s, A) + alpha * delta
        Contrast (off-policy Q-learning):
            target = R_{t+1} + gamma * max_a' Q(s', a')
        Bellman optimality target (what Q-learning chases):
            Q*(s,a) = E[R_{t+1} + gamma * max_a' Q*(s',a')]
        Return being estimated: G_t = sum_k gamma^k R_{t+k+1}.

    Args:
        episodes: Number of training episodes; must be positive.
        seed: Base RNG seed; drives both action sampling and per-episode env resets.
        scenario_ids: Scenarios cycled through across episodes (round-robin).
        alpha: Step size for the TD update.
        gamma: Discount factor for the return G_t.
        epsilon: Initial exploration probability for the epsilon-greedy behaviour policy.
        epsilon_decay: Multiplicative decay applied to epsilon after each episode.
        epsilon_min: Floor for epsilon so exploration never fully stops.
        horizon: Episode length passed to the environment.

    Returns:
        A ``SarsaResult`` with the learned Q-table and the per-episode training curve.

    Raises:
        ValueError: If ``episodes`` is not positive.

    RL concept: see docs/value-based-learning.md and docs/mdp-and-environment.md.
    """
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    rng = random.Random(seed)
    q_table: dict[StateKey, list[float]] = {}
    training_curve: list[dict[str, int | float]] = []
    current_epsilon = epsilon

    for episode in range(1, episodes + 1):
        scenario_id = scenario_ids[(episode - 1) % len(scenario_ids)]
        environment = StudentSupportEnvironment(horizon=horizon, reward_fn=default_reward)
        state = environment.reset(seed=seed + episode, scenario_id=scenario_id)
        total_reward = 0.0
        steps = 0

        # On-policy: commit to the FIRST action before the loop, then carry (s, A) forward.
        state_key = state.as_tuple()
        q_table.setdefault(state_key, [0.0] * len(ACTION_LABELS))
        action = _epsilon_greedy_action(q_table[state_key], current_epsilon, rng)

        while not environment.is_done():
            transition = environment.step(action)
            next_key = transition.state.as_tuple()
            q_table.setdefault(next_key, [0.0] * len(ACTION_LABELS))
            # On-policy: sample the ACTUALLY-chosen next action A' under epsilon-greedy.
            next_action = _epsilon_greedy_action(q_table[next_key], current_epsilon, rng)

            old_value = q_table[state_key][action]
            # SARSA backup bootstraps from Q[s'][A'] (the chosen A'), NOT max_a' Q[s'][a'].
            future_value = q_table[next_key][next_action]
            target = transition.reward + (0.0 if transition.done else gamma * future_value)
            # TD update: delta = target - Q(s, A); Q(s, A) += alpha * delta.
            q_table[state_key][action] = old_value + alpha * (target - old_value)

            total_reward += transition.reward
            steps += 1
            # Carry the (state, action) pair forward: tomorrow's s, A are today's s', A'.
            state_key, action = next_key, next_action

        training_curve.append(
            {
                "episode": episode,
                "scenario_id": scenario_id,
                "total_reward": round(total_reward, 4),
                "epsilon": round(current_epsilon, 4),
                "steps": steps,
            }
        )
        current_epsilon = max(epsilon_min, current_epsilon * epsilon_decay)

    return SarsaResult(q_table=q_table, training_curve=training_curve)


def _epsilon_greedy_action(
    action_values: list[float],
    epsilon: float,
    rng: random.Random,
) -> int:
    """Pick a uniform-random action with probability epsilon, else the greedy action.

    What + why: this is the single behaviour policy SARSA both follows and evaluates; because
    SARSA is on-policy, the exploration here directly shapes the learned values. RL concept:
    epsilon-greedy balances exploration vs exploitation over action values Q(s,a).

    Args:
        action_values: Current Q(s, .) estimates for the state.
        epsilon: Probability of taking a uniform-random (exploratory) action.
        rng: Seeded RNG for deterministic sampling.

    Returns:
        The chosen action index.

    RL concept: see docs/value-based-learning.md.
    """
    if rng.random() < epsilon:  # explore: probability epsilon
        return rng.randrange(len(action_values))
    return greedy_action(action_values)  # exploit: argmax_a Q(s,a)
