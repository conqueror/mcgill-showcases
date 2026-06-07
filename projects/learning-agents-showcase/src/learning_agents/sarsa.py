"""Tabular SARSA -- on-policy temporal-difference control for the agent-decision MDP.

What + why: SARSA is the on-policy sibling of Q-learning. It bootstraps from the value of the
action it *actually takes next* under its epsilon-greedy behaviour, so it learns the value of the
policy it follows (exploration included) rather than of a hypothetical greedy policy. Applied to the
agent-decision MDP (:mod:`learning_agents.environment`), it learns the orchestration policy --
when to answer, retrieve, clarify, or escalate -- directly from the judge-rubric reward
(:func:`learning_agents.reward.judge_reward`, re-exported as ``environment.default_reward``). Its
only structural difference from Q-learning is the backup target: SARSA uses the *chosen* next action
A', off-policy Q-learning uses the *max* over next actions.

RL concept:
    On-policy TD control -- the value-based rung of the ladder, beside off-policy Q-learning
    (contextual bandit -> MDP -> Q-learning -> SARSA -> DQN -> policy gradient -> actor-critic
    -> PPO). The learned table is consumed by :class:`learning_agents.policies.QTablePolicy`.

Math:
    TD target uses the next CHOSEN action A':  target = R_{t+1} + gamma * Q(s', A')  (0 if terminal)
    TD error: delta = target - Q(s, A);  update: Q(s, A) <- Q(s, A) + alpha * delta
    Contrast (off-policy Q-learning): target = R_{t+1} + gamma * max_a' Q(s', a').
    Return being estimated: G_t = sum_k gamma^k R_{t+k+1}.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from learning_agents.environment import (
    ACTION_LABELS,
    AgentDecisionEnvironment,
    default_reward,
)
from learning_agents.policies import QTablePolicy, greedy_action

StateKey = tuple[int, int, int, int, int, int, int]
"""Seven-int tabular state key produced by :meth:`learning_agents.environment.AgentState.as_tuple`.

RL concept: the discrete state key (step, intent, difficulty, ambiguity, evidence, attempts, budget)
behind tabular value lookup; its width matches the agent-decision state exactly.
"""


@dataclass
class SarsaResult:
    """Hold the learned action-value table and the per-episode training curve.

    What + why: SARSA is *on-policy* temporal-difference control; this container is shaped to mirror
    a Q-learning result so the two TD methods are directly comparable in notebooks and artifact
    pipelines. The learned ``q_table`` is the deliverable -- a greedy policy over it
    (:meth:`greedy_policy`) is the trained agent.

    Attributes:
        q_table: Action values Q(s, a), keyed by the seven-int state tuple
            (:meth:`learning_agents.environment.AgentState.as_tuple`); each value is a list of four
            floats, one per action in :data:`learning_agents.environment.ACTION_LABELS`.
        training_curve: One dict per episode with keys
            ``episode, scenario_id, total_reward, epsilon, steps`` -- the structured learning-curve
            row schema (suitable for a reporting module to serialize to CSV).

    RL concept: the action-value table Q(s, a) and its learning curve, the artifacts of value-based
    TD control.
    """

    q_table: dict[StateKey, list[float]]
    training_curve: list[dict[str, int | float]]

    def greedy_policy(self) -> QTablePolicy:
        """Wrap the learned table in a greedy, deterministic evaluation policy.

        What + why: training is exploratory (epsilon-greedy) but deployment is greedy; we wrap the
        table in :class:`learning_agents.policies.QTablePolicy` because greedy action selection over
        a tabular Q is identical regardless of how Q was estimated, and that policy already provides
        the safe all-zeros fallback for states never visited during training.

        Returns:
            A :class:`learning_agents.policies.QTablePolicy` selecting argmax_a Q(s, a) over this
            table.

        RL concept: the greedy policy a* = argmax_a Q(s, a) recovers the behaviour implied by the
        value function (value-based control).
        """
        return QTablePolicy(q_table=self.q_table)


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
    horizon: int = 5,
) -> SarsaResult:
    """Train tabular SARSA, the canonical on-policy contrast to off-policy Q-learning.

    What + why: SARSA bootstraps from the value of the action it *actually takes next* under its
    epsilon-greedy behaviour, so during exploration it learns the value of the policy it follows
    rather than of a hypothetical greedy policy. Because the target folds in the cost of exploratory
    moves, SARSA *can* learn a more conservative value function than Q-learning when those
    exploratory mistakes are expensive (the textbook Cliff-Walking result); whether that gap
    appears, and its sign, depends on the environment's reward structure and the seed, so treat it
    as something to observe rather than assume. RL concept: SARSA is on-policy TD control on the
    value-based rung of the ladder (contextual bandit -> MDP -> Q-learning -> SARSA -> DQN -> policy
    gradient -> actor-critic -> PPO); its only structural difference from Q-learning is the backup
    target.

    Math:
        TD target uses the next *chosen* action A' (not the max):
            target = R_{t+1} + gamma * Q(s', A')        (0 if terminal)
        TD error: delta = target - Q(s, A)
        Update:   Q(s, A) <- Q(s, A) + alpha * delta
        Contrast (off-policy Q-learning):
            target = R_{t+1} + gamma * max_a' Q(s', a')
        Bellman optimality target (what Q-learning chases):
            Q*(s, a) = E[R_{t+1} + gamma * max_a' Q*(s', a')]
        Return being estimated: G_t = sum_k gamma^k R_{t+k+1}.

    Args:
        episodes: Number of training episodes; must be positive.
        seed: Base RNG seed; drives both action sampling and per-episode env resets, so a fixed seed
            reproduces the whole run.
        scenario_ids: Scenarios cycled through across episodes (round-robin) so one table covers the
            whole start-state distribution.
        alpha: Step size for the TD update.
        gamma: Discount factor for the return G_t.
        epsilon: Initial exploration probability for the epsilon-greedy behaviour policy.
        epsilon_decay: Multiplicative decay applied to epsilon after each episode.
        epsilon_min: Floor for epsilon so exploration never fully stops.
        horizon: Episode length H passed to the environment.

    Returns:
        A :class:`SarsaResult` with the learned Q-table and the per-episode training curve.

    Raises:
        ValueError: If ``episodes`` is not positive.

    RL concept: on-policy temporal-difference control -- learning Q(s, a) for the epsilon-greedy
    behaviour policy from sampled (s, a, r, s', a') transitions.
    """
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    rng = random.Random(seed)
    q_table: dict[StateKey, list[float]] = {}
    training_curve: list[dict[str, int | float]] = []
    current_epsilon = epsilon

    for episode in range(1, episodes + 1):
        scenario_id = scenario_ids[(episode - 1) % len(scenario_ids)]
        environment = AgentDecisionEnvironment(horizon=horizon, reward_fn=default_reward)
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
            target = _sarsa_backup_target(
                reward=transition.reward,
                gamma=gamma,
                next_action_values=q_table[next_key],
                next_action=next_action,
                done=transition.done,
            )
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


def _sarsa_backup_target(
    *,
    reward: float,
    gamma: float,
    next_action_values: list[float],
    next_action: int,
    done: bool,
) -> float:
    """Compute the on-policy SARSA bootstrap target R + gamma * Q(s', A') (0 future if terminal).

    What + why: this is the *one* line that makes SARSA on-policy rather than off-policy, factored
    out so the training loop and the tests exercise the exact same backup. It indexes the next-state
    action values at the *chosen* next action ``A'`` -- the action the epsilon-greedy behaviour
    policy actually takes -- and deliberately does NOT take ``max`` over next actions. Swapping this
    indexing for ``max(next_action_values)`` would turn the algorithm into off-policy Q-learning;
    keeping it as ``next_action_values[next_action]`` is what makes the method learn the value of
    the behaviour policy it follows (exploration included).

    Args:
        reward: The immediate reward R_{t+1} from the transition.
        gamma: Discount factor applied to the bootstrapped future value.
        next_action_values: The Q(s', .) row for the next state.
        next_action: The chosen next action A' under the epsilon-greedy behaviour policy; its value
            (not the max) is bootstrapped from.
        done: Whether the transition was terminal; if True there is no future value to bootstrap.

    Returns:
        ``reward`` if ``done`` else ``reward + gamma * next_action_values[next_action]``.

    RL concept: the on-policy TD(0) target -- SARSA bootstraps from the *chosen* next action A',
    in contrast to off-policy Q-learning's ``reward + gamma * max_a' Q(s', a')``.
    """
    if done:
        return reward
    # On-policy: bootstrap from the CHOSEN next action A', not max_a' Q(s', a') (off-policy).
    return reward + gamma * next_action_values[next_action]


def _epsilon_greedy_action(
    action_values: list[float],
    epsilon: float,
    rng: random.Random,
) -> int:
    """Pick a uniform-random action with probability epsilon, else the greedy action.

    What + why: this is the single behaviour policy SARSA both follows and evaluates; because SARSA
    is on-policy, the exploration here directly shapes the learned values (unlike off-policy
    Q-learning, whose target ignores how the next action was actually picked). RL concept:
    epsilon-greedy balances exploration vs exploitation over action values Q(s, a).

    Args:
        action_values: Current Q(s, .) estimates for the state.
        epsilon: Probability of taking a uniform-random (exploratory) action.
        rng: Seeded RNG for deterministic sampling.

    Returns:
        The chosen action index in ``range(len(action_values))``.

    RL concept: the epsilon-greedy behaviour policy whose value SARSA estimates on-policy.
    """
    if rng.random() < epsilon:  # explore: probability epsilon
        return rng.randrange(len(action_values))
    return greedy_action(action_values)  # exploit: argmax_a Q(s, a)
