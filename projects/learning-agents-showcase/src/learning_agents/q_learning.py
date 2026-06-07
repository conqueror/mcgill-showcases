"""Tabular off-policy temporal-difference control via Q-learning for the agent-decision MDP.

What + why: this module learns the agent's *orchestration policy* -- when to answer, retrieve,
clarify, or escalate -- by one-step Q-learning, the off-policy companion to SARSA. The agent behaves
with an exploratory epsilon-greedy policy but bootstraps every backup from ``max_a' Q(s', a')`` --
the value of a *greedy* next action it may not actually take. That decoupling of the behaviour
policy from the target is exactly what makes Q-learning off-policy and lets it chase the Bellman
optimality action values Q*(s,a) regardless of how exploratory the data generation was. On the
algorithm ladder
(contextual bandit -> MDP -> Q-learning -> DQN -> policy gradient -> actor-critic -> PPO) this is
the value-based rung that DQN later scales to function approximation.

It composes directly on :mod:`learning_agents.environment` (the agent-decision MDP),
:mod:`learning_agents.reward` (the judge-rubric reward, via the environment's default), and
:mod:`learning_agents.policies` (``QTablePolicy`` wraps the learned table, ``greedy_action`` is the
argmax). It imports only from this package.

RL concept:
    Off-policy TD control / Q-learning, contrasted with on-policy SARSA; the learned table
    approaches the Bellman optimality fixed point Q*(s,a).

Math:
    TD target (greedy bootstrap):  target = R_{t+1} + gamma * max_a' Q(s', a')   (0 if terminal)
    TD error:                      delta = target - Q(s, A)
    Update:                        Q(s, A) <- Q(s, A) + alpha * delta
    Bellman optimality (the fixed point Q-learning chases):
        Q*(s,a) = E[R_{t+1} + gamma * max_a' Q*(s',a')]
    Return being estimated:        G_t = sum_k gamma^k R_{t+k+1}.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from learning_agents.environment import ACTION_LABELS, AgentDecisionEnvironment, default_reward
from learning_agents.policies import QTablePolicy, greedy_action

__all__ = [
    "QLearningResult",
    "StateKey",
    "q_table_rows",
    "train_q_learning",
]

StateKey = tuple[int, int, int, int, int, int, int]
"""Seven-int tabular state key produced by :meth:`AgentState.as_tuple`.

The fields are ``(step, intent, difficulty, ambiguity, evidence, attempts, budget)`` -- the discrete
state of the agent-decision MDP, in declaration order, used to key the Q-table.
"""


@dataclass
class QLearningResult:
    """Hold the learned action-value table and the per-episode training curve.

    What + why: bundles the estimate of Q*(s,a) with a diagnostic learning curve so a Q-learning run
    is reproducible and directly comparable to its on-policy twin (an identically shaped SARSA
    result) in notebooks and artifact pipelines. RL concept: it sits on the value-based rung of
    the ladder (contextual bandit -> MDP -> Q-learning -> DQN -> policy gradient -> actor-critic
    -> PPO), one step beside on-policy SARSA.

    Attributes:
        q_table: Action values Q(s,a), keyed by the seven-int state tuple
            (:meth:`AgentState.as_tuple`); each value is a list of four floats, one per action in
            :data:`learning_agents.environment.ACTION_LABELS`.
        training_curve: One dict per episode with columns
            ``episode, scenario_id, total_reward, epsilon, steps``.

    RL concept: tabular action-value storage Q(s,a) plus the learning curve that diagnoses
    convergence.
    """

    q_table: dict[StateKey, list[float]]
    training_curve: list[dict[str, int | float]]

    def greedy_policy(self) -> QTablePolicy:
        """Wrap the learned table in a greedy, deterministic evaluation policy.

        What + why: training explores with epsilon-greedy, but because Q-learning is off-policy its
        target already estimates the *greedy* value Q*(s,a), so deployment simply reads off that
        greedy action. The returned :class:`~learning_agents.policies.QTablePolicy` also supplies a
        safe all-zeros fallback for states never seen in training (yielding ``answer_direct``). RL
        concept: greedy action a* = argmax_a Q(s,a) recovers the optimal policy implied by the
        learned value function.

        Returns:
            A :class:`~learning_agents.policies.QTablePolicy` selecting argmax_a Q(s,a) over this
            table (named ``"q_table"``).

        RL concept: deriving the greedy control policy from a learned Q-table (value-based control).
        """
        return QTablePolicy(q_table=self.q_table)


def train_q_learning(
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
) -> QLearningResult:
    """Train tabular Q-learning, the canonical off-policy contrast to on-policy SARSA.

    What + why: the agent explores with an epsilon-greedy *behaviour* policy but always backs up
    toward ``max_a' Q(s', a')`` -- the value of the *greedy* next action rather than the one it
    actually takes. That max is the entire off-policy mechanism: it lets the update target the
    optimal policy while the data is generated by an exploratory one, so the table converges toward
    the Bellman optimality action values Q*(s,a). Its only structural difference from SARSA is this
    backup target. RL concept: off-policy TD control on the value-based rung of the ladder
    (contextual bandit -> MDP -> Q-learning -> DQN -> policy gradient -> actor-critic -> PPO); DQN
    later replaces the table with a neural function approximator.

    The agent learns against the agent-decision MDP
    (:class:`~learning_agents.environment.AgentDecisionEnvironment`) under its default judge-rubric
    reward, so what it discovers is a routing policy: answer easy requests directly, retrieve to
    ground harder ones, clarify ambiguous ones, and escalate only when a human is truly warranted.

    Math:
        TD target uses the greedy next action (the max), not the chosen one:
            target = R_{t+1} + gamma * max_a' Q(s', a')   (0 if terminal)
        TD error: delta = target - Q(s, A)
        Update:   Q(s, A) <- Q(s, A) + alpha * delta
        Contrast (on-policy SARSA target):
            target = R_{t+1} + gamma * Q(s', A')          where A' is the chosen next action
        Bellman optimality target (the fixed point Q-learning chases):
            Q*(s,a) = E[R_{t+1} + gamma * max_a' Q*(s',a')]
        Return being estimated: G_t = sum_k gamma^k R_{t+k+1}.

    Args:
        episodes: Number of training episodes; must be positive.
        seed: Base RNG seed; drives both action sampling and per-episode env resets (start jitter).
        scenario_ids: Scenarios cycled through across episodes (round-robin); each indexes
            :data:`learning_agents.environment.SCENARIOS`.
        alpha: Step size for the TD update.
        gamma: Discount factor for the return G_t.
        epsilon: Initial exploration probability for the epsilon-greedy behaviour policy.
        epsilon_decay: Multiplicative decay applied to epsilon after each episode.
        epsilon_min: Floor for epsilon so exploration never fully stops.
        horizon: Episode length H passed to the environment.

    Returns:
        A :class:`QLearningResult` with the learned Q-table and the per-episode training curve.

    Raises:
        ValueError: If ``episodes`` is not positive.

    RL concept: off-policy temporal-difference control -- the greedy backup is what separates
    Q-learning from on-policy SARSA and points the estimate at Q*(s,a).
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

        while not environment.is_done():
            state_key = state.as_tuple()
            q_table.setdefault(state_key, [0.0] * len(ACTION_LABELS))
            # Behaviour policy: act epsilon-greedily (this is what makes the data exploratory).
            action = _epsilon_greedy_action(q_table[state_key], current_epsilon, rng)
            transition = environment.step(action)
            next_key = transition.state.as_tuple()
            q_table.setdefault(next_key, [0.0] * len(ACTION_LABELS))

            old_value = q_table[state_key][action]
            # Off-policy: bootstrap from max_a' Q[s'][a'] (greedy next action), NOT Q[s'][A'].
            future_value = max(q_table[next_key])
            target = transition.reward + (0.0 if transition.done else gamma * future_value)
            # TD update: delta = target - Q(s, A); Q(s, A) += alpha * delta.
            q_table[state_key][action] = old_value + alpha * (target - old_value)

            total_reward += transition.reward
            steps += 1
            state = transition.state

        training_curve.append(
            {
                "episode": episode,
                "scenario_id": scenario_id,
                "total_reward": round(total_reward, 4),
                "epsilon": round(current_epsilon, 4),
                "steps": steps,
            }
        )
        # Multiplicative epsilon decay, floored at epsilon_min so exploration never stops.
        current_epsilon = max(epsilon_min, current_epsilon * epsilon_decay)

    return QLearningResult(q_table=q_table, training_curve=training_curve)


def q_table_rows(q_table: dict[StateKey, list[float]]) -> list[dict[str, int | float]]:
    """Flatten the learned Q-table into one tidy CSV row per (state, action) pair.

    What + why: unpacks each seven-int state key into named columns and emits one row per action
    with its value Q(s,a), giving a long-format table for export, plotting, or inspecting which
    orchestration move the policy prefers in each state. State keys are sorted so the output is
    deterministic and diff-friendly. RL concept: a direct read-out of the learned action-value
    function Q(s,a); no learning happens here.

    The artifact this row schema is intended for is
    ``artifacts/q_learning/q_table.csv`` (long format, one row per state-action pair).

    Args:
        q_table: Action values Q(s,a) keyed by the seven-int state tuple
            (:meth:`AgentState.as_tuple`).

    Returns:
        Rows with columns
        ``step, intent, difficulty, ambiguity, evidence, attempts, budget, action, q_value``
        (one row per state-action pair, q_value rounded to 6 dp).

    RL concept: reading off the learned action-value function Q(s,a) for inspection and export.
    """
    rows: list[dict[str, int | float]] = []
    for state_key in sorted(q_table):
        step, intent, difficulty, ambiguity, evidence, attempts, budget = state_key
        for action, value in enumerate(q_table[state_key]):
            rows.append(
                {
                    "step": step,
                    "intent": intent,
                    "difficulty": difficulty,
                    "ambiguity": ambiguity,
                    "evidence": evidence,
                    "attempts": attempts,
                    "budget": budget,
                    "action": action,
                    "q_value": round(value, 6),
                }
            )
    return rows


def _epsilon_greedy_action(
    action_values: list[float],
    epsilon: float,
    rng: random.Random,
) -> int:
    """Pick a uniform-random action with probability epsilon, else the greedy action.

    What + why: this is Q-learning's *behaviour* policy -- the exploratory rule that generates
    training data. Because Q-learning is off-policy, this exploration is decoupled from the greedy
    *target* the update chases, so wide exploration here does not bias the estimate of Q*(s,a). RL
    concept: epsilon-greedy balances exploration vs exploitation over action values Q(s,a).

    Args:
        action_values: Current Q(s, .) estimates for the state.
        epsilon: Probability of taking a uniform-random (exploratory) action.
        rng: Seeded RNG for deterministic sampling.

    Returns:
        The chosen action index.

    RL concept: epsilon-greedy exploration as the off-policy behaviour distribution.
    """
    if rng.random() < epsilon:  # explore: probability epsilon
        return rng.randrange(len(action_values))
    return greedy_action(action_values)  # exploit: argmax_a Q(s,a)
