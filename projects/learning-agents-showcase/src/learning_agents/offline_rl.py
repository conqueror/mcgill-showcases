"""Learn an orchestration policy from a *fixed log* of trajectories (offline / batch RL).

What + why: every other learner in this showcase is *online* -- it interacts with the live
:class:`~learning_agents.environment.AgentDecisionEnvironment` and chooses which transitions to
collect. Offline (batch) RL is the opposite and the realistic setting for agents: you are handed a
frozen log of (state, action, reward, next_state) tuples that some *behaviour* policy already
produced in production, and you must learn a better policy from that log alone, with no new
interaction. That constraint is the whole lesson -- you can only trust what the data covers, and a
policy that wanders off the logged support has no evidence behind it (the distribution-shift /
out-of-distribution-action problem at the heart of offline RL).

This module has two pieces:

* :func:`collect_logged_dataset` simulates that production log. It rolls out an *epsilon-soft*
  behaviour policy (a sensible base router that takes a uniform-random action with probability
  ``epsilon``) and records each transition together with the behaviour policy's probability of the
  action it took. That logged probability is what makes the dataset reusable for off-policy
  evaluation later (importance sampling needs it); here it documents how the data was generated.
* :func:`fitted_q_iteration` is the offline learner: tabular Fitted-Q Iteration. It sweeps the
  *static* dataset repeatedly, applying the Bellman optimality backup to the logged
  (state, action) pairs only, until the table stops changing. No environment is touched during
  learning -- every target is computed from the log and the current table.

Where it sits on the ladder: offline RL is a sibling of the model-free control rung (Q-learning /
SARSA), distinguished by *where the data comes from* (a fixed log, not fresh rollouts) rather than
by the backup. FQI's backup is exactly Q-learning's; batching it over a frozen dataset is what makes
it offline.

RL concept:
    Offline / batch reinforcement learning and Fitted-Q Iteration; the data-coverage constraint
    (a policy is only as trustworthy as the logged support behind its chosen actions).

Math:
    Behaviour policy (epsilon-soft over a deterministic base ``b``):
        pi_beta(a | s) = epsilon / |A| + (1 - epsilon) * 1[a = b(s)]
    Fitted-Q Iteration backup over the fixed dataset D, for each logged (s, a):
        Q_{k+1}(s, a) = mean over D of [ R + (0 if done else gamma * max_a' Q_k(s', a')) ]
    converging to the Bellman optimality fixed point restricted to D's support.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from learning_agents.environment import (
    ACTION_LABELS,
    AgentDecisionEnvironment,
    AgentState,
    RewardFunction,
    default_reward,
)
from learning_agents.policies import HeuristicRouterPolicy, Policy

# StateKey is the 7-int discrete state key (AgentState.as_tuple()): the hashable index a tabular
# value/policy lookup is keyed on. Defined locally so this module depends only on the environment.
StateKey = tuple[int, int, int, int, int, int, int]

__all__ = [
    "LoggedDataset",
    "LoggedTransition",
    "OfflineRLResult",
    "StateKey",
    "behavior_action_probabilities",
    "collect_logged_dataset",
    "fitted_q_iteration",
]


@dataclass(frozen=True)
class LoggedTransition:
    """One logged ``(s, a, R, s', done)`` tuple plus the behaviour policy's action probability.

    What + why: this is a single row of the frozen production log offline RL learns from. Beyond the
    standard transition it records ``behavior_action_prob`` -- the probability the logging
    (behaviour) policy assigned to the action it actually took -- because off-policy evaluation by
    importance sampling needs ``pi_beta(a | s)`` and it can only be known at logging time. Storing
    it now keeps the dataset self-describing and reusable.

    Attributes:
        state: The state ``s`` at which the action was taken.
        action: The action ``a`` taken (a key of
            :data:`~learning_agents.environment.ACTION_LABELS`).
        reward: The reward ``R_{t+1}`` observed after acting.
        next_state: The resulting state ``s'``.
        done: Whether the transition terminated the episode.
        behavior_action_prob: ``pi_beta(a | s)`` -- the behaviour policy's probability of ``a`` in
            ``s`` (in ``(0, 1]``); recorded for later importance-sampling-based evaluation.
        scenario_id: The scenario the episode was reset from (for grouping/inspection).

    RL concept: a logged transition with its behaviour-policy propensity -- the atom of offline RL
    and off-policy evaluation.
    """

    state: AgentState
    action: int
    reward: float
    next_state: AgentState
    done: bool
    behavior_action_prob: float
    scenario_id: int


@dataclass(frozen=True)
class LoggedDataset:
    """An immutable batch of logged transitions plus how the log was generated.

    What + why: bundles the frozen list of :class:`LoggedTransition` rows with the metadata that
    documents the behaviour policy (its base name and exploration rate). Offline learners consume
    ``transitions`` and never touch the environment; the metadata lets reports state exactly which
    behaviour policy and ``epsilon`` produced the data, which governs what the log covers.

    Attributes:
        transitions: The logged ``(s, a, r, s', done, prob)`` rows in collection order.
        behavior_policy_name: Name of the base policy the epsilon-soft behaviour wrapped.
        epsilon: Exploration rate of the epsilon-soft behaviour policy used for logging.

    RL concept: a fixed offline dataset -- the sole input to batch RL; its coverage caps what can be
    learned.
    """

    transitions: tuple[LoggedTransition, ...]
    behavior_policy_name: str
    epsilon: float

    def __len__(self) -> int:
        """Return the number of logged transitions."""
        return len(self.transitions)


@dataclass(frozen=True)
class OfflineRLResult:
    """The Q-table learned offline plus the per-sweep convergence curve.

    Attributes:
        q_table: Learned action values ``Q(s, a)`` keyed by the 7-int state key, defined for every
            state appearing in the dataset (as a source or successor state). States seen only as
            successors -- never as a logged decision point -- keep their all-zeros initialisation,
            which is exactly the out-of-distribution gap offline RL must respect.
        training_curve: One dict per Fitted-Q sweep with keys ``sweep``, ``bellman_residual`` (the
            max table change that sweep), and ``updated_state_action_pairs`` (the count of logged
            (s, a) cells refreshed) -- the offline analogue of an online training curve.

    RL concept: the offline-learned value table and its batch-convergence diagnostic.
    """

    q_table: dict[StateKey, list[float]]
    training_curve: list[dict[str, int | float]]


def behavior_action_probabilities(
    base_policy: Policy,
    state: AgentState,
    epsilon: float,
    num_actions: int = len(ACTION_LABELS),
) -> list[float]:
    """Return the epsilon-soft action distribution of the logging behaviour policy in ``state``.

    What + why: the behaviour policy that generates the log is an *epsilon-soft* wrapper around a
    deterministic base router -- with probability ``epsilon`` it acts uniformly at random, otherwise
    it follows the base policy. Making the distribution explicit (rather than only sampling from it)
    is what lets us record ``pi_beta(a | s)`` per logged step and reuse the log for importance-
    sampling-based off-policy evaluation. A strictly positive ``epsilon`` keeps every action's
    probability positive, so the log has full action support (a precondition for unbiased IS).

    Args:
        base_policy: The deterministic base policy ``b`` whose greedy action gets the exploit mass.
        state: The state ``s`` to score.
        epsilon: Exploration rate in ``[0, 1]``; the uniform-random mass spread over all actions.
        num_actions: Size of the action space ``|A|``.

    Returns:
        A list of probabilities (one per action index) summing to 1: ``epsilon/|A|`` everywhere plus
        ``(1 - epsilon)`` on the base policy's greedy action.

    RL concept: an epsilon-soft behaviour policy
    ``pi_beta(a | s) = epsilon/|A| + (1-epsilon) 1[a=b(s)]`` -- soft enough to cover every action,
    so the log supports off-policy evaluation.
    """
    greedy = base_policy.select_action(state)
    uniform = epsilon / num_actions
    probabilities = [uniform] * num_actions
    probabilities[greedy] += 1.0 - epsilon
    return probabilities


def collect_logged_dataset(
    *,
    base_policy: Policy | None = None,
    episodes: int = 400,
    epsilon: float = 0.3,
    seed: int = 7,
    scenario_ids: Sequence[int] = (0, 1, 2, 3, 4),
    horizon: int = 5,
    reward_fn: RewardFunction = default_reward,
) -> LoggedDataset:
    """Simulate a production log by rolling out an epsilon-soft behaviour policy.

    What + why: offline RL needs a *fixed* dataset that some behaviour policy already produced. This
    rolls out an epsilon-soft wrapper around ``base_policy`` (a sensible heuristic router by
    default) across the scenarios, sampling each action from
    :func:`behavior_action_probabilities` and recording the transition together with the chosen
    action's logged probability. The result is exactly the kind of log an offline learner is handed:
    decent-but-imperfect decisions with some exploration, so the data covers a useful slice of the
    state-action space without ever being optimal. Everything is seeded, so the same inputs always
    yield the same log (reproducible offline experiments).

    Args:
        base_policy: Deterministic base policy the behaviour wraps; defaults to
            :class:`~learning_agents.policies.HeuristicRouterPolicy`.
        episodes: Number of logged episodes (rollouts), cycled round-robin over ``scenario_ids``.
        epsilon: Exploration rate of the epsilon-soft behaviour policy (>0 keeps full action
            support).
        seed: Base RNG seed driving action sampling and per-episode env-reset jitter.
        scenario_ids: Scenario indices cycled across episodes.
        horizon: Episode length H passed to the environment.
        reward_fn: Reward function injected into the environment (defaults to the judge rubric).

    Returns:
        A :class:`LoggedDataset` of all transitions with their behaviour-policy probabilities.

    Raises:
        ValueError: If ``episodes`` is not positive or ``epsilon`` is outside ``[0, 1]``.

    RL concept: generating a behaviour-policy log -- the fixed dataset that offline RL and
    off-policy evaluation both consume.
    """
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")

    policy = base_policy if base_policy is not None else HeuristicRouterPolicy(horizon=horizon)
    rng = random.Random(seed)
    num_actions = len(ACTION_LABELS)
    transitions: list[LoggedTransition] = []

    for episode in range(episodes):
        scenario_id = scenario_ids[episode % len(scenario_ids)]
        policy.reset()
        environment = AgentDecisionEnvironment(horizon=horizon, reward_fn=reward_fn)
        state = environment.reset(seed=seed + episode, scenario_id=scenario_id)
        while not environment.is_done():
            probabilities = behavior_action_probabilities(policy, state, epsilon, num_actions)
            # Sample the logged action from the behaviour distribution and record its propensity.
            action = rng.choices(range(num_actions), weights=probabilities, k=1)[0]
            transition = environment.step(action)
            transitions.append(
                LoggedTransition(
                    state=state,
                    action=action,
                    reward=transition.reward,
                    next_state=transition.state,
                    done=transition.done,
                    behavior_action_prob=probabilities[action],
                    scenario_id=scenario_id,
                )
            )
            state = transition.state

    return LoggedDataset(
        transitions=tuple(transitions),
        behavior_policy_name=policy.name,
        epsilon=epsilon,
    )


def fitted_q_iteration(
    dataset: LoggedDataset,
    *,
    gamma: float = 0.9,
    sweeps: int = 100,
    tolerance: float = 1e-6,
) -> OfflineRLResult:
    """Learn Q* from a fixed log by tabular Fitted-Q Iteration (offline / batch RL).

    What + why: this is the offline learner. It never touches the environment -- it sweeps the
    frozen ``dataset`` repeatedly, and on each sweep replaces every logged ``Q(s, a)`` with the mean
    Bellman target computed from the *current* table. The backup is identical to Q-learning's
    (``r + gamma * max_a' Q(s', a')``); what makes it offline is that both the (s, a) updated and
    the successor ``s'`` bootstrapped from come only from the log. Because the agent-decision MDP is
    a finite-horizon deterministic model, sweeping to a fixed point recovers the optimal Q* *on the
    portion of the state-action space the log covers*.

    Coverage is the lesson: a state seen only as a successor (never as a logged decision point) is
    never updated and keeps its all-zeros row, so a greedy policy over the result falls back to
    ``answer_direct`` there -- it has no logged evidence for that state. Wrapping the returned table
    in :class:`~learning_agents.policies.QTablePolicy` yields the offline-learned policy.

    Args:
        dataset: The fixed :class:`LoggedDataset` to learn from.
        gamma: Discount factor for the Bellman backup.
        sweeps: Maximum number of full passes over the dataset.
        tolerance: Stop early once a sweep's maximum table change (Bellman residual) drops below
            this.

    Returns:
        An :class:`OfflineRLResult` with the learned ``q_table`` and per-sweep convergence curve.

    Raises:
        ValueError: If ``dataset`` has no transitions or ``sweeps`` is not positive.

    RL concept: Fitted-Q Iteration -- the batch (offline) form of value-based control; convergence
    is restricted to the dataset's support.

    Math:
        For each logged (s, a):
            Q_{k+1}(s, a) = mean_{D}[ R + (0 if done else gamma * max_a' Q_k(s', a')) ]
        and the Bellman residual ||Q_{k+1} - Q_k||_inf drives the stopping rule.
    """
    if not dataset.transitions:
        raise ValueError("dataset must contain at least one transition")
    if sweeps <= 0:
        raise ValueError("sweeps must be positive")

    num_actions = len(ACTION_LABELS)
    # Initialise a zero row for every state appearing as a source OR successor so bootstrapping
    # (max_a' Q[s']) is always defined; successor-only states stay zero -> the coverage gap.
    q_table: dict[StateKey, list[float]] = {}
    for transition in dataset.transitions:
        q_table.setdefault(transition.state.as_tuple(), [0.0] * num_actions)
        q_table.setdefault(transition.next_state.as_tuple(), [0.0] * num_actions)

    # Group transitions by the logged (state, action) cell so each sweep averages their targets.
    grouped: dict[tuple[StateKey, int], list[LoggedTransition]] = {}
    for transition in dataset.transitions:
        grouped.setdefault((transition.state.as_tuple(), transition.action), []).append(transition)

    training_curve: list[dict[str, int | float]] = []
    for sweep in range(1, sweeps + 1):
        # Batch update: compute every target from the CURRENT table, then apply together.
        updates: dict[tuple[StateKey, int], float] = {}
        for (state_key, action), rows in grouped.items():
            target_sum = 0.0
            for row in rows:
                future = 0.0 if row.done else gamma * max(q_table[row.next_state.as_tuple()])
                target_sum += row.reward + future
            updates[(state_key, action)] = target_sum / len(rows)

        residual = 0.0
        for (state_key, action), new_value in updates.items():
            residual = max(residual, abs(new_value - q_table[state_key][action]))
            q_table[state_key][action] = new_value

        training_curve.append(
            {
                "sweep": sweep,
                "bellman_residual": round(residual, 6),
                "updated_state_action_pairs": len(updates),
            }
        )
        if residual < tolerance:
            break

    return OfflineRLResult(q_table=q_table, training_curve=training_curve)
