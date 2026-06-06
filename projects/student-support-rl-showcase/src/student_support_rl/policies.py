"""Decision-making policies for the student-support MDP: baselines and learned wrappers.

A *policy* pi maps a state to an action; it is the object an RL agent ultimately learns. This
module collects every policy the showcase compares: hand-written baselines (random, heuristic,
the deliberately-bad advisor-heavy one) and learned wrappers that read a value table or a deep
model. Separating policies from the learning algorithms (q_learning.py, drl.py, ...) lets the
evaluation harness treat them uniformly through a single Protocol, so the same rollout code can
score a uniform-random agent and a trained DQN side by side.

On the RL ladder these policies span the whole range: RandomPolicy and HeuristicPolicy are
fixed (non-learned) references; QLearningPolicy is the greedy policy *derived* from a learned
action-value table Q (value-based control); ModelPolicy wraps a deep-RL model whose policy may
be value-based (DQN) or policy-gradient/actor-critic (PPO). Comparing a learned policy against
these baselines is how we tell whether learning actually helped.

RL concept:
    Policy pi(a|s) and greedy action selection from action values; see
    docs/value-based-learning.md and docs/glossary.md.

Math:
    A deterministic greedy policy acts as pi(s) = argmax_a Q(s,a); under the Bellman optimality
    relation Q*(s,a)=E[R_{t+1}+gamma*max_a' Q*(s',a')] this greedy policy is optimal.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, SupportsInt

from student_support_rl.environment import ACTION_LABELS, StudentState


class Policy(Protocol):
    """Define the structural interface every agent's behaviour must satisfy.

    A policy is the decision rule of an agent: given the current state it returns an action. By
    expressing this as a ``Protocol`` (structural typing) rather than a base class, any object
    with a ``name``, a ``reset``, and a ``select_action`` counts as a policy -- so baselines and
    learned agents are interchangeable in the evaluation harness without a shared parent. This
    is the policy pi at the centre of the RL ladder: tabular methods and deep methods alike are
    ultimately judged by the policy they induce.

    Attributes:
        name: Human-readable identifier used to label rows in evaluation tables and plots.

    RL concept:
        Policy pi(a|s) as the agent's behaviour; see docs/value-based-learning.md and
        docs/glossary.md.
    """

    name: str

    def reset(self) -> None:
        """Clear any per-episode internal state before a new rollout begins.

        Called by the evaluation harness at the start of each episode. Stateless policies make
        this a no-op; stochastic ones (e.g. RandomPolicy) reseed here so runs are reproducible.
        """
        ...

    def select_action(self, state: StudentState) -> int:
        """Return the action index the policy chooses in ``state``.

        This is the policy evaluation pi(s): the single decision the agent commits to this step.
        The returned integer must be a valid key of ACTION_LABELS (0..3 in this showcase).

        Args:
            state: The current ``StudentState`` observation.

        Returns:
            The chosen action index in ``ACTION_LABELS``.
        """
        ...


class PredictModel(Protocol):
    """Define the minimal ``predict`` surface a deep-RL model must expose to be wrapped.

    This Protocol captures exactly the slice of the Stable-Baselines3 (SB3) model API that
    ModelPolicy relies on, so the showcase can wrap a trained DQN/PPO agent without importing
    SB3 here. Structural typing keeps the policy layer dependency-light and makes the deep-RL
    bridge testable with a tiny stub. The wrapped model sits at the deep-RL rung of the ladder.

    RL concept:
        Function approximation for the policy/value function; see docs/deep-rl.md.
    """

    def predict(
        self,
        observation: Sequence[float],
        deterministic: bool = True,
    ) -> tuple[SupportsInt, object]:
        """Map an observation vector to an action, mirroring SB3's ``model.predict``.

        Args:
            observation: The state encoded as a numeric feature vector (the network input).
            deterministic: If True, take the greedy/mode action; if False, sample from the
                policy distribution (relevant for stochastic policies such as PPO).

        Returns:
            A ``(action, state)`` pair following the SB3 convention; only the first element (an
            int-like action index) is used by ModelPolicy, the second (recurrent state) is
            ignored.
        """
        ...


@dataclass
class RandomPolicy:
    """Choose actions uniformly at random, ignoring the state entirely.

    The simplest possible baseline: it pulls each action with equal probability and never
    learns. Its purpose is to establish a noise floor -- any agent worth its complexity must
    beat uniform-random return. Because it ignores the state it is not even a contextual policy;
    it sits below the entire ladder and exists only as a reference point for evaluation.

    Attributes:
        seed: Seed for the internal RNG so episodes are reproducible across runs.
        name: Identifier shown in evaluation output ("random").

    RL concept:
        Uniform-random baseline policy; see docs/value-based-learning.md and docs/glossary.md.
    """

    seed: int = 42
    name: str = "random"

    def __post_init__(self) -> None:
        """Initialise the seeded RNG that drives uniform action selection."""
        self._rng = random.Random(self.seed)

    def reset(self) -> None:
        """Reseed the RNG so each evaluation episode draws the same reproducible stream."""
        self._rng = random.Random(self.seed)

    def select_action(self, state: StudentState) -> int:
        """Return an action drawn uniformly from ACTION_LABELS, independent of the state.

        Args:
            state: Ignored; the choice does not depend on the observation.

        Returns:
            A uniformly random action index in ``range(len(ACTION_LABELS))``.
        """
        del state  # state-independent: uniform-random policy pi(a|s) = 1/|A|
        return self._rng.randrange(len(ACTION_LABELS))


@dataclass
class HeuristicPolicy:
    """Apply hand-written escalation rules that encode a domain expert's intuition.

    A fixed, non-learned policy: a cascade of if/else thresholds maps risk, pressure,
    completion, and engagement to an intervention of matching intensity (do nothing -> email ->
    TA session -> advisor meeting). It is the "sensible baseline" an advisor might write down,
    and the bar a learned agent should clear to justify training. It does not improve from
    experience, so it sits beside (not on) the learning ladder as a strong reference policy.

    Attributes:
        name: Identifier shown in evaluation output ("heuristic").

    RL concept:
        Hand-crafted reference policy; see docs/value-based-learning.md and docs/glossary.md.
    """

    name: str = "heuristic"

    def reset(self) -> None:
        """Do nothing; the policy is stateless and deterministic across episodes."""
        return None

    def select_action(self, state: StudentState) -> int:
        """Map the state to an intervention via ordered, decreasing-severity threshold rules.

        Rules are checked most-severe first so the first matching condition wins: highest-risk
        students under load get an advisor meeting (3), elevated risk or stalled completion gets
        a TA session (2), low engagement gets a resource email (1), otherwise no intervention.

        Args:
            state: The current ``StudentState`` observation.

        Returns:
            The action index selected by the first matching rule (0..3).
        """
        # Hand-written escalation ladder: each branch is a domain rule, severest condition first.
        if state.risk >= 3 and (state.pressure >= 3 or state.prior_interventions >= 2):
            return 3
        if state.risk >= 2 or state.completion <= 1:
            return 2
        if state.engagement <= 2:
            return 1
        return 0


@dataclass
class AdvisorHeavyPolicy:
    """Always escalate to an advisor meeting -- a deliberately BAD baseline for the demo.

    This policy ignores the state and returns the most expensive intervention (action 3) every
    step. It is intentionally bad: under a naively-shaped reward that over-credits "taking
    action", a constant-escalate agent can score deceptively well, which makes this the foil in
    the reward-hacking demonstration. It teaches that maximizing a proxy reward is not the same
    as solving the task, motivating careful reward design and the over-intervention penalty in
    the environment. As a constant policy it does not sit on the learning ladder at all.

    Attributes:
        name: Identifier shown in evaluation output ("advisor_heavy").

    RL concept:
        Reward hacking / specification gaming via a degenerate fixed policy; see
        docs/reward-design-and-hacking.md and docs/glossary.md.
    """

    name: str = "advisor_heavy"

    def reset(self) -> None:
        """Do nothing; the policy is stateless and always returns the same action."""
        return None

    def select_action(self, state: StudentState) -> int:
        """Return action 3 (advisor meeting) unconditionally, ignoring the state.

        Args:
            state: Ignored; the policy always escalates.

        Returns:
            The constant action index 3 (the most expensive intervention).
        """
        del state  # constant policy: always escalate -> exposes reward hacking under bad shaping
        return 3


@dataclass
class QLearningPolicy:
    """Act greedily with respect to a learned tabular action-value function Q(s,a).

    This is the *control* policy extracted after value-based learning: it looks up the row of
    action values for the current state in a Q-table and plays the highest-valued action. The
    table is produced by tabular Q-learning (q_learning.py); here we only consume it. This is
    the QLearningPolicy rung on the ladder -- the bridge from learned values to behaviour.

    Unseen-state fallback: if the state was never visited during training,
    ``q_table.get(...)`` returns an all-zeros value vector, so every action ties at 0.0 and the
    deterministic tie-break in greedy_action selects action 0 ("no_intervention"). This keeps
    the policy total over the whole state space without raising on novel states.

    Attributes:
        q_table: Mapping from a state's 6-tuple key (``StudentState.as_tuple``) to its list of
            per-action values; missing keys trigger the all-zeros fallback above.
        name: Identifier shown in evaluation output ("q_learning").

    RL concept:
        Greedy policy derived from a learned Q-table (value-based control); see
        docs/value-based-learning.md and docs/glossary.md.

    Math:
        pi(s) = argmax_a Q(s,a); the optimal Q satisfies the Bellman optimality relation
        Q*(s,a)=E[R_{t+1}+gamma*max_a' Q*(s',a')].
    """

    q_table: dict[tuple[int, int, int, int, int, int], list[float]]
    name: str = "q_learning"

    def reset(self) -> None:
        """Do nothing; the learned table is fixed, so behaviour is stateless across episodes."""
        return None

    def select_action(self, state: StudentState) -> int:
        """Return argmax_a Q(state, a), defaulting unseen states to all-zeros (-> action 0).

        Args:
            state: The current ``StudentState``; its ``as_tuple`` key indexes the Q-table.

        Returns:
            The greedy action index. For a state absent from the table the all-zeros fallback
            makes every action tie, and greedy_action's deterministic tie-break returns 0.
        """
        # Greedy control: pi(s) = argmax_a Q(s,a); unseen state -> all-zeros vector -> action 0.
        return greedy_action(self.q_table.get(state.as_tuple(), [0.0] * len(ACTION_LABELS)))


@dataclass
class ModelPolicy:
    """Adapt a trained deep-RL model (SB3) to the showcase's Policy interface.

    This wrapper bridges a neural-network agent into the same harness as the tabular baselines.
    It encodes the structured ``StudentState`` into the numeric vector the network expects via
    ``observation_fn``, calls the model's ``predict``, and returns the chosen action. The
    wrapped model may be value-based (DQN) or policy-gradient/actor-critic (PPO), so this is the
    showcase's entry point to the deep-RL rungs of the ladder.

    Attributes:
        model: Any object satisfying ``PredictModel`` (e.g. an SB3 DQN/PPO agent).
        observation_fn: Encodes a ``StudentState`` into the model's input feature vector,
            keeping observation engineering out of the model itself.
        name: Identifier shown in evaluation output.
        deterministic: Forwarded to ``predict``; True takes the greedy/mode action, False
            samples from the policy distribution.

    RL concept:
        Deep-RL policy via function approximation; see docs/deep-rl.md and
        docs/policy-gradient-and-actor-critic.md.
    """

    model: PredictModel
    observation_fn: Callable[[StudentState], Sequence[float]]
    name: str
    deterministic: bool = True

    def reset(self) -> None:
        """Do nothing; the trained model holds its own (frozen) parameters across episodes."""
        return None

    def select_action(self, state: StudentState) -> int:
        """Encode the state, query the model, and return its chosen action index.

        Args:
            state: The current ``StudentState`` observation.

        Returns:
            The model's action as an int (the second ``predict`` return value is discarded).
        """
        # Encode state -> feature vector, then defer the policy decision to the learned model.
        action, _ = self.model.predict(
            self.observation_fn(state),
            deterministic=self.deterministic,
        )
        return int(action)


def greedy_action(action_values: list[float]) -> int:
    """Return the index of the maximal action value, breaking ties by lowest index.

    Implements the greedy operator argmax_a that turns a row of action values into a single
    decision -- the core of value-based control and the exploit step of epsilon-greedy
    exploration. The tie-break is deterministic on purpose: scanning left-to-right returns the
    FIRST index attaining the maximum, so equal values (e.g. an untrained all-zeros row) always
    yield action 0. This determinism keeps evaluation reproducible.

    Args:
        action_values: Per-action values Q(s,.) for a fixed state, indexed by action.

    Returns:
        The index of the first action achieving ``max(action_values)``.

    Raises:
        RuntimeError: If ``action_values`` is empty (no action could be selected); note
            ``max([])`` raises ``ValueError`` first, so the explicit guard covers the
            defensive fall-through.

    RL concept:
        Greedy action selection / argmax operator; see docs/value-based-learning.md.

    Math:
        pi(s) = argmax_a Q(s,a), with ties resolved to the smallest index.
    """
    best_value = max(action_values)  # greedy target: the maximal action value max_a Q(s,a)
    # Deterministic tie-break: first index attaining the max wins (all-zeros row -> action 0).
    for index, value in enumerate(action_values):
        if value == best_value:
            return index
    raise RuntimeError("greedy_action requires a non-empty action value list")
