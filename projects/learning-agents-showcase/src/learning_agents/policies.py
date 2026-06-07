"""Decision-making policies for the agent-decision MDP: baselines and learned wrappers.

What + why: a *policy* pi maps a state to an action; it is the object an RL agent ultimately
learns. This module collects every policy the showcase compares for the agent's orchestration
problem: hand-written baselines (uniform random, a sensible heuristic router, a deliberately bad
always-escalate) and learned wrappers that read a Q-table or a deep model. Separating policies from
the learning algorithms (q_learning, sarsa, reinforce, ...) lets the evaluation harness treat them
uniformly through a single Protocol, so the same rollout code can score a random agent and a trained
model side by side.

On the RL ladder these policies span the whole range: :class:`RandomPolicy` and
:class:`AlwaysEscalatePolicy` are fixed, non-learned references; :class:`HeuristicRouterPolicy` is
the strong hand-written baseline a learned policy must beat; :class:`QTablePolicy` is the greedy
policy *derived* from a learned action-value table (value-based control); :class:`ModelPolicy` wraps
a deep-RL model whose policy may be value-based (DQN) or policy-gradient (REINFORCE/PPO).

RL concept: policy pi(a|s) and greedy action selection from action values. Comparing a learned
policy against these baselines is how we tell whether learning actually helped.

Math:
    A deterministic greedy policy acts as pi(s) = argmax_a Q(s,a); under the Bellman optimality
    relation Q*(s,a) = E[R_{t+1} + gamma*max_a' Q*(s',a')] this greedy policy is optimal.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, SupportsInt

from learning_agents.environment import ACTION_LABELS, AgentState
from learning_agents.reward import evidence_is_adequate


class Policy(Protocol):
    """Define the structural interface every agent's behaviour must satisfy.

    What + why: a policy is the decision rule of an agent -- given the current state it returns an
    action. By expressing this as a ``Protocol`` (structural typing) rather than a base class, any
    object with a ``name``, a ``reset``, and a ``select_action`` counts as a policy, so baselines
    and learned agents are interchangeable in the evaluation harness without a shared parent. This
    is the policy pi at the centre of the RL ladder: tabular and deep methods alike are judged by
    the policy they induce.

    Attributes:
        name: Human-readable identifier used to label rows in evaluation tables and plots.

    RL concept: policy pi(a|s) as the agent's behaviour, exposed structurally so every learner and
    baseline plugs into one harness.
    """

    name: str

    def reset(self) -> None:
        """Clear any per-episode internal state before a new rollout begins.

        What + why: the evaluation harness calls this at the start of each episode. Stateless
        policies make it a no-op; stochastic ones (e.g. :class:`RandomPolicy`) reseed here so runs
        are reproducible.
        """
        ...

    def select_action(self, state: AgentState) -> int:
        """Return the action index the policy chooses in ``state``.

        What + why: this is the policy evaluation pi(s) -- the single decision the agent commits to
        this step. The returned integer must be a valid key of :data:`ACTION_LABELS` (0..3 here).

        Args:
            state: The current :class:`AgentState` observation.

        Returns:
            The chosen action index in :data:`ACTION_LABELS`.
        """
        ...


class PredictModel(Protocol):
    """Define the minimal ``predict`` surface a deep-RL model must expose to be wrapped.

    What + why: this Protocol captures exactly the slice of the Stable-Baselines3 (SB3) model API
    that :class:`ModelPolicy` relies on, so the showcase can wrap a trained DQN/PPO agent without
    importing SB3 here. Structural typing keeps the policy layer dependency-light and makes the
    deep-RL bridge testable with a tiny stub. The wrapped model sits at the deep-RL rung of the
    ladder.

    RL concept: function approximation for the policy/value function, abstracted behind a tiny
    structural interface.
    """

    def predict(
        self,
        observation: Sequence[float],
        deterministic: bool = True,
    ) -> tuple[SupportsInt, object]:
        """Map an observation vector to an action, mirroring SB3's ``model.predict``.

        Args:
            observation: The state encoded as a numeric feature vector (the network input).
            deterministic: If True, take the greedy/mode action; if False, sample from the policy
                distribution (relevant for stochastic policies such as PPO).

        Returns:
            A ``(action, state)`` pair following the SB3 convention; only the first element (an
            int-like action index) is used by :class:`ModelPolicy`, the second (recurrent state) is
            ignored.
        """
        ...


def greedy_action(action_values: list[float]) -> int:
    """Return the index of the maximal action value, breaking ties by lowest index.

    What + why: implements the greedy operator argmax_a that turns a row of action values into a
    single decision -- the core of value-based control and the exploit step of epsilon-greedy
    exploration. The tie-break is deterministic on purpose: scanning left-to-right returns the FIRST
    index attaining the maximum, so equal values (e.g. an untrained all-zeros row) always yield
    action 0. This determinism keeps evaluation reproducible.

    Args:
        action_values: Per-action values Q(s, .) for a fixed state, indexed by action.

    Returns:
        The index of the first action achieving ``max(action_values)``.

    Raises:
        RuntimeError: If ``action_values`` is empty (no action could be selected).

    RL concept: greedy action selection / the argmax operator behind value-based control.

    Math:
        pi(s) = argmax_a Q(s, a), with ties resolved to the smallest index.
    """
    if not action_values:
        raise RuntimeError("greedy_action requires a non-empty action value list")
    best_value = max(action_values)  # greedy target: the maximal action value max_a Q(s,a)
    # Deterministic tie-break: first index attaining the max wins (all-zeros row -> action 0).
    for index, value in enumerate(action_values):
        if value == best_value:
            return index
    raise RuntimeError("greedy_action requires a non-empty action value list")


@dataclass
class RandomPolicy:
    """Choose actions uniformly at random, ignoring the state entirely.

    What + why: the simplest possible baseline -- it pulls each action with equal probability and
    never learns. Its purpose is to establish a noise floor: any agent worth its complexity must
    beat uniform-random return. Because it ignores the state it is not even a contextual policy; it
    sits below the entire ladder and exists only as a reference point for evaluation.

    Attributes:
        seed: Seed for the internal RNG so episodes are reproducible across runs.
        name: Identifier shown in evaluation output ("random").

    RL concept: the uniform-random baseline policy that sets the floor every learner must clear.
    """

    seed: int = 42
    name: str = "random"

    def __post_init__(self) -> None:
        """Initialise the seeded RNG that drives uniform action selection."""
        self._rng = random.Random(self.seed)

    def reset(self) -> None:
        """Reseed the RNG so each evaluation episode draws the same reproducible stream."""
        self._rng = random.Random(self.seed)

    def select_action(self, state: AgentState) -> int:
        """Return an action drawn uniformly from :data:`ACTION_LABELS`, independent of the state.

        Args:
            state: Ignored; the choice does not depend on the observation.

        Returns:
            A uniformly random action index in ``range(len(ACTION_LABELS))``.
        """
        del state  # state-independent: uniform-random policy pi(a|s) = 1/|A|
        return self._rng.randrange(len(ACTION_LABELS))


@dataclass
class HeuristicRouterPolicy:
    """Route requests with hand-written rules encoding a sensible orchestration strategy.

    What + why: a fixed, non-learned policy an engineer might write down as a first-pass router, and
    the bar a learned agent should clear to justify training. The rules express the intended
    trade-offs directly: resolve ambiguity before answering, gather grounding before answering a
    hard request, hand off genuinely hard/ambiguous requests to a human only when stuck (out of
    budget or steps), and otherwise answer. It does not improve from experience, so it sits beside
    (not on) the learning ladder as a strong reference policy.

    Decision order (most specific first, first match wins):
        1. If ambiguity remains and there is budget/steps to clarify -> ``clarify`` (2).
        2. If evidence is not yet adequate for the difficulty and there is budget/steps to retrieve
           -> ``retrieve`` (1).
        3. If the request is genuinely hard or still ambiguous but we are out of budget/steps to fix
           it -> ``escalate`` (3) (a safe hand-off rather than a bad answer).
        4. Otherwise the request is adequately grounded and unambiguous -> ``answer_direct`` (0).

    Attributes:
        clarify_cost_tenths: Budget (in tenths) a ``clarify`` needs; used to decide affordability.
        retrieve_cost_tenths: Budget (in tenths) a ``retrieve`` needs; used to decide affordability.
        horizon: Episode length, so the router knows when it is about to run out of steps.
        name: Identifier shown in evaluation output ("heuristic_router").

    RL concept: a hand-crafted reference policy -- the human baseline a learned policy is measured
    against.
    """

    clarify_cost_tenths: int = 3
    retrieve_cost_tenths: int = 5
    horizon: int = 5
    name: str = "heuristic_router"

    def reset(self) -> None:
        """Do nothing; the policy is stateless and deterministic across episodes."""
        return None

    def _can_afford(self, state: AgentState, cost_tenths: int) -> bool:
        """Return whether another non-terminal action fits within budget and the step horizon.

        What + why: clarify/retrieve are only sensible while the agent can still pay for them and
        has a step left before the clock forces a commit; this guard centralizes that check so the
        router degrades to escalate/answer instead of wasting a doomed action.

        Args:
            state: The current state s.
            cost_tenths: The budget cost (in tenths) of the action being considered.

        Returns:
            True iff the action's cost fits the remaining budget and ``step`` is below the horizon.
        """
        return state.budget - cost_tenths >= 0 and state.step < self.horizon

    def select_action(self, state: AgentState) -> int:
        """Map the state to an orchestration action via ordered router rules.

        What + why: rules are checked most-specific first so the first match wins -- disambiguate,
        then ground, then (if stuck on a genuinely hard/ambiguous request) escalate, else answer.

        Args:
            state: The current :class:`AgentState` observation.

        Returns:
            The action index selected by the first matching rule (0..3).
        """
        grounded = evidence_is_adequate(evidence=state.evidence, difficulty=state.difficulty)
        # 1. Resolve ambiguity first, while we can still afford and have a step for it.
        if state.ambiguity > 0 and self._can_afford(state, self.clarify_cost_tenths):
            return 2
        # 2. Then gather grounding for a hard request, while affordable and time remains.
        if not grounded and self._can_afford(state, self.retrieve_cost_tenths):
            return 1
        # 3. Stuck on a genuinely hard/ambiguous request with no room to fix it -> safe hand-off.
        if (not grounded or state.ambiguity > 0) and (state.difficulty >= 2 or state.ambiguity > 0):
            return 3
        # 4. Adequately grounded and unambiguous -> answer directly.
        return 0


@dataclass
class AlwaysEscalatePolicy:
    """Always escalate to a human -- a deliberately BAD baseline for the reward-hacking demo.

    What + why: this policy ignores the state and returns the most expensive action (escalate, 3)
    every step. It is intentionally bad: under a naively-shaped reward that over-credits escalation
    (see :func:`learning_agents.reward.hackable_reward`), a constant-escalate agent can score
    deceptively well, which makes this the foil in the reward-hacking demonstration. It teaches that
    maximizing a proxy reward is not the same as solving the task. As a constant policy it does not
    sit on the learning ladder at all.

    Attributes:
        name: Identifier shown in evaluation output ("always_escalate").

    RL concept: reward hacking / specification gaming via a degenerate fixed policy.
    """

    name: str = "always_escalate"

    def reset(self) -> None:
        """Do nothing; the policy is stateless and always returns the same action."""
        return None

    def select_action(self, state: AgentState) -> int:
        """Return action 3 (escalate) unconditionally, ignoring the state.

        Args:
            state: Ignored; the policy always escalates.

        Returns:
            The constant action index 3 (the most expensive action).
        """
        del state  # constant policy: always escalate -> exposes reward hacking under bad shaping
        return 3


@dataclass
class QTablePolicy:
    """Act greedily with respect to a learned tabular action-value function Q(s, a).

    What + why: this is the *control* policy extracted after value-based learning -- it looks up the
    row of action values for the current state in a Q-table and plays the highest-valued action. The
    table is produced by tabular Q-learning/SARSA; here we only consume it. This is the bridge from
    learned values to behaviour.

    Unseen-state fallback: if the state was never visited during training, ``q_table.get(...)``
    returns the supplied all-zeros vector, so every action ties at 0.0 and the deterministic
    tie-break in :func:`greedy_action` selects action 0 (``answer_direct``). This keeps the policy
    total over the whole state space without raising on novel states.

    Attributes:
        q_table: Mapping from a state's 7-tuple key (:meth:`AgentState.as_tuple`) to its list of
            per-action values; missing keys trigger the all-zeros fallback above.
        name: Identifier shown in evaluation output ("q_table").
        num_actions: Size of the action space, used to build the all-zeros fallback row.

    RL concept: the greedy policy derived from a learned Q-table (value-based control).

    Math:
        pi(s) = argmax_a Q(s, a); the optimal Q satisfies the Bellman optimality relation
        Q*(s, a) = E[R_{t+1} + gamma*max_a' Q*(s', a')].
    """

    q_table: dict[tuple[int, int, int, int, int, int, int], list[float]]
    name: str = "q_table"
    num_actions: int = len(ACTION_LABELS)

    def reset(self) -> None:
        """Do nothing; the learned table is fixed, so behaviour is stateless across episodes."""
        return None

    def select_action(self, state: AgentState) -> int:
        """Return argmax_a Q(state, a), defaulting unseen states to all-zeros (-> action 0).

        Args:
            state: The current :class:`AgentState`; its :meth:`AgentState.as_tuple` key indexes the
                Q-table.

        Returns:
            The greedy action index. For a state absent from the table the all-zeros fallback makes
            every action tie, and :func:`greedy_action`'s deterministic tie-break returns 0.
        """
        # Greedy control: pi(s) = argmax_a Q(s,a); unseen state -> all-zeros vector -> action 0.
        return greedy_action(self.q_table.get(state.as_tuple(), [0.0] * self.num_actions))


@dataclass
class ModelPolicy:
    """Adapt a trained deep-RL model (SB3-style) to the showcase's :class:`Policy` interface.

    What + why: this wrapper bridges a neural-network agent into the same harness as the tabular
    baselines. It encodes the structured :class:`AgentState` into the numeric vector the network
    expects via ``observation_fn``, calls the model's ``predict``, and returns the chosen action.
    The wrapped model may be value-based (DQN) or policy-gradient/actor-critic (PPO), so this is the
    showcase's entry point to the deep-RL rungs of the ladder.

    Attributes:
        model: Any object satisfying :class:`PredictModel` (e.g. an SB3 DQN/PPO agent).
        observation_fn: Encodes an :class:`AgentState` into the model's input feature vector,
            keeping observation engineering out of the model itself.
        name: Identifier shown in evaluation output.
        deterministic: Forwarded to ``predict``; True takes the greedy/mode action, False samples
            from the policy distribution.

    RL concept: a deep-RL policy via function approximation, wrapped to match the tabular baselines.
    """

    model: PredictModel
    observation_fn: Callable[[AgentState], Sequence[float]]
    name: str
    deterministic: bool = True

    def reset(self) -> None:
        """Do nothing; the trained model holds its own (frozen) parameters across episodes."""
        return None

    def select_action(self, state: AgentState) -> int:
        """Encode the state, query the model, and return its chosen action index.

        Args:
            state: The current :class:`AgentState` observation.

        Returns:
            The model's action as an int (the second ``predict`` return value is discarded).
        """
        # Encode state -> feature vector, then defer the policy decision to the learned model.
        action, _ = self.model.predict(
            self.observation_fn(state),
            deterministic=self.deterministic,
        )
        return int(action)
