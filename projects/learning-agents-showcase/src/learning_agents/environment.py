"""Define the agent-decision MDP every module in this showcase plans and learns against.

What + why: reinforcement learning needs a precise *environment* -- the world the agent acts on
and the source of the reward signal. This module is that world, but unlike a classic control task
it models an *assistant's own control loop*: a single user request arrives, and at each step the
agent picks one orchestration action -- answer directly, retrieve more evidence, ask a clarifying
question, or escalate to a human. The situation evolves deterministically, the episode ends when
the agent commits (answers or escalates) or runs out of steps/budget, and a judge-rubric reward
(:mod:`learning_agents.reward`) scores how well the agent balanced answer quality, grounding,
cost, and safety. Every other module (bandit, value iteration, Q-learning, SARSA, REINFORCE,
offline RL) plugs into this one object, so it is the shared anchor at the bottom of the ladder.

The MDP tuple (S, A, P, R, gamma, H):
    S -- a finite, fully observed state of seven discrete variables
        (step, intent, difficulty, ambiguity, evidence, attempts, budget).
    A -- four actions {0: answer_direct, 1: retrieve, 2: clarify, 3: escalate} with rising costs.
    P -- the transition kernel. HONESTY: P is *deterministic* given (s, a) -- :meth:`_transition`
        is a pure function with no sampling, so s' is fixed once (s, a) are fixed. The ``seed``
        argument to :meth:`AgentDecisionEnvironment.reset` only jitters the *start* state; it
        never injects randomness into a step. Treat this as a deterministic finite-horizon MDP.
    R -- :func:`reward.judge_reward` (re-exported as :func:`default_reward`), emitted as R_{t+1}.
    gamma -- the discount; owned by the *agent*, not stored here.
    H -- the horizon (default 5 steps); the episode terminates once step exceeds H.

RL concept: Markov decision process and environment design. The novelty here is that the *agent's
own decision process* is the MDP: states are request-handling situations, actions are orchestration
moves, and the reward is a judge rubric over the committed answer -- so we *learn the policy that
drives an agent*, not the answer text itself.

Math:
    Deterministic transition s' = T(s, a) realized by :meth:`_transition`.
    Reward after acting: R_{t+1} = R(s, a, s', done) from :func:`default_reward`.
    Discounted return the agent maximizes: G_t = sum_k gamma^k R_{t+k+1}.
    Finite horizon: the episode ends once ``step`` exceeds H (or on a terminal/illegal action).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

# ACTION_COSTS and evidence_is_adequate live in reward.py (the reward is their natural owner) and
# are re-exported here so callers can use ``environment.ACTION_COSTS``; this also keeps a single
# source of truth and avoids an environment<->reward import cycle (environment imports reward only).
from learning_agents.reward import (
    ACTION_COSTS,
    evidence_is_adequate,
    judge_reward,
)

# Public surface of this module. ACTION_COSTS and evidence_is_adequate are intentionally
# re-exported from reward.py so callers can rely on ``environment.ACTION_COSTS`` /
# ``environment.evidence_is_adequate`` even though their canonical home is the reward module.
__all__ = [
    "ACTION_COSTS",
    "ACTION_LABELS",
    "MAX_ACTION",
    "MAX_AMBIGUITY",
    "MAX_DIFFICULTY",
    "MAX_EVIDENCE",
    "STARTING_BUDGET",
    "AgentDecisionEnvironment",
    "AgentState",
    "RequestScenario",
    "RewardFunction",
    "SCENARIOS",
    "TransitionResult",
    "default_reward",
    "evidence_is_adequate",
    "scenario_catalog",
]

# Action space A: four discrete orchestration moves, indexed 0..MAX_ACTION.
MAX_ACTION = 3
ACTION_LABELS = {
    0: "answer_direct",
    1: "retrieve",
    2: "clarify",
    3: "escalate",
}

# Caps that keep the joint state finite so tabular methods see a closed state set.
MAX_EVIDENCE = 3
MAX_DIFFICULTY = 2
MAX_AMBIGUITY = 2

# Reward signature R(s, a, s', done) -> R_{t+1}: lets callers swap in alternative reward designs.
RewardFunction = Callable[["AgentState", int, "AgentState", bool], float]


@dataclass(frozen=True)
class AgentState:
    """Represent one fully observed state s of the agent-decision MDP as seven discrete variables.

    What + why: the state is everything the agent sees and everything the dynamics depend on. It
    captures the *situation while handling one request*: how far into the interaction we are, what
    kind of request it is, how hard and how ambiguous it is, how much evidence has been gathered,
    how many tool/clarify attempts have been spent, and how much budget remains. Because all seven
    fields are integers the joint state is finite and discrete, which is what lets the tabular
    methods key a Q-table or policy directly on it. The state is frozen (immutable) so a transition
    always produces a *new* object, keeping trajectories safe to store and replay. This is full
    observability: the observation equals the state (no hidden information, no POMDP).

    Attributes:
        step: Current step index 0..H; the within-episode clock (0 at reset).
        intent: Coarse request category in [0, 4] (e.g. factual, how-to, debug); a context label.
        difficulty: How hard the request is in [0, ``MAX_DIFFICULTY``] (higher needs more evidence).
        ambiguity: How under-specified the request is in [0, ``MAX_AMBIGUITY``] (higher = murkier).
        evidence: Amount of grounding gathered so far in [0, ``MAX_EVIDENCE``] (higher is better).
        attempts: Cumulative count of retrieve/clarify actions taken (a cost/effort proxy).
        budget: Remaining cost budget in tenths of a unit (integer); a step is blocked if the
            action cost would drive it negative.

    RL concept: state representation in an MDP -- the state is an *engineered, fully observed*
    snapshot of the agent's situation, so the policy it learns is a function of the situation,
    not of raw text.
    """

    step: int
    intent: int
    difficulty: int
    ambiguity: int
    evidence: int
    attempts: int
    budget: int

    def as_tuple(self) -> tuple[int, int, int, int, int, int, int]:
        """Return the state as a plain integer tuple for use as a hashable table key.

        What + why: tabular agents index a dict by the discrete state, so they need a lightweight,
        hashable, order-fixed view of s. The field order matches the dataclass declaration and is
        treated as a stable contract by callers and tests.

        Returns:
            The seven state fields as ``(step, intent, difficulty, ambiguity, evidence, attempts,
            budget)``.

        RL concept: the discrete state key behind tabular value/policy lookup.
        """
        return (
            self.step,
            self.intent,
            self.difficulty,
            self.ambiguity,
            self.evidence,
            self.attempts,
            self.budget,
        )

    def as_normalized_vector(self, *, horizon: int) -> tuple[float, ...]:
        """Return the state as a [0, 1]-scaled float feature vector for function approximation.

        What + why: deep agents (the DQN/REINFORCE rungs) consume features, not table keys.
        Dividing each field by its natural maximum puts every input on a comparable scale, which
        conditions a neural network far better than raw integers. ``step``/``attempts`` are scaled
        by the horizon since both grow with episode length; the others use their fixed caps.

        Args:
            horizon: Episode length H, used to scale the time-like fields; floored at 1 to avoid
                division by zero.

        Returns:
            Seven floats in [0, 1], each rounded to 6 decimals, for ``(step, intent, difficulty,
            ambiguity, evidence, attempts, budget)``. ``budget`` is scaled by the budget a fresh
            episode starts with (``horizon`` steps' worth of tenths) and clamped to [0, 1].

        RL concept: feature encoding for value-function approximation -- raw states are normalized
        before being fed to a network so no single coordinate dominates the gradient.
        """
        safe_horizon = max(1, horizon)
        # A fresh episode starts with STARTING_BUDGET tenths; scale budget by that and clamp.
        budget_scale = float(max(1, STARTING_BUDGET))
        return (
            # ``step`` can reach ``horizon + 1`` on a horizon-terminated episode, so clamp like
            # ``attempts`` below to honour the documented [0, 1] range for every reachable state.
            round(min(1.0, self.step / safe_horizon), 6),
            round(self.intent / 4.0, 6),
            round(self.difficulty / float(MAX_DIFFICULTY), 6),
            round(self.ambiguity / float(MAX_AMBIGUITY), 6),
            round(self.evidence / float(MAX_EVIDENCE), 6),
            round(min(1.0, self.attempts / safe_horizon), 6),
            round(min(1.0, max(0.0, self.budget / budget_scale)), 6),
        )


# A fresh episode's budget, in tenths of a cost unit (so it stays integer in the state). Enough to
# afford several retrieve/clarify actions but not unlimited escalation, forcing real trade-offs.
STARTING_BUDGET = 30


@dataclass(frozen=True)
class RequestScenario:
    """Hold the starting profile for one named request type the environment can be reset to.

    What + why: a scenario fixes the request features (intent, difficulty, ambiguity) that
    :meth:`AgentDecisionEnvironment.reset` builds the start state S_0 from. Distinct scenarios give
    learners a spread of situations -- an easy factual lookup, a medium how-to, an ambiguous query,
    a hard debug, and one that genuinely needs a human -- so a single trained policy can be
    evaluated across the request landscape rather than one lucky request.

    Attributes:
        scenario_id: Stable index into :data:`SCENARIOS`; also seeds the reset jitter.
        name: Human-readable label surfaced in transition info and reports.
        intent: Initial request category in [0, 4].
        difficulty: Initial difficulty in [0, ``MAX_DIFFICULTY``].
        ambiguity: Initial ambiguity in [0, ``MAX_AMBIGUITY``].

    RL concept: the start-state distribution of the MDP -- evaluating a policy across all scenarios
    measures generalization over the request landscape, not performance on a single request.
    """

    scenario_id: int
    name: str
    intent: int
    difficulty: int
    ambiguity: int


# Start-state distribution of the MDP: the catalog of request profiles a reset can draw from.
# They span the action space's reason-for-existing: easy_factual rewards answering immediately,
# howto_medium rewards a little retrieval, ambiguous_query rewards clarifying first, hard_debug
# rewards heavy grounding, and needs_escalation is the case where escalate is the right call.
SCENARIOS: tuple[RequestScenario, ...] = (
    RequestScenario(0, "easy_factual", intent=0, difficulty=0, ambiguity=0),
    RequestScenario(1, "howto_medium", intent=1, difficulty=1, ambiguity=0),
    RequestScenario(2, "ambiguous_query", intent=2, difficulty=1, ambiguity=2),
    RequestScenario(3, "hard_debug", intent=3, difficulty=2, ambiguity=1),
    RequestScenario(4, "needs_escalation", intent=4, difficulty=2, ambiguity=2),
)


def default_reward(
    previous_state: AgentState,
    action: int,
    next_state: AgentState,
    done: bool,
) -> float:
    """Score one transition as the MDP reward R_{t+1} (the judge-rubric reward).

    What + why: this is the environment's default objective. It simply delegates to
    :func:`learning_agents.reward.judge_reward`, the multi-criterion judge rubric that rewards a
    correctly grounded answer, penalizes an under-grounded one, charges for needless tool use, and
    gives escalation a modest safe payoff minus its high cost. Keeping the default here as a thin
    re-export lets the environment own a sensible objective while the reward logic (and its
    hackable counterpart) live in one place.

    Args:
        previous_state: The state s before acting.
        action: The action a taken (indexes :data:`ACTION_COSTS`).
        next_state: The resulting state s'.
        done: Whether this transition terminated the episode.

    Returns:
        The scalar reward R_{t+1} from :func:`judge_reward`, rounded to 4 decimals.

    RL concept: reward design -- the reward function *is* the objective; here the objective is a
    judge rubric over the agent's committed answer rather than a hand-shaped control reward.
    """
    return judge_reward(previous_state, action, next_state, done)


@dataclass(frozen=True)
class TransitionResult:
    """Bundle the outcome of one environment step: (s', R_{t+1}, done, info).

    What + why: this is the standard RL step tuple every agent consumes to learn. ``state`` is the
    next state s', ``reward`` is the reward emitted after acting (R_{t+1}), and ``done`` flags
    episode termination so the agent knows to stop bootstrapping past the horizon. The ``info``
    dict carries diagnostics (chosen action, its cost, scenario name, why the episode ended) for
    logging and reward analysis; agents must not learn from it.

    Attributes:
        state: The next state s' after the transition.
        reward: The scalar reward R_{t+1} returned by the reward function.
        done: True once the episode has terminated.
        info: Auxiliary diagnostics; not part of the state and not for learning.

    RL concept: the agent-environment step interface -- the (s', r, done) loop the agent learns
    from, with ``info`` reserved for logging only.
    """

    state: AgentState
    reward: float
    done: bool
    info: dict[str, int | float | str]


@dataclass
class AgentDecisionEnvironment:
    """Simulate the agent-decision MDP as a Gym-style reset/step environment.

    What + why: this is the concrete world agents interact with -- the object that realizes the MDP
    tuple (S, A, P, R, gamma, H) for the rest of the showcase, modelling an assistant's control
    loop over a single request. It holds the current state as a private mutable cursor and exposes
    the classic loop: ``reset`` to draw a start state S_0 from a scenario, ``step(a)`` to apply one
    orchestration action and return (s', R_{t+1}, done, info), and ``observe`` / ``is_done`` to
    inspect. HONESTY: dynamics are *deterministic* -- given (s, a) the next state is fixed; only the
    *start* state can be jittered (via ``reset(seed=...)``). gamma lives with the agent, not here.

    Termination has three causes: committing (``answer_direct`` or ``escalate`` are terminal), the
    clock running out (``step`` exceeds ``horizon``), and an action whose cost would drive the
    budget negative (the action is *not* applied; the episode ends as a forced-commit failure).

    Attributes:
        horizon: Episode length H in steps (default 5); the episode terminates once step exceeds H.
        reward_fn: The reward function R(s, a, s', done) -> R_{t+1}; defaults to
            :func:`default_reward` (the judge rubric) but can be swapped to study other designs.

    RL concept: the agent-environment interface of an MDP -- reset/step semantics with terminal
    *commit* actions, a finite horizon, and a hard budget constraint that bounds exploration.
    """

    horizon: int = 5
    reward_fn: RewardFunction = default_reward

    def __post_init__(self) -> None:
        """Initialize the private episode cursor before the first reset.

        What + why: the dataclass fields are configuration; the mutable per-episode state lives in
        private attributes set here so the environment exists in a defined (un-reset) state.
        ``_state is None`` marks "not yet reset" and guards :meth:`observe`.
        """
        self._state: AgentState | None = None
        self._done = False
        self._scenario_name = ""

    def reset(self, seed: int | None = None, scenario_id: int = 0) -> AgentState:
        """Draw a start state S_0 for a chosen scenario and begin a fresh episode.

        What + why: every episode begins here. The selected scenario fixes the request features; an
        optional ``seed`` jitters difficulty and ambiguity by +/- 1 (within their caps) so a
        scenario yields a small family of nearby starts for more robust training and evaluation.
        HONESTY: the seed only perturbs the *start* state -- it is consumed entirely in this method
        and never touches :meth:`step`, so the same seed reproduces the same S_0 exactly and the
        dynamics remain deterministic. ``step`` is set to 0, ``evidence``/``attempts`` to 0, and
        ``budget`` to :data:`STARTING_BUDGET`.

        Args:
            seed: Optional jitter seed for the start state only; ``None`` uses the scenario's exact
                request features.
            scenario_id: Index into :data:`SCENARIOS` selecting the request profile.

        Returns:
            The start state S_0.

        Raises:
            ValueError: If ``scenario_id`` is outside ``[0, len(SCENARIOS))``.

        RL concept: the start-state distribution of an episodic MDP -- S_0 is drawn from a scenario
        (optionally jittered) so training sees a family of related requests.
        """
        if scenario_id < 0 or scenario_id >= len(SCENARIOS):
            raise ValueError("scenario_id out of range")
        scenario = SCENARIOS[scenario_id]
        if seed is not None:
            # Start-state jitter ONLY: this RNG is consumed here and never reaches step(), so the
            # transition stays deterministic; same seed -> same S_0.
            rng = random.Random((seed + 1) * (scenario_id + 7))
            difficulty = _clamp(scenario.difficulty + rng.choice((-1, 0, 1)), 0, MAX_DIFFICULTY)
            ambiguity = _clamp(scenario.ambiguity + rng.choice((-1, 0, 1)), 0, MAX_AMBIGUITY)
        else:
            difficulty = scenario.difficulty
            ambiguity = scenario.ambiguity

        self._scenario_name = scenario.name
        self._state = AgentState(
            step=0,
            intent=scenario.intent,
            difficulty=difficulty,
            ambiguity=ambiguity,
            evidence=0,
            attempts=0,
            budget=STARTING_BUDGET,
        )
        self._done = False
        return self._state

    def observe(self) -> AgentState:
        """Return the current state s without advancing the environment.

        What + why: under full observability the agent's observation is exactly the state, so this
        hands back the current cursor for the agent to choose its next action from.

        Returns:
            The current state s.

        Raises:
            RuntimeError: If called before :meth:`reset` (no state exists yet).

        RL concept: observation == state under full observability.
        """
        if self._state is None:
            raise RuntimeError("environment must be reset before observe")
        return self._state

    def is_done(self) -> bool:
        """Report whether the current episode has terminated.

        Returns:
            True once the agent has committed (answer/escalate), the clock has run out, or a budget
            violation forced the episode to end.

        RL concept: episode termination in an episodic MDP.
        """
        return self._done

    @property
    def scenario_name(self) -> str:
        """Return the human-readable name of the scenario the current episode was reset from.

        Returns:
            The scenario label (empty string before the first :meth:`reset`).
        """
        return self._scenario_name

    def step(self, action: int) -> TransitionResult:
        """Advance the MDP one step: apply (P, R) for the chosen action and return the outcome.

        What + why: this is the single environment transition that drives all learning. It reads
        the current state s, checks the budget, applies the deterministic dynamics to get s',
        decides termination (commit action, clock, or budget violation), scores the move with the
        reward function (R_{t+1}), then commits the new state and done flag.

        Three termination paths:
            * Commit -- ``answer_direct`` (0) and ``escalate`` (3) are terminal by definition.
            * Clock -- if applying the action pushes ``step`` past ``horizon``, the episode ends.
            * Budget -- if the action's cost would drive ``budget`` negative, the action is *not*
              applied; the episode ends with ``done=True`` on a state that only advanced the clock,
              modelling a forced give-up. The reward then judges this non-commit terminal state
              (typically an under-grounded outcome), teaching the agent to commit before it runs
              dry.

        Args:
            action: The orchestration action a to take; must be a valid key of
                :data:`ACTION_LABELS`.

        Returns:
            A :class:`TransitionResult` with (s', R_{t+1}, done, info).

        Raises:
            ValueError: If ``action`` is not one of the four defined actions.
            RuntimeError: If called after the episode has already terminated, or (via
                :meth:`observe`) before the first :meth:`reset`.

        RL concept: the (s, a) -> (s', r, done) transition of the MDP -- here the transition also
        enforces a hard resource constraint, so the agent must learn *when to stop*.

        Math:
            s' = T(s, a) (deterministic); R_{t+1} = R(s, a, s', done).
        """
        if action not in ACTION_LABELS:
            raise ValueError(f"unknown action: {action}")
        if self._done:
            raise RuntimeError("episode has terminated; call reset before stepping again")
        previous_state = self.observe()

        cost_tenths = int(round(ACTION_COSTS[action] * 10))
        # Budget guard: a forced give-up. The action is NOT applied; only the clock advances, and
        # the episode ends so the reward judges a non-commit (typically under-grounded) outcome.
        if previous_state.budget - cost_tenths < 0:
            next_state = AgentState(
                step=previous_state.step + 1,
                intent=previous_state.intent,
                difficulty=previous_state.difficulty,
                ambiguity=previous_state.ambiguity,
                evidence=previous_state.evidence,
                attempts=previous_state.attempts,
                budget=previous_state.budget,
            )
            done = True
            termination = "budget_exhausted"
            reward = self.reward_fn(previous_state, action, next_state, done)
            self._state = next_state
            self._done = done
            return TransitionResult(
                state=next_state,
                reward=reward,
                done=done,
                info={
                    "action": action,
                    "action_label": ACTION_LABELS[action],
                    "action_cost": ACTION_COSTS[action],
                    "scenario_name": self._scenario_name,
                    "termination": termination,
                },
            )

        # Deterministic transition s' = T(s, a); no sampling here.
        next_state = self._transition(previous_state, action, cost_tenths)
        # Commit actions are terminal; otherwise the clock can still end the episode.
        if action in (0, 3):
            done = True
            termination = ACTION_LABELS[action]
        elif next_state.step > self.horizon:
            done = True
            termination = "horizon"
        else:
            done = False
            termination = "ongoing"
        # Reward emitted after acting: R_{t+1} = R(s, a, s', done).
        reward = self.reward_fn(previous_state, action, next_state, done)
        self._state = next_state
        self._done = done
        return TransitionResult(
            state=next_state,
            reward=reward,
            done=done,
            info={
                "action": action,
                "action_label": ACTION_LABELS[action],
                "action_cost": ACTION_COSTS[action],
                "scenario_name": self._scenario_name,
                "termination": termination,
            },
        )

    def _transition(self, state: AgentState, action: int, cost_tenths: int) -> AgentState:
        """Compute the next state s' = T(s, a) under the deterministic dynamics.

        What + why: this is the transition kernel P of the MDP and the heart of the simulator. It
        is a pure, deterministic function: the same (s, a) always yields the same s', with no
        sampling. Each action encodes one orchestration move:

            * ``answer_direct`` (0): TERMINAL commit. The situation is unchanged except the clock;
              quality is judged by the reward (good only if evidence is adequate for the difficulty
              AND ambiguity is low).
            * ``retrieve`` (1): ``evidence += 1`` (capped at ``MAX_EVIDENCE``), ``attempts += 1``,
              budget debited; non-terminal grounding.
            * ``clarify`` (2): ``ambiguity -= 1`` (floored at 0), ``attempts += 1``, budget
              debited; non-terminal disambiguation.
            * ``escalate`` (3): TERMINAL safe hand-off. The situation is unchanged except the clock
              and the budget debit; the reward gives it a modest payoff minus its high cost.

        Args:
            state: The current state s.
            action: The action a in {0, 1, 2, 3}.
            cost_tenths: The action's cost in tenths of a unit, already validated against budget by
                the caller; debited from ``budget`` here.

        Returns:
            The next state s'.

        RL concept: the (deterministic) transition function T(s, a) of the MDP -- each action moves
        exactly one situational variable, which keeps the dynamics legible for teaching.

        Math:
            s' = T(s, a); evidence in [0, MAX_EVIDENCE]; ambiguity in [0, MAX_AMBIGUITY];
            attempts += 1 for a in {1, 2}; budget -= cost_tenths; step += 1.
        """
        evidence = state.evidence
        ambiguity = state.ambiguity
        attempts = state.attempts

        if action == 1:
            # Retrieve: add one unit of grounding (capped), count the attempt.
            evidence = min(MAX_EVIDENCE, state.evidence + 1)
            attempts = state.attempts + 1
        elif action == 2:
            # Clarify: resolve one unit of ambiguity (floored at 0), count the attempt.
            ambiguity = max(0, state.ambiguity - 1)
            attempts = state.attempts + 1
        # answer_direct (0) and escalate (3) leave the situation untouched (terminal commits).

        return AgentState(
            step=state.step + 1,  # advance the within-episode clock by one step
            intent=state.intent,
            difficulty=state.difficulty,
            ambiguity=ambiguity,
            evidence=evidence,
            attempts=attempts,
            budget=state.budget - cost_tenths,
        )


def _clamp(value: int, lower: int, upper: int) -> int:
    """Clamp an integer field into ``[lower, upper]`` to keep the state space finite.

    What + why: jittered start values can fall outside their valid band; clamping bounds each field
    so the joint state stays finite and tabular methods see a closed state set.

    Args:
        value: The raw value before bounding.
        lower: Inclusive floor.
        upper: Inclusive ceiling.

    Returns:
        ``value`` confined to ``[lower, upper]``.

    RL concept: keeping the discrete state space closed and finite for tabular methods.
    """
    return max(lower, min(upper, value))


def scenario_catalog() -> tuple[RequestScenario, ...]:
    """Return the full tuple of available start-state scenarios.

    What + why: a stable public accessor for the set of request profiles, so runners and evaluation
    can enumerate every scenario (e.g. to score a policy across the whole start-state distribution)
    without importing the module-level constant directly.

    Returns:
        The immutable tuple :data:`SCENARIOS` of request scenarios.

    RL concept: enumerating the start-state distribution for evaluation across the request
    landscape.
    """
    return SCENARIOS
