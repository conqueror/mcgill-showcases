"""Define the student-support MDP every agent in this showcase plans and learns against.

What + why: reinforcement learning needs a precise *environment* -- the world the agent
acts on and the source of the reward signal. This module is that world. It models a single
student over a six-week term: each week an academic-support agent chooses one of four
interventions, the student's situation evolves, and a reward scores how well progress and
risk were balanced against the cost of intervening. Every other module (bandit, value
iteration, Q-learning, SARSA, DQN, REINFORCE) plugs into this one object, so it is the
shared anchor at the bottom of the ladder
(contextual bandit -> MDP -> Q-learning -> DQN -> policy gradient -> actor-critic -> PPO).

The MDP tuple (S, A, P, R, gamma, H):
    S -- a finite, fully observed state of six discrete variables
        (week, engagement, completion, pressure, risk, prior_interventions).
    A -- four actions {0: no_intervention, 1: resource_email, 2: ta_session,
        3: advisor_meeting} with strictly rising costs.
    P -- the transition kernel. HONESTY: P is *deterministic* given (s, a) -- ``_transition``
        is a pure function with no sampling, so s' is fixed once (s, a) are fixed. The
        ``seed`` argument to :meth:`StudentSupportEnvironment.reset` only jitters the
        *start* state; it never injects randomness into a step. Treat this as a
        deterministic finite-horizon MDP.
    R -- :func:`default_reward`, the reward emitted as R_{t+1} after acting in s.
    gamma -- the discount; owned by the *agent*, not stored here.
    H -- the horizon (default 6 weeks); the episode terminates after week H.

RL concept: Markov decision process and environment design -- see docs/mdp-and-environment.md
(states, actions, transition, termination), docs/reward-design-and-hacking.md (why the
reward is shaped the way it is and how it can be gamed), docs/math-notes.md (the MDP tuple
and return), and docs/glossary.md (state vs. observation, episodic vs. continuing).

Math:
    Deterministic transition s' = T(s, a) realized by ``_transition``.
    Reward after acting: R_{t+1} = R(s, a, s', done) from :func:`default_reward`.
    Discounted return the agent maximizes: G_t = sum_k gamma^k R_{t+k+1}.
    Finite horizon: the episode ends once week exceeds H.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

# Action space A: four discrete interventions, indexed 0..MAX_ACTION.
MAX_ACTION = 3
ACTION_LABELS = {
    0: "no_intervention",
    1: "resource_email",
    2: "ta_session",
    3: "advisor_meeting",
}
# Action cost c(a): strictly rising, so the reward must trade efficacy against price.
ACTION_COSTS = {
    0: 0.0,
    1: 0.2,
    2: 0.7,
    3: 1.2,
}

@dataclass(frozen=True)
class ScenarioDefinition:
    """Hold the starting profile for one named student scenario.

    What + why: a scenario fixes the initial metrics that :meth:`StudentSupportEnvironment.reset`
    builds the start state S_0 from. Distinct scenarios give learners a spread of difficulties
    (low risk through high workload through fatigued-by-prior-help) so a single trained agent
    can be evaluated across the support landscape rather than one lucky student.

    Attributes:
        scenario_id: Stable index into :data:`SCENARIOS`; also seeds the reset jitter.
        name: Human-readable label surfaced in transition info and reports.
        engagement: Initial engagement metric in [0, 4] (higher is better).
        completion: Initial assignment-completion metric in [0, 4] (higher is better).
        pressure: Initial workload-pressure metric in [0, 4] (higher is worse).
        prior_interventions: Count of supports already received before week 1; drives
            both the over-intervention penalty and transition fatigue.

    RL concept: the start-state distribution of the MDP -- see docs/mdp-and-environment.md
    (how S_0 is chosen) and docs/evaluation-and-governance.md (evaluating across scenarios).
    """

    scenario_id: int
    name: str
    engagement: int
    completion: int
    pressure: int
    prior_interventions: int = 0


# Start-state distribution of the MDP: the catalog of student profiles a reset can draw from.
SCENARIOS: tuple[ScenarioDefinition, ...] = (
    ScenarioDefinition(0, "low_risk_student", 4, 4, 1),
    ScenarioDefinition(1, "medium_risk_student", 2, 2, 2),
    ScenarioDefinition(2, "high_risk_student", 1, 1, 3),
    ScenarioDefinition(3, "high_workload_pressure", 3, 2, 4),
    ScenarioDefinition(4, "repeated_prior_interventions", 2, 2, 2, prior_interventions=3),
)
# Reward signature R(s, a, s', done) -> R_{t+1}: lets callers swap in alternative reward designs.
RewardFunction = Callable[["StudentState", int, "StudentState", bool], float]


@dataclass(frozen=True)
class StudentState:
    """Represent one fully observed state s of the MDP as six discrete variables.

    What + why: the state is everything the agent sees and everything the dynamics depend on.
    Because all six fields are integers, the joint state is finite and discrete, which is what
    lets the tabular methods on the lower rungs of the ladder key a Q-table or policy directly
    on it. The state is frozen (immutable) so a transition always produces a *new* object,
    keeping trajectories safe to store and replay. This is full observability: the observation
    equals the state, so there is no hidden information (no POMDP).

    Attributes:
        week: Current week index 1..H; the within-episode clock.
        engagement: Engagement metric in [0, 4] (higher is better).
        completion: Assignment-completion metric in [0, 4] (higher is better).
        pressure: Workload-pressure metric in [0, 4] (higher is worse).
        risk: Heuristic risk level in [0, 3] derived from the other metrics by
            :func:`risk_from_metrics` (3 = highest risk).
        prior_interventions: Cumulative count of non-null interventions taken so far.

    RL concept: state representation in an MDP -- see docs/mdp-and-environment.md (state
    space S, full observability) and docs/glossary.md (state vs. observation).
    """

    week: int
    engagement: int
    completion: int
    pressure: int
    risk: int
    prior_interventions: int

    def as_tuple(self) -> tuple[int, int, int, int, int, int]:
        """Return the state as a plain integer tuple for use as a hashable table key.

        What + why: tabular agents index a dict by the discrete state, so they need a
        lightweight, hashable, order-fixed view of s. The field order matches the dataclass
        declaration and is treated as a stable contract by callers and tests.

        Returns:
            The six state fields as ``(week, engagement, completion, pressure, risk,
            prior_interventions)``.

        RL concept: the discrete state key behind tabular value/policy lookup --
        see docs/value-based-learning.md.
        """
        return (
            self.week,
            self.engagement,
            self.completion,
            self.pressure,
            self.risk,
            self.prior_interventions,
        )

    def as_normalized_vector(
        self,
        *,
        horizon: int,
    ) -> tuple[float, float, float, float, float, float]:
        """Return the state as a roughly [0, 1]-scaled float feature vector for function approx.

        What + why: deep agents (the DQN rung) consume features, not table keys. Dividing each
        field by its natural maximum puts every input on a comparable scale, which conditions a
        neural network far better than raw integers. Week and prior_interventions are scaled by
        the horizon since both grow with episode length; metrics use their fixed caps (4 and 3).

        Args:
            horizon: Episode length H, used to scale the time-like fields; floored at 1 to
                avoid division by zero.

        Returns:
            Six floats, each rounded to 6 decimals, for ``(week, engagement, completion,
            pressure, risk, prior_interventions)``.

        RL concept: feature encoding for value-function approximation -- see docs/deep-rl.md
        (why raw states are normalized before a network).
        """
        return (
            round(self.week / max(1, horizon), 6),
            round(self.engagement / 4.0, 6),
            round(self.completion / 4.0, 6),
            round(self.pressure / 4.0, 6),
            round(self.risk / 3.0, 6),
            round(self.prior_interventions / float(max(1, horizon)), 6),
        )


@dataclass(frozen=True)
class TransitionResult:
    """Bundle the outcome of one environment step: (s', R_{t+1}, done, info).

    What + why: this is the standard RL step tuple every agent consumes to learn. ``state`` is
    the next state s', ``reward`` is the reward emitted after acting (R_{t+1}), and ``done``
    flags episode termination so the agent knows to stop bootstrapping past the horizon. The
    ``info`` dict carries diagnostics (chosen action, its cost, realized risk reduction,
    scenario name) for logging and reward analysis; agents must not learn from it.

    Attributes:
        state: The next state s' after the transition (week-clamped if terminal).
        reward: The scalar reward R_{t+1} returned by the reward function.
        done: True once the episode has terminated (week exceeded the horizon).
        info: Auxiliary diagnostics; not part of the state and not for learning.

    RL concept: the agent-environment step interface -- see docs/mdp-and-environment.md
    (the (s', r, done) loop) and docs/evaluation-and-governance.md (logging via info).
    """

    state: StudentState
    reward: float
    done: bool
    info: dict[str, int | float | str]


def default_reward(
    previous_state: StudentState,
    action: int,
    next_state: StudentState,
    done: bool,
) -> float:
    """Score one transition as the MDP reward R_{t+1}, balancing gains against cost.

    What + why: the reward function is the *objective* of the whole problem -- everything the
    agent optimizes flows from this single number. This design is a deliberately shaped reward
    that rewards real student gains (more engagement/completion, lower risk) while charging for
    the cost and the side effects of intervening, so the optimal policy is *not* "always send
    the most expensive help." It is the running example for reward hacking: each term exists to
    close a loophole an agent would otherwise exploit (e.g. over-intervening, or coasting into a
    high-risk finish). The formula weights risk reduction (1.4) above raw progress (1.0)
    because de-risking a student is the primary mission.

    Args:
        previous_state: The state s before acting.
        action: The action a taken; indexes :data:`ACTION_COSTS` for its price.
        next_state: The resulting state s'.
        done: Whether this transition terminated the episode; gates the end-of-term penalty.

    Returns:
        The scalar reward R_{t+1}, rounded to 4 decimals.

    RL concept: reward design and reward hacking -- see docs/reward-design-and-hacking.md
    (each term as an anti-gaming guardrail) and docs/mdp-and-environment.md (R in the MDP tuple).

    Math:
        R_{t+1} = 1.0 * progress + 1.4 * risk_reduction - c(a)
                  - over_intervention_penalty - unresolved_high_risk_penalty,
        where progress and risk_reduction are differences between s' and s.
    """
    # Progress term: gain in engagement + completion from s to s' (the main "did the student
    # improve" signal).
    progress = (next_state.engagement + next_state.completion) - (
        previous_state.engagement + previous_state.completion
    )
    # Risk-reduction term: drop in heuristic risk level, weighted 1.4x as the primary objective.
    risk_reduction = previous_state.risk - next_state.risk
    # Over-intervention penalty: charges for each cumulative intervention beyond 2 (diminishing
    # social returns / discourages spamming help).
    over_intervention_penalty = 0.6 * max(0, next_state.prior_interventions - 2)
    # Terminal high-risk penalty: extra cost if the episode ends with the student still at risk.
    unresolved_high_risk_penalty = 1.2 if done and next_state.risk >= 2 else 0.0
    return round(
        (1.0 * progress)
        + (1.4 * risk_reduction)
        - ACTION_COSTS[action]  # action cost c(a): price of the chosen intervention
        - over_intervention_penalty
        - unresolved_high_risk_penalty,
        4,
    )


def _clamp(value: int, lower: int = 0, upper: int = 4) -> int:
    """Clamp an integer metric into ``[lower, upper]`` to keep the state space finite.

    What + why: transition deltas can push a metric past its valid band; clamping bounds every
    metric to [0, 4] so the joint state stays finite and tabular methods see a closed state set.

    Args:
        value: The raw metric value before bounding.
        lower: Inclusive floor (default 0).
        upper: Inclusive ceiling (default 4).

    Returns:
        ``value`` confined to ``[lower, upper]``.
    """
    return max(lower, min(upper, value))


def risk_from_metrics(
    *,
    engagement: int,
    completion: int,
    pressure: int,
    prior_interventions: int,
) -> int:
    """Map the other metrics to a 0..3 risk level via a fixed heuristic score.

    What + why: ``risk`` is a derived feature of the state, not an independent variable -- it
    summarizes how worried we are about the student. HONESTY: this is a hand-tuned *heuristic*
    scoring rule, not a learned or calibrated risk model; it exists to give the agent and the
    reward a single salient danger signal. Lower engagement/completion and higher pressure raise
    the score, as does a history of many prior interventions (a fatigue/escalation proxy);
    thresholds then bucket the score into four ordered levels.

    Args:
        engagement: Engagement metric in [0, 4].
        completion: Completion metric in [0, 4].
        pressure: Pressure metric in [0, 4].
        prior_interventions: Cumulative non-null interventions so far.

    Returns:
        Discrete risk level in {0, 1, 2, 3}, where 3 is highest risk.

    RL concept: a derived state feature (heuristic, not a learned model) -- see
    docs/mdp-and-environment.md (engineered state variables) and
    docs/reward-design-and-hacking.md (risk as the de-risking target).

    Math:
        score = 5 - engagement - completion + pressure + max(0, prior_interventions - 2);
        risk = 3 if score >= 5, 2 if score >= 3, 1 if score >= 1, else 0.
    """
    # Heuristic risk score (NOT a learned model): low engagement/completion and high pressure
    # raise danger; many prior interventions add an escalation/fatigue surcharge.
    score = 5 - engagement - completion + pressure + max(0, prior_interventions - 2)
    if score >= 5:
        return 3
    if score >= 3:
        return 2
    if score >= 1:
        return 1
    return 0


@dataclass
class StudentSupportEnvironment:
    """Simulate the student-support MDP as a Gym-style reset/step environment.

    What + why: this is the concrete world agents interact with -- the object that realizes the
    MDP tuple (S, A, P, R, gamma, H) for the rest of the showcase. It holds the current state as
    private mutable cursor and exposes the classic loop: ``reset`` to draw a start state S_0,
    ``step(a)`` to advance one week and return (s', R_{t+1}, done, info), and ``observe`` /
    ``is_done`` to inspect. HONESTY: dynamics are *deterministic* -- given (s, a), the next state
    is fixed; only the *start* state can be jittered (via ``reset(seed=...)``). gamma lives with
    the agent, not here.

    Attributes:
        horizon: Episode length H in weeks (default 6); the episode terminates after week H.
        reward_fn: The reward function R(s, a, s', done) -> R_{t+1}; defaults to
            :func:`default_reward` but can be swapped to study alternative reward designs.

    RL concept: the agent-environment interface of an MDP -- see docs/mdp-and-environment.md
    (reset/step semantics, termination) and docs/reward-design-and-hacking.md (pluggable reward).
    """

    horizon: int = 6
    reward_fn: RewardFunction = default_reward

    def __post_init__(self) -> None:
        """Initialize the private episode cursor before the first reset.

        What + why: the dataclass fields are configuration; the mutable per-episode state lives
        in private attributes set here so the environment exists in a defined (un-reset) state.
        ``_state is None`` marks "not yet reset" and guards :meth:`observe`.
        """
        self._state: StudentState | None = None
        self._done = False
        self._scenario_name = ""

    def reset(self, seed: int | None = None, scenario_id: int = 0) -> StudentState:
        """Draw a start state S_0 for a chosen scenario and begin a fresh episode.

        What + why: every episode begins here. The selected scenario fixes the baseline metrics;
        an optional ``seed`` jitters each metric by +/- 1 so a scenario yields a small family of
        nearby starts for more robust training and evaluation. HONESTY: the seed only perturbs
        the *start* state -- it is consumed entirely in this method and never touches
        :meth:`step`, so the same seed reproduces the same S_0 exactly and the dynamics remain
        deterministic. The derived ``risk`` of S_0 is computed from the (possibly jittered)
        metrics via :func:`risk_from_metrics`, and the week clock is set to 1.

        Args:
            seed: Optional jitter seed for the start state only; ``None`` uses the scenario's
                exact baseline metrics.
            scenario_id: Index into :data:`SCENARIOS` selecting the student profile.

        Returns:
            The start state S_0.

        Raises:
            ValueError: If ``scenario_id`` is outside ``[0, len(SCENARIOS))``.

        RL concept: the start-state distribution of an episodic MDP -- see
        docs/mdp-and-environment.md (S_0 and episode boundaries).
        """
        if scenario_id < 0 or scenario_id >= len(SCENARIOS):
            raise ValueError("scenario_id out of range")
        scenario = SCENARIOS[scenario_id]
        if seed is not None:
            # Start-state jitter ONLY: this RNG is consumed here and never reaches step(), so the
            # transition stays deterministic; same seed -> same S_0.
            rng = random.Random((seed + 1) * (scenario_id + 7))
            engagement = _clamp(scenario.engagement + rng.choice((-1, 0, 1)))
            completion = _clamp(scenario.completion + rng.choice((-1, 0, 1)))
            pressure = _clamp(scenario.pressure + rng.choice((-1, 0, 1)))
        else:
            engagement = scenario.engagement
            completion = scenario.completion
            pressure = scenario.pressure

        prior_interventions = scenario.prior_interventions
        self._scenario_name = scenario.name
        self._state = StudentState(
            week=1,
            engagement=engagement,
            completion=completion,
            pressure=pressure,
            risk=risk_from_metrics(
                engagement=engagement,
                completion=completion,
                pressure=pressure,
                prior_interventions=prior_interventions,
            ),
            prior_interventions=prior_interventions,
        )
        self._done = False
        return self._state

    def observe(self) -> StudentState:
        """Return the current state s without advancing the environment.

        What + why: under full observability the agent's observation is exactly the state, so
        this hands back the current cursor for the agent to choose its next action from.

        Returns:
            The current state s.

        Raises:
            RuntimeError: If called before :meth:`reset` (no state exists yet).

        RL concept: observation == state under full observability -- see
        docs/mdp-and-environment.md.
        """
        if self._state is None:
            raise RuntimeError("environment must be reset before observe")
        return self._state

    def is_done(self) -> bool:
        """Report whether the current episode has terminated.

        Returns:
            True once a step has pushed the week past the horizon H.

        RL concept: episode termination in an episodic MDP -- see docs/mdp-and-environment.md.
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
        """Advance the MDP one week: apply (P, R) for the chosen action and return the outcome.

        What + why: this is the single environment transition that drives all learning. It reads
        the current state s, applies the deterministic dynamics to get s', decides termination by
        comparing the week to the horizon, scores the move with the reward function (R_{t+1}),
        then commits the new state and done flag. On the terminal step the week is clamped to H so
        the reported terminal state never advertises a week beyond the horizon; the other fields
        of s' are preserved. ``done`` is passed into the reward so the end-of-term penalty can
        fire.

        Args:
            action: The intervention a to take; must be a valid key of :data:`ACTION_LABELS`.

        Returns:
            A :class:`TransitionResult` with (s', R_{t+1}, done, info).

        Raises:
            ValueError: If ``action`` is not one of the four defined actions.
            RuntimeError: Propagated from :meth:`observe` if called before :meth:`reset`.

        RL concept: the (s, a) -> (s', r, done) transition of the MDP -- see
        docs/mdp-and-environment.md (the step loop) and docs/reward-design-and-hacking.md
        (where R_{t+1} comes from).

        Math:
            s' = T(s, a) (deterministic); R_{t+1} = R(s, a, s', done).
        """
        if action not in ACTION_LABELS:
            raise ValueError(f"unknown action: {action}")
        previous_state = self.observe()
        # Deterministic transition s' = T(s, a); no sampling here.
        next_state = self._transition(previous_state, action)
        # Finite-horizon termination: the episode ends once the new week exceeds H.
        done = next_state.week > self.horizon
        if done:
            # Terminal week-clamp: report week == H rather than H + 1; metrics/risk unchanged.
            next_state = StudentState(
                week=self.horizon,
                engagement=next_state.engagement,
                completion=next_state.completion,
                pressure=next_state.pressure,
                risk=next_state.risk,
                prior_interventions=next_state.prior_interventions,
            )
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
                "risk_reduction": previous_state.risk - next_state.risk,
                "scenario_name": self._scenario_name,
            },
        )

    def _transition(self, state: StudentState, action: int) -> StudentState:
        """Compute the next state s' = T(s, a) under the deterministic dynamics.

        What + why: this is the transition kernel P of the MDP and the heart of the simulator.
        It is a pure, deterministic function: the same (s, a) always yields the same s', with no
        sampling. Each action applies hand-crafted deltas to engagement/completion/pressure that
        encode the domain story -- doing nothing lets a pressured or risky student slip; cheaper
        supports help modestly; costlier supports help more and relieve pressure. Two structural
        effects shape the policy: (1) *fatigue* -- repeated interventions yield diminishing
        engagement gains; (2) *pressure-sensitive drift* -- under no intervention, high pressure
        erodes engagement and low completion raises pressure further. After the deltas, metrics
        are clamped to [0, 4], the intervention counter increments for any non-null action, and
        ``risk`` is recomputed from the new metrics, advancing the week by one.

        Args:
            state: The current state s.
            action: The action a in {0, 1, 2, 3}.

        Returns:
            The next state s'.

        RL concept: the (deterministic) transition function T(s, a) of the MDP -- see
        docs/mdp-and-environment.md (dynamics) and docs/math-notes.md (deterministic P).

        Math:
            s' = T(s, a); engagement/completion/pressure clamped to [0, 4];
            prior_interventions += 1 for a != 0; risk = risk_from_metrics(...).
        """
        # Fatigue: repeated prior interventions blunt the engagement payoff of active supports
        # (diminishing returns from over-helping).
        fatigue = max(0, state.prior_interventions - 1)

        if action == 0:
            # No intervention: pressure-sensitive drift -- high pressure erodes engagement, risk
            # erodes completion, and low completion ratchets pressure up.
            engagement_delta = -1 if state.pressure >= 3 else 0
            completion_delta = -1 if state.risk >= 2 else 0
            pressure_delta = 1 if state.completion <= 2 else 0
        elif action == 1:
            # Resource email (cheap): small engagement bump, completion only if already engaged.
            engagement_delta = 1
            completion_delta = 1 if state.engagement >= 2 else 0
            pressure_delta = 0
        elif action == 2:
            # TA session: engagement gain net of fatigue; strong completion help; eases pressure.
            engagement_delta = 1 - fatigue
            completion_delta = 2 if state.completion <= 2 else 1
            pressure_delta = -1
        else:
            # Advisor meeting (costliest): largest engagement gain (net of fatigue) when at risk,
            # steady completion help, and the biggest pressure relief.
            engagement_delta = (2 if state.risk >= 2 else 1) - fatigue
            completion_delta = 1
            pressure_delta = -2

        # Clamp metrics back into the valid band so the state space stays finite.
        engagement = _clamp(state.engagement + engagement_delta)
        completion = _clamp(state.completion + completion_delta)
        pressure = _clamp(state.pressure + pressure_delta)
        # Count this intervention unless it was the null action (action == 0).
        prior_interventions = state.prior_interventions + (1 if action else 0)
        # Recompute the derived risk feature from the updated metrics.
        risk = risk_from_metrics(
            engagement=engagement,
            completion=completion,
            pressure=pressure,
            prior_interventions=prior_interventions,
        )
        return StudentState(
            week=state.week + 1,  # advance the within-episode clock by one week
            engagement=engagement,
            completion=completion,
            pressure=pressure,
            risk=risk,
            prior_interventions=prior_interventions,
        )


def scenario_catalog() -> tuple[ScenarioDefinition, ...]:
    """Return the full tuple of available start-state scenarios.

    What + why: a stable public accessor for the set of student profiles, so runners and
    evaluation can enumerate every scenario (e.g. to score a policy across the whole start-state
    distribution) without importing the module-level constant directly.

    Returns:
        The immutable tuple :data:`SCENARIOS` of scenario definitions.

    RL concept: enumerating the start-state distribution for evaluation -- see
    docs/evaluation-and-governance.md.
    """
    return SCENARIOS
