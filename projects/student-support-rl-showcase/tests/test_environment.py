"""Pin the MDP contract of the student-support environment as executable documentation.

These tests fix the agent-environment loop that every later algorithm relies on: a
``reset`` that yields a defined initial state s_0, a ``step(a)`` that returns the
(s', R_{t+1}, done) transition tuple, a horizon-bounded episode, and a validated action
space A = {0,1,2,3}. The environment is the bottom rung of the ladder (contextual bandit ->
MDP -> Q-learning -> DQN -> policy gradient -> actor-critic -> PPO); pinning its determinism
is what makes the value-based and policy-gradient tests reproducible.

RL concept:
    Markov decision process and the agent-environment interface; see
    docs/mdp-and-environment.md.
"""

from __future__ import annotations

import pytest

from student_support_rl.environment import StudentSupportEnvironment


def test_environment_reset_and_step_are_deterministic() -> None:
    """Verify reset gives a fixed s_0 and one step yields the exact (s', R_{t+1}) tuple.

    Pins that the transition kernel is a deterministic teaching simulator: with no RNG seed,
    a TA-session action (a=2) on the high-risk scenario maps the start state to one known
    successor with a strictly positive reward, the episode is not yet terminal, and
    ``observe`` reports that same successor. This determinism is the precondition for every
    downstream Q-learning / policy-gradient reproducibility test.

    RL concept:
        Deterministic MDP transition and the s -> (s', R_{t+1}) step contract; see
        docs/mdp-and-environment.md.
    """
    env = StudentSupportEnvironment(horizon=4)

    state = env.reset(scenario_id=2)

    # Initial state s_0 for the high-risk scenario is fixed when no seed is supplied.
    assert state.as_tuple() == (1, 1, 1, 3, 3, 0)

    transition = env.step(2)

    # One deterministic transition: s -> s' with reward R_{t+1} and a not-yet-terminal flag.
    assert transition.state.as_tuple() == (2, 2, 3, 2, 1, 1)
    assert transition.reward > 0
    assert transition.done is False
    assert env.observe().as_tuple() == transition.state.as_tuple()
    assert env.scenario_name == "high_risk_student"


def test_environment_marks_done_at_horizon() -> None:
    """Verify the episode terminates exactly when the week index passes the horizon.

    Pins the finite-horizon boundary: with horizon H=2, the agent acts twice and only the
    second transition carries ``done=True``, after which ``is_done`` stays latched. A
    correct terminal flag is what stops bootstrapping in the TD target (the next-state value
    term must be dropped at episode end).

    RL concept:
        Episodic/finite-horizon termination; the terminal flag gates the bootstrap term in
        the TD update. See docs/mdp-and-environment.md and docs/value-based-learning.md.
    """
    env = StudentSupportEnvironment(horizon=2)
    env.reset(scenario_id=1, seed=0)

    assert env.is_done() is False

    env.step(1)
    assert env.is_done() is False

    terminal = env.step(0)
    # done flips True only after the horizon-th decision; later bootstrapping must stop here.
    assert terminal.done is True
    assert env.is_done() is True


def test_environment_rejects_invalid_actions() -> None:
    """Verify the action space is closed: an out-of-range action raises ``ValueError``.

    Pins that A = {0,1,2,3} (no-op, resource email, TA session, advisor meeting) is the only
    admissible action set, so stepping with a=9 fails loudly. A guarded action space keeps
    every learner's argmax and sampling confined to legal interventions.

    RL concept:
        Discrete action space A(s); see docs/mdp-and-environment.md.
    """
    env = StudentSupportEnvironment()
    env.reset()

    # Action 9 is outside A = {0,1,2,3}; the environment must reject it rather than guess.
    with pytest.raises(ValueError, match="unknown action"):
        env.step(9)


def test_environment_seed_changes_initial_state_deterministically() -> None:
    """Verify the seeded reset is reproducible: same (seed, scenario) gives the same s_0.

    Pins that seeding makes initial-state randomization a pure function of (seed,
    scenario_id): two resets with seed=3 on scenario 1 return identical states. Reproducible
    starts are what let training curves and policy comparisons be compared run-to-run.

    RL concept:
        Reproducible stochastic initial-state distribution rho_0; see
        docs/mdp-and-environment.md and docs/evaluation-and-governance.md.
    """
    env = StudentSupportEnvironment()
    first = env.reset(seed=3, scenario_id=1)
    second = env.reset(seed=3, scenario_id=1)

    # Same (seed, scenario) -> identical s_0: the start distribution is reproducible.
    assert first.as_tuple() == second.as_tuple()
