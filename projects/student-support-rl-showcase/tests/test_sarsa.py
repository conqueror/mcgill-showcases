"""Tests pinning SARSA as on-policy TD control, distinct from off-policy Q-learning.

SARSA and Q-learning differ in exactly one place: the backup target. SARSA bootstraps from the
value of the action it *actually takes next* under its epsilon-greedy behaviour, ``Q[s'][A']``,
whereas Q-learning bootstraps from ``max_a' Q[s'][a']``. These tests guard that distinction at
two levels of rigor. A black-box test asserts the two learners' value functions diverge on a
substantial fraction of shared states under matched hyperparameters. A white-box test removes
the RNG-schedule ambiguity by reconstructing the SARSA loop twice on the SAME random stream --
once with the genuine on-policy backup, once with a ``max`` backup -- and proves ``train_sarsa``
matches the on-policy one and not the ``max`` one. Shape and guard-clause tests round out the
contract. SARSA sits on the value-based rung of the ladder (contextual bandit -> MDP ->
Q-learning -> DQN -> policy gradient -> actor-critic -> PPO), one step beside Q-learning.

RL concept:
    On-policy vs off-policy temporal-difference control; see docs/value-based-learning.md and
    docs/mdp-and-environment.md.

Math:
    SARSA target uses the next *chosen* action A' (not the max):
        target = R_{t+1} + gamma * Q(s', A')   (0 if terminal);  delta = target - Q(s, A)
    Contrast (off-policy Q-learning):  target = R_{t+1} + gamma * max_a' Q(s', a').
"""

from __future__ import annotations

import random

from student_support_rl.environment import (
    ACTION_LABELS,
    StudentSupportEnvironment,
    default_reward,
)
from student_support_rl.policies import greedy_action
from student_support_rl.q_learning import train_q_learning
from student_support_rl.sarsa import StateKey, train_sarsa

_CURVE_COLUMNS = {"episode", "scenario_id", "total_reward", "epsilon", "steps"}


def test_sarsa_training_returns_curve_and_q_table() -> None:
    """Shape contract: curve length matches episodes, table is populated, columns match."""
    result = train_sarsa(episodes=24, seed=11, scenario_ids=(0, 1, 2, 3, 4))

    assert len(result.training_curve) == 24
    assert result.q_table
    assert _CURVE_COLUMNS == set(result.training_curve[0])
    # Curve rounding contract: total_reward and epsilon are 4dp floats; steps/ids are ints.
    row = result.training_curve[0]
    assert isinstance(row["episode"], int)
    assert isinstance(row["scenario_id"], int)
    assert isinstance(row["steps"], int)
    assert round(float(row["total_reward"]), 4) == row["total_reward"]
    assert round(float(row["epsilon"]), 4) == row["epsilon"]


def test_sarsa_and_q_learning_disagree_on_many_shared_states() -> None:
    """SARSA and Q-learning produce different value functions under matched hyperparameters.

    Both learners share seed and hyperparameters and use the same epsilon-greedy behaviour
    rule, so they explore the same *kind* of policy. They differ in the backup target (SARSA
    bootstraps from ``Q[s'][A']`` for the actually-chosen ``A'``; Q-learning from
    ``max_a' Q[s'][a']``). Note: this is *not* the only mechanical difference -- SARSA commits
    to its first action before the step loop, so the two consume the shared RNG on slightly
    different schedules and visit overlapping-but-not-identical state sets. This black-box
    test therefore only asserts the weaker, robust fact that the value functions diverge on a
    substantial fraction of shared states; the on-policy backup itself is pinned precisely by
    ``test_sarsa_backup_is_on_policy_not_max`` below.
    """
    sarsa = train_sarsa(
        episodes=40,
        seed=3,
        scenario_ids=(0, 1, 2, 3, 4),
        alpha=0.35,
        gamma=0.9,
        epsilon=0.4,
        epsilon_decay=0.97,
        epsilon_min=0.05,
        horizon=6,
    )
    q_learning = train_q_learning(
        episodes=40,
        seed=3,
        scenario_ids=(0, 1, 2, 3, 4),
        alpha=0.35,
        gamma=0.9,
        epsilon=0.4,
        epsilon_decay=0.97,
        epsilon_min=0.05,
        horizon=6,
    )

    shared_keys = set(sarsa.q_table) & set(q_learning.q_table)
    assert len(shared_keys) >= 50, "expected substantial state overlap for a fair comparison"

    differing = [
        key
        for key in shared_keys
        if [round(v, 6) for v in sarsa.q_table[key]]
        != [round(v, 6) for v in q_learning.q_table[key]]
    ]
    # A regression that silently weakened learning could leave most cells at their 0.0 seed
    # value, making the tables agree. Require a meaningful fraction to disagree, not just one
    # cell. (Empirically ~47% disagree at this seed; 25% is a comfortable, stable floor.)
    assert len(differing) >= len(shared_keys) // 4, (
        f"only {len(differing)}/{len(shared_keys)} shared states differ; "
        "expected on-policy vs off-policy backups to diverge on many states"
    )


def _rebuild_sarsa_table(
    *,
    rule: str,
    seed: int,
    scenario_ids: tuple[int, ...],
    alpha: float,
    gamma: float,
    epsilon: float,
    episodes: int,
    horizon: int,
) -> tuple[dict[StateKey, list[float]], bool]:
    """Re-implement the SARSA loop with a swappable backup rule, mirroring RNG usage exactly.

    Returns the learned table plus a flag recording whether any non-terminal step actually
    selected a next action ``A'`` that was NOT greedy while ``Q[s']`` was already non-flat --
    i.e. a step where the on-policy target ``Q[s'][A']`` provably differs from ``max Q[s']``.
    That flag is what guarantees the two ``rule`` values can diverge, so a test built on this
    has real discriminating power rather than passing vacuously.
    """
    rng = random.Random(seed)
    table: dict[StateKey, list[float]] = {}
    diverging_step_seen = False

    def choose(action_values: list[float]) -> int:
        # epsilon is fixed (no decay) in this reconstruction; with epsilon=1.0 every action
        # is uniform-random, making the whole trajectory RNG-determined and reproducible.
        if rng.random() < epsilon:
            return rng.randrange(len(action_values))
        return greedy_action(action_values)

    for episode in range(1, episodes + 1):
        scenario_id = scenario_ids[(episode - 1) % len(scenario_ids)]
        environment = StudentSupportEnvironment(horizon=horizon, reward_fn=default_reward)
        state = environment.reset(seed=seed + episode, scenario_id=scenario_id)
        state_key = state.as_tuple()
        table.setdefault(state_key, [0.0] * len(ACTION_LABELS))
        action = choose(table[state_key])

        while not environment.is_done():
            transition = environment.step(action)
            next_key = transition.state.as_tuple()
            table.setdefault(next_key, [0.0] * len(ACTION_LABELS))
            next_action = choose(table[next_key])

            if (
                not transition.done
                and next_action != greedy_action(table[next_key])
                and abs(table[next_key][next_action] - max(table[next_key])) > 1e-9
            ):
                diverging_step_seen = True

            future = (
                table[next_key][next_action] if rule == "on_policy" else max(table[next_key])
            )
            old_value = table[state_key][action]
            target = transition.reward + (0.0 if transition.done else gamma * future)
            table[state_key][action] = old_value + alpha * (target - old_value)

            state_key, action = next_key, next_action

    return table, diverging_step_seen


def test_sarsa_backup_is_on_policy_not_max() -> None:
    """White-box guard: the SARSA backup uses ``Q[s'][A']`` (chosen A'), never ``max Q[s']``.

    The black-box comparison test is necessary but not sufficient: because SARSA and
    Q-learning consume the RNG on different schedules, their tables would diverge *even if*
    SARSA secretly used a ``max`` backup. This test removes that ambiguity. Under a fully
    RNG-determined regime (``epsilon=1.0``), it reconstructs the SARSA loop twice with the
    SAME RNG schedule -- once with the genuine on-policy backup, once with a ``max`` backup --
    and asserts:

      * ``train_sarsa`` reproduces the on-policy reconstruction EXACTLY, and
      * the on-policy and ``max`` reconstructions actually DIFFER here, and
      * a non-terminal step really chose a non-greedy ``A'`` over a non-flat ``Q[s']``.

    The middle and last assertions prove the test is discriminating: a regression that
    replaced ``Q[s'][A']`` with ``max Q[s']`` would make ``train_sarsa`` match the ``max``
    reconstruction and fail the first assertion. Config chosen so the two rules provably part.
    """
    on_policy_table, diverged = _rebuild_sarsa_table(
        rule="on_policy",
        seed=0,
        scenario_ids=(1, 2),
        alpha=0.5,
        gamma=0.9,
        epsilon=1.0,
        episodes=5,
        horizon=6,
    )
    max_table, _ = _rebuild_sarsa_table(
        rule="max",
        seed=0,
        scenario_ids=(1, 2),
        alpha=0.5,
        gamma=0.9,
        epsilon=1.0,
        episodes=5,
        horizon=6,
    )

    # Sanity: the reconstruction actually hits a step where the chosen A' is non-greedy with a
    # non-flat Q[s'], so on-policy and max targets genuinely diverge -> the test has teeth.
    assert diverged, "reconstruction never exercised a step where Q[s'][A'] != max Q[s']"

    # Teeth: the two backup rules must yield genuinely different tables for this config.
    rules_differ = set(on_policy_table) != set(max_table) or any(
        on_policy_table[k] != max_table.get(k) for k in on_policy_table
    )
    assert rules_differ, "config failed to separate on-policy from max backups; pick another"

    result = train_sarsa(
        seed=0,
        scenario_ids=(1, 2),
        alpha=0.5,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=1.0,
        epsilon_min=1.0,
        episodes=5,
        horizon=6,
    )

    # The real implementation must match the ON-POLICY reconstruction exactly...
    assert set(result.q_table) == set(on_policy_table)
    assert all(result.q_table[k] == on_policy_table[k] for k in on_policy_table)

    # ...and must NOT match the max reconstruction (else the backup is off-policy).
    matches_max = set(result.q_table) == set(max_table) and all(
        result.q_table[k] == max_table.get(k) for k in result.q_table
    )
    assert not matches_max, "train_sarsa matched a max backup; the update is not on-policy"


def test_sarsa_rejects_non_positive_episodes() -> None:
    """Guard clause: ``episodes <= 0`` raises ``ValueError`` (matches the documented Raises)."""
    import pytest

    with pytest.raises(ValueError, match="episodes must be positive"):
        train_sarsa(episodes=0)
