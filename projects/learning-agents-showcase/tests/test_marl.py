"""Pin multi-agent coordination (locus C): independent learners miscoordinate, joint learning wins.

These tests anchor the MARL lane on the cooperative Climbing game. They pin the cooperative optimum,
that joint-action (centralised) learning reliably reaches it across seeds, and that independent
Q-learning reliably fails to -- retreating to a safe, suboptimal joint action (relative
overgeneralisation under non-stationarity). The gap between the two success rates is the lesson, and
it is checked robustly across many seeds rather than from a single run.

RL concept:
    Locus of learning C -- multi-agent coordination; decentralised (independent) vs centralised
    (joint-action) learning on a cooperative game.
"""

from __future__ import annotations

from learning_agents.marl import (
    CLIMBING_GAME,
    coordination_success_rate,
    marl_comparison_rows,
    optimal_joint_action,
    optimal_team_reward,
    team_reward,
    train_independent_q_learning,
    train_joint_action_learner,
)

_SEEDS = 12
_EPISODES = 3000


def test_optimum_is_the_thorough_detailed_corner() -> None:
    """The cooperative optimum is (deep_research, detailed) with the top team reward.

    Pins the coordination target: the joint action maximising team reward is index (0, 0) -- the
    relabelled Climbing-game optimum -- worth 11, and ``team_reward`` reads the matrix correctly.
    """
    assert optimal_joint_action(CLIMBING_GAME) == (0, 0)
    assert optimal_team_reward(CLIMBING_GAME) == 11.0
    assert team_reward(CLIMBING_GAME, 0, 0) == 11.0
    assert team_reward(CLIMBING_GAME, 2, 2) == 5.0  # the safe, suboptimal corner


def test_joint_action_learning_reaches_the_optimum() -> None:
    """Centralised joint-action learning converges to the cooperative optimum, every seed.

    Pins the centralised upside: with the full joint view the learner reaches (deep_research,
    detailed) on a representative seed and across all seeds (success rate 1.0).
    """
    result = train_joint_action_learner(CLIMBING_GAME, episodes=_EPISODES, seed=0)
    assert result.final_joint_action == (0, 0)
    assert result.reached_optimum
    assert result.final_team_reward == 11.0
    assert coordination_success_rate(
        CLIMBING_GAME, method="joint", seeds=_SEEDS, episodes=_EPISODES
    ) == 1.0


def test_independent_learning_miscoordinates() -> None:
    """Independent Q-learning fails to reach the optimum and settles for a safe, lower-value joint.

    Pins the decentralised failure: on a representative seed independent learners do not reach the
    optimum and earn less than 11, and across seeds their success rate is strictly below the
    joint-action learner's -- the relative-overgeneralisation / non-stationarity lesson.
    """
    result = train_independent_q_learning(CLIMBING_GAME, episodes=_EPISODES, seed=0)
    assert not result.reached_optimum
    assert result.final_team_reward < optimal_team_reward(CLIMBING_GAME)

    independent_rate = coordination_success_rate(
        CLIMBING_GAME, method="independent", seeds=_SEEDS, episodes=_EPISODES
    )
    joint_rate = coordination_success_rate(
        CLIMBING_GAME, method="joint", seeds=_SEEDS, episodes=_EPISODES
    )
    assert independent_rate < joint_rate  # independence coordinates far less reliably


def test_training_curve_and_determinism() -> None:
    """Both learners log a convergence curve and reproduce exactly under a fixed seed."""
    first = train_joint_action_learner(CLIMBING_GAME, episodes=_EPISODES, seed=3)
    second = train_joint_action_learner(CLIMBING_GAME, episodes=_EPISODES, seed=3)
    assert first == second  # frozen dataclass value-equality -> deterministic
    assert first.training_curve  # convergence checkpoints are recorded
    assert set(first.training_curve[0]) == {"episode", "greedy_team_reward"}


def test_marl_comparison_rows_show_the_coordination_gap() -> None:
    """The comparison artifact contrasts independent vs joint with the expected schema and gap.

    Pins the headline MARL artifact: one row per method with the documented columns, the joint
    learner's success rate strictly exceeding the independent learner's, and both referencing the
    same optimal team reward.
    """
    rows = marl_comparison_rows(seeds=_SEEDS, episodes=_EPISODES)
    by_method = {str(row["method"]): row for row in rows}
    assert set(by_method) == {"independent", "joint"}
    assert set(rows[0]) == {
        "method",
        "coordination_success_rate",
        "final_joint_action",
        "final_team_reward",
        "optimal_team_reward",
    }
    assert float(by_method["joint"]["coordination_success_rate"]) > float(
        by_method["independent"]["coordination_success_rate"]
    )
    assert float(by_method["joint"]["optimal_team_reward"]) == 11.0
