"""Pin off-policy evaluation: estimating a target policy's value from a behaviour log only.

These tests anchor OPE -- the governance-critical ability to vet a candidate policy from logged data
without running it. They pin that: a deterministic target's propensity is a point mass; trajectories
are reconstructed from the flat log; and, graded against the simulator's true value, the four
estimators (IS, WIS, FQE direct method, doubly-robust) are accurate for *well-covered* targets but
degrade for a target far from the behaviour policy -- the overlap/coverage requirement at the heart
of OPE, with weighted IS the most robust under poor overlap.

RL concept:
    Off-policy evaluation (importance sampling, direct method, doubly-robust) and its dependence on
    behaviour/target overlap.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

import pytest

from learning_agents.dynamic_programming import optimal_action_values
from learning_agents.environment import AgentDecisionEnvironment
from learning_agents.offline_rl import collect_logged_dataset
from learning_agents.ope import (
    ope_estimates,
    ope_report_rows,
    target_action_probability,
    true_policy_value,
)
from learning_agents.policies import (
    AlwaysEscalatePolicy,
    HeuristicRouterPolicy,
    Policy,
    QTablePolicy,
)
from scripts import run_ope, run_showcase

# A behaviour log from an epsilon-soft heuristic router: covers the router's neighbourhood well.
_LOG = collect_logged_dataset(episodes=800, epsilon=0.3, seed=7)
_TRUTH_EPISODES = 40
_COVERED_TOLERANCE = 0.15  # observed errors are <=0.06 for well-covered targets; generous bound


def test_target_action_probability_is_a_point_mass() -> None:
    """A deterministic target assigns probability 1 to its action and 0 to all others.

    Pins the propensity used as the importance-ratio numerator: for the heuristic router on the
    ambiguous-query start state, the action it would take scores 1.0 and every other action 0.0.
    """
    state = AgentDecisionEnvironment().reset(scenario_id=2)
    router = HeuristicRouterPolicy()
    chosen = router.select_action(state)
    assert target_action_probability(router, state, chosen) == 1.0
    for action in range(4):
        if action != chosen:
            assert target_action_probability(router, state, action) == 0.0


def test_ope_is_accurate_for_a_well_covered_target() -> None:
    """All four estimators recover the heuristic router's true value from the log.

    The target is the same router the behaviour policy is an epsilon-soft version of, so the log
    covers it well and every estimator -- IS, WIS, direct method, doubly-robust -- lands within a
    tight tolerance of the simulator's true value. This is OPE working as intended on in-support
    targets.
    """
    router = HeuristicRouterPolicy()
    truth = true_policy_value(router, episodes_per_scenario=_TRUTH_EPISODES)
    estimates = ope_estimates(_LOG, router, gamma=1.0)
    for name, estimate in estimates.items():
        assert abs(estimate - truth) < _COVERED_TOLERANCE, (name, estimate, truth)


def test_ope_evaluates_a_divergent_but_covered_target() -> None:
    """OPE estimates the DP-optimal policy's value from a router log it did not generate.

    The real OPE use case: estimate a *different, better* candidate (the planning optimum) from logs
    the behaviour policy produced. Because the optimum stays within the well-explored region, the
    direct method and doubly-robust estimators recover its true value within tolerance -- vetting a
    new policy before ever deploying it.
    """
    optimum = QTablePolicy(q_table=optimal_action_values(), name="dp_optimal")
    truth = true_policy_value(optimum, episodes_per_scenario=_TRUTH_EPISODES)
    estimates = ope_estimates(_LOG, optimum, gamma=1.0)
    # The lower-variance estimators are reliable here; assert them tightly.
    assert abs(estimates["direct_method"] - truth) < _COVERED_TOLERANCE
    assert abs(estimates["doubly_robust"] - truth) < _COVERED_TOLERANCE
    assert abs(estimates["weighted_importance_sampling"] - truth) < _COVERED_TOLERANCE


def test_ope_degrades_under_poor_overlap() -> None:
    """A target far from the behaviour policy is estimated poorly -- the coverage requirement.

    A deterministic always-escalate target has no support in an answer-only log. No estimator
    can recover its value from that log; self-normalising cannot repair absent coverage.
    """
    dataset = collect_logged_dataset(episodes=1, scenario_ids=(0,), epsilon=0.0, seed=0)
    assert [row.action for row in dataset.transitions] == [0]
    target = AlwaysEscalatePolicy()
    truth = true_policy_value(target, scenario_ids=(0,), episodes_per_scenario=1)
    # Easy escalation: 0.6 payoff - 1.5 cost = -0.9; the log has no escalation.
    assert truth == -0.9
    estimates = ope_estimates(dataset, target)
    assert set(estimates.values()) == {0.0}


@pytest.mark.parametrize("runner", [run_ope.main, run_showcase.main])
def test_ope_runners_use_reproducible_deterministic_targets(
    tmp_path: Path, runner: Callable[[list[str] | None], int]
) -> None:
    args = ["--quick", "--output-dir", str(tmp_path)]
    assert runner(args) == 0
    artifact = tmp_path / "ope" / "estimator_comparison.csv"
    before = artifact.read_bytes()
    with artifact.open(newline="") as handle:
        targets = {row["target"] for row in csv.DictReader(handle)}
    assert targets == {"heuristic_router", "dp_optimal", "always_escalate"}
    assert runner(args) == 0
    assert artifact.read_bytes() == before


def test_ope_report_compares_the_same_discounted_return() -> None:
    env = AgentDecisionEnvironment()
    start = env.reset(seed=0, scenario_id=0)
    assert (start.difficulty, start.ambiguity) == (0, 0)
    target = QTablePolicy(q_table={start.as_tuple(): [0.0, 1.0, 0.0, 0.0]})
    dataset = collect_logged_dataset(
        episodes=1, scenario_ids=(0,), base_policy=target, epsilon=0.0, seed=0
    )
    assert [row.reward for row in dataset.transitions] == [-0.7, 2.0]
    # Needless retrieval: -0.5 cost - 0.2 effort; then a grounded answer: +2.
    # G(gamma=0.5) = -0.7 + 0.5*2 = 0.3, rather than the undiscounted 1.3.
    rows = ope_report_rows(
        dataset, [("retrieve_then_answer", target)], gamma=0.5,
        scenario_ids=(0,), episodes_per_scenario=1,
    )
    assert len(rows) == 4
    for row in rows:
        assert row["true_value"] == pytest.approx(0.3)
        assert row["estimate"] == pytest.approx(0.3)
        assert row["abs_error"] == 0.0


def test_ope_report_rows_have_the_expected_schema() -> None:
    """The OPE report yields one row per (target, estimator) with estimate, truth, and error.

    Pins the artifact contract for the OPE table: every target contributes four estimator rows, each
    carrying the estimate, the true value, and their absolute error for at-a-glance accuracy.
    """
    targets: list[tuple[str, Policy]] = [
        ("heuristic_router", HeuristicRouterPolicy()),
        ("dp_optimal", QTablePolicy(q_table=optimal_action_values(), name="dp_optimal")),
    ]
    rows = ope_report_rows(_LOG, targets, episodes_per_scenario=_TRUTH_EPISODES)
    assert len(rows) == len(targets) * 4  # four estimators per target
    assert set(rows[0]) == {"target", "estimator", "estimate", "true_value", "abs_error"}
    estimators = {str(row["estimator"]) for row in rows}
    assert estimators == {
        "importance_sampling",
        "weighted_importance_sampling",
        "direct_method",
        "doubly_robust",
    }
