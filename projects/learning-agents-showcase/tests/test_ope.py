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
    HeuristicRouterPolicy,
    Policy,
    QTablePolicy,
    RandomPolicy,
)

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

    Evaluating a uniform-random target from a heuristic-router log violates overlap: the log rarely
    follows random trajectories, so importance sampling is badly off. This pins the central OPE
    caveat (no coverage -> no trust) and that weighted IS is more robust than ordinary IS under that
    poor overlap.
    """
    random_target = RandomPolicy(seed=1)
    truth = true_policy_value(random_target, episodes_per_scenario=_TRUTH_EPISODES)
    estimates = ope_estimates(_LOG, random_target, gamma=1.0)

    is_error = abs(estimates["importance_sampling"] - truth)
    wis_error = abs(estimates["weighted_importance_sampling"] - truth)
    # Ordinary IS is unreliable here -- a much larger error than on a covered target.
    assert is_error > 0.2
    # Self-normalisation makes weighted IS more robust under poor overlap.
    assert wis_error < is_error


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
