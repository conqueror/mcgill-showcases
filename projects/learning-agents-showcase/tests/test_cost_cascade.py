"""Pin the cost-aware cascade and its cost/quality Pareto frontier.

These tests anchor the cost-cascade rung: a tunable cascade that spends cheap grounding before
falling through to an expensive human escalation, and the cost-vs-quality sweep it traces. They pin
that a zero-budget cascade commits immediately (escalating hard requests it cannot afford to
ground), that more effort buys quality (reward up, escalation down), that the operating curve
carries both money and latency cost, and that the cost-efficient frontier correctly drops dominated
operating points and selects the best one.

RL concept:
    Cost-aware control and the cost/quality (money + latency vs reward) Pareto frontier.
"""

from __future__ import annotations

from learning_agents.cost_cascade import (
    EffortCascadePolicy,
    cost_cascade_curve,
    cost_efficient_frontier,
    recommended_operating_point,
)
from learning_agents.environment import AgentDecisionEnvironment, scenario_catalog
from learning_agents.evaluation import evaluate_policies

ALL_SCENARIOS = tuple(range(len(scenario_catalog())))
HARD_DEBUG = next(i for i, s in enumerate(scenario_catalog()) if s.name == "hard_debug")
EASY_FACTUAL = next(i for i, s in enumerate(scenario_catalog()) if s.name == "easy_factual")


def test_zero_budget_cascade_commits_immediately() -> None:
    """With no effort budget the cascade commits on the first step: answer if ready, else escalate.

    Pins the cheapest tier: on an already-grounded, unambiguous easy request it answers directly; on
    a hard, under-grounded request it cannot afford to ground (budget 0), so it falls through to the
    expensive escalation immediately.
    """
    cascade = EffortCascadePolicy(effort_budget=0)
    easy = AgentDecisionEnvironment().reset(scenario_id=EASY_FACTUAL)
    hard = AgentDecisionEnvironment().reset(scenario_id=HARD_DEBUG)
    assert cascade.select_action(easy) == 0  # answer_direct (ready)
    assert cascade.select_action(hard) == 3  # escalate (cannot ground within a zero budget)


def test_more_effort_grounds_instead_of_escalating() -> None:
    """A larger effort budget grounds the hard request and answers it instead of escalating.

    Pins the cascade's core behaviour: investing cheap grounding lets it handle a hard request
    itself -- so on the hard scenario the high-budget cascade never escalates and solves the
    request, unlike the zero-budget cascade that escalates every time.
    """
    zero_summary, _ = evaluate_policies(
        policies=[EffortCascadePolicy(effort_budget=0)],
        scenario_ids=(HARD_DEBUG,),
        episodes_per_scenario=8,
    )
    high_summary, _ = evaluate_policies(
        policies=[EffortCascadePolicy(effort_budget=4)],
        scenario_ids=(HARD_DEBUG,),
        episodes_per_scenario=8,
    )
    assert float(zero_summary[0]["avg_escalation_rate"]) == 1.0  # always escalates the hard request
    assert float(high_summary[0]["avg_escalation_rate"]) == 0.0  # grounds and answers instead
    assert float(high_summary[0]["solved_rate"]) >= float(zero_summary[0]["solved_rate"])


def test_cost_cascade_curve_schema_and_quality_trend() -> None:
    """The sweep carries money + latency cost columns; more effort raises reward, lowers escalation.

    Pins the artifact contract (the expected columns) and the qualitative trend the cascade is meant
    to show: as the effort budget grows the average reward is non-decreasing and the escalation rate
    is non-increasing -- spending cheap grounding steadily replaces expensive escalations.
    """
    rows = cost_cascade_curve(effort_levels=(0, 1, 2, 3, 4), episodes_per_scenario=12)
    assert {row["effort_budget"] for row in rows} == {0, 1, 2, 3, 4}
    assert set(rows[0]) == {
        "effort_budget",
        "avg_action_cost",
        "avg_steps",
        "total_cost",
        "avg_reward",
        "avg_escalation_rate",
        "solved_rate",
    }
    rewards = [float(r["avg_reward"]) for r in rows]
    escalations = [float(r["avg_escalation_rate"]) for r in rows]
    assert rewards == sorted(rewards)  # non-decreasing reward with more effort
    assert escalations == sorted(escalations, reverse=True)  # non-increasing escalation


def test_frontier_drops_dominated_operating_points() -> None:
    """The cost-efficient frontier is a strict, non-dominated subset of the swept curve.

    Pins the Pareto logic: at least one operating point is dominated (another is both no costlier on
    ``total_cost`` and no worse on reward), so the frontier is smaller than the full curve, and no
    frontier point dominates another.
    """
    rows = cost_cascade_curve(effort_levels=(0, 1, 2, 3, 4), episodes_per_scenario=12)
    frontier = cost_efficient_frontier(rows)
    assert 0 < len(frontier) < len(rows)  # some point is dominated and dropped
    # sorted by total_cost ascending; along the frontier reward must strictly increase with cost
    costs = [float(r["total_cost"]) for r in frontier]
    rewards = [float(r["avg_reward"]) for r in frontier]
    assert costs == sorted(costs)
    assert rewards == sorted(rewards)  # cheaper frontier points have strictly lower reward


def test_recommended_operating_point_maximises_reward_on_frontier() -> None:
    """The recommended point is the highest-reward operating choice on the frontier.

    Pins the selection rule: among non-dominated points the recommendation has the greatest average
    reward, and it lies on the frontier (it is not a dominated setting).
    """
    rows = cost_cascade_curve(effort_levels=(0, 1, 2, 3, 4), episodes_per_scenario=12)
    recommended = recommended_operating_point(rows)
    frontier = cost_efficient_frontier(rows)
    assert recommended in frontier
    assert float(recommended["avg_reward"]) == max(float(r["avg_reward"]) for r in rows)


def test_cost_cascade_curve_is_deterministic() -> None:
    """Two sweeps with the same settings produce identical curves (reproducible evaluation)."""
    first = cost_cascade_curve(effort_levels=(0, 2, 4), episodes_per_scenario=6)
    second = cost_cascade_curve(effort_levels=(0, 2, 4), episodes_per_scenario=6)
    assert first == second
