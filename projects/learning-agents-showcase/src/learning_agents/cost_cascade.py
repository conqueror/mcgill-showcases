"""A cost-aware orchestration cascade and the cost-vs-quality frontier it traces.

What + why: a real agent does not pay for its most expensive option on every request. It runs a
*cascade* -- try the cheap moves first (answer directly when already grounded; otherwise spend a
little on retrieval/clarification) and fall through to the expensive human escalation only when the
cheap tiers cannot resolve the request. How far up that cost ladder you are willing to climb before
committing is an operating choice, and different choices trade money/latency for answer quality.

This module makes that trade-off explicit and measurable:

* :class:`EffortCascadePolicy` is a cascade with one knob -- an ``effort_budget`` (how many cheap
  grounding steps it will spend before it must commit). ``effort_budget = 0`` commits immediately
  (cheapest: answer if ready, else escalate); a larger budget invests in retrieval/clarification so
  it can answer hard requests itself instead of paying for a human.
* :func:`cost_cascade_curve` sweeps that knob and records, at each setting, the realised cost and
  quality (average action cost, reward, solved rate, escalation rate) from the same evaluation
  harness used everywhere else -- so the cost/quality trade-off is data, not assertion.
* :func:`cost_efficient_frontier` keeps only the non-dominated operating points (no other setting is
  both cheaper and better), the Pareto frontier a practitioner actually chooses an operating point
  from, and :func:`recommended_operating_point` picks the highest-reward setting on it.

Where it sits: this is an *evaluation/deployment* concern layered on the learned/heuristic policies,
not a new learning algorithm -- the same cost and quality metrics the governance rung already
reports, swept across a cost knob to expose the frontier.

RL concept:
    Cost-aware control and the cost/quality (cost/reward) Pareto frontier -- spend cheap actions
    before an expensive escalation, and choose an operating point on the non-dominated frontier.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from learning_agents.environment import AgentState, RewardFunction, default_reward
from learning_agents.evaluation import evaluate_policies
from learning_agents.reward import evidence_is_adequate

__all__ = [
    "EffortCascadePolicy",
    "cost_cascade_curve",
    "cost_efficient_frontier",
    "recommended_operating_point",
]


@dataclass
class EffortCascadePolicy:
    """A cost-aware cascade: spend up to ``effort_budget`` cheap grounding steps, then commit.

    What + why: this is the tunable cascade. On each step it first decides whether it is still
    *allowed* and *able* to invest in cheap grounding (it has not exhausted ``effort_budget``, the
    request still needs work, and it can afford the action); if so it clarifies (when ambiguous) or
    retrieves (when under-grounded). Once the budget is spent -- or the request is already ready --
    it commits to the cheapest adequate terminal action: a direct answer when grounded and
    unambiguous, otherwise an escalation to a human (the expensive last tier). Sweeping
    ``effort_budget`` moves the policy along the cost/quality frontier: a small budget leans on
    cheap immediate commits (and pays for escalations it could have avoided), a larger budget pays
    for grounding so it can answer hard requests itself.

    Attributes:
        effort_budget: Maximum number of cheap grounding steps (clarify/retrieve) to spend before
            the policy must commit. ``0`` commits on the first step.
        clarify_cost_tenths: Budget (tenths) a ``clarify`` needs; used for the affordability gate.
        retrieve_cost_tenths: Budget (in tenths) a ``retrieve`` needs; used for the affordability
            gate.
        horizon: Episode length, so the cascade knows when a step would overrun the clock.
        name: Identifier shown in evaluation output (defaults to ``cascade``; the sweep relabels it
            per effort level).

    RL concept: a cost-aware cascade policy -- cheap tiers first, expensive escalation last, with a
    tunable effort budget.
    """

    effort_budget: int
    clarify_cost_tenths: int = 3
    retrieve_cost_tenths: int = 5
    horizon: int = 5
    name: str = "cascade"

    def reset(self) -> None:
        """Do nothing; the cascade is a stateless function of the observed state."""
        return None

    def _can_afford(self, state: AgentState, cost_tenths: int) -> bool:
        """Return whether a cheap grounding action fits the remaining budget and the step horizon.

        Args:
            state: The current state s.
            cost_tenths: The budget cost (in tenths) of the grounding action being considered.

        Returns:
            True iff the action's cost fits the remaining budget and ``step`` is below the horizon.
        """
        return state.budget - cost_tenths >= 0 and state.step < self.horizon

    def select_action(self, state: AgentState) -> int:
        """Choose the next cascade action: cheap grounding within budget, else commit.

        What + why: while the effort budget allows and the request still needs work, prefer the
        cheapest useful grounding move (clarify before retrieve); otherwise commit to the cheapest
        adequate terminal action (answer when ready, else escalate to a human).

        Args:
            state: The current :class:`~learning_agents.environment.AgentState`.

        Returns:
            The chosen action index (0 answer_direct, 1 retrieve, 2 clarify, 3 escalate).
        """
        grounded = evidence_is_adequate(evidence=state.evidence, difficulty=state.difficulty)
        needs_work = state.ambiguity > 0 or not grounded
        # Cheap tier: invest in grounding only while the budget allows and it is still affordable.
        if state.step < self.effort_budget and needs_work:
            if state.ambiguity > 0 and self._can_afford(state, self.clarify_cost_tenths):
                return 2  # clarify (cheapest disambiguation)
            if not grounded and self._can_afford(state, self.retrieve_cost_tenths):
                return 1  # retrieve (cheap grounding)
        # Commit tier: answer when ready; otherwise fall through to the expensive human escalation.
        if grounded and state.ambiguity == 0:
            return 0  # answer_direct (free and correct)
        if state.difficulty >= 2 or state.ambiguity > 0:
            return 3  # escalate (the expensive last tier)
        return 0  # adequately grounded and easy -> answer directly


def cost_cascade_curve(
    *,
    effort_levels: Sequence[int] = (0, 1, 2, 3, 4),
    scenario_ids: Sequence[int] = (0, 1, 2, 3, 4),
    episodes_per_scenario: int = 12,
    latency_cost_per_step: float = 0.3,
    horizon: int = 5,
    reward_fn: RewardFunction = default_reward,
) -> list[dict[str, int | float]]:
    """Sweep the cascade's effort budget and record realised cost and quality at each setting.

    What + why: traces the cost/quality trade-off by evaluating an :class:`EffortCascadePolicy` at
    each effort level on the fixed scenarios and reading the metrics straight from the shared
    evaluation harness. Operational cost has two parts here: ``avg_action_cost`` (the resource/money
    price of the actions, where a human escalation is by far the dearest) and *latency* (how many
    orchestration steps the request took). ``total_cost`` combines them as
    ``avg_action_cost + latency_cost_per_step * avg_steps``. Reporting both is what creates a real
    frontier: spending cheap grounding avoids dear escalations (money falls) but adds steps (latency
    rises), so the cheapest-fastest and the highest-quality settings are different operating points.

    Args:
        effort_levels: The effort-budget settings to sweep (one cascade policy per level).
        scenario_ids: Scenarios to evaluate each cascade on.
        episodes_per_scenario: Seeded rollouts per (policy, scenario) pair.
        latency_cost_per_step: Weight converting one extra orchestration step (latency) into the
            reward's cost units; tune it to how much latency matters in your deployment. The
            frontier shape depends on this weight.
        horizon: Episode length H.
        reward_fn: Reward function injected into the environment (defaults to the judge rubric).

    Returns:
        One row per effort level with columns ``effort_budget``, ``avg_action_cost``, ``avg_steps``,
        ``total_cost``, ``avg_reward``, ``avg_escalation_rate``, and ``solved_rate`` -- the
        cost-vs-quality sweep, ordered by effort.

    RL concept: a cost/quality operating curve -- realised (money + latency) cost and reward as a
    function of how much cheap effort the cascade spends before committing.
    """
    rows: list[dict[str, int | float]] = []
    for effort in effort_levels:
        policy = EffortCascadePolicy(
            effort_budget=effort,
            horizon=horizon,
            name=f"cascade_effort_{effort}",
        )
        summary, _ = evaluate_policies(
            policies=[policy],
            scenario_ids=scenario_ids,
            episodes_per_scenario=episodes_per_scenario,
            horizon=horizon,
            reward_fn=reward_fn,
        )
        metrics = summary[0]
        action_cost = float(metrics["avg_action_cost"])
        steps = float(metrics["avg_steps"])
        rows.append(
            {
                "effort_budget": effort,
                "avg_action_cost": action_cost,
                "avg_steps": steps,
                # Operational cost = money (action cost) + latency (weighted steps).
                "total_cost": round(action_cost + latency_cost_per_step * steps, 4),
                "avg_reward": float(metrics["avg_reward"]),
                "avg_escalation_rate": float(metrics["avg_escalation_rate"]),
                "solved_rate": float(metrics["solved_rate"]),
            }
        )
    return rows


def cost_efficient_frontier(
    curve_rows: Sequence[dict[str, int | float]],
) -> list[dict[str, int | float]]:
    """Keep only the cost-efficient (Pareto non-dominated) operating points of a cost-cascade curve.

    What + why: an operating point is *dominated* if another point is at least as cheap (on
    ``total_cost`` = money + latency) AND at least as rewarding while strictly better on one of the
    two -- nobody would ever choose a dominated point. Filtering them out leaves the cost/reward
    Pareto frontier, the real menu of sensible operating choices. This is the standard tool for
    reading a cost/quality sweep.

    Args:
        curve_rows: Rows from :func:`cost_cascade_curve` (each with ``total_cost`` and
            ``avg_reward``).

    Returns:
        The non-dominated subset, sorted by ``total_cost`` ascending (cheapest first). Lower total
        cost is better and higher reward is better.

    RL concept: the cost/quality Pareto frontier -- the non-dominated operating points of the
    cost-aware cascade.
    """
    frontier: list[dict[str, int | float]] = []
    for candidate in curve_rows:
        candidate_cost = float(candidate["total_cost"])
        candidate_reward = float(candidate["avg_reward"])
        dominated = any(
            float(other["total_cost"]) <= candidate_cost
            and float(other["avg_reward"]) >= candidate_reward
            and (
                float(other["total_cost"]) < candidate_cost
                or float(other["avg_reward"]) > candidate_reward
            )
            for other in curve_rows
            if other is not candidate
        )
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda row: float(row["total_cost"]))


def recommended_operating_point(
    curve_rows: Sequence[dict[str, int | float]],
) -> dict[str, int | float]:
    """Pick the highest-reward operating point on the cost-efficient frontier.

    What + why: among the non-dominated points, this returns the one with the greatest average
    reward (ties broken toward the cheaper point) -- a simple, defensible default operating choice
    when reward is the objective and cost is a soft constraint. A real deployment might instead cap
    cost and take the best reward under that cap; this picks the unconstrained reward-maximiser on
    the frontier.

    Args:
        curve_rows: Rows from :func:`cost_cascade_curve`.

    Returns:
        The chosen frontier row.

    Raises:
        ValueError: If ``curve_rows`` is empty.

    RL concept: selecting an operating point on the cost/quality frontier.
    """
    if not curve_rows:
        raise ValueError("curve_rows must be non-empty")
    frontier = cost_efficient_frontier(curve_rows)
    # Highest reward first; break ties toward the cheaper point so the choice is deterministic.
    return min(frontier, key=lambda row: (-float(row["avg_reward"]), float(row["total_cost"])))
