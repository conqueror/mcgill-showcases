"""Pin that the reward-hacking measurement reproduces the proxy-vs-aligned rank reversal.

These tests guard the reward-design rung: comparing the same fixed policies under the aligned judge
rubric and the hackable proxy must reproduce the signature of reward hacking -- the degenerate
always-escalate policy wins under the proxy (which overpays escalation) yet loses under the aligned
reward, while its solved rate stays low throughout. The report and spec builders are pinned to be
heading-led Markdown so the artifact-contract validator accepts them.

RL concept:
    Reward design and reward hacking -- a measurable proxy/true-objective rank reversal.
"""

from __future__ import annotations

from learning_agents.environment import scenario_catalog
from learning_agents.policies import AlwaysEscalatePolicy, HeuristicRouterPolicy
from learning_agents.reward_study import (
    compare_reward_models,
    reward_hacking_report,
    reward_model_specs,
)

ALL_SCENARIOS = tuple(range(len(scenario_catalog())))


def _reward_by_key(rows: list[dict[str, int | float | str]]) -> dict[tuple[str, str], float]:
    """Index (reward_model, policy) -> avg_reward for convenient assertions."""
    return {
        (str(row["reward_model"]), str(row["policy"])): float(row["avg_reward"]) for row in rows
    }


def test_compare_reward_models_tags_both_objectives() -> None:
    """The comparison emits one tagged row per (reward model, policy) pair.

    Pins that swapping only the reward function yields a 2 x P table -- both the ``"bad"`` proxy and
    the ``"good"`` judge rubric scored over the same policies -- so the downstream report can
    address every cell of the comparison.
    """
    rows = compare_reward_models(
        policies=[AlwaysEscalatePolicy(), HeuristicRouterPolicy()],
        scenario_ids=ALL_SCENARIOS,
    )
    assert len(rows) == 4  # 2 reward models x 2 policies
    assert {str(row["reward_model"]) for row in rows} == {"bad", "good"}
    assert {str(row["policy"]) for row in rows} == {"always_escalate", "heuristic_router"}


def test_reward_models_reverse_the_ranking() -> None:
    """The hackable proxy reverses the aligned ranking -- the reward-hacking signature.

    Under the aligned judge rubric the heuristic router beats always-escalate; under the hackable
    proxy (which overpays escalation) the ranking flips. This is the controlled experiment that
    isolates the reward as the cause of the rank reversal.
    """
    rows = compare_reward_models(
        policies=[AlwaysEscalatePolicy(), HeuristicRouterPolicy()],
        scenario_ids=ALL_SCENARIOS,
    )
    reward = _reward_by_key(rows)
    # Aligned objective: the genuinely good router wins.
    assert reward[("good", "heuristic_router")] > reward[("good", "always_escalate")]
    # Misspecified proxy: the degenerate always-escalate policy wins -> reward hacking.
    assert reward[("bad", "always_escalate")] > reward[("bad", "heuristic_router")]


def test_reward_hacking_report_is_heading_led_markdown() -> None:
    """The report names both policies and starts with a Markdown heading (contract-valid)."""
    rows = compare_reward_models(
        policies=[AlwaysEscalatePolicy(), HeuristicRouterPolicy()],
        scenario_ids=ALL_SCENARIOS,
    )
    report = reward_hacking_report(rows)
    assert report.startswith("#")
    assert "always-escalate" in report.lower()
    assert "heuristic-router" in report.lower()


def test_reward_model_specs_are_heading_led() -> None:
    """Both reward specs are non-empty, heading-led Markdown the validator accepts."""
    specs = reward_model_specs()
    assert set(specs) == {"good", "bad"}
    for body in specs.values():
        assert body.startswith("#")
        assert body.strip()
