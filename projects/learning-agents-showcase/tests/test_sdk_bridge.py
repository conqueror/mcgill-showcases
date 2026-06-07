"""Pin the OpenAI Agents SDK bridge (locus of learning A): policy drives the agent loop.

These tests anchor the offline, SDK-independent heart of Lane A: the action -> SDK-construct
mapping, and the learned policy driving an agent loop whose every step is annotated with the SDK
role it would execute (tool call / handoff / final output). They also pin that the live-SDK adapter
is correctly *gated* on the optional dependency -- it raises a clear error when the SDK is absent
and constructs an agent when present -- and that the bridge report states the policy/framework split
and the live status. The live agent is never *run* here (network); only the wiring is checked.

RL concept:
    Locus of learning A -- RL learns the orchestration policy; the agent framework executes/logs it.
"""

from __future__ import annotations

from learning_agents.environment import ACTION_LABELS, scenario_catalog
from learning_agents.policies import HeuristicRouterPolicy
from learning_agents.sdk_bridge import (
    OptionalSDKError,
    action_tool_mapping,
    bridge_report_markdown,
    build_agents_sdk_agent,
    run_bridged_episode,
    sdk_available,
)

HARD_DEBUG = next(i for i, s in enumerate(scenario_catalog()) if s.name == "hard_debug")
_VALID_ROLES = {"final_output", "tool_call", "handoff"}


def test_action_tool_mapping_covers_every_action() -> None:
    """The crosswalk maps each of the MDP's actions to a valid SDK construct.

    Pins the locus-of-learning A interface: every action label the environment defines appears once,
    each with a recognised SDK role (final output, tool call, or handoff) and a named target.
    """
    rows = action_tool_mapping()
    assert {row["action_label"] for row in rows} == set(ACTION_LABELS.values())
    for row in rows:
        assert row["sdk_role"] in _VALID_ROLES
        assert row["sdk_target"]
    # escalate is the handoff; answer_direct is the run-completing final output.
    by_label = {row["action_label"]: row for row in rows}
    assert by_label["escalate"]["sdk_role"] == "handoff"
    assert by_label["answer_direct"]["sdk_role"] == "final_output"


def test_run_bridged_episode_annotates_each_step_with_its_sdk_role() -> None:
    """Rolling the policy out yields a per-step trace tagged with the SDK construct per step.

    Pins the offline demonstration: every step carries the action, its SDK role/target, the reward,
    and a terminal flag; roles are valid; and the episode ends on a terminal commit (final output or
    handoff) -- exactly what an Agents-SDK run would log, with the decisions made by the policy.
    """
    rows = run_bridged_episode(policy=HeuristicRouterPolicy(), scenario_id=HARD_DEBUG)
    assert rows
    expected_keys = {
        "step",
        "scenario_name",
        "action_label",
        "sdk_role",
        "sdk_target",
        "reward",
        "terminal",
    }
    for row in rows:
        assert expected_keys <= set(row)
        assert row["sdk_role"] in _VALID_ROLES
    assert int(rows[-1]["terminal"]) == 1  # the loop ends on a terminal commit
    assert rows[-1]["sdk_role"] in {"final_output", "handoff"}  # answer or escalate ends the run


def test_run_bridged_episode_is_seed_reproducible() -> None:
    """The same seed reproduces the same SDK-annotated trace (deterministic demonstration)."""
    first = run_bridged_episode(policy=HeuristicRouterPolicy(), scenario_id=HARD_DEBUG, seed=3)
    second = run_bridged_episode(policy=HeuristicRouterPolicy(), scenario_id=HARD_DEBUG, seed=3)
    assert first == second


def test_build_agents_sdk_agent_is_gated_on_the_optional_dependency() -> None:
    """The live adapter raises when the SDK is absent and builds an agent when it is present.

    Pins the gating contract: without the optional ``sdk`` extra, :func:`build_agents_sdk_agent`
    raises :class:`OptionalSDKError` (so the core path never hard-depends on the SDK); with it, the
    call returns a constructed agent. The agent is not run -- only its construction is exercised.
    """
    if sdk_available():
        agent = build_agents_sdk_agent()
        assert agent is not None
    else:
        try:
            build_agents_sdk_agent()
        except OptionalSDKError as error:
            assert "sdk" in str(error).lower()
        else:  # pragma: no cover - only reached if the SDK is unexpectedly importable
            raise AssertionError("expected OptionalSDKError when the SDK is absent")


def test_bridge_report_states_the_policy_framework_split() -> None:
    """The report is heading-led, states the locus-of-learning thesis, and reflects the SDK status.

    Pins the narrative artifact: it starts with a Markdown heading, frames RL as learning the
    orchestration policy while the SDK executes it, lists the action mapping, and reports whether
    the live SDK is installed (matching :func:`sdk_available`).
    """
    report = bridge_report_markdown(sdk_present=sdk_available())
    assert report.startswith("#")
    lowered = report.lower()
    assert "orchestration policy" in lowered
    assert "executor" in lowered or "executes" in lowered
    assert "answer_direct" in report and "handoff" in lowered
    expected_status = "installed" if sdk_available() else "not installed"
    assert expected_status in lowered
