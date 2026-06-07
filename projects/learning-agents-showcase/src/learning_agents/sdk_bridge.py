"""Bridge a learned orchestration policy to the OpenAI Agents SDK (locus of learning: Lane A).

What + why: the central reframing of this showcase is that an "agent" has several places learning
*could* live -- the **orchestration policy** (which tool/handoff to choose), the **LLM weights**, or
the **multi-agent coordination**. This module makes Lane A concrete: the thing RL learns here is the
*orchestration policy*, and a framework like the OpenAI Agents SDK is the **environment, executor,
and logger** that runs it -- NOT the trainer. The SDK decides nothing about *how* to route; it just
executes the routing decisions and records traces. RL (everything else in this package) learns those
decisions; the SDK carries them out.

The bridge has two layers, deliberately separated by what can be run offline:

* A pure-Python demonstration that needs no SDK and no network: :func:`action_tool_mapping` is the
  crosswalk from this MDP's four actions to the SDK constructs they correspond to (function tools, a
  handoff, a final output), and :func:`run_bridged_episode` rolls the learned policy out in this
  package's own environment while annotating each step with the SDK role it would drive. This is the
  testable heart of the lane: it shows the learned policy *driving an agent loop* without depending
  on the SDK being installed or reachable.
* A gated adapter to the real SDK: :func:`build_agents_sdk_agent` constructs an actual
  ``agents.Agent`` (with function tools and a human handoff) so the same mapping wires straight into
  the live framework. It is imported dynamically and raises :class:`OptionalSDKError` when the
  optional ``sdk`` extra (``openai-agents``) is not installed, so the core showcase never depends on
  it. Running that agent against a model needs network/credentials and is out of the offline path.

RL concept:
    Locus of learning A -- RL learns the agent's *orchestration policy*; an agent framework executes
    and logs it. The framework is the environment/runtime, not the learning algorithm.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Sequence
from typing import Any

from learning_agents.environment import (
    ACTION_LABELS,
    AgentDecisionEnvironment,
    RewardFunction,
    default_reward,
)
from learning_agents.policies import Policy

__all__ = [
    "SDK_CONSTRUCT_BY_ACTION",
    "OptionalSDKError",
    "action_tool_mapping",
    "build_agents_sdk_agent",
    "bridge_report_markdown",
    "run_bridged_episode",
    "sdk_available",
]


class OptionalSDKError(RuntimeError):
    """Raised when the optional OpenAI Agents SDK is required but not installed.

    What + why: the live-SDK adapter is gated behind the optional ``sdk`` extra so the core showcase
    stays dependency-light and offline. Callers that want the real ``agents.Agent`` catch this and
    fall back to the pure-Python demonstration (or print install instructions).
    """


# Crosswalk from this MDP's actions to the OpenAI Agents SDK construct each one drives.
# answer_direct is the agent's final output; retrieve/clarify are function-tool calls; escalate is
# a handoff to a human-specialist agent -- "what the orchestration policy controls in the SDK".
SDK_CONSTRUCT_BY_ACTION: dict[int, tuple[str, str, str]] = {
    0: ("final_output", "assistant_answer", "Emit the final answer to the user (run completes)."),
    1: ("tool_call", "retrieve_context", "Call the retrieval function tool to gather grounding."),
    2: ("tool_call", "ask_clarifying_question", "Call the clarification function tool."),
    3: ("handoff", "human_specialist", "Hand off to a human-specialist agent (escalation)."),
}


def sdk_available() -> bool:
    """Return whether the optional OpenAI Agents SDK (``agents``) can be imported.

    What + why: a cheap, import-free probe (it only checks the module spec) used to gate the
    live-SDK adapter and to record in the bridge report whether the live path is enabled here.

    Returns:
        True iff the ``agents`` package (``openai-agents``) is importable.

    RL concept: gating an optional dependency so the core (RL) path never requires it.
    """
    return importlib.util.find_spec("agents") is not None


def action_tool_mapping() -> list[dict[str, str]]:
    """Build the action -> SDK-construct crosswalk rows (the locus-of-learning A table).

    What + why: states, in data, exactly which SDK construct each orchestration action drives. This
    is what makes "RL learns the policy; the SDK executes it" concrete: the learned policy's output
    (an action) becomes a function-tool call, a handoff, or a final output in the framework.

    Returns:
        One row per action with ``action`` (index), ``action_label``, ``sdk_role``, ``sdk_target``,
        and ``description``.

    RL concept: the interface between the learned orchestration policy and the agent framework.
    """
    return [
        {
            "action": str(action),
            "action_label": ACTION_LABELS[action],
            "sdk_role": role,
            "sdk_target": target,
            "description": description,
        }
        for action, (role, target, description) in sorted(SDK_CONSTRUCT_BY_ACTION.items())
    ]


def run_bridged_episode(
    *,
    policy: Policy,
    scenario_id: int,
    seed: int | None = None,
    horizon: int = 5,
    reward_fn: RewardFunction = default_reward,
) -> list[dict[str, int | float | str]]:
    """Roll the learned policy out as an agent loop, annotating each step with its SDK role.

    What + why: this is the offline demonstration of Lane A. It runs the policy inside this
    package's own :class:`~learning_agents.environment.AgentDecisionEnvironment` (the stand-in
    runtime) and, for each decision, records the SDK construct that decision maps to -- a
    function-tool call, a handoff, or the final output. The resulting trace is exactly what an
    Agents-SDK run would log, except the *decisions* came from the learned policy rather than from
    free-running LLM tool-choice. No SDK and no network are needed, so this is the bridge's
    testable core.

    Args:
        policy: The orchestration policy to drive the loop (any
            :class:`~learning_agents.policies.Policy`, e.g. the learned Q-table policy or the
            heuristic router).
        scenario_id: Scenario index to start from.
        seed: Optional env-reset jitter seed.
        horizon: Episode length H.
        reward_fn: Reward function injected into the environment (defaults to the judge rubric).

    Returns:
        One row per step with ``step``, ``scenario_name``, ``action_label``, ``sdk_role``,
        ``sdk_target``, ``reward``, and ``terminal`` -- the SDK-annotated orchestration trace.

    RL concept: a learned orchestration policy driving an agent's tool/handoff/output loop -- the
    framework executes what RL chose.
    """
    policy.reset()
    environment = AgentDecisionEnvironment(horizon=horizon, reward_fn=reward_fn)
    state = environment.reset(seed=seed, scenario_id=scenario_id)
    rows: list[dict[str, int | float | str]] = []
    while not environment.is_done():
        action = policy.select_action(state)
        role, target, _description = SDK_CONSTRUCT_BY_ACTION[action]
        transition = environment.step(action)
        rows.append(
            {
                "step": state.step,
                "scenario_name": environment.scenario_name,
                "action_label": ACTION_LABELS[action],
                "sdk_role": role,
                "sdk_target": target,
                "reward": transition.reward,
                "terminal": int(transition.done),
            }
        )
        state = transition.state
    return rows


def build_agents_sdk_agent(
    *,
    instructions: str | None = None,
    tool_targets: Sequence[str] = ("retrieve_context", "ask_clarifying_question"),
) -> object:
    """Construct a live ``agents.Agent`` wired with this MDP's tools and a human handoff (gated).

    What + why: proves the action -> SDK-construct mapping wires into the *real* OpenAI Agents SDK,
    not just the offline simulation. It dynamically imports ``agents`` (so the core package never
    statically depends on it), defines function tools for the retrieve/clarify actions and a
    human-specialist agent for the escalate handoff, and returns the assembled orchestrator
    ``Agent``. Construction needs no network; actually *running* the agent against a model does, and
    is intentionally left to the caller. Raises :class:`OptionalSDKError` when the SDK is absent so
    callers degrade to :func:`run_bridged_episode`.

    Args:
        instructions: System instructions for the orchestrator agent; a sensible default describing
            the routing policy is used when ``None``.
        tool_targets: Which function-tool names to attach (defaults to retrieve and clarify).

    Returns:
        The constructed ``agents.Agent`` (typed ``object`` so the core package needs no SDK types).

    Raises:
        OptionalSDKError: If the optional ``agents`` (``openai-agents``) package is not importable.

    RL concept: the learned orchestration policy's action space realised as real SDK tools/handoffs.
    """
    if not sdk_available():
        raise OptionalSDKError(
            "The OpenAI Agents SDK is not installed. Install the optional extra with "
            "`uv sync --extra sdk` (adds openai-agents) to enable the live bridge; the offline "
            "demonstration via run_bridged_episode needs no SDK."
        )
    # Dynamic import keeps the dependency optional and avoids a static import mypy cannot resolve;
    # the module is typed Any so the SDK's (untyped, to us) helpers compose under mypy --strict.
    agents: Any = importlib.import_module("agents")

    def _retrieve_context(query: str) -> str:
        """Retrieve grounding evidence for the user's request."""
        return f"(stub) retrieved grounding for: {query}"

    def _ask_clarifying_question(question: str) -> str:
        """Ask the user a clarifying question to resolve ambiguity."""
        return f"(stub) asked the user: {question}"

    # function_tool reads each function's name and docstring to build the tool schema (call form,
    # not decorator form, so mypy --strict does not flag an untyped decorator).
    available_tools = {
        "retrieve_context": agents.function_tool(_retrieve_context),
        "ask_clarifying_question": agents.function_tool(_ask_clarifying_question),
    }
    tools = [available_tools[name] for name in tool_targets if name in available_tools]
    human_specialist = agents.Agent(
        name="human_specialist",
        instructions="You are a human specialist who handles escalated, genuinely hard requests.",
    )
    return agents.Agent(
        name="orchestrator",
        instructions=instructions
        or (
            "Route each request using the learned orchestration policy: answer directly when "
            "grounded and unambiguous, call the retrieval tool to gather evidence, call the "
            "clarification tool to resolve ambiguity, and hand off to the human specialist only "
            "when a human is genuinely warranted."
        ),
        tools=tools,
        handoffs=[human_specialist],
    )


def bridge_report_markdown(*, sdk_present: bool) -> str:
    """Render the Lane A narrative: the policy/framework split and the live-SDK status.

    What + why: produces the bridge report artifact. It states the locus-of-learning A thesis (RL
    learns the orchestration policy; the SDK executes and logs it, it is not the trainer), lists the
    action -> SDK-construct mapping, and records whether the optional SDK is installed in this
    environment with instructions to enable the live path.

    Args:
        sdk_present: Whether :func:`sdk_available` is True in this environment.

    Returns:
        A heading-led Markdown report.

    RL concept: locus of learning A -- orchestration-policy learning, framework as executor.
    """
    status = (
        "installed -- the live `agents.Agent` bridge is available."
        if sdk_present
        else "not installed -- the offline demonstration runs; `uv sync --extra sdk` enables the "
        "live `agents.Agent` bridge."
    )
    mapping_lines = "\n".join(
        f"- `{row['action_label']}` -> {row['sdk_role']} (`{row['sdk_target']}`): "
        f"{row['description']}"
        for row in action_tool_mapping()
    )
    return (
        "# OpenAI Agents SDK Bridge (Locus of Learning A)\n\n"
        "## Thesis\n\n"
        "Reinforcement learning here learns the agent's **orchestration policy** -- which tool to "
        "call, when to clarify, and when to hand off to a human. The OpenAI Agents SDK is the "
        "**environment, executor, and logger** that runs those decisions; it is not the trainer. "
        "The learned policy decides; the framework carries it out and records the trace.\n\n"
        "## Action to SDK construct\n\n"
        f"{mapping_lines}\n\n"
        "## Live SDK status\n\n"
        f"The optional OpenAI Agents SDK is {status}\n\n"
        "## How to run\n\n"
        "- Offline (no SDK, no network): `run_bridged_episode` rolls the learned policy out in the "
        "local environment and logs the SDK role of every decision.\n"
        "- Live (optional): `build_agents_sdk_agent` constructs a real `agents.Agent` with the "
        "same tools and handoff; running it against a model requires network and credentials.\n"
    )
