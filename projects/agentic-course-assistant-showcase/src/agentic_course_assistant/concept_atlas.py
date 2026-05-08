"""Concept taxonomy for the agent framework course assistant showcase."""
# ruff: noqa: E501

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

OPENAI_AGENTS_GUIDE = "https://developers.openai.com/api/docs/guides/agents"
OPENAI_TOOLS_GUIDE = "https://developers.openai.com/api/docs/guides/tools#usage-in-the-agents-sdk"
OPENAI_RUNNING_GUIDE = "https://developers.openai.com/api/docs/guides/agents/running-agents"
OPENAI_ORCHESTRATION_GUIDE = (
    "https://developers.openai.com/api/docs/guides/agents/orchestration"
)
OPENAI_GUARDRAILS_GUIDE = (
    "https://developers.openai.com/api/docs/guides/agents/guardrails-approvals"
)
OPENAI_OBSERVABILITY_GUIDE = (
    "https://developers.openai.com/api/docs/guides/agents/integrations-observability"
)
OPENAI_EVALS_GUIDE = "https://developers.openai.com/api/docs/guides/agent-evals"

GOOGLE_ADK_OVERVIEW = "https://adk.dev/get-started/about/"
GOOGLE_ADK_FUNCTION_TOOLS = "https://adk.dev/tools-custom/function-tools/"
GOOGLE_ADK_CONFIRMATION = "https://adk.dev/tools-custom/confirmation/"
GOOGLE_ADK_MCP = "https://adk.dev/tools-custom/mcp-tools/"
GOOGLE_ADK_MULTI_AGENT = "https://adk.dev/agents/multi-agents/"
GOOGLE_ADK_SESSIONS = "https://adk.dev/sessions/"
GOOGLE_ADK_MEMORY = "https://adk.dev/sessions/memory/"
GOOGLE_ADK_CALLBACKS = "https://adk.dev/callbacks/"
GOOGLE_ADK_TRACES = "https://adk.dev/observability/traces/"
GOOGLE_ADK_EVALS = "https://adk.dev/evaluate/"
GOOGLE_ADK_A2A = "https://adk.dev/a2a/"
GOOGLE_ADK_ARTIFACTS = "https://adk.dev/artifacts/"
GOOGLE_ADK_PLUGINS = "https://adk.dev/plugins/"
GOOGLE_ADK_SKILLS = "https://adk.dev/skills/"

EXPECTED_CONCEPT_COLUMNS = [
    "concept_id",
    "name",
    "category",
    "refined_question",
    "definition",
    "openai_agents_sdk",
    "google_adk",
    "student_build",
    "showcase_artifact",
    "evaluation_prompt",
    "risk_to_watch",
    "source_refs",
]

REQUESTED_CONCEPT_IDS = {
    "tool-calls-function-tools",
    "guardrails-policy-checks",
    "tracing-observability",
    "agentic-workflows",
    "evals",
    "agent-as-judge",
    "multi-agent-orchestration",
    "handoff-triage",
    "a2a-agent-protocol",
    "sessions",
    "memory",
    "skills",
    "harness",
}


@dataclass(frozen=True)
class AgenticConcept:
    """One comparable concept across OpenAI Agents SDK, Google ADK, and this project."""

    concept_id: str
    name: str
    category: str
    refined_question: str
    definition: str
    openai_agents_sdk: str
    google_adk: str
    student_build: str
    showcase_artifact: str
    evaluation_prompt: str
    risk_to_watch: str
    source_refs: tuple[str, ...]

    def to_csv_row(self) -> dict[str, str]:
        row = asdict(self)
        row["source_refs"] = "; ".join(self.source_refs)
        return {column: str(row[column]) for column in EXPECTED_CONCEPT_COLUMNS}


REFINED_QUESTIONS = (
    "What is an agent, and what contract does it own in a student project?",
    "When should logic be a tool, a specialist agent, a handoff, or a workflow step?",
    "How do OpenAI Agents SDK and Google ADK each model function tools and tool schemas?",
    "Where do guardrails live: input, output, tool call, callback, or human approval?",
    "What trace evidence proves the agent chose the right route, tool, and stopping point?",
    "Which behaviors should be evaluated by unit tests, trace graders, datasets, or a judge?",
    "When is an LLM-as-judge appropriate, and how do we keep the judge from becoming vague?",
    "How should multi-agent orchestration differ from a single agent with several tools?",
    "When should a triage agent hand off ownership instead of calling a specialist as a tool?",
    "What does A2A add beyond local sub-agents, MCP tools, or direct function calls?",
    "What belongs in a session, what belongs in durable memory, and what should never persist?",
    "How do skills package instructions, tools, and resources without bloating context?",
    "What public-safe harness evidence should gate an educational agent showcase?",
    "How should artifacts, manifests, and verification scripts make behavior reproducible?",
    "How do callbacks, plugins, and hooks support policy, logging, and metrics?",
    "What deployment, auth, cost, latency, and rollback questions appear after the demo works?",
)


AGENTIC_CONCEPTS: tuple[AgenticConcept, ...] = (
    AgenticConcept(
        concept_id="agent-contract",
        name="Agent Contract",
        category="foundation",
        refined_question="What job does one agent own, and how do we know it is done?",
        definition=(
            "A named worker with instructions, capabilities, model choices, and an output contract."
        ),
        openai_agents_sdk=(
            "Use an Agent definition for one specialist and Runner/run to execute its loop."
        ),
        google_adk="Use an LlmAgent or workflow/custom agent as the fundamental worker unit.",
        student_build="Define Concept coach, Practice designer, Debug mentor, and Project planner.",
        showcase_artifact="artifacts/agent_trace.json",
        evaluation_prompt="Did the selected specialist match the student's intent?",
        risk_to_watch="Vague agent roles produce overlapping routes and noisy traces.",
        source_refs=(OPENAI_AGENTS_GUIDE, GOOGLE_ADK_OVERVIEW),
    ),
    AgenticConcept(
        concept_id="model-provider-selection",
        name="Model And Provider Selection",
        category="foundation",
        refined_question="Which model should power each agent, and what can stay deterministic?",
        definition=(
            "The runtime choice of hosted model, provider, local deterministic logic, or hybrid path."
        ),
        openai_agents_sdk=(
            "Agents SDK supports code-first model configuration around SDK-managed runs."
        ),
        google_adk="ADK is optimized for Gemini but exposes model interfaces for other backends.",
        student_build="Keep the default offline path deterministic and make hosted SDKs optional.",
        showcase_artifact="projects/agentic-course-assistant-showcase/README.md",
        evaluation_prompt="Can the project pass all default gates with no API credentials?",
        risk_to_watch="Credential-gated demos fail in classrooms and CI.",
        source_refs=(OPENAI_AGENTS_GUIDE, GOOGLE_ADK_OVERVIEW),
    ),
    AgenticConcept(
        concept_id="runner-event-loop",
        name="Runner And Event Loop",
        category="runtime",
        refined_question="What loop executes model calls, tools, handoffs, and final answers?",
        definition=(
            "The engine that advances a turn until it reaches a final output, pause, or failure."
        ),
        openai_agents_sdk=(
            "Runner/run loops through model output, tool calls, handoffs, and final output."
        ),
        google_adk=(
            "Runner coordinates agent execution with sessions, events, tools, and services."
        ),
        student_build="Represent the loop with explicit trace steps in the offline assistant.",
        showcase_artifact="artifacts/agent_trace.json",
        evaluation_prompt="Does the trace show route, tool lookup, specialist draft, and guardrail?",
        risk_to_watch="Students may mistake one prompt response for the whole runtime loop.",
        source_refs=(OPENAI_RUNNING_GUIDE, GOOGLE_ADK_OVERVIEW),
    ),
    AgenticConcept(
        concept_id="tool-calls-function-tools",
        name="Tool Calls And Function Tools",
        category="capability",
        refined_question="When should an agent call code instead of answering from memory?",
        definition="Typed functions or hosted capabilities that let an agent act or fetch context.",
        openai_agents_sdk=(
            "Use function_tool or SDK tool definitions; attach tools to agents or specialists."
        ),
        google_adk=(
            "Assign Python functions to tools so ADK wraps them as FunctionTool schemas."
        ),
        student_build="Use search_resources as the deterministic course catalog tool.",
        showcase_artifact="artifacts/resource_matches.csv",
        evaluation_prompt="Did the tool receive a useful query and return the expected resources?",
        risk_to_watch="Poor function signatures make tool arguments ambiguous.",
        source_refs=(OPENAI_TOOLS_GUIDE, GOOGLE_ADK_FUNCTION_TOOLS),
    ),
    AgenticConcept(
        concept_id="hosted-tools-connectors-mcp",
        name="Hosted Tools, Connectors, And MCP",
        category="capability",
        refined_question="Which external tools belong inside the agent loop?",
        definition=(
            "External capabilities exposed through hosted tools, connectors, or MCP servers."
        ),
        openai_agents_sdk=(
            "Attach hosted tools, function tools, hosted MCP, or SDK-managed MCP servers."
        ),
        google_adk="Use MCP toolsets or expose ADK tools through MCP when interoperability matters.",
        student_build="Keep MCP out of the first run, but document it as the next integration layer.",
        showcase_artifact="artifacts/concepts/openai_vs_adk_concepts.json",
        evaluation_prompt="Is the tool boundary documented before adding remote capabilities?",
        risk_to_watch="Remote tools expand trust, auth, latency, and approval surfaces.",
        source_refs=(OPENAI_OBSERVABILITY_GUIDE, GOOGLE_ADK_MCP),
    ),
    AgenticConcept(
        concept_id="structured-output-schema",
        name="Structured Output And Schemas",
        category="quality",
        refined_question="What shape must the agent output so downstream code can trust it?",
        definition="A typed response or artifact contract that separates valid output from prose.",
        openai_agents_sdk="Use structured output types and result surfaces when code consumes output.",
        google_adk="Use tool schemas, state fields, artifacts, and eval records as stable contracts.",
        student_build="Validate JSON and CSV artifacts with scripts/verify_artifacts.py.",
        showcase_artifact="artifacts/manifest.json",
        evaluation_prompt="Does verification fail clearly when a required key or column is missing?",
        risk_to_watch="Unstructured prose makes evals and regression tests brittle.",
        source_refs=(OPENAI_RUNNING_GUIDE, GOOGLE_ADK_ARTIFACTS),
    ),
    AgenticConcept(
        concept_id="guardrails-policy-checks",
        name="Guardrails And Policy Checks",
        category="safety",
        refined_question="What should block, warn, redact, or pause before the agent continues?",
        definition="Automatic checks around input, output, tools, or workflow state.",
        openai_agents_sdk=(
            "Use input, output, and tool guardrails; add approvals for sensitive side effects."
        ),
        google_adk=(
            "Use callbacks, plugins, safety controls, and tool confirmations as intervention points."
        ),
        student_build="Add scope and secret-handling guardrail notes to every answer.",
        showcase_artifact="artifacts/agent_trace.json",
        evaluation_prompt="Does a secret-related question trigger a non-leaking guardrail note?",
        risk_to_watch="Guardrails placed only at the agent boundary miss tool-specific risks.",
        source_refs=(OPENAI_GUARDRAILS_GUIDE, GOOGLE_ADK_CALLBACKS, GOOGLE_ADK_PLUGINS),
    ),
    AgenticConcept(
        concept_id="human-approval-interruptions",
        name="Human Approval And Interruptions",
        category="safety",
        refined_question="When should a run pause for review instead of executing a side effect?",
        definition="A deliberate pause that waits for a person or policy system to approve action.",
        openai_agents_sdk=(
            "Tools can require approval and return interruptions with resumable state."
        ),
        google_adk="Tool confirmation can pause execution for yes/no or structured approval.",
        student_build="Model approvals as a future extension before writeback or external actions.",
        showcase_artifact="artifacts/concepts/student_learning_path.md",
        evaluation_prompt="Does the design require approval before mutation, cancellation, or writes?",
        risk_to_watch="Starting a fresh turn after approval can duplicate work or lose context.",
        source_refs=(OPENAI_GUARDRAILS_GUIDE, GOOGLE_ADK_CONFIRMATION),
    ),
    AgenticConcept(
        concept_id="tracing-observability",
        name="Tracing And Observability",
        category="observability",
        refined_question="What evidence shows why the agent behaved the way it did?",
        definition="A structured record of model calls, tool calls, handoffs, guardrails, and spans.",
        openai_agents_sdk="Tracing is built into normal SDK paths and feeds trace grading.",
        google_adk="ADK traces connect LLM reasoning, tool calls, external APIs, and latency.",
        student_build="Write agent_trace.json and ask students to inspect each step.",
        showcase_artifact="artifacts/agent_trace.json",
        evaluation_prompt="Can a reviewer reconstruct the route without re-running the model?",
        risk_to_watch="No trace means failures become anecdotal and hard to debug.",
        source_refs=(OPENAI_OBSERVABILITY_GUIDE, GOOGLE_ADK_TRACES),
    ),
    AgenticConcept(
        concept_id="agentic-workflows",
        name="Agentic Workflows",
        category="orchestration",
        refined_question="What workflow steps happen before and after any model response?",
        definition="The planned sequence of routing, reasoning, tool use, checks, and artifacts.",
        openai_agents_sdk="Compose workflows with agents, tools, handoffs, state, and approvals.",
        google_adk=(
            "Use workflow agents such as SequentialAgent, ParallelAgent, and LoopAgent."
        ),
        student_build="Implement a five-step offline workflow before enabling live SDK calls.",
        showcase_artifact="artifacts/course_assistant_response.md",
        evaluation_prompt="Does the workflow have one observable artifact after each important step?",
        risk_to_watch="Agentic workflow diagrams drift if tests only check the final prose.",
        source_refs=(OPENAI_RUNNING_GUIDE, GOOGLE_ADK_MULTI_AGENT),
    ),
    AgenticConcept(
        concept_id="multi-agent-orchestration",
        name="Multi-Agent Orchestration",
        category="orchestration",
        refined_question="When do multiple specialists improve clarity rather than add noise?",
        definition="Coordination among specialized agents with clear ownership boundaries.",
        openai_agents_sdk="Use handoffs when ownership transfers or agents-as-tools when it does not.",
        google_adk="Build hierarchical teams, explicit workflow agents, or LLM-driven routing.",
        student_build="Route to four specialists while keeping a single deterministic outer loop.",
        showcase_artifact="artifacts/agent_trace.json",
        evaluation_prompt="Are specialists distinct by instructions, tools, policy, or output contract?",
        risk_to_watch="Splitting too early creates extra prompts, traces, and failure modes.",
        source_refs=(OPENAI_ORCHESTRATION_GUIDE, GOOGLE_ADK_MULTI_AGENT),
    ),
    AgenticConcept(
        concept_id="handoff-triage",
        name="Handoff And Triage",
        category="orchestration",
        refined_question="Who owns the next answer after classification?",
        definition="A triage decision that routes work to the right specialist or owner.",
        openai_agents_sdk="Handoffs transfer control to a specialist agent for that branch.",
        google_adk="Agent routing can transfer among sub-agents or route through workflow control.",
        student_build="Classify questions as concept, exercise, debug, or project.",
        showcase_artifact="artifacts/agent_trace.json",
        evaluation_prompt="Does the trace record the selected intent and specialist?",
        risk_to_watch="A triage step without ownership rules becomes a decorative classifier.",
        source_refs=(OPENAI_ORCHESTRATION_GUIDE, GOOGLE_ADK_MULTI_AGENT),
    ),
    AgenticConcept(
        concept_id="agent-as-tool",
        name="Agent As A Tool",
        category="orchestration",
        refined_question="When should the manager stay in control while calling a specialist?",
        definition="A specialist packaged as a bounded capability rather than the reply owner.",
        openai_agents_sdk="Expose a specialist through as_tool so the manager synthesizes the reply.",
        google_adk="Use AgentTool when an agent should be invoked as a tool inside another agent.",
        student_build="Compare future summarizer-as-tool against the current handoff-style route.",
        showcase_artifact="artifacts/concepts/openai_vs_adk_concepts.json",
        evaluation_prompt="Can the student explain who owns the final answer?",
        risk_to_watch="Confusing tool invocation with ownership transfer muddies debugging.",
        source_refs=(OPENAI_ORCHESTRATION_GUIDE, GOOGLE_ADK_FUNCTION_TOOLS),
    ),
    AgenticConcept(
        concept_id="a2a-agent-protocol",
        name="Agent2Agent Protocol",
        category="interoperability",
        refined_question="How can one agent safely collaborate with a remote agent?",
        definition="An interoperability pattern for agents to expose and consume remote capabilities.",
        openai_agents_sdk="Model the boundary with tools, MCP, or protocol adapters in your runtime.",
        google_adk="ADK includes A2A guides for exposing and consuming remote agents.",
        student_build="Keep A2A as an advanced capstone after local handoffs are understood.",
        showcase_artifact="artifacts/concepts/student_learning_path.md",
        evaluation_prompt="Does the design distinguish local sub-agents, MCP tools, and remote agents?",
        risk_to_watch="Remote agents need authentication, trust boundaries, and failure handling.",
        source_refs=(OPENAI_OBSERVABILITY_GUIDE, GOOGLE_ADK_A2A),
    ),
    AgenticConcept(
        concept_id="sessions",
        name="Sessions",
        category="state",
        refined_question="What state belongs to the current conversation thread?",
        definition="Short-term continuity for one conversation, including events and temporary state.",
        openai_agents_sdk=(
            "Use sessions for durable chat state, resumable approvals, and app-controlled storage."
        ),
        google_adk="Session and State track one conversation thread, events, and working data.",
        student_build="Treat one CLI run as one session and avoid hidden persistent state.",
        showcase_artifact="artifacts/agent_trace.json",
        evaluation_prompt="Can students rerun the showcase and get deterministic session artifacts?",
        risk_to_watch="Mixing replay, session, and server-managed state can duplicate context.",
        source_refs=(OPENAI_RUNNING_GUIDE, GOOGLE_ADK_SESSIONS),
    ),
    AgenticConcept(
        concept_id="memory",
        name="Memory",
        category="state",
        refined_question="What should the assistant remember across sessions?",
        definition="Durable, searchable knowledge outside a single conversation's temporary state.",
        openai_agents_sdk="Use app-owned storage or session strategies deliberately for continuity.",
        google_adk="MemoryService ingests sessions/events/items and supports search_memory.",
        student_build="Do not persist student secrets; document memory as a future opt-in extension.",
        showcase_artifact="artifacts/concepts/student_learning_path.md",
        evaluation_prompt="Does the memory plan specify retention, deletion, and search behavior?",
        risk_to_watch="Long-term memory can silently retain sensitive or stale information.",
        source_refs=(OPENAI_RUNNING_GUIDE, GOOGLE_ADK_MEMORY),
    ),
    AgenticConcept(
        concept_id="skills",
        name="Skills",
        category="capability-packaging",
        refined_question="How do we package reusable abilities without loading everything at once?",
        definition="A self-contained unit of instructions, resources, and tools for a task.",
        openai_agents_sdk="Represent skills as reusable agents, tools, prompts, or MCP-backed surfaces.",
        google_adk="ADK Skills can be loaded through SkillToolset from code or the filesystem.",
        student_build="Use this repo's skills as implementation guidance and teach skill packaging.",
        showcase_artifact="artifacts/concepts/refined_questions.md",
        evaluation_prompt="Does each skill have a narrow job, resources, and testable outcome?",
        risk_to_watch="Huge skill bundles waste context and hide which capability was used.",
        source_refs=(OPENAI_TOOLS_GUIDE, GOOGLE_ADK_SKILLS),
    ),
    AgenticConcept(
        concept_id="harness",
        name="Harness",
        category="governance",
        refined_question="What evidence gates keep an agent showcase honest?",
        definition="A lightweight control plane for planning, file claims, gates, review, and evidence.",
        openai_agents_sdk="Pair SDK traces/evals with local tests and artifact verification.",
        google_adk="Pair ADK evals/traces with local harness-lite commands and run ledgers.",
        student_build="Use Make targets, manifest verification, tests, and docs/agents run ledgers.",
        showcase_artifact="docs/agents/runs/2026-05-08-agentic-course-assistant-showcase-implementation.md",
        evaluation_prompt="Can another student reproduce the run and verify the same artifacts?",
        risk_to_watch="Harness language can become performative unless gates are executable.",
        source_refs=(OPENAI_EVALS_GUIDE, GOOGLE_ADK_EVALS),
    ),
    AgenticConcept(
        concept_id="evals",
        name="Evals",
        category="quality",
        refined_question="How do we know an agent got better after a change?",
        definition="Repeatable scoring of workflow behavior, tool use, outputs, and regressions.",
        openai_agents_sdk="Start from traces, then use trace grading, datasets, and eval runs.",
        google_adk="Use ADK eval files, CLI/web evals, custom metrics, and user simulation.",
        student_build="Validate routing with pytest and define a judge rubric for later trace grading.",
        showcase_artifact="artifacts/evals/agent_judge_rubric.json",
        evaluation_prompt="Does the eval distinguish correctness, traceability, safety, and usefulness?",
        risk_to_watch="Vague evals reward fluent text instead of correct workflow behavior.",
        source_refs=(OPENAI_EVALS_GUIDE, GOOGLE_ADK_EVALS),
    ),
    AgenticConcept(
        concept_id="agent-as-judge",
        name="Agent As A Judge",
        category="quality",
        refined_question="When should another model judge the agent's output?",
        definition="A grader agent that scores outputs or traces against an explicit rubric.",
        openai_agents_sdk="Use trace graders or eval graders to assess tool choice and policy adherence.",
        google_adk="Use custom metrics or evaluator agents around eval cases when deterministic checks end.",
        student_build="Ship a rubric for judging answer usefulness, routing, guardrails, and evidence.",
        showcase_artifact="artifacts/evals/agent_judge_rubric.json",
        evaluation_prompt="Would two judges using the rubric produce similar scores?",
        risk_to_watch="A judge without examples and thresholds becomes another subjective chatbot.",
        source_refs=(OPENAI_EVALS_GUIDE, GOOGLE_ADK_EVALS),
    ),
    AgenticConcept(
        concept_id="artifacts",
        name="Artifacts",
        category="evidence",
        refined_question="What should the agent write so humans and tests can inspect behavior?",
        definition="Versioned or stable files that capture outputs, traces, data, and reports.",
        openai_agents_sdk="Persist run results, traces, and evaluation inputs in app-owned artifacts.",
        google_adk="Artifacts manage named, versioned binary data scoped to sessions or users.",
        student_build="Write Markdown, JSON, CSV, concept, and eval artifacts after each run.",
        showcase_artifact="artifacts/manifest.json",
        evaluation_prompt="Does the manifest enumerate every file the verifier expects?",
        risk_to_watch="Generated files drift when docs and manifest do not share stable paths.",
        source_refs=(OPENAI_RUNNING_GUIDE, GOOGLE_ADK_ARTIFACTS),
    ),
    AgenticConcept(
        concept_id="events-state",
        name="Events And State",
        category="state",
        refined_question="How does the runtime record actions and update working data?",
        definition="Events describe what happened; state stores the working facts for the run.",
        openai_agents_sdk="Use run items, results, state snapshots, and histories to continue turns.",
        google_adk="Events form session history and State stores conversation-local data.",
        student_build="Expose trace steps as student-readable event-like records.",
        showcase_artifact="artifacts/agent_trace.json",
        evaluation_prompt="Can each event be tied to one workflow decision or side effect?",
        risk_to_watch="State updates hidden in prose cannot be audited or replayed.",
        source_refs=(OPENAI_RUNNING_GUIDE, GOOGLE_ADK_SESSIONS),
    ),
    AgenticConcept(
        concept_id="callbacks-plugins-hooks",
        name="Callbacks, Plugins, And Hooks",
        category="extension",
        refined_question="Where can developers observe or modify the agent lifecycle?",
        definition="Lifecycle extension points for logging, policy checks, caching, and metrics.",
        openai_agents_sdk="Use SDK tracing, custom spans, guardrails, and runtime hooks in app code.",
        google_adk="Callbacks and Plugins run before/after agent, model, tool, event, and runner stages.",
        student_build="Treat guardrail_notes and artifact verification as starter lifecycle checks.",
        showcase_artifact="artifacts/concepts/openai_vs_adk_concepts.json",
        evaluation_prompt="Is each hook observability-only, policy-enforcing, or behavior-changing?",
        risk_to_watch="Hooks can hide business logic outside the obvious agent definition.",
        source_refs=(OPENAI_OBSERVABILITY_GUIDE, GOOGLE_ADK_CALLBACKS, GOOGLE_ADK_PLUGINS),
    ),
    AgenticConcept(
        concept_id="streaming-realtime-voice",
        name="Streaming, Realtime, And Voice",
        category="runtime",
        refined_question="How does the same workflow behave while output is still arriving?",
        definition="Incremental text, audio, or multimodal interaction using the same state model.",
        openai_agents_sdk="Streaming uses the same agent loop and must settle before final decisions.",
        google_adk="ADK includes streaming agents and Gemini Live API Toolkit paths.",
        student_build="Leave streaming out of the first CLI demo but include it in the learning path.",
        showcase_artifact="artifacts/concepts/student_learning_path.md",
        evaluation_prompt="Does the design wait for streams to finish before approving side effects?",
        risk_to_watch="Partial outputs can be mistaken for final, policy-checked answers.",
        source_refs=(OPENAI_RUNNING_GUIDE, GOOGLE_ADK_OVERVIEW),
    ),
    AgenticConcept(
        concept_id="sandbox-code-execution",
        name="Sandbox And Code Execution",
        category="runtime",
        refined_question="When may an agent run code, commands, or notebooks?",
        definition="A controlled execution environment for computation, files, or generated code.",
        openai_agents_sdk="Use sandbox agents when the workflow needs files, commands, or packages.",
        google_adk="ADK supports code execution through tools and execution-oriented integrations.",
        student_build="Use local Python scripts, not model-generated shell commands, in the first build.",
        showcase_artifact="projects/agentic-course-assistant-showcase/Makefile",
        evaluation_prompt="Can every executable path be run intentionally from Make targets?",
        risk_to_watch="Uncontrolled code execution creates security and reproducibility risks.",
        source_refs=(OPENAI_AGENTS_GUIDE, GOOGLE_ADK_OVERVIEW),
    ),
    AgenticConcept(
        concept_id="auth-permissions-secrets",
        name="Auth, Permissions, And Secrets",
        category="safety",
        refined_question="What credentials or permissions can the agent see or use?",
        definition="The access boundary around model calls, tools, memory, logs, and artifacts.",
        openai_agents_sdk="Your runtime owns tool execution, approvals, secrets, and external access.",
        google_adk="ADK tools, MCP, deployment, and services require explicit auth boundaries.",
        student_build="Default to no credentials and warn users not to paste secrets.",
        showcase_artifact="artifacts/course_assistant_response.md",
        evaluation_prompt="Does a credential-related prompt avoid echoing secrets into artifacts?",
        risk_to_watch="Secrets in prompts, traces, or artifacts are classroom foot-guns.",
        source_refs=(OPENAI_AGENTS_GUIDE, GOOGLE_ADK_MCP),
    ),
    AgenticConcept(
        concept_id="error-handling-retries-timeouts",
        name="Errors, Retries, And Timeouts",
        category="reliability",
        refined_question="How should an agent fail when a model, tool, or policy check fails?",
        definition="Explicit behavior for validation errors, tool failures, cancellations, and timeouts.",
        openai_agents_sdk="Separate validation failures from expected approval pauses and resume state.",
        google_adk="Use runner events, services, callbacks, and tooling to surface failures clearly.",
        student_build="Raise ValueError for empty questions and fail verifier errors with messages.",
        showcase_artifact="scripts/verify_artifacts.py",
        evaluation_prompt="Does the failure message identify the broken contract exactly?",
        risk_to_watch="Retrying blindly can repeat side effects or mask schema defects.",
        source_refs=(OPENAI_RUNNING_GUIDE, GOOGLE_ADK_CALLBACKS),
    ),
    AgenticConcept(
        concept_id="cost-latency-context-budget",
        name="Cost, Latency, And Context Budget",
        category="operations",
        refined_question="What makes an agent too slow, expensive, or context-heavy?",
        definition="The practical budget across model calls, tool calls, traces, memory, and skills.",
        openai_agents_sdk="Use traces to inspect model/tool costs before formal evals or optimization.",
        google_adk="Use metrics, traces, plugins, and context controls to manage overhead.",
        student_build="Use deterministic local logic so the first run is fast and free.",
        showcase_artifact="artifacts/concepts/student_learning_path.md",
        evaluation_prompt="Does every added agent or tool earn its latency and complexity cost?",
        risk_to_watch="Over-orchestration turns a teaching demo into a slow Rube Goldberg machine.",
        source_refs=(OPENAI_OBSERVABILITY_GUIDE, GOOGLE_ADK_TRACES, GOOGLE_ADK_PLUGINS),
    ),
    AgenticConcept(
        concept_id="deployment-runtime",
        name="Deployment Runtime",
        category="operations",
        refined_question="Where does the agent run after it leaves the laptop?",
        definition="The packaging, service boundary, runtime config, and operational environment.",
        openai_agents_sdk="Run from your application server when you own orchestration and state.",
        google_adk="ADK provides local web/CLI loops and deployment paths such as Agent Runtime.",
        student_build="Keep this showcase local; propose deployment as a later production module.",
        showcase_artifact="projects/agentic-course-assistant-showcase/docs/sdk-comparison.md",
        evaluation_prompt="Is the deployment plan explicit about runtime, state, secrets, and rollback?",
        risk_to_watch="Deploying before evals and observability turns learning bugs into ops bugs.",
        source_refs=(OPENAI_AGENTS_GUIDE, GOOGLE_ADK_OVERVIEW),
    ),
    AgenticConcept(
        concept_id="versioning-release-rollback",
        name="Versioning, Release, And Rollback",
        category="operations",
        refined_question="How do we compare prompt, tool, and routing changes over time?",
        definition="A release discipline for prompts, skills, tools, datasets, and artifact contracts.",
        openai_agents_sdk="Use eval datasets and traces to compare SDK workflow revisions.",
        google_adk="Use eval sets and artifacts to compare agent revisions before promotion.",
        student_build="Version behavior through tests, manifest paths, docs, and stable Make targets.",
        showcase_artifact="artifacts/evals/concept_coverage.json",
        evaluation_prompt="Can an older trace still be scored against the current rubric?",
        risk_to_watch="Changing prompts without versioned evals makes regressions invisible.",
        source_refs=(OPENAI_EVALS_GUIDE, GOOGLE_ADK_EVALS),
    ),
    AgenticConcept(
        concept_id="prompt-instruction-design",
        name="Prompt And Instruction Design",
        category="design",
        refined_question="What should instructions say, and what should code enforce?",
        definition="The boundary between natural-language task guidance and deterministic guarantees.",
        openai_agents_sdk="Give specialists narrow instructions and enforce risky behavior in code.",
        google_adk="Use agent instructions, workflow structure, callbacks, and tools together.",
        student_build="Keep route keywords in code and explain specialist behavior in docs.",
        showcase_artifact="projects/agentic-course-assistant-showcase/docs/learning-guide.md",
        evaluation_prompt="Could a student change one instruction without breaking the verifier?",
        risk_to_watch="Prompts used as the only contract are hard to audit.",
        source_refs=(OPENAI_ORCHESTRATION_GUIDE, GOOGLE_ADK_CALLBACKS),
    ),
    AgenticConcept(
        concept_id="grounding-retrieval",
        name="Grounding And Retrieval",
        category="capability",
        refined_question="What evidence should an answer cite instead of hallucinating?",
        definition="The use of retrieved resources, files, search, or databases to ground output.",
        openai_agents_sdk="Use tools, hosted tools, MCP, or app retrieval around the agent loop.",
        google_adk="Use tools, grounding integrations, MCP, and artifacts to supply context.",
        student_build="Return explicit course resource matches for every answer.",
        showcase_artifact="artifacts/resource_matches.csv",
        evaluation_prompt="Does the answer cite retrieved resources that match the question?",
        risk_to_watch="Grounding that is not shown in artifacts cannot be inspected.",
        source_refs=(OPENAI_TOOLS_GUIDE, GOOGLE_ADK_OVERVIEW),
    ),
    AgenticConcept(
        concept_id="privacy-retention",
        name="Privacy, Retention, And Data Boundaries",
        category="safety",
        refined_question="What should never enter traces, memory, eval datasets, or artifacts?",
        definition="Policies for excluding secrets, personal data, and private records from evidence.",
        openai_agents_sdk="App-owned storage and tracing controls must respect privacy boundaries.",
        google_adk="Memory, sessions, artifacts, traces, and plugins all need retention choices.",
        student_build="Use public course examples and verify artifacts contain no credentials.",
        showcase_artifact="artifacts/evals/agent_judge_rubric.json",
        evaluation_prompt="Does the judge rubric penalize secret leakage and unsupported private data?",
        risk_to_watch="Evals and traces can accidentally become sensitive datasets.",
        source_refs=(OPENAI_GUARDRAILS_GUIDE, GOOGLE_ADK_MEMORY, GOOGLE_ADK_ARTIFACTS),
    ),
)


def list_concepts() -> tuple[AgenticConcept, ...]:
    """Return the stable concept atlas used by docs, tests, and artifacts."""

    return AGENTIC_CONCEPTS


def required_concept_coverage() -> dict[str, bool]:
    """Return coverage for concepts explicitly requested by the user."""

    available = {concept.concept_id for concept in AGENTIC_CONCEPTS}
    return {concept_id: concept_id in available for concept_id in sorted(REQUESTED_CONCEPT_IDS)}


def write_concept_artifacts(output_dir: Path) -> list[Path]:
    """Write comprehensive concept, question, and evaluator artifacts."""

    concepts_dir = output_dir / "concepts"
    evals_dir = output_dir / "evals"
    concepts_dir.mkdir(parents=True, exist_ok=True)
    evals_dir.mkdir(parents=True, exist_ok=True)

    csv_path = concepts_dir / "agentic_concepts.csv"
    comparison_path = concepts_dir / "openai_vs_adk_concepts.json"
    questions_path = concepts_dir / "refined_questions.md"
    learning_path = concepts_dir / "student_learning_path.md"
    rubric_path = evals_dir / "agent_judge_rubric.json"
    coverage_path = evals_dir / "concept_coverage.json"

    _write_concept_csv(csv_path)
    comparison_path.write_text(
        json.dumps(_comparison_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    questions_path.write_text(_render_refined_questions(), encoding="utf-8")
    learning_path.write_text(_render_learning_path(), encoding="utf-8")
    rubric_path.write_text(
        json.dumps(_judge_rubric(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    coverage_path.write_text(
        json.dumps(_coverage_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return [csv_path, comparison_path, questions_path, learning_path, rubric_path, coverage_path]


def _write_concept_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPECTED_CONCEPT_COLUMNS)
        writer.writeheader()
        for concept in AGENTIC_CONCEPTS:
            writer.writerow(concept.to_csv_row())


def _comparison_payload() -> dict[str, object]:
    return {
        "showcase": "agentic-course-assistant-showcase",
        "frameworks": {
            "openai_agents_sdk": {
                "summary": (
                    "Best for code-first orchestration when the application owns tools, "
                    "approvals, state, handoffs, tracing, and eval loops."
                ),
                "source": OPENAI_AGENTS_GUIDE,
            },
            "google_adk": {
                "summary": (
                    "Best for building, running, evaluating, and deploying agents with "
                    "ADK primitives such as agents, tools, sessions, memory, artifacts, "
                    "callbacks, plugins, and A2A."
                ),
                "source": GOOGLE_ADK_OVERVIEW,
            },
        },
        "concepts": [asdict(concept) for concept in AGENTIC_CONCEPTS],
    }


def _render_refined_questions() -> str:
    question_lines = "\n".join(f"{index}. {question}" for index, question in enumerate(REFINED_QUESTIONS, 1))
    return (
        "# Refined Questions For Agent Framework Study\n\n"
        "These questions expand the original prompt into a buildable course-assistant "
        "curriculum. Use them before adding live SDK credentials.\n\n"
        f"{question_lines}\n\n"
        "## Completeness Check\n\n"
        "- The list covers tools, guardrails, tracing, workflows, evals, judging, "
        "multi-agent routing, handoff/triage, A2A, sessions, memory, skills, and harnesses.\n"
        "- Additional operational concepts include artifacts, callbacks/plugins, schemas, "
        "deployment, auth, errors, budgets, versioning, grounding, and privacy.\n"
    )


def _render_learning_path() -> str:
    return (
        "# Student Learning Path\n\n"
        "## Stage 1: Offline Workflow\n\n"
        "Build one deterministic triage loop, call one local course-catalog tool, "
        "write trace artifacts, and validate them with `make verify`.\n\n"
        "## Stage 2: SDK Shape\n\n"
        "Map the same workflow to OpenAI Agents SDK and Google ADK without changing "
        "the artifact contract. Keep credentials optional.\n\n"
        "## Stage 3: Multi-Agent Design\n\n"
        "Compare handoffs, agents-as-tools, ADK workflow agents, and local routing. "
        "Add specialists only when the ownership contract changes.\n\n"
        "## Stage 4: Safety And State\n\n"
        "Add guardrails, human approvals, sessions, memory, and retention policies. "
        "Do not persist secrets or private student data.\n\n"
        "## Stage 5: Evaluation And Harness\n\n"
        "Use pytest for deterministic routing, trace/eval rubrics for subjective quality, "
        "and harness-lite ledgers for reproducible evidence.\n\n"
        "## Stage 6: Interoperability And Operations\n\n"
        "Explore MCP, A2A, deployment, streaming, observability, cost, latency, "
        "versioning, and rollback only after the offline contract is stable.\n"
    )


def _judge_rubric() -> dict[str, object]:
    return {
        "name": "agentic_course_assistant_trace_judge",
        "purpose": (
            "Score a course-assistant response and trace after deterministic tests pass."
        ),
        "scoring_scale": "0 to 4 per criterion; 4 means strong evidence and no material gap.",
        "criteria": [
            {
                "id": "routing_correctness",
                "weight": 0.2,
                "question": "Did triage select the specialist that best matches the question?",
            },
            {
                "id": "tool_grounding",
                "weight": 0.2,
                "question": "Did the answer use relevant resource matches from the catalog tool?",
            },
            {
                "id": "trace_completeness",
                "weight": 0.2,
                "question": "Can the route, tool call, specialist step, and guardrail be reconstructed?",
            },
            {
                "id": "safety_and_privacy",
                "weight": 0.15,
                "question": "Did the answer avoid secrets, unsupported private data, and risky advice?",
            },
            {
                "id": "student_usefulness",
                "weight": 0.15,
                "question": "Does the response give a clear next step a student can perform locally?",
            },
            {
                "id": "artifact_contract",
                "weight": 0.1,
                "question": "Do generated artifacts satisfy the manifest and verifier contract?",
            },
        ],
        "blocking_failures": [
            "The response includes a secret or tells the user to paste one.",
            "The trace omits the selected intent or tool lookup.",
            "The verifier fails for generated artifacts.",
            "The answer is unrelated to the student question.",
        ],
    }


def _coverage_payload() -> dict[str, object]:
    coverage = required_concept_coverage()
    missing = [concept_id for concept_id, covered in coverage.items() if not covered]
    return {
        "requested_concepts_covered": coverage,
        "missing_requested_concepts": missing,
        "total_concepts": len(AGENTIC_CONCEPTS),
        "categories": sorted({concept.category for concept in AGENTIC_CONCEPTS}),
        "verdict": "pass" if not missing else "fail",
    }
