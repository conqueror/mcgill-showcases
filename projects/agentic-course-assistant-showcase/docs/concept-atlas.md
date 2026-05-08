# Agent Framework Concept Atlas

This showcase treats OpenAI Agents SDK and Google ADK as two concrete ways to teach
the same agent-engineering concepts. The default code path is offline and
deterministic, but the concepts are mapped to official SDK/ADK primitives so students
can extend the project safely.

## Refined Question Set

The original question was broad: "What are the concepts of OpenAI Agents SDK and
Google ADK, and what should students build?" The buildable version is:

1. What is an agent, and what contract does it own?
2. When should logic be a function tool, hosted tool, MCP tool, specialist agent, or
   workflow step?
3. Where do guardrails live: input, output, tool call, callback, plugin, or human
   approval?
4. What trace evidence proves the route, tool choice, handoff, and stopping point?
5. Which behaviors need unit tests, trace grading, eval datasets, or an agent judge?
6. How should multi-agent orchestration differ from one agent with several tools?
7. When should triage hand off ownership instead of calling a specialist as a tool?
8. What does A2A add beyond local sub-agents, MCP tools, and function calls?
9. What belongs in session state, durable memory, artifacts, and logs?
10. How do skills package reusable instructions, resources, and tools?
11. What local harness evidence proves the project is reproducible?
12. What else matters after the demo: auth, privacy, cost, latency, deployment,
    streaming, versioning, rollback, and data retention?

## Concept Coverage

The generated `artifacts/concepts/agentic_concepts.csv` covers these major groups:

- Foundation: agent contracts, model/provider selection, runner/event loop.
- Capability: function tools, hosted tools, MCP, grounding, skills.
- Safety: guardrails, human approval, auth, privacy, retention.
- Orchestration: workflows, multi-agent systems, triage, handoffs, agents-as-tools,
  A2A.
- State: sessions, memory, events, working state.
- Observability and quality: traces, evals, agent-as-judge, schemas, artifacts.
- Operations: errors, retries, timeouts, budgets, deployment, versioning, rollback.
- Extension: callbacks, plugins, hooks, streaming, sandbox/code execution.

## OpenAI Agents SDK Lens

Use the OpenAI Agents SDK when the application owns code-first orchestration:

- `Agent` defines the specialist contract.
- `Runner.run` or `run` executes the agent loop.
- `function_tool` wraps local code.
- Handoffs transfer answer ownership to a specialist.
- `agent.as_tool` keeps the manager in control while calling a specialist.
- Guardrails and human approvals block, pause, or resume risky work.
- Built-in tracing captures model calls, tool calls, handoffs, guardrails, and spans.
- Agent evals start with traces and grow into repeatable graders/datasets.

Sources:

- [OpenAI Agents SDK guide](https://developers.openai.com/api/docs/guides/agents)
- [OpenAI tools in the Agents SDK](https://developers.openai.com/api/docs/guides/tools#usage-in-the-agents-sdk)
- [OpenAI orchestration and handoffs](https://developers.openai.com/api/docs/guides/agents/orchestration)
- [OpenAI guardrails and human review](https://developers.openai.com/api/docs/guides/agents/guardrails-approvals)
- [OpenAI agent evals](https://developers.openai.com/api/docs/guides/agent-evals)

## Google ADK Lens

Use Google ADK when you want a structured agent project with local development,
runtime services, evaluation, and deployment paths:

- `LlmAgent`, workflow agents, and custom agents define worker units.
- Function tools wrap local functions through signatures and docstrings.
- `Runner`, `Session`, `State`, `Events`, `MemoryService`, and `ArtifactService`
  model runtime context and evidence.
- Sequential, parallel, and loop workflow agents support deterministic orchestration.
- Agent routing, sub-agents, and `AgentTool` cover specialist composition.
- Callbacks and plugins provide lifecycle checks for logging, policy, metrics, and
  behavior changes.
- Traces, eval sets, custom metrics, and user simulation support quality loops.
- A2A guides cover exposing and consuming remote agents.

Sources:

- [Google ADK technical overview](https://adk.dev/get-started/about/)
- [Google ADK function tools](https://adk.dev/tools-custom/function-tools/)
- [Google ADK sessions, state, and memory](https://adk.dev/sessions/)
- [Google ADK multi-agent systems](https://adk.dev/agents/multi-agents/)
- [Google ADK evaluation](https://adk.dev/evaluate/)
- [Google ADK A2A](https://adk.dev/a2a/)

## Student Build Recommendation

The first comprehensive build should stay small: a course assistant that routes a
student question, calls a course-catalog tool, writes a trace, applies guardrail
notes, and validates artifacts. That gives students the mental model before they add
credentials, remote tools, memory, A2A, or deployment.

The generated files are the teaching surface:

- `artifacts/concepts/agentic_concepts.csv`: concept-by-concept comparison.
- `artifacts/concepts/openai_vs_adk_concepts.json`: machine-readable comparison.
- `artifacts/concepts/refined_questions.md`: expanded question list.
- `artifacts/concepts/student_learning_path.md`: staged build path.
- `artifacts/evals/agent_judge_rubric.json`: agent-as-judge rubric.
- `artifacts/evals/concept_coverage.json`: coverage check for requested concepts.
