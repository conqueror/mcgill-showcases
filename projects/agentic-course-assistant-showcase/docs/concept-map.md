# Concept Map

This map connects the deterministic course-assistant harness to the agent-framework concepts students will see in OpenAI Agents SDK and Google ADK. Use it after running `make smoke` so every concept has a local artifact to inspect.

## Big Picture

```text
student question
  -> triage and intent classification
  -> deterministic course-catalog tool
  -> specialist response
  -> guardrail notes
  -> trace, resources, concepts, eval rubric, manifest
  -> optional live OpenAI or Google ADK extension
```

The offline path is not a weaker version of the project. It is the control sample that lets students understand routing, tools, traces, and evals before hosted model behavior adds variability.

## Core Concept Map

| Concept | Offline Harness | OpenAI Agents SDK Lens | Google ADK Lens | Artifact To Inspect |
|---|---|---|---|---|
| Agent contract | A function owns the assistant workflow and returns structured results. | `Agent` owns instructions, tools, handoffs, and output behavior. | `LlmAgent`, workflow agent, or custom agent owns the worker boundary. | `artifacts/agent_trace.json` |
| Triage | `classify_question` selects `concept`, `exercise`, `debug`, or `project`. | A triage agent can hand off to specialists. | Routing can use sub-agents or workflow branches. | `artifacts/agent_trace.json` |
| Tool use | `search_resources` returns public course resources. | A `function_tool` wraps local code. | A function tool exposes Python logic to an agent. | `artifacts/resource_matches.csv` |
| Specialist behavior | The selected route shapes the answer. | Handoffs transfer ownership or agents-as-tools keep a manager in control. | Sub-agents or `AgentTool` model specialist composition. | `artifacts/course_assistant_response.md` |
| Guardrails | Scope and secret-handling reminders are written into the trace. | Guardrails and human review block or pause risky paths. | Callbacks, plugins, and confirmation flows can enforce policy. | `artifacts/agent_trace.json` |
| Tracing | The JSON trace records route, resources, and guardrail notes. | SDK traces show model calls, tool calls, handoffs, and spans. | ADK traces and events show runtime behavior and services. | `artifacts/agent_trace.json` |
| Evals | The rubric separates deterministic checks from judge-style checks. | Trace grading and eval datasets can score agent behavior. | ADK eval sets and custom metrics can score agent behavior. | `artifacts/evals/agent_judge_rubric.json` |
| Sessions | The first run is single-turn and stateless. | Sessions can continue or resume conversations. | `Session`, `State`, and `Events` model conversation context. | `artifacts/concepts/agentic_concepts.csv` |
| Memory | Memory is discussed but not required by the first run. | Apps should own durable memory deliberately. | `MemoryService` can add searchable long-term memory. | `artifacts/concepts/student_learning_path.md` |
| A2A and MCP | Remote agents and external tools stay out of the first run. | Protocol adapters or MCP tools can expose remote capabilities. | ADK includes A2A and MCP paths for later interoperability. | `artifacts/concepts/openai_vs_adk_concepts.json` |
| Harness evidence | `make verify` checks stable local artifacts. | Local tests should stay alongside SDK traces and evals. | Local checks should stay alongside ADK evals and traces. | `artifacts/manifest.json` |

## Student Progression

1. Read the response artifact.
2. Prove the route from the trace.
3. Prove the tool output from the CSV.
4. Read the eval rubric and name what is deterministic.
5. Read the concept CSV and pick one framework concept to explain.
6. Install optional SDK extras only after the offline contract is clear.
7. Compare live behavior against the offline trace instead of replacing the offline trace.

## Concept Boundaries

Use these boundaries to keep the project student-friendly:

- A tool is a typed capability, not magic model knowledge.
- A guardrail should have a visible enforcement or evidence surface.
- A trace should answer "what happened?" without needing to trust the final answer.
- Memory is not needed until the assistant has a clear reason to remember.
- A2A and MCP are interoperability extensions, not first-lab requirements.
- Live SDK calls are optional experiments, not the default CI contract.

## Framework Comparison Questions

OpenAI Agents SDK:

- Which code owns the `Agent` instructions?
- Which local function should become a `function_tool`?
- Should a specialist receive a handoff or be called as a tool?
- What trace evidence would make a live run reviewable?

Google ADK:

- What should be the `root_agent`?
- Which tool functions are safe to expose?
- Which state belongs in a session versus an artifact?
- When would ADK evals add value beyond `make verify`?

## Public-Safe Extension Ideas

- Add a new route for assessment or rubric questions.
- Add one more public course resource to the catalog.
- Add a deterministic trace field for policy decisions.
- Add one eval case to the rubric before using an agent-as-judge.
- Add an optional live demo script without importing SDKs in default tests.

## Read Next

- [`lab-guide.md`](lab-guide.md)
- [`learning-guide.md`](learning-guide.md)
- [`sdk-comparison.md`](sdk-comparison.md)
- [`concept-atlas.md`](concept-atlas.md)
