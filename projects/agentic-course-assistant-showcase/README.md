# Agentic Course Assistant Showcase

Learn how agent frameworks route work, call tools, apply guardrails, and leave evidence a student can inspect.

This project builds a small course assistant for machine-learning students. The default path is deterministic and offline, so you can understand the workflow before connecting to a hosted model. Optional modules show the same design shape with the OpenAI Agents SDK and Google ADK, and the generated concept atlas maps the broader agent-engineering curriculum.

## Learning Outcomes

By the end of this project, you should be able to:

- explain the difference between a deterministic agent workflow and a hosted SDK workflow,
- route a student question to a specialist agent intent,
- use a small course catalog as a tool,
- inspect an agent trace and identify each workflow step,
- compare OpenAI Agents SDK concepts with Google ADK concepts,
- explain sessions, memory, A2A, evals, agent-as-judge, skills, artifacts, and harness gates,
- add one guardrail that keeps an assistant scoped to public learning support.

## Prerequisites

- Python 3.11+
- `uv`
- Basic Python functions, dictionaries, and dataclasses
- Optional: an OpenAI API key plus the `openai` extra for `openai_agents_example.py`
- Optional: a Gemini or Vertex setup plus the `adk` extra for `google_adk_example.py`

## Quickstart

```bash
cd projects/agentic-course-assistant-showcase
make sync
make smoke
make verify
```

Ask your own question:

```bash
make run QUESTION="I understand train/test splits, but why is leakage so dangerous?"
```

Run quality checks:

```bash
make check
```

Install live SDK extras only when you are ready to run the hosted examples:

```bash
make sync-openai  # installs openai-agents
make sync-adk     # installs google-adk
make sync-live    # installs both optional SDK extras
```

## Key Artifacts

After `make run`, inspect:

- `artifacts/course_assistant_response.md`: the routed answer, matched resources, and trace.
- `artifacts/agent_trace.json`: machine-readable routing, guardrail, and resource evidence.
- `artifacts/resource_matches.csv`: course resources returned by the lookup tool.
- `artifacts/concepts/agentic_concepts.csv`: a comprehensive concept map across OpenAI Agents SDK, Google ADK, and this offline build.
- `artifacts/concepts/openai_vs_adk_concepts.json`: machine-readable framework comparison.
- `artifacts/concepts/refined_questions.md`: expanded questions students should answer before building.
- `artifacts/concepts/student_learning_path.md`: staged path from offline workflow to SDK, eval, A2A, and deployment extensions.
- `artifacts/evals/agent_judge_rubric.json`: rubric for an agent-as-judge or trace-grading extension.
- `artifacts/evals/concept_coverage.json`: coverage proof for requested concepts.
- `artifacts/manifest.json`: the artifact contract used by `make verify`.

## How The Assistant Works

The default offline workflow is intentionally small:

1. `triage_agent` receives the question.
2. `classify_question` routes it to `concept`, `exercise`, `debug`, or `project`.
3. `search_resources` acts as a deterministic course-catalog tool.
4. The selected specialist composes a short answer.
5. `guardrail_notes` adds scope and secret-handling reminders.
6. `write_artifacts` writes the response, trace, and matched resources.
7. `write_concept_artifacts` writes the concept atlas, refined questions, learning path, and eval rubric.

This shape maps cleanly to hosted SDKs without making the first student run depend on API credentials.

## SDK Examples

The repository did not previously include a direct OpenAI Agents SDK or Google ADK example. This showcase adds both as optional reference modules:

- `src/agentic_course_assistant/openai_agents_example.py` uses `Agent`, `Runner`, `function_tool`, tools, and handoffs.
- `src/agentic_course_assistant/google_adk_example.py` defines the ADK `root_agent` and function tool.
- `adk_course_assistant/agent.py` is an ADK-discoverable wrapper for `uv run adk run adk_course_assistant`.

The default tests do not import these modules because credentials are optional and the first classroom path stays offline. Use `make sync-openai`, `make sync-adk`, or `make sync-live` to install the dependency-managed SDK extras.

## Comprehensive Concept Atlas

The concept atlas covers the requested topics and the missing-but-important surrounding topics:

- tool calls and function tools,
- guardrails, human approval, callbacks, plugins, and policy checks,
- tracing, observability, artifacts, and reproducible evidence,
- agentic workflows, multi-agent orchestration, handoffs, triage, and agents-as-tools,
- evals, trace grading, and agent-as-judge rubrics,
- A2A, MCP, hosted tools, connectors, and remote-agent boundaries,
- sessions, state, memory, events, privacy, and retention,
- skills, harness gates, tests, deployment, auth, cost, latency, versioning, and rollback.

Read `docs/concept-atlas.md` after running `make smoke`, then inspect the generated files under `artifacts/concepts/` and `artifacts/evals/`.

## Common Failure Modes

- Starting with a large multi-agent design before a one-agent trace works.
- Treating a tool call as magic instead of a typed function with inputs and outputs.
- Forgetting that model-backed agents need guardrails and trace inspection.
- Adding A2A, memory, or deployment before the offline artifact contract is stable.
- Pasting API keys into prompts, logs, screenshots, or artifacts.
- Comparing OpenAI Agents SDK and Google ADK without separating routing, tools, state, and deployment concerns.

## Suggested Next Projects

- `../autoresearch/README.md`
- `../modern-nlp-pipeline-showcase/README.md`
- `../model-release-rollout-showcase/README.md`

## Project Structure

```text
agentic-course-assistant-showcase/
├── README.md
├── Makefile
├── pyproject.toml
├── docs/
├── scripts/
├── src/agentic_course_assistant/
├── tests/
└── artifacts/
```
