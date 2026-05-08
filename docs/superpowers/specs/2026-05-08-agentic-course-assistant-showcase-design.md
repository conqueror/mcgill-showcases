# Agentic Course Assistant Showcase Design

## Scope

Build a new showcase project under `projects/agentic-course-assistant-showcase` that teaches agent routing, tool use, guardrails, traceability, evals, agent-as-judge, multi-agent orchestration, A2A/session/memory concepts, skills, harness evidence, and the conceptual bridge between deterministic local workflows and hosted agent SDKs.

## Primary Learning Outcome

Learners should be able to explain and test the core parts of an agent workflow before plugging the same design into OpenAI Agents SDK or Google ADK.

## Deep QNA Synthesis

Original questions:

- Does the codebase already contain an OpenAI Agents SDK or Google ADK example?
- What is a good example for students to build?
- What showcases could we have?
- Which skills should we employ to implement them?

Refined questions:

- What is present in the repository today versus what we are inferring from external SDK docs?
- Which example is useful, safe, laptop-friendly, and coherent enough for one showcase?
- Which future showcases should remain separate projects rather than bloating the first one?
- Which harness and engineering skills are needed for design, implementation, review, and verification?
- What concepts beyond the initial list should be covered before students build live agents?
- Which concepts belong in the first offline build, and which should remain extension stages?

Confirmed facts:

- Repo search found no direct OpenAI Agents SDK or Google ADK example before this project.
- The repo playbook requires a small script-first project with tests, artifacts, docs, and root integrations.
- OpenAI Agents SDK is the right fit when code-first orchestration needs agents, tools, handoffs, guardrails, and tracing.
- Google ADK is a code-first agent framework with Python setup, `root_agent`, function tools, local CLI/web loops, and multi-agent patterns.
- Official current docs also identify sessions/state, memory, artifacts, callbacks/plugins, traces, evals, skills, MCP, and A2A as important ADK concepts.
- Official current OpenAI docs also identify sessions/state strategies, handoffs, agents-as-tools, guardrails/approvals, tracing, MCP, and agent evals as important Agents SDK concepts.

Recommendation:

- Implement `agentic-course-assistant-showcase` first.
- Keep the default run offline and deterministic.
- Include optional SDK-shaped modules for OpenAI Agents SDK and Google ADK.
- Save richer future ideas as separate showcases.
- Add a generated concept atlas so the comprehensive answer is executable, inspectable, and test-covered.

## Design Decisions

### 1. Course assistant over generic demo

A course assistant maps directly to student needs: concept explanation, exercises, debugging help, and project planning. It avoids private enterprise data and gives students a useful local artifact.

### 2. Offline default, SDK references optional

The default path uses deterministic routing and a local catalog. Optional modules show real SDK code shapes without making API keys a prerequisite for tests.

### 3. Trace as the main teaching artifact

Students should inspect `agent_trace.json` to see routing, tool lookup, specialist selection, and guardrails.

### 4. Concept atlas as the curriculum artifact

The project should generate a concept atlas that maps each concept to:

- a refined student question,
- a plain definition,
- the OpenAI Agents SDK interpretation,
- the Google ADK interpretation,
- the student build action,
- the evidence artifact,
- an evaluation prompt,
- a risk to watch.

## Proposed Project Structure

```text
projects/agentic-course-assistant-showcase/
├── README.md
├── Makefile
├── pyproject.toml
├── docs/
│   ├── learning-guide.md
│   ├── concept-atlas.md
│   └── sdk-comparison.md
├── scripts/
│   ├── run_showcase.py
│   └── verify_artifacts.py
├── src/agentic_course_assistant/
├── tests/
└── artifacts/
```

## Artifact Contract

Required files:

- `artifacts/manifest.json`
- `artifacts/course_assistant_response.md`
- `artifacts/agent_trace.json`
- `artifacts/resource_matches.csv`
- `artifacts/concepts/agentic_concepts.csv`
- `artifacts/concepts/openai_vs_adk_concepts.json`
- `artifacts/concepts/refined_questions.md`
- `artifacts/concepts/student_learning_path.md`
- `artifacts/evals/agent_judge_rubric.json`
- `artifacts/evals/concept_coverage.json`

## Skills To Employ

- `core-qna-synthesis`: refine the multi-part question and choose the best project.
- `core-harness-flow`: route the run through public harness-lite.
- `core-brainstorming`: compare candidate showcases.
- `core-writing-plans`: persist the implementation plan.
- `core-executing-plans`: implement the chosen slice.
- `core-test-driven-development`: keep behavior covered by focused tests.
- `eng-python-engineer`: implement the Python package, scripts, and tests.
- `plat-a2a-mcp-protocols`: future extension only, if an A2A or MCP showcase is added later.
- `plat-google-adk-integration`: optional future deepening for a full Google ADK project.
- `plat-openai-agents-sdk-integration`: optional future deepening for a full OpenAI Agents SDK project.

## Required Concept Coverage

- Tool calls and function tools.
- Hosted tools, connectors, and MCP.
- Guardrails, human approval, callbacks, plugins, and policy checks.
- Tracing, observability, artifacts, and reproducible evidence.
- Agentic workflows, multi-agent orchestration, handoff/triage, and agents-as-tools.
- Evals, trace grading, and agent-as-judge.
- A2A and remote-agent interoperability.
- Sessions, state, events, memory, privacy, and retention.
- Skills and reusable capability packaging.
- Harness gates, tests, manifests, and run ledgers.
- Deployment, auth, errors, retries, timeouts, budgets, streaming, sandbox/code execution, versioning, rollback, grounding, and prompt/instruction design.

## Future Showcase Options

- Agentic course assistant with live SDK execution and trace comparison.
- Research digest agent over the repository docs.
- Experiment-review agent for model metrics and rollout decisions.
- Data-quality triage agent for EDA and leakage artifacts.
- Local API contract assistant grounded in OpenAPI JSON.

## Success Criteria

- `make smoke` generates the required artifacts.
- `make verify` validates the artifact contract.
- `make check` passes in the project directory.
- Concept coverage verifies every user-requested concept.
- Root docs, CI, Makefile, issue templates, learning path, and coverage matrix include the new showcase.
