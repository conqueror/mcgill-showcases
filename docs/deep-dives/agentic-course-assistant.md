# Agentic Course Assistant Deep Dive

Project:

- `projects/agentic-course-assistant-showcase`

## Why This Deep Dive

Use this project when students are ready to move from ordinary scripts to agentic systems, but still need a reproducible local path before hosted model calls enter the workflow.

The course assistant is intentionally small:

1. Classify a student question.
2. Route it to a specialist intent.
3. Call a deterministic course-catalog tool.
4. Write a trace and response artifact.
5. Validate the artifact contract.
6. Compare the same design shape with optional OpenAI Agents SDK and Google ADK examples.

The default path is offline and deterministic. It is the path students should run first, and it is the path that belongs in default CI.

## Quickstart

```bash
cd projects/agentic-course-assistant-showcase
make sync
make smoke
make verify
```

Ask a custom question:

```bash
make run QUESTION="I understand train/test splits, but why is leakage dangerous?"
```

Run the focused quality gate:

```bash
make check
```

## What To Inspect

| Artifact | What It Teaches |
|---|---|
| `artifacts/course_assistant_response.md` | The routed answer students can read first. |
| `artifacts/agent_trace.json` | The route, specialist, guardrail notes, and resource evidence. |
| `artifacts/resource_matches.csv` | The deterministic tool output. |
| `artifacts/concepts/agentic_concepts.csv` | The concept map across offline, OpenAI Agents SDK, and Google ADK lenses. |
| `artifacts/concepts/openai_vs_adk_concepts.json` | Machine-readable framework comparison. |
| `artifacts/evals/agent_judge_rubric.json` | A starter rubric for agent-as-judge or trace-grading discussions. |
| `artifacts/evals/concept_coverage.json` | Coverage proof for the requested agent-framework concepts. |
| `artifacts/manifest.json` | The artifact contract validated by `make verify`. |

## Lab Sequence

1. Run the offline smoke path.
2. Open `artifacts/course_assistant_response.md` and identify the selected intent.
3. Open `artifacts/agent_trace.json` and map each trace step to the response.
4. Open `artifacts/resource_matches.csv` and confirm the answer is grounded in public course resources.
5. Open `artifacts/evals/agent_judge_rubric.json` and decide which rubric checks are deterministic versus judgment-based.
6. Read the concept atlas to connect the local workflow to OpenAI Agents SDK and Google ADK terms.

For the full classroom walkthrough, see the project lab guide in the source project:

- [Lab guide](https://github.com/conqueror/mcgill-showcases/blob/main/projects/agentic-course-assistant-showcase/docs/lab-guide.md)
- [Concept map](https://github.com/conqueror/mcgill-showcases/blob/main/projects/agentic-course-assistant-showcase/docs/concept-map.md)

## Offline Harness First

The deterministic harness is the teaching anchor:

- It does not require API keys.
- It writes stable artifacts that can be inspected by another student.
- It makes routing and guardrails visible instead of hiding them inside a model call.
- It supports normal laptop execution with `uv` and local Python.
- It keeps default CI public-safe because optional SDK modules are not imported by default tests.

This is the recommended order for students:

1. Make the local trace understandable.
2. Make the artifact contract pass.
3. Add or edit one specialist route.
4. Add one deterministic test.
5. Only then install optional live SDK extras.

## Optional Live SDK Path

Live SDK usage is an extension, not a requirement.

OpenAI Agents SDK setup:

```bash
cd projects/agentic-course-assistant-showcase
make sync-openai
```

Then configure `OPENAI_API_KEY` outside source control and call the optional reference function from a local scratch command or notebook:

```bash
uv run python - <<'PY'
import asyncio
from agentic_course_assistant.openai_agents_example import run_openai_agents_course_assistant

question = "How should I debug a suspicious validation score?"
print(asyncio.run(run_openai_agents_course_assistant(question)))
PY
```

Google ADK setup:

```bash
cd projects/agentic-course-assistant-showcase
make sync-adk
```

After configuring the Gemini or Vertex credentials expected by ADK, run the ADK-discoverable wrapper:

```bash
uv run adk run adk_course_assistant
```

Use `make sync-live` only when a student intentionally wants both optional SDK extras in the same environment.

## Public-Safe Guardrails

Keep the live path classroom-safe:

- Do not paste API keys into prompts, notebooks, logs, screenshots, or artifacts.
- Do not make live SDK imports part of default tests.
- Do not require hosted credentials for default CI.
- Use public course resources as the grounding source.
- Treat traces and artifacts as evidence students can discuss without private data.

## How To Interpret Outputs

1. A good answer is not enough; the trace should prove how the answer was selected.
2. Tool output should be deterministic before an LLM is asked to synthesize.
3. Guardrails should state what they protect and what they do not protect.
4. Hosted SDK examples should preserve the same route, tool, and trace mental model.
5. Evals should start with local artifact checks before moving to agent-as-judge scoring.

## Next Step

Continue with the [Agent Frameworks track](../tracks/agent-frameworks.md), then compare this project with `projects/autoresearch` for a broader agent-guided research loop.
