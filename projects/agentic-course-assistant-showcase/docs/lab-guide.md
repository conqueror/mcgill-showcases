# Lab Guide

This lab walks through the agentic course assistant from a deterministic offline harness to optional live SDK usage. The default classroom and CI path uses local Python only. Hosted model calls are an extension for students who intentionally install optional SDK extras and configure credentials.

## Learning Contract

By the end of the lab, you should be able to:

- explain what the assistant routes and why,
- identify the course-catalog tool output used in the answer,
- read trace evidence for routing, guardrails, and artifacts,
- separate deterministic checks from judgment-based evals,
- describe how the same design maps to OpenAI Agents SDK and Google ADK,
- decide what must stay out of default CI when live SDKs are optional.

Expected runtime: 10-20 minutes for the offline path on a normal laptop.

## Part 1: Run The Offline Harness

```bash
cd projects/agentic-course-assistant-showcase
make sync
make smoke
make verify
```

The smoke path writes a course-assistant answer and the evidence needed to inspect it. The verifier checks the artifact contract without using an API key.

## Part 2: Ask Your Own Question

```bash
make run QUESTION="I understand train/test splits, but why is leakage so dangerous?"
make verify
```

Keep the question public and course-related. Do not include private student data, tokens, credentials, or unpublished course materials.

## Part 3: Inspect The Response

Open:

- `artifacts/course_assistant_response.md`
- `artifacts/agent_trace.json`
- `artifacts/resource_matches.csv`

Answer these questions:

1. Which intent was selected?
2. Which specialist answered?
3. Which course resources were matched?
4. Which guardrail note was added?
5. Which artifact proves the route?

The goal is not just to like the answer. The goal is to prove how the answer was produced.

## Part 4: Inspect The Concept And Eval Artifacts

Open:

- `artifacts/concepts/agentic_concepts.csv`
- `artifacts/concepts/openai_vs_adk_concepts.json`
- `artifacts/concepts/student_learning_path.md`
- `artifacts/evals/agent_judge_rubric.json`
- `artifacts/evals/concept_coverage.json`

Use the concept artifacts to connect local code to framework terms:

| Local Pattern | OpenAI Agents SDK Lens | Google ADK Lens |
|---|---|---|
| `classify_question` | Triage agent or routing logic | Routing agent, sub-agent, or workflow branch |
| `search_resources` | Function tool | Function tool |
| Specialist answer step | Handoff or agent-as-tool decision | Sub-agent or `AgentTool` decision |
| `agent_trace.json` | Trace and eval evidence | Events, traces, and artifact evidence |
| `make verify` | Artifact contract check | Local harness check before ADK evals |

## Part 5: Make One Safe Extension

Add a tiny design proposal before writing code:

```text
New route:
Example student question:
Expected specialist:
Resource needed:
Trace field that should change:
Verifier or test that should catch regressions:
```

For this docs-discovery slice, do not edit code. In a later implementation slice, the smallest safe code change would be one route, one resource, and one test.

## Part 6: Move To Optional OpenAI Agents SDK

Only start this part after the offline artifacts pass.

```bash
cd projects/agentic-course-assistant-showcase
make sync-openai
```

Configure `OPENAI_API_KEY` outside source control. Then run the optional reference function locally:

```bash
uv run python - <<'PY'
import asyncio
from agentic_course_assistant.openai_agents_example import run_openai_agents_course_assistant

question = "How should I debug a suspicious validation score?"
print(asyncio.run(run_openai_agents_course_assistant(question)))
PY
```

Compare the live answer with the offline artifacts:

- Did the live path still use a course-resource tool?
- Did the route or specialist choice change?
- What trace evidence would you need before trusting the live behavior?
- Which parts of the offline verifier should still run?

Do not add this live command to default CI.

## Part 7: Move To Optional Google ADK

Only start this part after the offline artifacts pass.

```bash
cd projects/agentic-course-assistant-showcase
make sync-adk
```

Configure the Gemini or Vertex credentials expected by ADK outside source control. Then run the ADK-discoverable wrapper:

```bash
uv run adk run adk_course_assistant
```

Compare ADK concepts with the local harness:

- `root_agent` maps to the first live agent boundary.
- The lookup function maps to the deterministic course-catalog tool.
- ADK sessions, state, memory, and artifacts are extensions, not requirements for the first run.
- ADK evals are useful after the local artifact contract is stable.

Do not add this live command to default CI.

If you use a project `.env` file, start from `.env.example`. The showcase reads
`OPENAI_API_KEY`, `GEMINI_API_KEY`, `OPENAI_MODEL`, and `GEMINI_MODEL`, keeps
`.env` out of git, and copies `GEMINI_API_KEY` into `GOOGLE_API_KEY` for ADK
when `GOOGLE_API_KEY` is not already set.

## Public-Safe Checklist

- Keep the default path offline.
- Keep credentials out of prompts, logs, artifacts, screenshots, and committed files.
- Use public course examples only.
- Run `make verify` after generating artifacts.
- Treat live SDK runs as optional local experiments.
- Document any live behavior difference before changing the deterministic harness.

## Exit Ticket

Before leaving the lab, write three short answers:

1. What did the assistant route?
2. Which artifact proves the route and tool output?
3. What would need to be true before a live SDK path could be trusted in a class demo?
