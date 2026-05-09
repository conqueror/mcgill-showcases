# Agent Frameworks Track

This track teaches agentic systems through a public-safe progression: deterministic local workflows first, optional hosted SDKs second, and reproducible evidence throughout.

## Recommended Sequence

1. `projects/autoresearch`
2. `projects/agentic-course-assistant-showcase`
3. `projects/modern-nlp-pipeline-showcase`
4. `projects/model-release-rollout-showcase`

## Core Skills Covered

- Separating deterministic workflow logic from model-backed agent behavior.
- Routing questions to specialist intents.
- Turning course resources into tool outputs that can be inspected.
- Reading traces for route, tool, guardrail, and artifact evidence.
- Comparing OpenAI Agents SDK and Google ADK concepts without requiring live credentials for the default path.
- Designing eval rubrics before adding agent-as-judge scoring.
- Deciding when memory, A2A, MCP, hosted tools, sessions, callbacks, plugins, and deployment are actually needed.

## Primary Showcase

Start with `projects/agentic-course-assistant-showcase`.

```bash
cd projects/agentic-course-assistant-showcase
make sync
make smoke
make verify
```

Then read:

- [Agentic Course Assistant deep dive](../deep-dives/agentic-course-assistant.md)
- [`docs/lab-guide.md`](https://github.com/conqueror/mcgill-showcases/blob/main/projects/agentic-course-assistant-showcase/docs/lab-guide.md)
- [`docs/concept-map.md`](https://github.com/conqueror/mcgill-showcases/blob/main/projects/agentic-course-assistant-showcase/docs/concept-map.md)

## Evidence Artifacts To Inspect

- `projects/agentic-course-assistant-showcase/artifacts/course_assistant_response.md`
- `projects/agentic-course-assistant-showcase/artifacts/agent_trace.json`
- `projects/agentic-course-assistant-showcase/artifacts/resource_matches.csv`
- `projects/agentic-course-assistant-showcase/artifacts/concepts/agentic_concepts.csv`
- `projects/agentic-course-assistant-showcase/artifacts/concepts/openai_vs_adk_concepts.json`
- `projects/agentic-course-assistant-showcase/artifacts/evals/agent_judge_rubric.json`
- `projects/agentic-course-assistant-showcase/artifacts/evals/concept_coverage.json`

## Optional Live Framework Extension

The live path is intentionally opt-in:

1. Run the offline smoke path and verify artifacts.
2. Install only the optional SDK extra you need.
3. Configure credentials outside source control.
4. Run the optional reference module locally.
5. Compare live behavior with the deterministic trace.

OpenAI Agents SDK:

```bash
cd projects/agentic-course-assistant-showcase
make sync-openai
```

Google ADK:

```bash
cd projects/agentic-course-assistant-showcase
make sync-adk
uv run adk run adk_course_assistant
```

Do not add live SDK execution to default CI. The public classroom contract is the offline harness and its artifact verifier.

## Suggested Reflection Prompts

- Which parts of the assistant should remain deterministic even after adding a hosted model?
- What does the trace prove that the final answer alone does not prove?
- When should a specialist be a handoff instead of a tool?
- Which guardrail belongs in code rather than a prompt?
- What would convince you that memory improved the experience without leaking private data?
- What is the smallest eval dataset that would catch a routing regression?
- Which live SDK feature is worth adding first, and which should stay out of the first build?
