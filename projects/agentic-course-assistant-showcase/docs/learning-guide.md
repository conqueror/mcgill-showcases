# Learning Guide

## The Core Idea

An agent framework is useful when a model-driven workflow needs clear ownership of steps: routing, tool use, specialist behavior, guardrails, and traceability. This showcase keeps those ideas visible without requiring an external model call for the default run.

## What To Inspect First

1. Run `make smoke`.
2. Open `artifacts/course_assistant_response.md`.
3. Find the selected intent and specialist.
4. Open `artifacts/agent_trace.json`.
5. Match each trace step to the source code in `src/agentic_course_assistant/assistant.py`.
6. Open `artifacts/concepts/agentic_concepts.csv`.
7. Open `artifacts/evals/agent_judge_rubric.json`.

## How To Use The Concept Atlas

Each row in `agentic_concepts.csv` answers four practical questions:

- What does the concept mean in plain engineering terms?
- How does OpenAI Agents SDK express it?
- How does Google ADK express it?
- What small part of this showcase lets a student build or test it?

Start with the requested core concepts: tools, guardrails, tracing, workflows,
evals, agent-as-judge, multi-agent orchestration, handoff/triage, A2A, sessions,
memory, skills, and harness evidence. Then continue into the "what else?" concepts:
artifacts, callbacks, plugins, schemas, auth, privacy, errors, budgets, deployment,
streaming, versioning, rollback, grounding, and retention.

## Interpretation Prompts

- Which part of the workflow is deterministic?
- Which part would become model-driven in a hosted SDK?
- What does the resource lookup tool guarantee?
- What does the guardrail catch, and what does it not catch?
- How would you test a new specialist before adding a real LLM?
- When should a specialist be a handoff instead of an agent-as-tool?
- What evidence would convince you that memory helped rather than leaked data?
- Which concept from the atlas should remain out of the first build, and why?

## Extension Exercise

Add a new `assessment` route for questions about quizzes and grading rubrics.

Keep the extension small:

- add route keywords,
- add one specialist name,
- add one resource,
- add or update one test,
- run `make check`,
- run `make smoke`.
