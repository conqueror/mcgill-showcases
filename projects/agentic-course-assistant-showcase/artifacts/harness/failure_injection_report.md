# Harness Failure Injection Report

Question: I understand train test split basics, but I am confused about data leakage in feature engineering.

## Simulated Failures

- Tool failure: the course catalog returns no matches, so the verifier should reject an empty resource list.
- Guardrail trip: a prompt includes an API key or secret, so the policy agent blocks the request.
- Routing ambiguity: a question mixes project planning and debugging, so the router trace must show the selected specialist.
- Loop runaway: a refinement loop must stop after the bounded quality threshold instead of iterating forever.
- Trace corruption: a missing trace event should fail `make trace-check` before a student trusts the answer.

## Current Judge Verdicts

- `sequential_course_plan`: `pass`
- `loop_refinement`: `pass`
- `parallel_resource_review`: `pass`
- `router_triage`: `pass`
- `custom_policy_agent`: `pass`
- `route_project_agent`: `pass`
- `block_secret_request`: `pass`
- `sequential_plan_resources`: `pass`
- `parallel_review_count`: `pass`
- `bounded_loop`: `pass`
