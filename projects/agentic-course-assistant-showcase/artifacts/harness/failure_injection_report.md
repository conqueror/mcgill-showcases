# Harness Failure Injection Report

Question: Help me build an agent project for debugging leakage without pasting any API keys.

Run ID: 6a17318eb67930dfb6a041f977bde5c675c97ed451872301507aa0b5cdda72f2

## Executed Boundary Checks

The tool and trace checks inject corrupt outputs into the artifact verifier. The loop check injects an excessive round count into the workflow judge. The policy check sends a flagged prompt through the assistant, and the routing check exercises a keyword tie.

- Tool failure: `pass`; observed: ["agent_trace.json resource_ids must be a non-empty list", "resource_matches.csv must contain at least one resource row"]
- Trace corruption: `pass`; observed: ["agent_trace.json trace missing step for tool_call"]
- Guardrail trip: `pass`; observed: "question contains sensitive terms; use a public example instead"
- Routing ambiguity: `pass`; observed: "Tied project/debug keywords selected debug."
- Loop runaway: `pass`; observed: {"bounded_loop_completed": false, "state_present": true, "stop_reason_present": true, "summary_present": true, "trace_present": true}

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
