# Lane A: The Agent Framework as Environment (OpenAI Agents SDK)

This page is the first of three "locus of learning" lanes. Each lane fixes a different answer to one question: *when we say an agent "learns", which component actually changes?* Lane A's answer is the most surprising to newcomers, because it puts the learning somewhere most people would not look.

## Thesis: RL learns the orchestration policy, not the framework

An LLM-powered agent has several places where learning *could* live:

- the **orchestration policy** — which tool to call, when to ask a clarifying question, when to hand off to a human;
- the **LLM weights** — the parameters of the underlying language model;
- the **multi-agent coordination** — how several agents divide and combine work.

Lane A isolates the first one. Here, reinforcement learning learns the **orchestration policy** `pi(a|s)`: a mapping from the agent's current state `s` to a probability distribution over routing actions `a`. The local simulator represents the **environment, executor, and logger** and labels actions with SDK constructs. The optional SDK builder is a separate construction example; it does not connect the learned policy to live execution.

The split is worth stating bluntly, because the framework's marketing rarely does:

```text
learned policy pi(a|s)   decides which action to take
local simulator         executes that action and records an SDK-labelled trace
RL training loop         improves pi(a|s) from the recorded rewards
```

In the offline loop, the learned policy chooses an action and the simulator applies it and logs its SDK label. The intended architectural split is that a framework executes a policy rather than training it. Connecting that policy to a real SDK runtime still requires caller implementation.

The implementation lives in `src/learning_agents/sdk_bridge.py`, and the rest of the `learning_agents` package is the RL that produces the policy the offline bridge runs.

## The action-to-SDK-construct mapping

The bridge between "an RL action" and "a thing an agent framework does" is a small crosswalk. The MDP has four discrete actions (defined in `src/learning_agents/environment.py` as `ACTION_LABELS`), and each maps to exactly one SDK construct. From the `SDK_CONSTRUCT_BY_ACTION` table in `src/learning_agents/sdk_bridge.py`:

| Action `a` | SDK role | SDK target | What the framework does |
| --- | --- | --- | --- |
| `answer_direct` | `final_output` | `assistant_answer` | Emit the final answer; the run completes. |
| `retrieve` | `tool_call` | `retrieve_context` | Call the retrieval function tool to gather grounding. |
| `clarify` | `tool_call` | `ask_clarifying_question` | Call the clarification function tool. |
| `escalate` | `handoff` | `human_specialist` | Hand off to a human-specialist agent. |

Read this table as the *type signature* of the interface. The learned policy emits an integer in `{0, 1, 2, 3}`; the framework interprets that integer as one of three SDK primitives — a **final output** (the agent's answer, which ends the run), a **function-tool call** (retrieve or clarify), or a **handoff** (escalate to a different agent). Two distinct actions, `retrieve` and `clarify`, both surface as tool calls but to *different* tools, which is why the mapping carries a separate `sdk_target` alongside the `sdk_role`.

The crosswalk labels action `3` as a `handoff` to `human_specialist` in a local trace. It does not execute an SDK handoff. The optional builder's `human_specialist` is another software agent, not a human approval mechanism.

## How the policy drives the loop

Here is the control flow, with the responsibilities color-separated. The policy is the only learning component; everything to its right is the framework doing as it is told.

```mermaid
flowchart LR
    S["State s<br/>(evidence, difficulty,<br/>ambiguity, step)"] --> P{"Learned policy<br/>pi(a|s)<br/>chooses action a"}
    P -->|"a = answer_direct"| F["final_output<br/>(assistant_answer)"]
    P -->|"a = retrieve / clarify"| T["tool_call<br/>(retrieve_context /<br/>ask_clarifying_question)"]
    P -->|"a = escalate"| H["handoff<br/>(human_specialist)"]
    F --> X["Local simulator<br/>applies the action"]
    T --> X
    H --> X
    X --> L["Logger writes one<br/>trace row: step, action,<br/>sdk_role, reward, terminal"]
    L -->|"next state s'"| S
```

Two layers make this real, deliberately separated by *what can run offline*:

1. **The pure-Python demonstration** — `run_bridged_episode` in `src/learning_agents/sdk_bridge.py`. It rolls the learned policy out inside this package's own `AgentDecisionEnvironment` and annotates each step with the SDK construct it would represent. This is a simulator rollout, not an SDK run. No SDK and no network are required.
2. **The optional SDK construction example** — `build_agents_sdk_agent` in the same module. It constructs an `agents.Agent` with stub retrieval/clarification tools and a software-agent handoff. It does not receive the learned policy or route SDK execution through it.

The state `s` the offline policy reasons over is the MDP state from `src/learning_agents/environment.py`: how much evidence has been gathered, the request's difficulty and ambiguity, and the step index. The local simulator applies the selected action; the SDK builder receives neither this state nor a learned action.

## The gated optional dependency

A core design constraint of this showcase is that it runs **offline, with no SDK and no network, by default**. The live-SDK path is hidden behind an optional dependency so the core never requires it.

The gate works in two pieces, both in `src/learning_agents/sdk_bridge.py`:

- `sdk_available()` is a cheap, import-free probe. It checks only the module spec (`importlib.util.find_spec("agents")`), so it can report whether the live path is enabled without importing or running anything.
- `build_agents_sdk_agent` raises `OptionalSDKError` when the SDK is absent, instructing the caller to run `uv sync --extra sdk` (which adds the `openai-agents` package). Callers that want the real `agents.Agent` catch this and fall back to the offline `run_bridged_episode`.

So the two paths are:

```text
default:           offline demonstration; no SDK, no network
uv sync --extra sdk:  agents.Agent construction example available
```

Even with the extra installed, *constructing* the agent needs no network; only *running* it against a model does, and that is intentionally left to the caller. In the environment that generated the artifacts here, the SDK is **not installed**, so the offline demonstration is what produced the trace below (see `artifacts/sdk_bridge/bridge_report.md`, "Live SDK status").

You can learn and evaluate the entire policy with the SDK absent. Supplying that policy to a live runtime requires integration beyond the provided builder.

## The orchestration trace

`run_bridged_episode` returns one row per decision step, and the runner writes them to `artifacts/sdk_bridge/orchestration_trace.csv`. The columns are `step`, `scenario_name`, `action_label`, `sdk_role`, `sdk_target`, `reward`, and `terminal`. This is a local simulator trace with SDK labels; equivalence to the native SDK trace format has not been verified.

The recorded trace covers five scenarios:

| step | scenario | action | SDK role | reward | terminal |
| --- | --- | --- | --- | --- | --- |
| 0 | `easy_factual` | `answer_direct` | `final_output` | 2.0 | yes |
| 0 | `howto_medium` | `retrieve` | `tool_call` | -0.5 | no |
| 1 | `howto_medium` | `answer_direct` | `final_output` | 2.0 | yes |
| 0 | `ambiguous_query` | `escalate` | `handoff` | 0.45 | yes |
| 0 | `hard_debug` | `escalate` | `handoff` | 0.45 | yes |
| 0 | `needs_escalation` | `escalate` | `handoff` | 0.9 | yes |

Read the trace as a sequence of `(s, a, r)` tuples. The easy factual question is answered in one step for reward `2.0`. The medium how-to takes two steps: a `retrieve` tool call that costs `-0.5` (gathering evidence is not free), followed by an `answer_direct` worth `2.0`. The three harder scenarios all `escalate`, and notice the rewards differ — escalation earns only `0.45` when a cheaper action might have sufficed, but `0.9` on `needs_escalation`, the one scenario where a handoff is genuinely the right call. That reward gradient is precisely the signal RL uses to learn *when* escalation pays off.

## The bridge report

`bridge_report_markdown` renders the narrative artifact at `artifacts/sdk_bridge/bridge_report.md`. It is generated, not hand-written, so it cannot drift from the code: it restates the thesis, prints the action-to-SDK-construct mapping straight from `action_tool_mapping()`, and records the live-SDK status for the current environment (here: not installed, with the `uv sync --extra sdk` instruction to enable it). If you change the crosswalk in `SDK_CONSTRUCT_BY_ACTION`, regenerating the report updates the table automatically.

## Which policy should the framework run?

Lane A defines the *interface*; the rest of the showcase decides *which* policy belongs on the other side of it. That distinction matters, because not every learned policy is fit to drive a live agent. The framework will faithfully execute whatever you hand it — including a bad policy — so the choice is a governance decision, not a framework one.

From `artifacts/eval/policy_comparison.csv` (avg reward / escalation rate / avg steps / solved rate):

| policy | avg reward | escalation rate | avg steps | solved rate |
| --- | --- | --- | --- | --- |
| `dp_optimal` | 1.2142 | 0.2833 | 2.05 | 1.0 |
| `offline_fqi` | 1.2067 | 0.30 | 2.0 | 1.0 |
| `heuristic_router` | 1.16 | 0.0 | 3.0667 | 1.0 |
| `q_learning` | 0.8525 | 0.65 | 1.2167 | 1.0 |
| `random` | -1.1817 | 1.0 | 3.0 | 0.5333 |

The honest tensions here are the lesson:

- `dp_optimal` is the **planning ceiling** — exact `Q*` computed by backward induction in `src/learning_agents/dynamic_programming.py`. It is what perfect knowledge of the MDP buys you.
- `offline_fqi` (offline Fitted-Q learned from the heuristic log) reaches `1.2067`, **nearly matching the ceiling** without ever interacting online.
- `q_learning` — online tabular, trained 400 episodes — scores only `0.8525` and escalates 65% of the time. It **over-escalates**, and is **governance-REJECTED** for exactly that reason. The framework would happily run it; the governance gate is what stops you.
- `random` is the floor at `-1.1817`, escalating on every request.

So the framework is policy-agnostic by design, and that is a feature with a sharp edge: the safety of a live agent is determined entirely upstream, by which policy you certify before wiring it to the executor. See [evaluation and governance](evaluation-and-governance.md) for how that gate is run, and the [RL ladder](rl-ladder.md) for how these policies are produced.

## What this lane is and is not

- It **is** a testable simulator demonstration: `run_bridged_episode` labels local actions with SDK constructs. `build_agents_sdk_agent` constructs an SDK agent with stubs; it does not execute the learned policy or prove native trace equivalence.
- The SDK example does not train a policy. The numbers on this page come from simulator evaluation, not SDK execution.
- It is **not** a benchmark of the OpenAI Agents SDK's performance. The artifacts here are generated offline, with the SDK absent, precisely to show that the policy and its evaluation do not depend on the framework being present.

## See also

- [Locus of learning](locus-of-learning.md) — the three lanes (orchestration policy, LLM weights, coordination) and why this one is Lane A.
- [Showcase architecture](showcase-architecture.md) — how `src/learning_agents/sdk_bridge.py`, the environment, and the artifact contract fit together.
- [Evaluation and governance](evaluation-and-governance.md) — how a policy is certified (or rejected) before a framework runs it.
- [The RL ladder](rl-ladder.md) — how `dp_optimal`, `offline_fqi`, and `q_learning` are produced.
