# Policy Export And Agent Bridge

The export story in this project is intentionally modest.

We do not claim that a trained policy should replace the entire assistant.

Instead, the export artifacts show how the learned controller can plug back into a deterministic assistant as a small decision module. That is why the bridge artifacts are simple:

- `artifacts/bridge/policy_router.json`
- `artifacts/bridge/action_mapping.md`

Those files answer a practical question:

> if the learned policy says "give_hint" or "slow_down_and_rephrase", what does the assistant actually do with that?

That is a much healthier deployment boundary than letting the policy write the answer text itself.

It is also a deliberately modest export boundary:

- `policy_router.json` is an assistant-side action contract,
- it does **not** contain learned weights,
- it does **not** claim to be a serialized champion policy ready for deployment.

## Component Ownership

Use this table when students ask where the "agent" stops and where the "learning" starts.

| Component | Owned by | Why |
|---|---|---|
| Tool wiring, handoffs, sessions, traces | Agent framework such as the OpenAI Agents SDK | Those are runtime orchestration concerns. |
| Student-question classification and safe workflow scaffolding | Deterministic assistant or framework logic | This keeps the control problem small enough to inspect. |
| Next intervention choice | Learned policy | This is the decision surface the simulator actually optimizes. |
| Final answer wording | Deterministic or model-backed assistant | This project does not train answer generation. |
| Deployment/governance decision | Offline evaluation and human review | Reward alone is not a launch decision. |

## What Counts As A Learning Agent Here

In this repo, a learning agent means the behavior policy is learned from interaction or simulated feedback.

That means:

- the optional OpenAI Agents SDK example in the sibling project is **not** a learning agent by itself,
- this adaptive showcase **is** a learning-agent story for intervention selection,
- DQN and PPO are optional DRL comparison baselines around the same boundary,
- MARL is intentionally out of scope because it introduces multiple learning policies, non-stationarity, and coordination concerns that would obscure the core lesson.

## The Canonical Story Artifact

Run `make run-learning-agent-story` or `make run`, then read:

- `artifacts/bridge/learning_agent_story.md`

That artifact walks through one fixed tutoring question and shows:

1. what the deterministic assistant already handled,
2. what state the learned controller saw,
3. which intervention the policy chose,
4. how an SDK agent would execute that choice without pretending the SDK itself was trained.
