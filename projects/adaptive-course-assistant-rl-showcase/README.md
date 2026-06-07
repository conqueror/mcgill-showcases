# Adaptive Course Assistant RL Showcase

This showcase teaches one narrow but important idea: how a course assistant can keep its normal deterministic workflow, then hand just the **next intervention choice** to a learned policy.

That boundary is the whole point. We are not training a model to generate the assistant's answers from scratch. We are training a smaller controller that decides whether the assistant should clarify, retrieve, hint, slow down, assign practice, check understanding, or escalate.

## Learning Outcomes

By the end of this project, you should be able to:

- explain why the first intervention choice can be treated as a contextual bandit,
- explain why multi-turn tutoring needs an MDP,
- define a usable tutoring state, action space, reward, and horizon,
- compare a rule-based baseline with Q-learning, SARSA, REINFORCE, and the optional DQN/PPO bridge,
- explain the progression from Q-learning to DQN to policy gradients to actor-critic to PPO,
- explain how a learned intervention policy could sit underneath an agent framework such as the OpenAI Agents SDK without claiming the framework agent itself is being trained,
- read the artifact trail and decide whether a learned tutoring policy is safe enough even for shadow mode.

## Prerequisites

- Python 3.11+
- `uv`
- Basic Python and CSV reading
- Helpful, but not required: `projects/agentic-course-assistant-showcase`
- Helpful, but not required: `projects/student-support-rl-showcase`

## Quickstart

```bash
cd projects/adaptive-course-assistant-rl-showcase
make sync
make smoke
make verify
```

Run the full core path:

```bash
make run
```

Run the optional DRL comparison:

```bash
make sync-drl
make run-drl-optional
make verify-full
```

Run checks:

```bash
make check
```

## What Makes This Showcase Different

`agentic-course-assistant-showcase` already explains routing, tools, traces, and agent-framework concepts.

`student-support-rl-showcase` already teaches the broad RL ladder in a general student-support domain.

This project sits between them. It asks a more specific question: once a deterministic assistant already understands the request, **what should it do next?**

## What This Project Does And Does Not Claim

- `yes`: this is a standalone showcase for a bounded learning-agent story where the learned part chooses the next pedagogical intervention.
- `yes`: the project uses contextual bandits, tabular RL, and an optional DRL bridge with DQN and PPO.
- `no`: the optional OpenAI Agents SDK example in the sibling project is not itself being trained here.
- `no`: this project does not claim a full end-to-end DRL agent that learns answer generation.
- `no`: this is not MARL. There is one learned controller in the simulator, not multiple learning agents with coordination or competition.

## Key Artifacts

After `make run`, inspect:

- `artifacts/concepts/mdp_spec.md`
- `artifacts/concepts/algorithm_progression.md`
- `artifacts/concepts/state_action_reward_schema.csv`
- `artifacts/concepts/agentic_rl_bridge.md`
- `artifacts/concepts/interpretation_prompts.md`
- `artifacts/assistant/episode_trace.json`
- `artifacts/assistant/resource_matches.csv`
- `artifacts/bandit/contextual_policy_metrics.csv`
- `artifacts/bandit/regret_trace.csv`
- `artifacts/bandit/action_breakdown.csv`
- `artifacts/mdp/sample_episodes.csv`
- `artifacts/policy/rule_policy_summary.csv`
- `artifacts/policy/intervention_decisions.csv`
- `artifacts/q_learning/training_curve.csv`
- `artifacts/q_learning/q_table.csv`
- `artifacts/sarsa/training_curve.csv`
- `artifacts/policy_gradient/training_curve.csv`
- `artifacts/eval/offline_policy_eval.csv`
- `artifacts/eval/scenario_rollups.csv`
- `artifacts/eval/safety_summary.md`
- `artifacts/reward/reward_hacking_report.md`
- `artifacts/bridge/policy_router.json`
- `artifacts/bridge/action_mapping.md`
- `artifacts/bridge/learning_agent_story.md`
- `artifacts/business/deployment_recommendation.md`
- `artifacts/manifest.json`

After `make run-drl-optional`, inspect:

- `artifacts/drl_optional/dqn_training_summary.csv`
- `artifacts/drl_optional/ppo_training_summary.csv`
- `artifacts/drl_optional/rl_family_comparison.csv`
- `artifacts/drl_optional/scenario_rollups.csv`
- `artifacts/drl_optional/policy_gradient_bridge_notes.md`

Use this table when reading the most important outputs:

| Artifact | Question it answers | What to inspect | Do not overclaim |
|---|---|---|---|
| `artifacts/bandit/contextual_policy_metrics.csv` | Which first intervention works for each context? | `scenario_name`, `action`, `reward`, `optimal_action` | A bandit does not model future tutoring turns. |
| `artifacts/q_learning/q_table.csv` | What action value did tabular Q-learning learn for a state? | decoded state columns, `action`, `q_value` | The table chooses interventions; it does not write answer text. |
| `artifacts/policy_gradient/training_curve.csv` | Did direct policy optimization improve returns? | `episode`, `total_reward`, `baseline` | REINFORCE is policy gradient, not actor-critic. |
| `artifacts/eval/offline_policy_eval.csv` | How do policies behave in local simulator replay? | `policy`, `avg_reward`, `solved_rate`, safety columns | This is not formal off-policy evaluation over logged production data. |
| `artifacts/bridge/policy_router.json` | What can an agent framework safely call? | `decision_boundary`, `allowed_actions`, export flags | It is a routing contract, not learned model weights. |
| `artifacts/drl_optional/rl_family_comparison.csv` | How do tabular value learning, DQN, and PPO compare on the same simulator? | `family`, reward, solved rate, safety columns | Better simulator reward is not deployment readiness. |

## Reading Order

1. Start with [docs/00-start-here.md](docs/00-start-here.md).
2. Read [docs/system-boundary.md](docs/system-boundary.md) so the project scope is clear.
3. Read [docs/state-action-reward.md](docs/state-action-reward.md) before looking at training curves.
4. Read [docs/algorithm-ladder.md](docs/algorithm-ladder.md) while comparing Q-learning, DQN, and PPO.
5. Read `artifacts/bridge/learning_agent_story.md` so the agent-framework boundary becomes concrete.
6. Finish with [docs/policy-export-and-agent-bridge.md](docs/policy-export-and-agent-bridge.md) and [docs/evaluation-and-governance.md](docs/evaluation-and-governance.md).

## Common Failure Modes

- Treating the learned policy as if it writes the whole assistant response.
- Looking only at reward and missing safety or grounding problems.
- Forgetting that a contextual bandit sees only the first decision, not a whole episode.
- Letting the optional DRL path overshadow the simpler, more readable tabular path.
- Reusing the same domain or artifact structure in coursework instead of transferring the pattern.

## Suggested Next Projects

- `../agentic-course-assistant-showcase/README.md`
- `../student-support-rl-showcase/README.md`
- `../model-release-rollout-showcase/README.md`

## If You Want The OpenAI Agents SDK Version To Be A Learning Agent

Use the sibling `agentic-course-assistant-showcase` to learn the runtime pieces: tools, handoffs, traces, and optional hosted execution.

Use this project to learn the policy pieces: state, action, reward, tabular RL, and the optional DQN/PPO bridge.

The honest way to combine them is:

1. let the SDK agent own orchestration,
2. let the learned policy return only the next intervention,
3. keep evaluation and governance outside the final answer text.

That is exactly what `artifacts/bridge/learning_agent_story.md` is trying to make visible.

## Project Structure

```text
adaptive-course-assistant-rl-showcase/
├── README.md
├── Makefile
├── pyproject.toml
├── docs/
├── scripts/
├── src/adaptive_course_assistant_rl/
├── tests/
└── artifacts/
```
