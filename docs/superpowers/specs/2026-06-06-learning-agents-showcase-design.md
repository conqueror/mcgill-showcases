# Product Spec: Learning Agents Showcase

Date: 2026-06-06
Project dir: `projects/learning-agents-showcase`
Input: "Make the agentic course-assistant learn with DRL; multi-agent RL; an OpenAI Agents SDK
agent that learns with DRL." (Design analysis lives in this repo's conversation; this spec is the
agreed scope.)

## Locked decisions (from planning)
- **Name:** `learning-agents-showcase`.
- **Independence:** **fully self-contained.** It vendors its own RL algorithm library (ported from
  the proven `student-support-rl` modules as a *correctness reference*, but with **no import
  dependency**) and its own **judge-rubric reward module** (modeled on the course-assistant's
  rubric, copied in — not imported). No code dependency on any other showcase; conceptual references
  in docs only. ("Independent" = no cross-showcase coupling; optional external libs like the
  `openai-agents` SDK are still allowed in gated lanes.)
- **Reward source:** a **judge-rubric reward** (a self-contained, multi-criterion rubric), with a
  deliberately-hackable variant for the reward-hacking lesson.
- **Scope:** the core orchestration-policy lane is the current deliverable. The SDK bridge,
  RLHF-concepts lane, and MARL lane are staged future extensions for this separate capstone.

## Vision
A self-contained capstone about **where learning lives in an agent** and how to make an agent
**learn its decisions** with RL/DRL by learning the **orchestration policy**: routing, tool choice,
escalation, and stop decisions. The core stays deterministic and laptop-friendly. The optional
future lanes cover OpenAI Agents SDK integration, RLHF/DPO/GRPO ideas at toy scale, and simulated
MARL. The claims stay narrow on purpose.

## The central teaching idea (the "locus of learning")
| Locus | What is optimized | Methods | This showcase |
|---|---|---|---|
| **A. Orchestration policy** | the agent's discrete control decisions | bandits, Q-learning/SARSA, DQN/PPO, offline RL | **Core — built for real** |
| **B. LLM weights** | the token policy | RLHF-PPO, DPO, GRPO, RLAIF, RLVR | **Concepts + toy loop** |
| **C. Multi-agent coordination** | sub-agent policies that co-adapt | MARL: IQL, MAPPO, COMA, QMIX | **Simulated lane** |

## User Stories
1. As an RL-track grad, I want to apply bandits/Q-learning/DQN/PPO to a *new* domain (agent
   decisions) so that I practice transfer, not memorization.
2. As an agentic-track grad, I want to watch a hand-written router become a *learned* policy so that
   I connect agent architecture to RL.
3. As a learner confused by "RL for LLMs," I want a clear taxonomy of *where* learning happens so
   that I can separate hype from reality.
4. As an instructor, I want a deterministic, laptop/CI-friendly core plus clearly-gated optional
   lanes so that I can teach without GPUs/keys yet still show the real SDK/RLHF/MARL ideas.
5. As a practitioner, I want to learn an agent policy *offline from logged traces* and produce a
   deploy/shadow/reject recommendation so that I see the production-realistic loop.

## Features

### Core (Lane 1) — learn the orchestration policy
- **Agent-decision simulator** — seeded; state = request/context features; actions = route / call-tool
  / answer / ask-clarifying-Q / escalate / stop; reward = judge-rubric score − cost − safety penalty.
- **Judge-rubric reward** — self-contained multi-criterion rubric + a hackable variant.
- **Vendored RL library** — contextual bandit, Q-learning, SARSA, dynamic-programming optimum,
  policy gradient, evaluation, reporting (ported, self-contained, tested).
- **Contextual-bandit routing** with regret vs oracle.
- **Sequential control** (Q-learning/SARSA/DQN/PPO) incl. the stop decision; learned-vs-handwritten-
  vs-oracle comparison; exact DP ground truth.
- **Reward-hacking report** (agent games the hackable rubric).
- **Offline evaluation + deploy/shadow/reject governance** (multi-objective: quality/cost/safety).

### Enhanced
- **Offline RL from logged traces** + off-policy-evaluation honesty.
- **Cost-sensitive cascade** (cheap→escalate).
- Exercises (with solutions), glossary, math notes.

### Optional gated lanes (future extensions for this separate capstone)
- **Lane A — OpenAI Agents SDK bridge:** instrument a real SDK agent (handoffs/tools/guardrails/
  tracing), collect traces, learn a routing policy offline, inject it at the handoff decision,
  compare vs SDK default. Deterministic fallback (sample traces + simulator) when no key.
- **Lane B — RLHF/DPO/GRPO concepts (toy):** preference data → small reward model → policy-gradient/
  GRPO-style update on a tiny softmax policy; show KL-to-reference, reward over-optimization,
  sycophancy hacking. Pure-CPU; explicitly not a real LLM fine-tune.
- **Lane C — MARL (simulated):** planner/solver/critic co-adapt under cooperative reward; independent
  learners (non-stationarity) vs centralized critic (credit assignment).

These lanes belong here only because this project is already the separate capstone. They do not
belong in `adaptive-course-assistant-rl-showcase`, which stays focused on one learned controller
around a deterministic assistant.

## Architecture (high level only)
- **Stack:** Python 3.11+, `uv`, numpy/pandas; deterministic-by-seed; script-first; artifact
  contract; Makefile gates; no GPU/network on the core path. Optional gated extras: `openai-agents`
  (Lane A); pure-numpy tiny policy preferred for Lane B; none for Lane C.
- **Key components (conceptual):** agent-decision environment; vendored RL library; judge-rubric
  reward (+ hackable variant); offline-RL + OPE; optional SDK/RLHF/MARL lanes; reporting/governance;
  layered docs.
- **Data model (entities):** State (context features), Action (discrete decision), Trajectory/Trace
  (logged run), Reward spec, Policy, Evaluation rows, Artifact contract.

## Scope Boundaries
- **In scope now:** the core lane, fully self-contained, with deterministic offline evaluation and
  honest governance around learned orchestration decisions.
- **Future scope:** the optional SDK, RLHF-concepts, and MARL lanes in this separate capstone.
- **Out of scope:** real GPU LLM fine-tuning; online RL on live LLM calls in the core; production
  deployment; large benchmarks; full MARL on live LLMs.
- **Future:** heavier SDK integration; real offline-RL libraries; verifiable-reward tool tasks;
  competitive self-play; the staged SDK, RLHF, and MARL lanes described above.

## Open Questions (remaining, minor)
1. Lane B implementation: pure-numpy tiny policy (default — zero deps, deterministic) vs a small
   torch model. — default to pure-numpy.
2. Lane A CI determinism: confirm the sample-trace + simulator fallback fully covers CI. — default
   yes; SDK live path is opt-in only.
3. How literally to mirror the course-assistant's rubric criteria in the vendored judge. — default:
   model it (intent-match, groundedness, cost, safety), not copy verbatim.

## Spec self-review
- **Ambition:** ship the core orchestration-policy lesson first, then leave room for later SDK,
  RLHF, and MARL extensions in this separate capstone.
- **Level:** "what/why" only; implementation specifics live in the plan.
- **Cascade:** independence + reward-source + determinism are now *decided*, not floating.
- **Coherence:** every lane serves one vision — teach where learning lives in an agent, honestly.
